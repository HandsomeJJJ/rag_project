import hashlib
import json
import os
import threading
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import config
from infra.model_factory import build_embeddings
from ingestion.legal_chunker import build_chunks_from_article_units, parse_legal_article_units
from ingestion.legal_preprocess import preprocess_legal_text

ProgressCallback = Callable[[str, int, str], None]
_REGISTRY_LOCK = threading.Lock()

# 由于 DashScope 对单次写入 Chroma 的文本数量有限制（上限为 10），因此需要分批写入。
def add_texts_in_batches(
	chroma,
	texts,
	metadatas,
	batch_size: int = 10,
	progress_cb: Callable[[int, int], None] | None = None,
):
	"""按 DashScope 允许的上限分批写入 Chroma。"""
	if len(texts) != len(metadatas):
		raise ValueError("texts 和 metadatas 数量必须一致")

	for start in range(0, len(texts), batch_size):
		end = start + batch_size
		chroma.add_texts(
			texts[start:end],
			metadatas=metadatas[start:end],
		)
		if progress_cb is not None:
			progress_cb(min(end, len(texts)), len(texts))

# 通过 MD5 校验文本内容是否已经存在知识库中，避免重复入库。
def check_md5(md5_str: str):
	# 如果不存在，创建一个新的 MD5 文件，并返回 False，表示内容未入库。
	if not os.path.exists(config.md5_path):
		open(config.md5_path, "w", encoding="utf-8").close()
		return False

	for line in open(config.md5_path, "r", encoding="utf-8").readlines():
		line = line.strip()
		if line == md5_str:
			return True
	return False

# 将新的 MD5 值追加写入文件，记录已经入库的文本内容。
def save_md5(md5_str: str):
	with open(config.md5_path, "a", encoding="utf-8") as f:
		f.write(md5_str + "\n")

# 计算输入字符串的 MD5 值，用于内容去重校验。
def get_string_md5(input_str: str, encoding="utf-8"):
	str_bytes = input_str.encode(encoding=encoding)
	md5_obj = hashlib.md5()
	md5_obj.update(str_bytes)
	return md5_obj.hexdigest()


def _remove_md5_line(md5_hex: str) -> None:
	path = Path(config.md5_path)
	if not path.exists() or not md5_hex:
		return
	lines = [
		line.strip() for line in path.read_text(encoding="utf-8").splitlines()
		if line.strip() and line.strip() != md5_hex
	]
	path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_registry() -> dict[str, dict]:
	path = Path(config.document_registry_path)
	if not path.exists():
		return {}
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return {}
	documents = data.get("documents") if isinstance(data, dict) else None
	return documents if isinstance(documents, dict) else {}


def _save_registry(documents: dict[str, dict]) -> None:
	path = Path(config.document_registry_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps({"documents": documents}, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)


def extract_text_from_file(filename: str, raw: bytes) -> str:
	"""把上传文件解析为纯文本，支持 TXT / Markdown / PDF / DOCX。"""
	suffix = Path(filename).suffix.lower()
	if suffix in {".txt", ".md"}:
		for encoding in ("utf-8-sig", "gb18030"):
			try:
				return raw.decode(encoding)
			except UnicodeDecodeError:
				continue
		raise ValueError("文件编码无法识别，请使用 UTF-8 或 GB18030 编码")
	if suffix == ".pdf":
		try:
			from pypdf import PdfReader
		except ImportError as exc:
			raise RuntimeError("缺少 pypdf 依赖，无法解析 PDF") from exc
		reader = PdfReader(BytesIO(raw))
		return "\n".join((page.extract_text() or "") for page in reader.pages)
	if suffix == ".docx":
		try:
			from docx import Document as DocxDocument
		except ImportError as exc:
			raise RuntimeError("缺少 python-docx 依赖，无法解析 DOCX") from exc
		document = DocxDocument(BytesIO(raw))
		return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
	raise ValueError(f"不支持的文件格式：{suffix or '未知'}")


def _fallback_chunks(text: str) -> tuple[list[str], list[dict]]:
	"""非法律结构文本使用通用递归分块，保证 Markdown/PDF 等也能入库。"""
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=config.chunk_size,
		chunk_overlap=config.chunk_overlap,
		separators=["\n\n", "\n", "。", "！", "？", "；", " "],
	)
	chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	metadatas = [
		{
			"source": "",
			"create_time": now,
			"operator": "",
			"part": "",
			"chapter": "",
			"section": "",
			"article_no": "",
			"article_end": "",
			"chunk_article_end": "",
			"chunk_index": index,
		}
		for index, _ in enumerate(chunks)
	]
	return chunks, metadatas


class KnowledgeBaseService:
	"""知识库入库与文档管理服务。"""

	def __init__(self):
		os.makedirs(config.persist_directory, exist_ok=True)
		self.chroma = self._build_chroma()
		self.chunk_size = config.chunk_size
		self.chunk_overlap_articles = config.chunk_overlap_articles

	def _build_chroma(self) -> Chroma:
		return Chroma(
			collection_name=config.collection_name,
			embedding_function=build_embeddings(),
			persist_directory=config.persist_directory,
		)

	@staticmethod
	def _emit(progress_cb: ProgressCallback | None, stage: str, percent: int, message: str) -> None:
		if progress_cb is not None:
			progress_cb(stage, max(0, min(100, percent)), message)

	def _delete_source_vectors(self, source: str) -> int:
		"""按 source 元数据删除向量，返回删除的分块数。"""
		try:
			data = self.chroma._collection.get(where={"source": source}, include=["metadatas"])  # noqa: SLF001
			ids = list(data.get("ids") or [])
			if ids:
				self.chroma._collection.delete(ids=ids)  # noqa: SLF001
			return len(ids)
		except Exception:
			return 0

	def _registered_md5(self, source: str) -> str:
		return str((_load_registry().get(source) or {}).get("md5") or "")

	def upload_by_str(
		self,
		data: str,
		filename: str,
		progress_cb: ProgressCallback | None = None,
	):
		filename = Path(filename).name
		data = preprocess_legal_text(data)
		if not data.strip():
			return "[Warn] 文件内容为空或未解析到有效文本"

		self._emit(progress_cb, "parsing", 5, "正在校验文件内容…")
		md5_hex = get_string_md5(data)
		if check_md5(md5_hex):
			return "[Repeat] 内容已存在知识库"

		registry = _load_registry()
		existing = registry.get(filename)
		if existing and existing.get("md5") and existing.get("md5") != md5_hex:
			# 同名文件内容更新时，先移除旧分块，避免检索到过期法条
			self._delete_source_vectors(filename)
			_remove_md5_line(str(existing["md5"]))

		self._emit(progress_cb, "chunking", 12, "正在解析法律结构并分块…")
		article_units = parse_legal_article_units(data)
		knowledge_chunks, metadata_list = build_chunks_from_article_units(
			article_units=article_units,
			max_chars=self.chunk_size,
			overlap_articles=self.chunk_overlap_articles,
		)
		if not knowledge_chunks:
			knowledge_chunks, metadata_list = _fallback_chunks(data)
		if not knowledge_chunks:
			return "[Warn] 未解析到可入库的内容"

		for metadata in metadata_list:
			metadata["source"] = filename

		self._emit(
			progress_cb,
			"embedding",
			20,
			f"共 {len(knowledge_chunks)} 个分块，正在向量化并写入…",
		)

		def batch_progress(done: int, total: int) -> None:
			percent = 20 + int(done / max(total, 1) * 75)
			self._emit(progress_cb, "embedding", percent, f"已写入 {done}/{total} 个分块")

		add_texts_in_batches(
			self.chroma,
			knowledge_chunks,
			metadata_list,
			batch_size=10,
			progress_cb=batch_progress,
		)
		save_md5(md5_hex)

		article_numbers = [str(meta.get("article_no") or "") for meta in metadata_list if meta.get("article_no")]
		with _REGISTRY_LOCK:
			registry = _load_registry()
			registry[filename] = {
				"md5": md5_hex,
				"chunks": len(knowledge_chunks),
				"chars": len(data),
				"create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				"first_article": article_numbers[0] if article_numbers else "",
				"last_article": article_numbers[-1] if article_numbers else "",
				"extension": Path(filename).suffix.lower(),
			}
			_save_registry(registry)

		self._emit(progress_cb, "done", 100, "入库完成")
		return f"[Success] 已入库 {len(knowledge_chunks)} 个分块（{len(data)} 字）"

	def list_documents(self) -> list[dict]:
		"""返回知识库文档列表，兼容没有注册表的历史文档。"""
		registry = _load_registry()
		documents: list[dict] = []
		seen: set[str] = set()
		for source, info in registry.items():
			documents.append(
				{
					"source": source,
					"chunks": int(info.get("chunks") or 0),
					"chars": int(info.get("chars") or 0),
					"create_time": str(info.get("create_time") or ""),
					"first_article": str(info.get("first_article") or ""),
					"last_article": str(info.get("last_article") or ""),
					"extension": str(info.get("extension") or Path(source).suffix.lower()),
					"registered": True,
				}
			)
			seen.add(source)

		try:
			data = self.chroma._collection.get(include=["metadatas"])  # noqa: SLF001
			counts = Counter(
				str(meta.get("source") or "")
				for meta in (data.get("metadatas") or [])
				if isinstance(meta, dict) and meta.get("source")
			)
			for source, count in counts.items():
				if source in seen:
					continue
				documents.append(
					{
						"source": source,
						"chunks": count,
						"chars": 0,
						"create_time": "",
						"first_article": "",
						"last_article": "",
						"extension": Path(source).suffix.lower(),
						"registered": False,
					}
				)
		except Exception:
			pass

		documents.sort(key=lambda item: (item["create_time"], item["source"]), reverse=True)
		return documents

	def delete_document(self, source: str) -> str:
		source = Path(source).name
		if not source:
			raise ValueError("文档名不能为空")
		removed = self._delete_source_vectors(source)
		with _REGISTRY_LOCK:
			registry = _load_registry()
			info = registry.pop(source, None)
			if info and info.get("md5"):
				_remove_md5_line(str(info["md5"]))
			_save_registry(registry)
		self._emit(None, "done", 100, "删除完成")
		return f"[Success] 已删除 {source}（{removed} 个分块）"

	def rebuild_from_data_directory(self, progress_cb: ProgressCallback | None = None) -> str:
		"""清空索引并重新入库 data/ 目录下的全部支持文件。"""
		data_dir = Path(config.data_directory)
		files = sorted(
			path for path in data_dir.rglob("*")
			if path.is_file() and path.suffix.lower() in config.supported_document_extensions
		)
		if not files:
			return "[Warn] data/ 目录下没有可重建的文件"

		self._emit(progress_cb, "reset", 3, "正在清空向量索引…")
		try:
			self.chroma.delete_collection()
		except Exception:
			pass
		Path(config.md5_path).write_text("", encoding="utf-8")
		with _REGISTRY_LOCK:
			_save_registry({})
		self.chroma = self._build_chroma()

		results = []
		total = len(files)
		for index, path in enumerate(files):
			base_percent = int(index / total * 90)
			self._emit(
				progress_cb,
				"rebuild",
				5 + base_percent,
				f"正在处理 {path.name}（{index + 1}/{total}）",
			)
			try:
				text = extract_text_from_file(path.name, path.read_bytes())

				def sub_progress(stage: str, percent: int, message: str, _index=index) -> None:
					overall = 5 + int((_index + percent / 100) / total * 90)
					self._emit(progress_cb, stage, overall, message)

				result = self.upload_by_str(text, path.name, progress_cb=sub_progress)
				results.append(f"{path.name}: {result}")
			except Exception as exc:
				results.append(f"{path.name}: [Error] {exc}")

		self._emit(progress_cb, "done", 100, "索引重建完成")
		success = sum(1 for item in results if "[Success]" in item)
		return f"[Success] 索引重建完成，成功 {success}/{total} 个文件\n" + "\n".join(results)


__all__ = [
	"KnowledgeBaseService",
	"add_texts_in_batches",
	"check_md5",
	"save_md5",
	"get_string_md5",
	"extract_text_from_file",
]
