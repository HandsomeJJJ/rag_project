from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from core import config
from core.legal_numbers import extract_article_numbers


def _doc_metadata_article_keys(doc: Document) -> set[str]:
    metadata = doc.metadata or {}
    keys = extract_article_numbers(str(metadata.get("article_no") or ""))
    keys |= extract_article_numbers(str(metadata.get("chunk_article_end") or ""))
    return keys


def _doc_article_keys(doc: Document) -> set[str]:
    return _doc_metadata_article_keys(doc) | extract_article_numbers(doc.page_content)


def _char_tokenize(text: str) -> list[str]:
    """使用字符级分词，避免额外依赖。"""
    return [c for c in text if not c.isspace()]

# 生成文档唯一键，基于 metadata 和内容前 120 字符的 hash。
def _doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    return "|".join(
        [
            str(md.get("source", "")),
            str(md.get("chapter", "")),
            str(md.get("article_no", "")),
            str(md.get("chunk_article_end", "")),
            str(hash(doc.page_content[:120])),
        ]
    )

# RRF 融合算法实现，输入多个 ranked list，输出融合后的 top_k 文档列表。
def rrf_fuse(
    ranked_lists: list[list[Document]],
    top_k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """Reciprocal Rank Fusion。

    score(d) = Σ 1 / (rrf_k + rank_i(d))
    """
    score_map: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for docs in ranked_lists:
        for idx, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            doc_map[key] = doc
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (rrf_k + idx)

    ranked_keys = sorted(score_map.keys(), key=lambda k: score_map[k], reverse=True)
    return [doc_map[k] for k in ranked_keys[:top_k]]


@dataclass
class HybridRetrieverService:
    """混合召回：向量并行 BM25，RRF 融合后再做精排。"""

    get_vector_docs: Callable[..., list[Document]]
    get_all_docs: Callable[..., list[Document]]
    vector_k: int = 20
    bm25_k: int = 20
    final_k: int = 6
    rrf_k: int = 60
    rerank_fn: Callable[[str, list[Document], int], list[Document]] | None = None
    # 按检索范围缓存 BM25 索引，避免切换文档范围时互相覆盖
    _bm25_cache: dict[str, tuple[int, BM25Retriever]] = field(default_factory=dict)

    def invalidate_cache(self) -> None:
        """文档库变更后清理 BM25 缓存。"""
        self._bm25_cache.clear()

    def _ensure_bm25(
        self,
        documents: list[Document],
        source_filter: str = "",
    ) -> BM25Retriever | None:
        count = len(documents)
        cache_key = source_filter or "__all__"
        if count == 0:
            self._bm25_cache.pop(cache_key, None)
            return None

        cached = self._bm25_cache.get(cache_key)
        if cached is not None and cached[0] == count:
            return cached[1]

        retriever = BM25Retriever.from_documents(documents, preprocess_func=_char_tokenize)
        retriever.k = self.bm25_k
        self._bm25_cache[cache_key] = (count, retriever)
        return retriever

    @staticmethod
    def _prioritize_exact(documents: list[Document], exact_keys: set[str]) -> list[Document]:
        """法条号精确命中的文档排到最前，其余保持原顺序。"""
        if not exact_keys:
            return documents
        exact_ids = {_doc_key(doc) for doc in documents if _doc_article_keys(doc) & exact_keys}
        if not exact_ids:
            return documents
        return (
            [doc for doc in documents if _doc_key(doc) in exact_ids]
            + [doc for doc in documents if _doc_key(doc) not in exact_ids]
        )

    def retrieve(self, query: str, source_filter: str = "") -> list[Document]:
        vector_docs = self.get_vector_docs(query, self.vector_k, source_filter)

        all_docs = self.get_all_docs(source_filter)
        bm25_docs: list[Document] = []
        bm25_retriever = self._ensure_bm25(all_docs, source_filter)
        if bm25_retriever is not None:
            bm25_docs = bm25_retriever.invoke(query)

        exact_keys = extract_article_numbers(query)
        exact_docs: list[Document] = []
        if exact_keys:
            metadata_matches = [
                doc for doc in all_docs
                if _doc_metadata_article_keys(doc) & exact_keys
            ]
            matched_keys = {_doc_key(doc) for doc in metadata_matches}
            content_matches = [
                doc for doc in all_docs
                if _doc_key(doc) not in matched_keys and _doc_article_keys(doc) & exact_keys
            ]
            exact_docs = (metadata_matches or content_matches)[: self.final_k * 2]

        if not vector_docs and not bm25_docs and not exact_docs:
            return []

        candidate_k = max(self.final_k, config.rerank_candidate_k) if self.rerank_fn else self.final_k
        fused_docs = rrf_fuse(
            [vector_docs, bm25_docs, exact_docs],
            top_k=candidate_k,
            rrf_k=self.rrf_k,
        )
        if self.rerank_fn is None:
            final_docs = fused_docs
        else:
            final_docs = self.rerank_fn(query, fused_docs, self.final_k)

        metadata_exact = [
            doc for doc in exact_docs
            if _doc_metadata_article_keys(doc) & exact_keys
        ]
        if metadata_exact:
            head_keys = {_doc_key(doc) for doc in metadata_exact[: self.final_k]}
            head = [doc for doc in exact_docs if _doc_key(doc) in head_keys][: self.final_k]
            tail = [doc for doc in final_docs if _doc_key(doc) not in head_keys]
            return (head + tail)[: self.final_k]
        return self._prioritize_exact(final_docs, exact_keys)[: self.final_k]
