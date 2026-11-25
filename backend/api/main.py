from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from core import config
from generation.rag_service import RagService
from ingestion.ingest_service import KnowledgeBaseService, extract_text_from_file
from memory.history_store import (
    delete_history,
    get_history,
    get_session_metadata,
    get_session_source_filter,
    get_session_updated_at,
    list_session_ids,
    replace_history_message,
    set_auto_session_title,
    set_session_pinned,
    set_session_source_filter,
    set_session_title,
)
from memory.share_store import create_share, delete_session_shares, resolve_share

logger = logging.getLogger(__name__)
USER_COOKIE = "rag_uid"
USER_PATTERN = re.compile(r"^user_[A-Za-z0-9_-]{6,64}$")
_FEEDBACK_LOCK = threading.Lock()
_INGEST_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create expensive RAG and ingestion services once per application process."""
    app.state.rag_service = RagService()
    app.state.knowledge_base_service = KnowledgeBaseService()
    yield


app = FastAPI(title="Legal RAG API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_user_id(request: Request, call_next):
    """为每个浏览器分配稳定的匿名用户 ID，实现会话隔离。"""
    user_id = request.cookies.get(USER_COOKIE, "")
    is_new = not USER_PATTERN.fullmatch(user_id or "")
    if is_new:
        user_id = f"user_{uuid4().hex[:16]}"
    request.state.user_id = user_id
    response = await call_next(request)
    if is_new:
        response.set_cookie(
            USER_COOKIE,
            user_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="lax",
        )
    return response


def _uid(request: Request) -> str:
    return getattr(request.state, "user_id", "default")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    session_id: str | None = None
    regenerate_message_index: int | None = Field(default=None, ge=0)
    source_filter: str | None = Field(default=None, max_length=200)


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class PinSessionRequest(BaseModel):
    pinned: bool


class SourceFilterRequest(BaseModel):
    source_filter: str = Field(default="", max_length=200)


class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int = Field(default=0, ge=0)
    rating: str = Field(pattern="^(up|down|citation_wrong)$")
    question: str = Field(default="", max_length=20000)
    answer: str = Field(default="", max_length=40000)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    comment: str = Field(default="", max_length=2000)


@dataclass
class JobState:
    job_id: str
    user_id: str
    kind: str
    filename: str = ""
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = "等待处理"
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)


_JOBS: dict[str, JobState] = {}
_JOBS_LOCK = threading.Lock()


def _prune_jobs() -> None:
    cutoff = time.time() - 3600
    stale = [
        job_id for job_id, job in _JOBS.items()
        if job.status in {"done", "error"} and job.created_at < cutoff
    ]
    for job_id in stale:
        _JOBS.pop(job_id, None)


def _create_job(user_id: str, kind: str, filename: str = "") -> JobState:
    job = JobState(
        job_id=f"job_{uuid4().hex[:12]}",
        user_id=user_id,
        kind=kind,
        filename=filename,
    )
    with _JOBS_LOCK:
        _prune_jobs()
        _JOBS[job.job_id] = job
    return job


def _update_job(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)


def _job_payload(job: JobState) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "filename": job.filename,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "error": job.error,
    }


def _job_progress_callback(job_id: str):
    def callback(stage: str, percent: int, message: str) -> None:
        _update_job(
            job_id,
            status="running",
            stage=stage,
            progress=max(0, min(100, int(percent))),
            message=message,
        )

    return callback


def _run_ingest_job(
    job_id: str,
    filename: str,
    raw: bytes,
    knowledge_service: KnowledgeBaseService,
    rag_service: RagService,
) -> None:
    with _INGEST_LOCK:
        _update_job(
            job_id,
            status="running",
            stage="parsing",
            progress=2,
            message=f"正在解析 {filename}…",
        )
        try:
            text = extract_text_from_file(filename, raw)
            if not text.strip():
                raise ValueError("文件内容为空或未解析到有效文本")
            progress = _job_progress_callback(job_id)
            result = knowledge_service.upload_by_str(text, filename, progress_cb=progress)
            rag_service.refresh_vector_store()
            _update_job(
                job_id,
                status="done",
                stage="done",
                progress=100,
                message="入库完成",
                result=result,
            )
        except Exception as exc:
            logger.exception("Knowledge base upload failed")
            _update_job(
                job_id,
                status="error",
                stage="error",
                progress=100,
                message="入库失败",
                error=str(exc),
            )


def _run_rebuild_job(
    job_id: str,
    knowledge_service: KnowledgeBaseService,
    rag_service: RagService,
) -> None:
    with _INGEST_LOCK:
        _update_job(
            job_id,
            status="running",
            stage="reset",
            progress=1,
            message="正在重建索引…",
        )
        try:
            result = knowledge_service.rebuild_from_data_directory(
                progress_cb=_job_progress_callback(job_id)
            )
            rag_service.refresh_vector_store()
            _update_job(
                job_id,
                status="done",
                stage="done",
                progress=100,
                message="索引重建完成",
                result=result,
            )
        except Exception as exc:
            logger.exception("Knowledge base rebuild failed")
            _update_job(
                job_id,
                status="error",
                stage="error",
                progress=100,
                message="索引重建失败",
                error=str(exc),
            )


def _validate_session_id(session_id: str) -> None:
    if not re.fullmatch(r"chat_[A-Za-z0-9_-]{1,120}", session_id):
        raise HTTPException(status_code=422, detail="session_id 格式无效")


def _require_session(session_id: str, user_id: str) -> None:
    _validate_session_id(session_id)
    if session_id not in list_session_ids(user_id):
        raise HTTPException(status_code=404, detail="会话不存在")


def _visible_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [message for message in messages if message.type in {"human", "ai"}]


def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in _visible_messages(messages):
        additional_kwargs = message.additional_kwargs or {}
        payload.append(
            {
                "role": "user" if message.type == "human" else "assistant",
                "content": str(message.content),
                "sources": additional_kwargs.get("sources") or [],
                "citation_warning": str(additional_kwargs.get("citation_warning") or ""),
            }
        )
    return payload


def _session_title(session_id: str, user_id: str, messages: list[BaseMessage]) -> str:
    metadata = get_session_metadata(session_id, user_id)
    if metadata.get("title"):
        return str(metadata["title"])
    for message in messages:
        if message.type == "human" and isinstance(message.content, str):
            return message.content.strip().replace("\n", " ")[:28] or "新会话"
    return "新会话"


def _session_payload(session_id: str, user_id: str) -> dict[str, Any]:
    messages = get_history(session_id, user_id).messages
    metadata = get_session_metadata(session_id, user_id)
    return {
        "session_id": session_id,
        "title": _session_title(session_id, user_id, messages),
        "pinned": bool(metadata.get("pinned", False)),
        "source_filter": str(metadata.get("source_filter") or ""),
        "updated_at": datetime.fromtimestamp(
            get_session_updated_at(session_id, user_id), timezone.utc
        ).isoformat(),
        "messages": _serialize_messages(messages),
    }


def _session_summary(session_id: str, user_id: str) -> dict[str, Any]:
    """会话列表只返回元数据，消息在打开会话时再按需加载。"""
    metadata = get_session_metadata(session_id, user_id)
    title = str(metadata.get("title") or "")
    message_count = metadata.get("message_count")
    if not title or message_count is None:
        messages = get_history(session_id, user_id).messages
        title = title or _session_title(session_id, user_id, messages)
        if message_count is None:
            message_count = len(_visible_messages(messages))
    return {
        "session_id": session_id,
        "title": title,
        "pinned": bool(metadata.get("pinned", False)),
        "source_filter": str(metadata.get("source_filter") or ""),
        "updated_at": datetime.fromtimestamp(
            get_session_updated_at(session_id, user_id), timezone.utc
        ).isoformat(),
        "message_count": int(message_count),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sessions")
async def sessions(request: Request) -> list[dict[str, Any]]:
    user_id = _uid(request)
    try:
        return [_session_summary(sid, user_id) for sid in list_session_ids(user_id)]
    except Exception as exc:
        logger.exception("Could not load chat sessions")
        raise HTTPException(status_code=500, detail="无法读取会话列表") from exc


@app.get("/sessions/{session_id}")
async def session_detail(session_id: str, request: Request):
    user_id = _uid(request)
    _require_session(session_id, user_id)
    return _session_payload(session_id, user_id)


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, payload: RenameSessionRequest, request: Request):
    user_id = _uid(request)
    _require_session(session_id, user_id)
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="标题不能为空")
    metadata = set_session_title(session_id, title, manual=True, user_id=user_id)
    return {"session_id": session_id, "title": metadata["title"]}


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str, request: Request):
    user_id = _uid(request)
    _require_session(session_id, user_id)
    delete_history(session_id, user_id)
    delete_session_shares(user_id, session_id)
    return {"session_id": session_id, "deleted": True}


@app.post("/sessions/{session_id}/pin")
async def pin_session(session_id: str, payload: PinSessionRequest, request: Request):
    user_id = _uid(request)
    _require_session(session_id, user_id)
    set_session_pinned(session_id, payload.pinned, user_id)
    return {"session_id": session_id, "pinned": payload.pinned}


@app.post("/sessions/{session_id}/source-filter")
async def update_source_filter(
    session_id: str,
    payload: SourceFilterRequest,
    request: Request,
):
    user_id = _uid(request)
    _require_session(session_id, user_id)
    source_filter = payload.source_filter.strip()
    set_session_source_filter(session_id, source_filter, user_id)
    return {"session_id": session_id, "source_filter": source_filter}


@app.post("/sessions/{session_id}/share")
async def share_session(session_id: str, request: Request):
    user_id = _uid(request)
    _require_session(session_id, user_id)
    token = create_share(user_id, session_id)
    return {"session_id": session_id, "token": token, "path": f"/share/{token}"}


@app.get("/share/{token}")
async def view_shared_session(token: str):
    resolved = resolve_share(token)
    if resolved is None:
        raise HTTPException(status_code=404, detail="分享链接不存在或已失效")
    user_id = resolved["user_id"]
    session_id = resolved["session_id"]
    messages = get_history(session_id, user_id).messages
    if not _visible_messages(messages):
        raise HTTPException(status_code=404, detail="分享的会话已不存在")
    metadata = get_session_metadata(session_id, user_id)
    return {
        "session_id": session_id,
        "title": _session_title(session_id, user_id, messages),
        "source_filter": str(metadata.get("source_filter") or ""),
        "messages": _serialize_messages(messages),
    }


@app.get("/sessions/{session_id}/export")
async def export_session(session_id: str, request: Request):
    user_id = _uid(request)
    _require_session(session_id, user_id)
    messages = get_history(session_id, user_id).messages
    title = _session_title(session_id, user_id, messages)
    lines = [
        f"# {title}",
        "",
        f"> 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for message in _visible_messages(messages):
        role = "你" if message.type == "human" else "法律助手"
        lines.extend([f"## {role}", "", str(message.content).strip(), ""])
        sources = (message.additional_kwargs or {}).get("sources") or []
        if message.type == "ai" and sources:
            lines.append("**参考来源**")
            lines.append("")
            for source in sources:
                label = " · ".join(
                    str(source.get(key) or "")
                    for key in ("source", "article_no", "chapter")
                    if source.get(key)
                )
                lines.append(f"- [{source.get('index', '')}] {label}")
            lines.append("")
    content = "\n".join(lines)
    filename = f"{title or session_id}.md".replace("/", "_")
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
    }
    return PlainTextResponse(content, media_type="text/markdown; charset=utf-8", headers=headers)


@app.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question 不能为空")
    user_id = _uid(request)
    session_id = payload.session_id or f"chat_{uuid4().hex[:12]}"
    _validate_session_id(session_id)
    rag_service: RagService = request.app.state.rag_service

    if payload.source_filter is not None:
        set_session_source_filter(session_id, payload.source_filter.strip(), user_id)
    source_filter = get_session_source_filter(session_id, user_id)

    base_history: list[BaseMessage] = list(get_history(session_id, user_id).messages)
    replacement_index: int | None = None
    if payload.regenerate_message_index is not None:
        _require_session(session_id, user_id)
        original_messages = list(get_history(session_id, user_id).messages)
        visible_indexes = [
            index for index, message in enumerate(original_messages)
            if message.type in {"human", "ai"}
        ]
        visible_index = payload.regenerate_message_index
        if visible_index >= len(visible_indexes):
            raise HTTPException(status_code=409, detail="目标回答已不存在")
        replacement_index = visible_indexes[visible_index]
        target_answer = original_messages[replacement_index]
        if target_answer.type != "ai":
            raise HTTPException(status_code=409, detail="目标消息不是助手回答")
        previous_user_index = next(
            (index for index in reversed(visible_indexes[:visible_index])
             if original_messages[index].type == "human"),
            None,
        )
        if previous_user_index is None:
            raise HTTPException(status_code=409, detail="找不到对应的问题")
        previous_question = str(original_messages[previous_user_index].content).strip()
        if previous_question != question:
            raise HTTPException(status_code=409, detail="问题与目标回答不匹配")
        base_history = original_messages[:previous_user_index]

    async def event_stream():
        client_cancelled = threading.Event()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def emit(item: tuple[str, Any]) -> None:
            if client_cancelled.is_set():
                return
            try:
                asyncio.run_coroutine_threadsafe(queue.put(item), loop).result(timeout=5)
            except Exception:
                # 事件循环已关闭或客户端已中断，忽略推送，但继续完成落盘
                pass

        yield f"event: session\ndata: {json.dumps({'session_id': session_id}, ensure_ascii=False)}\n\n"

        def produce() -> None:
            try:
                for event in rag_service.stream_answer(question, base_history, source_filter):
                    if event.get("type") == "done":
                        answer = str(event.get("content") or "").strip()
                        sources = event.get("sources") or []
                        warning = str(event.get("citation_warning") or "")
                        if answer:
                            if replacement_index is not None:
                                replace_history_message(
                                    session_id,
                                    replacement_index,
                                    answer,
                                    sources=sources,
                                    citation_warning=warning,
                                    user_id=user_id,
                                )
                            else:
                                get_history(session_id, user_id).add_messages(
                                    [
                                        HumanMessage(content=question),
                                        AIMessage(
                                            content=answer,
                                            additional_kwargs={
                                                "sources": sources,
                                                "citation_warning": warning,
                                            },
                                        ),
                                    ]
                                )
                                set_auto_session_title(
                                    session_id,
                                    question.replace("\n", " ")[:28],
                                    user_id,
                                )
                    emit(("event", event))
            except Exception as exc:  # forwarded to the SSE client below
                logger.exception("RAG stream failed")
                emit(("error", exc))
            finally:
                emit(("finished", None))

        worker = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                kind, value = await queue.get()
                if kind == "event":
                    event = value
                    event_type = str(event.get("type") or "")
                    if event_type == "token":
                        data = {"content": event.get("content") or ""}
                    elif event_type == "sources":
                        data = {
                            "sources": event.get("sources") or [],
                            "query": event.get("query") or "",
                            "intent": event.get("intent") or "local",
                        }
                    elif event_type == "status":
                        data = {
                            "stage": event.get("stage") or "",
                            "text": event.get("text") or "",
                        }
                    else:
                        data = {
                            "content": event.get("content") or "",
                            "sources": event.get("sources") or [],
                            "citation_warning": event.get("citation_warning") or "",
                            "tool_trace": event.get("tool_trace") or [],
                            "query": event.get("query") or "",
                            "intent": event.get("intent") or "local",
                            "latency": event.get("latency") or {},
                        }
                    yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                elif kind == "error":
                    yield f"event: error\ndata: {json.dumps({'detail': '回答生成失败，请检查服务配置后重试'}, ensure_ascii=False)}\n\n"
                    break
                else:
                    break
            await worker
        except asyncio.CancelledError:
            client_cancelled.set()
            raise
        except Exception:
            logger.exception("Unexpected chat streaming failure")
            yield f"event: error\ndata: {json.dumps({'detail': '服务暂时不可用'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    user_id = _uid(request)
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in config.supported_document_extensions:
        allowed = "、".join(config.supported_document_extensions)
        raise HTTPException(status_code=415, detail=f"目前支持 {allowed}")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="文件内容不能为空")
    if len(raw) > config.max_upload_bytes:
        raise HTTPException(status_code=413, detail="文件过大，请上传 30MB 以内的文件")

    job = _create_job(user_id, "upload", filename)
    threading.Thread(
        target=_run_ingest_job,
        args=(
            job.job_id,
            filename,
            raw,
            request.app.state.knowledge_base_service,
            request.app.state.rag_service,
        ),
        daemon=True,
    ).start()
    return {"job_id": job.job_id, "filename": filename}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None or job.user_id != _uid(request):
        raise HTTPException(status_code=404, detail="任务不存在")
    return _job_payload(job)


@app.get("/documents")
async def list_documents(request: Request):
    service: KnowledgeBaseService = request.app.state.knowledge_base_service
    try:
        documents = await asyncio.to_thread(service.list_documents)
        return {"documents": documents}
    except Exception as exc:
        logger.exception("Could not list documents")
        raise HTTPException(status_code=500, detail="无法读取知识库文档") from exc


@app.delete("/documents/{source}")
async def delete_document(source: str, request: Request):
    service: KnowledgeBaseService = request.app.state.knowledge_base_service
    rag_service: RagService = request.app.state.rag_service
    try:
        result = await asyncio.to_thread(service.delete_document, source)
        rag_service.refresh_vector_store()
        return {"source": source, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Could not delete document")
        raise HTTPException(status_code=500, detail="删除文档失败") from exc


@app.post("/documents/rebuild")
async def rebuild_documents(request: Request):
    user_id = _uid(request)
    job = _create_job(user_id, "rebuild")
    threading.Thread(
        target=_run_rebuild_job,
        args=(
            job.job_id,
            request.app.state.knowledge_base_service,
            request.app.state.rag_service,
        ),
        daemon=True,
    ).start()
    return {"job_id": job.job_id}


@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest, request: Request):
    user_id = _uid(request)
    _require_session(payload.session_id, user_id)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "session_id": payload.session_id,
        "message_index": payload.message_index,
        "rating": payload.rating,
        "question": payload.question,
        "answer": payload.answer,
        "sources": payload.sources,
        "comment": payload.comment,
    }
    path = Path(config.feedback_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _FEEDBACK_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}
