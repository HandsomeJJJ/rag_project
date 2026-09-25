from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core import config
from generation.rag_service import RagService
from ingestion.ingest_service import KnowledgeBaseService
from memory.history_store import (
    delete_history,
    get_history,
    get_session_metadata,
    get_session_updated_at,
    list_session_ids,
    replace_history_message,
    save_history_messages,
    set_auto_session_title,
    set_session_pinned,
    set_session_title,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create expensive RAG and ingestion services once per application process."""
    app.state.rag_service = RagService()
    app.state.knowledge_base_service = KnowledgeBaseService()
    yield


app = FastAPI(title="Legal RAG API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    session_id: str | None = None
    regenerate_message_index: int | None = Field(default=None, ge=0)


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class PinSessionRequest(BaseModel):
    pinned: bool


def _session_label(session_id: str) -> str:
    metadata = get_session_metadata(session_id)
    if metadata.get("title"):
        return str(metadata["title"])
    for message in get_history(session_id).messages:
        if message.type == "human" and isinstance(message.content, str):
            return message.content.strip().replace("\n", " ")[:28] or "新会话"
    return "新会话"


def _validate_session_id(session_id: str) -> None:
    if not re.fullmatch(r"chat_[A-Za-z0-9_-]{1,120}", session_id):
        raise HTTPException(status_code=422, detail="session_id 格式无效")


def _require_session(session_id: str) -> None:
    _validate_session_id(session_id)
    if session_id not in list_session_ids():
        raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sessions")
async def sessions() -> list[dict[str, Any]]:
    try:
        return [
            {
                "session_id": sid,
                "title": _session_label(sid),
                "pinned": bool(get_session_metadata(sid).get("pinned", False)),
                "updated_at": datetime.fromtimestamp(
                    get_session_updated_at(sid), timezone.utc
                ).isoformat(),
                "messages": [
                    {"role": "user" if msg.type == "human" else "assistant", "content": str(msg.content)}
                    for msg in get_history(sid).messages
                    if not (msg.type == "system" and str(msg.content).startswith(config.memory_summary_tag))
                    and msg.type in {"human", "ai"}
                ],
            }
            for sid in list_session_ids()
        ]
    except Exception as exc:
        logger.exception("Could not load chat sessions")
        raise HTTPException(status_code=500, detail="无法读取会话列表") from exc


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, payload: RenameSessionRequest):
    _require_session(session_id)
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="标题不能为空")
    metadata = set_session_title(session_id, title, manual=True)
    return {"session_id": session_id, "title": metadata["title"]}


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    _require_session(session_id)
    delete_history(session_id)
    return {"session_id": session_id, "deleted": True}


@app.post("/sessions/{session_id}/pin")
async def pin_session(session_id: str, payload: PinSessionRequest):
    _require_session(session_id)
    set_session_pinned(session_id, payload.pinned)
    return {"session_id": session_id, "pinned": payload.pinned}


@app.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question 不能为空")
    session_id = payload.session_id or f"chat_{uuid4().hex[:12]}"
    _validate_session_id(session_id)
    rag_service: RagService = request.app.state.rag_service

    chain_session_id = session_id
    replacement_index: int | None = None
    if payload.regenerate_message_index is not None:
        _require_session(session_id)
        original_messages = list(get_history(session_id).messages)
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
        chain_session_id = f"regen_{uuid4().hex}"
        save_history_messages(
            chain_session_id,
            original_messages[:previous_user_index],
            compress=False,
            touch=False,
        )

    async def event_stream():
        client_cancelled = threading.Event()
        worker_started = False
        try:
            yield f"event: session\ndata: {json.dumps({'session_id': session_id}, ensure_ascii=False)}\n\n"
            # LangChain's existing chain owns retrieval, agent/MCP calls, and message history.
            # Run its synchronous stream off the event loop and forward each generated chunk.
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

            def produce() -> None:
                try:
                    for chunk in rag_service.chain.stream(
                        {"input": question}, config.build_session_config(chain_session_id)
                    ):
                        chunk_text = str(chunk)
                        if replacement_index is not None:
                            generated_parts.append(chunk_text)
                        if not client_cancelled.is_set():
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("chunk", chunk_text)), loop
                            ).result()
                    if replacement_index is not None:
                        if not client_cancelled.is_set():
                            replace_history_message(
                                session_id, replacement_index, "".join(generated_parts)
                            )
                    else:
                        set_auto_session_title(session_id, question.replace("\n", " ")[:28])
                except Exception as exc:  # forwarded to the SSE client below
                    if not client_cancelled.is_set():
                        asyncio.run_coroutine_threadsafe(queue.put(("error", exc)), loop)
                finally:
                    if replacement_index is not None:
                        delete_history(chain_session_id)
                    if not client_cancelled.is_set():
                        asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)

            loop = asyncio.get_running_loop()
            generated_parts: list[str] = []
            worker = asyncio.create_task(asyncio.to_thread(produce))
            worker_started = True
            while True:
                kind, value = await queue.get()
                if kind == "chunk":
                    text = str(value)
                    # RunnableLambda currently yields the final Agent answer as one
                    # LangChain chunk; split it into modest SSE frames for live UI rendering.
                    for start in range(0, len(text), 32):
                        yield f"data: {json.dumps({'content': text[start:start + 32]}, ensure_ascii=False)}\n\n"
                elif kind == "error":
                    logger.error("RAG stream failed: %s", value)
                    yield f"event: error\ndata: {json.dumps({'detail': '回答生成失败，请检查服务配置后重试'}, ensure_ascii=False)}\n\n"
                else:
                    break
            await worker
            yield "event: done\ndata: [DONE]\n\n"
        except asyncio.CancelledError:
            client_cancelled.set()
            raise
        except Exception:
            logger.exception("Unexpected chat streaming failure")
            yield f"event: error\ndata: {json.dumps({'detail': '服务暂时不可用'}, ensure_ascii=False)}\n\n"
        finally:
            if replacement_index is not None and not worker_started:
                delete_history(chain_session_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() != ".txt":
        raise HTTPException(status_code=415, detail="目前仅支持 TXT 文件")
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=422, detail="文件内容不能为空")
        text = raw.decode("utf-8-sig")
        service: KnowledgeBaseService = request.app.state.knowledge_base_service
        result = await asyncio.to_thread(service.upload_by_str, text, Path(file.filename).name)
        return {"filename": Path(file.filename).name, "result": result}
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="文件必须使用 UTF-8 编码") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Knowledge base upload failed")
        raise HTTPException(status_code=500, detail="文件入库失败，请检查服务配置") from exc
