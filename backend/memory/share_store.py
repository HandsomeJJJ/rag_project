from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any

from core import config


def _registry_path() -> Path:
    path = Path(config.share_registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load() -> dict[str, Any]:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict[str, Any]) -> None:
    with open(_registry_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_share(user_id: str, session_id: str) -> str:
    """为会话生成只读分享 token；同一会话复用已有 token。"""
    data = _load()
    for token, item in data.items():
        if item.get("user_id") == user_id and item.get("session_id") == session_id:
            return token

    token = secrets.token_urlsafe(18)
    data[token] = {
        "user_id": user_id,
        "session_id": session_id,
        "created_at": time.time(),
    }
    _save(data)
    return token


def resolve_share(token: str) -> dict[str, str] | None:
    """解析分享 token，返回用户与会话 ID。"""
    item = _load().get(token)
    if not isinstance(item, dict):
        return None
    user_id = str(item.get("user_id") or "")
    session_id = str(item.get("session_id") or "")
    if not user_id or not session_id:
        return None
    return {"user_id": user_id, "session_id": session_id}


def delete_session_shares(user_id: str, session_id: str) -> None:
    """会话被删除时清理其分享链接。"""
    data = _load()
    stale = [
        token for token, item in data.items()
        if item.get("user_id") == user_id and item.get("session_id") == session_id
    ]
    if not stale:
        return
    for token in stale:
        data.pop(token, None)
    _save(data)
