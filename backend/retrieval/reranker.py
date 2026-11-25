from __future__ import annotations

import logging
import time

import requests
from langchain_core.documents import Document

from core import config

logger = logging.getLogger(__name__)


class DashScopeReranker:
    """调用 DashScope 兼容工作区的文本重排接口，失败时退化为 RRF 顺序。"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.rerank_model_name
        self._disabled_until = 0.0
        self._endpoint = (
            f"{config.dashscope_native_api_base.rstrip('/')}"
            "/services/rerank/text-rerank/text-rerank"
        )

    def _request(self, query: str, documents: list[Document], top_n: int) -> list[dict]:
        headers = {
            "Authorization": f"Bearer {config.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": [doc.page_content[:4000] for doc in documents],
            },
            "parameters": {
                "return_documents": False,
                "top_n": top_n,
            },
        }
        response = requests.post(self._endpoint, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        output = data.get("output") if isinstance(data, dict) else None
        results = output.get("results") if isinstance(output, dict) else None
        return results or []

    def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        if len(documents) <= 1 or not config.dashscope_api_key:
            return documents[:top_n]
        if time.time() < self._disabled_until:
            return documents[:top_n]
        try:
            results = self._request(query, documents, top_n)
            ordered: list[Document] = []
            for item in results:
                index = item.get("index") if isinstance(item, dict) else getattr(item, "index", None)
                if index is None:
                    continue
                index = int(index)
                if 0 <= index < len(documents):
                    ordered.append(documents[index])
            if ordered:
                logger.info("Rerank 完成: model=%s docs=%s", self.model_name, len(ordered))
                return ordered[:top_n]
        except Exception as exc:  # 网络或模型不可用时不影响主链路
            # 连续失败时短暂熔断，避免每次问答都等待超时
            self._disabled_until = time.time() + 300
            logger.warning("Rerank 失败，5 分钟内回退 RRF 顺序: %s", exc)
        return documents[:top_n]
