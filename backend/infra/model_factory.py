from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from core import config


def uses_openai_compatible_endpoint() -> bool:
    """配置了兼容模式地址时，统一走 OpenAI 兼容协议。"""
    return bool(config.dashscope_base_url)


def build_chat_model(model_name: str | None = None):
    model_name = model_name or config.chat_model_name
    if uses_openai_compatible_endpoint():
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
            temperature=0.2,
            timeout=180,
            max_retries=1,
        )
    try:
        return init_chat_model(model=model_name, model_provider="tongyi")
    except Exception:
        return ChatTongyi(model=model_name)


def build_embeddings():
    if uses_openai_compatible_endpoint():
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.embedding_model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
            check_embedding_ctx_length=False,
            chunk_size=10,
            request_timeout=60,
            max_retries=1,
        )
    return DashScopeEmbeddings(model=config.embedding_model_name)
