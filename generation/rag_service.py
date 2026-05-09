from __future__ import annotations

import json
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from agent.mcp_client import MCPClientManager
from agent.tool_executor import MCPToolExecutor, build_mcp_tool
from core import config
from infra.vector_store import VectorStoreService
from memory.history_store import get_history
from retrieval.hybrid_retriever import HybridRetrieverService

# 方便观察每轮 Agent 交互的输出内容
def _print_debug_block(title: str, body: str) -> None:
    print("=" * 20)
    print(title)
    print(body)
    print("=" * 20)


class RagService:
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        # 混合检索服务
        self.hybrid_retriever = HybridRetrieverService(
            get_vector_docs=self.vector_service.get_vector_docs,
            get_all_docs=self.vector_service.get_all_documents,
            vector_k=config.hybrid_vector_k,
            bm25_k=config.hybrid_bm25_k,
            final_k=config.hybrid_final_k,
            rrf_k=config.hybrid_rrf_k,
        )
        self.mcp_manager = MCPClientManager.from_environment()
        self.mcp_executor = MCPToolExecutor(self.mcp_manager)
        self.chat_model = _build_chat_model()
        self.tool_model = self._build_tool_model()
        self.chain = self._build_chain()
    # 历史会话链
    def _build_chain(self):
        def invoke_agent(value: dict[str, Any]) -> str:
            question = str(value.get("input", "")).strip()
            history = value.get("history", []) or []
            return self._run_agent(question, history)

        chain = RunnableLambda(invoke_agent)

        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    # 绑定工具到模型
    def _build_tool_model(self):
        # 工作原理：判断 MCP 执行器有没有挂载工具（has_tools）。
        # 如果没有，直接用回普通聊天模型（self.chat_model）；
        # 如果有，就用 bind_tools 把它和工具绑定起来。
        # 这样既能支持带工具的 Agent，也能在工具缺失时优雅降级。
        if not self.mcp_executor.has_tools():
            return self.chat_model

        try:
            return self.chat_model.bind_tools([build_mcp_tool(self.mcp_executor)])
        except Exception:
            return self.chat_model

    # 格式化文档
    def _format_document(self, docs: list[Document]) -> str:
        if not docs:
            return "无相关参考资料"
        evidence_lines = []
        # 遍历检索到的文档
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            source = meta.get("source", "")
            chapter = meta.get("chapter", "")
            article = meta.get("article_no", "")
            evidence_lines.append(
                f"[{i}] source={source} chapter={chapter} article={article}\n"
                f"原文片段：{doc.page_content}"
            )
        return "\n\n".join(evidence_lines)

    def _build_system_prompt(self, local_context: str, tool_catalog: str) -> str:
        return (
            "你是一位拥有多年实务经验的**劳动法与教育法专业助手**，同时可以在需要时调用外部 MCP 工具获取最新信息。"
            "你的目标是：先结合本地检索证据回答；如果问题涉及最新法规、外部系统查询、企业信息或本地知识不足，再调用 MCP 工具补充数据。"
            "\n\n【输出结构要求】"
            "\n### ⚖️ 初步判定"
            "\n### 🔍 深度分析"
            "\n### ⚠️ 风险预警与待核实点"
            "\n### 📜 免责声明"
            "\n\n【规则】"
            "\n- 优先使用本地检索证据；证据不足时再调用工具。"
            "\n- 每次工具调用都要尽量最小化，调用后综合工具结果与本地证据再作答。"
            "\n- 若当前没有可用 MCP 工具，请诚实说明无法联网或查询外部系统。"
            "\n- 严格禁止编造法条、数据或外部查询结果。"
            "\n- 保持专业、理性且有同理心。"
            f"\n\n【本地检索证据】\n{local_context}"
            f"\n\n【可用外部 MCP 工具】\n{tool_catalog}"
        )
    # 兼容不同版本的 LangChain 和底层模型返回的 Tool Call 格式。
    # 有些模型返回字典，有些返回对象，这个函数把它们统统拍扁成统一的字典格式。
    def _parse_tool_call(self, message: AIMessage) -> list[dict[str, Any]]:
        tool_calls = getattr(message, "tool_calls", None) or []
        parsed_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            if isinstance(call, dict):
                parsed_calls.append(call)
                continue
            name = getattr(call, "name", None)
            args = getattr(call, "args", None)
            call_id = getattr(call, "id", None) or getattr(call, "tool_call_id", None)
            if name is not None:
                parsed_calls.append({"name": name, "args": args or {}, "id": call_id})
        return parsed_calls
    # 从工具调用结果构建工具消息
    def _tool_message_from_result(self, tool_call: dict[str, Any], result_text: str) -> ToolMessage:
        call_id = str(tool_call.get("id") or tool_call.get("tool_call_id") or "").strip()
        if not call_id:
            call_id = "mcp_call"
        return ToolMessage(
            content=result_text,
            tool_call_id=call_id,
            name=str(tool_call.get("name") or ""),
        )
    # 纯文本提取
    def _extract_text(self, message: AIMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "\n".join(parts).strip()
        return str(content).strip()
    # 规范化工具参数
    # 工作原理：大模型有时会把工具参数输出为标准的 JSON 字典，有时却是一坨 JSON 字符串。
    # 如果解析失败，就强行把它包在一个 {"value": ...} 字典里，保证后续代码不报错。
    def _normalize_tool_args(self, args: Any) -> dict[str, Any]:
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                return {"value": args}
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        return {"value": args}
# 运行Agent
    def _run_agent(self, question: str, history: list[BaseMessage]) -> str:
        local_docs = self.hybrid_retriever.retrieve(question)
        local_context = self._format_document(local_docs)
        try:
            tool_catalog = self.mcp_executor.describe_tools_text(refresh=False)
        except Exception as exc:
            tool_catalog = f"当前无法加载外部 MCP 工具: {exc}"
        # 构建系统提示
        messages: list[BaseMessage] = [
            SystemMessage(content=self._build_system_prompt(local_context, tool_catalog))
        ]
        # 添加历史消息
        messages.extend(history)
        # 添加当前问题
        messages.append(HumanMessage(content=question))

        if self.mcp_executor.has_tools():
            model = self.tool_model
        else:
            model = self.chat_model

        max_rounds = 3
        for round_idx in range(max_rounds):
            response = model.invoke(messages)
            if not isinstance(response, AIMessage):
                return str(response)

            tool_calls = self._parse_tool_call(response)
            if not tool_calls:
                final_text = self._extract_text(response)
                _print_debug_block(f"Agent round {round_idx + 1}", final_text)
                return final_text

            messages.append(response)
            for tool_call in tool_calls:
                tool_name = str(tool_call.get("name") or "")
                tool_args = self._normalize_tool_args(tool_call.get("args") or {})
                server_name = str(tool_args.get("server_name") or "").strip()
                requested_tool_name = str(tool_args.get("tool_name") or "").strip()
                request_arguments = self._normalize_tool_args(tool_args.get("arguments") or {})
                try:
                    result_text = self.mcp_executor.execute(
                        server_name=server_name,
                        tool_name=requested_tool_name or tool_name,
                        arguments=request_arguments,
                    )
                except Exception as exc:
                    result_text = f"MCP 工具调用失败: {exc}"
                messages.append(self._tool_message_from_result(tool_call, result_text))

        fallback_response = model.invoke(messages)
        if isinstance(fallback_response, AIMessage):
            return self._extract_text(fallback_response)
        return str(fallback_response)

# 初始化llm
def _build_chat_model():
    """优先使用 LangChain v1 的统一模型初始化语法。"""
    try:
        return init_chat_model(
            model=config.chat_model_name,
            model_provider="tongyi",
        )
    except Exception:
        # 若本地依赖未完成迁移，回退到社区模型实现，保证可运行。
        return ChatTongyi(model=config.chat_model_name)
