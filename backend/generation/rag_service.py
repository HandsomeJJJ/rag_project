from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Iterator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent.mcp_client import MCPClientManager
from agent.tool_executor import MCPToolExecutor, build_mcp_tool
from core import config
from core.legal_numbers import extract_article_numbers, normalize_article_number
from generation.query_understanding import QueryPlan, QueryResolver
from infra.model_factory import build_chat_model, build_embeddings
from infra.vector_store import VectorStoreService
from retrieval.hybrid_retriever import HybridRetrieverService
from retrieval.reranker import DashScopeReranker

_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千零〇两0-9]+\s*条")
logger = logging.getLogger(__name__)


# 方便观察每轮 Agent 交互的输出内容
def _print_debug_block(title: str, body: str) -> None:
    print("=" * 20)
    print(title)
    print(body)
    print("=" * 20)


class RagService:
    def __init__(self):
        self.vector_service = VectorStoreService(embedding=build_embeddings())
        reranker = DashScopeReranker() if config.rerank_enabled else None
        # 混合检索服务
        self.hybrid_retriever = HybridRetrieverService(
            get_vector_docs=self.vector_service.get_vector_docs,
            get_all_docs=self.vector_service.get_all_documents,
            vector_k=config.hybrid_vector_k,
            bm25_k=config.hybrid_bm25_k,
            final_k=config.hybrid_final_k,
            rrf_k=config.hybrid_rrf_k,
            rerank_fn=reranker.rerank if reranker else None,
        )
        self.mcp_manager = MCPClientManager.from_environment()
        self.mcp_executor = MCPToolExecutor(self.mcp_manager)
        self.query_resolver = QueryResolver()
        self.chat_model = build_chat_model()
        self.tool_model = self._build_tool_model()

    def refresh_vector_store(self) -> None:
        """知识库重建后刷新向量库连接与 BM25 缓存。"""
        self.vector_service = VectorStoreService(embedding=build_embeddings())
        self.hybrid_retriever.get_vector_docs = self.vector_service.get_vector_docs
        self.hybrid_retriever.get_all_docs = self.vector_service.get_all_documents
        self.hybrid_retriever.invalidate_cache()

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

    def _sources_payload(self, docs: list[Document]) -> list[dict[str, Any]]:
        """把检索证据整理成前端可渲染的引用卡片数据。"""
        payload: list[dict[str, Any]] = []
        for index, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            payload.append(
                {
                    "index": index,
                    "source": str(meta.get("source") or ""),
                    "part": str(meta.get("part") or ""),
                    "chapter": str(meta.get("chapter") or ""),
                    "section": str(meta.get("section") or ""),
                    "article_no": str(meta.get("article_no") or ""),
                    "article_end": str(meta.get("chunk_article_end") or meta.get("article_end") or ""),
                    "content": doc.page_content,
                }
            )
        return payload

    def _build_system_prompt(
        self,
        local_context: str,
        tool_catalog: str,
        plan: QueryPlan,
        allow_tools: bool,
    ) -> str:
        if plan.intent == "chat":
            return (
                "你是一位友好、简洁的中文法律助手。用户当前只是问候或闲聊，"
                "请用 1-2 句话自然回应，不要调用工具，不要套用法律分析结构；"
                "如果用户随后提出法律问题，再进入专业分析。"
            )

        if allow_tools:
            tool_section = (
                f"\n\n【可用外部工具】\n{tool_catalog}"
                "\n\n【工具使用规则】"
                "\n- 只有问题明确涉及最新法规、实时信息、外部系统查询，或本地证据完全不足时才调用工具。"
                "\n- 同一工具最多调用一次；工具无结果时不要重复调用，直接基于本地证据作答，并说明未获取到外部信息。"
                "\n- 工具结果只能作为补充，最终必须回到法律分析并标注本地证据编号。"
            )
            if plan.intent == "tool":
                tool_section += "\n- 本轮问题明确涉及时效或外部信息，可调用工具核实；若工具不可用，如实说明。"
        else:
            tool_section = (
                "\n\n【工具状态】"
                "\n本轮未启用外部工具。不要声称已经联网或查询外部系统；证据不足时如实说明。"
            )

        return (
            "你是一位严谨、务实的中文法律助手，熟悉中国现行法律、行政法规与司法解释，尤其擅长劳动法、教育法、刑法。"
            "请直接回答用户问题，不要输出检索、思考或工具调用过程。"
            "\n\n【回答要求】"
            "\n1. 结论先行：正文开头用 2-4 句话直接回答用户最关心的问题。"
            "\n2. 依据准确：引用本地证据时，在相关句末标注 [1][2]；编号必须对应该句实际依据的片段，不要把所有内容都标成 [1]；法条编号统一使用证据中的中文数字格式（如“第二百六十四条”），不要在同一答案中混用“第264条”；刑期、金额、时间必须与证据一致。"
            "\n3. 不编造：证据未覆盖的司法解释、地方标准、最新政策，明确写“本地知识库未收录，需进一步核实”，不要猜测具体数字。"
            "\n4. 简洁：简单概念题控制在 200-400 字；比较、程序或风险分析控制在 400-800 字。不要重复问题，不要写空泛套话。"
            "\n5. 格式：使用 Markdown 小标题和短列表；只有对比两种罪名/两种方案时才用表格；引用标记紧跟相关句末。"
            "\n6. 免责声明只写一句，放在最后。"
            "\n\n【输出结构】"
            "\n### 结论"
            "\n（2-4 句话直接回答）"
            "\n### 法律依据"
            "\n- 《法律名称》第X条：要点 [n]"
            "\n### 分析"
            "\n（按争议点分 2-4 点，必要时给一个简短例子）"
            "\n### 风险与建议"
            "\n（视问题需要输出；列 2-4 条可执行建议或待核实点）"
            "\n> 免责声明：以上内容仅供一般性法律参考，不构成正式法律意见。"
            "\n\n说明：简单概念题可省略“风险与建议”；若本地没有直接法条依据，在“法律依据”中如实说明。"
            f"\n\n【本地检索证据】\n{local_context}"
            f"{tool_section}"
        )

    def _build_messages(
        self,
        question: str,
        history: list[BaseMessage],
        local_context: str,
        tool_catalog: str,
        plan: QueryPlan,
        allow_tools: bool,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [
            SystemMessage(
                content=self._build_system_prompt(local_context, tool_catalog, plan, allow_tools)
            )
        ]
        messages.extend(history)
        messages.append(HumanMessage(content=question))
        return messages

    # 兼容不同版本的 LangChain 和底层模型返回的 Tool Call 格式。
    # 有些模型返回字典，有些返回对象，这个函数把它们统统拍平成一个统一的字典格式。
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
    def _extract_text(self, message: Any) -> str:
        content = getattr(message, "content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "\n".join(parts)
        return str(content)

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

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> str:
        tool_name = str(tool_call.get("name") or "")
        tool_args = self._normalize_tool_args(tool_call.get("args") or {})
        server_name = str(tool_args.get("server_name") or "").strip()
        requested_tool_name = str(tool_args.get("tool_name") or "").strip()
        request_arguments = self._normalize_tool_args(tool_args.get("arguments") or {})
        try:
            return self.mcp_executor.execute(
                server_name=server_name,
                tool_name=requested_tool_name or tool_name,
                arguments=request_arguments,
            )
        except Exception as exc:
            return f"MCP 工具调用失败: {exc}"

    def _citation_warning(self, answer: str, docs: list[Document]) -> str:
        """校验回答中引用的法条是否出现在本轮检索证据中。"""
        if not answer or not docs:
            return ""
        cited_map = {
            normalize_article_number(match.group(0)): match.group(0)
            for match in _ARTICLE_RE.finditer(answer)
        }
        cited_map.pop("", None)
        if not cited_map:
            return ""
        available: set[str] = set()
        for doc in docs:
            meta = doc.metadata or {}
            for key in ("article_no", "chunk_article_end", "article_end"):
                available |= extract_article_numbers(str(meta.get(key) or ""))
            available |= extract_article_numbers(doc.page_content)
        missing_keys = set(cited_map) - available
        if not missing_keys:
            return ""
        missing_display = [
            cited_map[key]
            for key in sorted(missing_keys, key=lambda value: int(value) if value.isdigit() else 0)
        ]
        return "以下法条未出现在本轮检索证据中，请谨慎核实：" + "、".join(missing_display[:5])

    def stream_answer(
        self,
        question: str,
        history: list[BaseMessage] | None = None,
        source_filter: str = "",
    ) -> Iterator[dict[str, Any]]:
        """真正的 token 级流式回答，同时推送检索、工具与引用事件。"""
        history = list(history or [])
        has_tools = self.mcp_executor.has_tools()
        started_at = time.perf_counter()
        first_token_ms: float | None = None

        yield {"type": "status", "stage": "understanding", "text": "正在理解问题…"}
        plan = self.query_resolver.resolve(question, history, has_tools)

        docs: list[Document] = []
        if plan.intent != "chat":
            yield {"type": "status", "stage": "retrieving", "text": "正在检索法律法规…"}
            docs = self.hybrid_retriever.retrieve(plan.query, source_filter)

        sources = self._sources_payload(docs)
        yield {
            "type": "sources",
            "sources": sources,
            "query": plan.query,
            "intent": plan.intent,
        }

        if plan.intent == "chat":
            local_context = "（本轮为日常对话，无需检索本地法条）"
        else:
            local_context = self._format_document(docs)

        # 普通法条问题不挂载工具，避免不必要的联网和额外延迟
        allow_tools = bool(
            has_tools
            and plan.intent != "chat"
            and (plan.intent == "tool" or not docs)
        )
        tool_catalog = ""
        if allow_tools:
            try:
                tool_catalog = self.mcp_executor.describe_tools_text(refresh=False)
            except Exception as exc:
                tool_catalog = f"当前无法加载外部 MCP 工具: {exc}"

        messages = self._build_messages(
            question, history, local_context, tool_catalog, plan, allow_tools
        )
        model = self.tool_model if allow_tools else self.chat_model

        yield {"type": "status", "stage": "generating", "text": "正在生成回答…"}

        collected: list[str] = []
        tool_trace: list[dict[str, Any]] = []
        needs_final_answer = False
        for round_idx in range(2):
            response = None
            for chunk in model.stream(messages):
                response = chunk if response is None else response + chunk
                chunk_text = self._extract_text(chunk)
                if chunk_text:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started_at) * 1000
                    collected.append(chunk_text)
                    yield {"type": "token", "content": chunk_text}

            if response is None:
                break

            tool_calls = self._parse_tool_call(response)
            if not tool_calls:
                needs_final_answer = False
                if round_idx == 0:
                    _print_debug_block("Agent streaming", "".join(collected))
                break

            needs_final_answer = True
            messages.append(response)
            for tool_call in tool_calls:
                tool_args = self._normalize_tool_args(tool_call.get("args") or {})
                display_name = str(tool_args.get("tool_name") or tool_call.get("name") or "MCP 工具")
                yield {"type": "status", "stage": "tool_running", "text": f"正在调用外部工具 {display_name}…"}
                result_text = self._execute_tool_call(tool_call)
                tool_trace.append(
                    {
                        "name": display_name,
                        "arguments": tool_args,
                        "result": result_text[:800],
                    }
                )
                messages.append(self._tool_message_from_result(tool_call, result_text))
                yield {"type": "status", "stage": "tool_done", "text": "工具调用完成，正在整理答案…"}
            # 工具只允许使用一轮，下一轮改用不带工具的模型，避免重复联网
            model = self.chat_model

        if needs_final_answer:
            # 达到工具轮次上限后强制使用不带工具的模型收尾，避免重复调用工具
            yield {"type": "status", "stage": "finalizing", "text": "正在汇总答案…"}
            for chunk in self.chat_model.stream(messages):
                chunk_text = self._extract_text(chunk)
                if chunk_text:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started_at) * 1000
                    collected.append(chunk_text)
                    yield {"type": "token", "content": chunk_text}

        answer = "".join(collected).strip()
        latency = {
            "first_token_ms": round(first_token_ms, 1) if first_token_ms is not None else None,
            "total_ms": round((time.perf_counter() - started_at) * 1000, 1),
        }
        logger.info(
            "stream_answer completed intent=%s first_token_ms=%s total_ms=%s",
            plan.intent,
            latency["first_token_ms"],
            latency["total_ms"],
        )
        yield {
            "type": "done",
            "content": answer,
            "sources": sources,
            "citation_warning": self._citation_warning(answer, docs),
            "tool_trace": tool_trace,
            "query": plan.query,
            "intent": plan.intent,
            "latency": latency,
        }


# 初始化llm
def _build_chat_model():
    """兼容旧调用入口，实际模型由 model_factory 选择协议。"""
    return build_chat_model()
