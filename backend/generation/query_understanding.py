from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import BaseMessage

from core import config
from infra.model_factory import build_chat_model

_CHAT_PATTERN = re.compile(
    r"^(你好|您好|hi|hello|嗨|谢谢|感谢|再见|你是谁|你能做什么|在吗|早上好|晚上好)"
    r"[\s!！。,.，？?~～]*$",
    re.IGNORECASE,
)
_TOOL_KEYWORDS = (
    "最新", "实时", "联网", "上网查", "检索最新", "新闻",
    "政策变化", "企业信息", "工商信息", "官网", "现行有效版本",
)


@dataclass
class QueryPlan:
    query: str
    intent: str = "local"  # local | tool | chat
    reason: str = ""


class QueryResolver:
    """多轮查询改写与意图路由：轻量模型优先，失败时使用规则兜底。"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.light_model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = build_chat_model(self.model_name)
        return self._model

    @staticmethod
    def _recent_dialogue(history: list[BaseMessage]) -> str:
        if not history:
            return ""
        rounds = max(1, int(config.query_rewrite_history_rounds))
        usable = [
            message for message in history
            if message.type in {"human", "ai"} and str(message.content or "").strip()
        ][-rounds * 2:]
        lines = []
        for message in usable:
            role = "用户" if message.type == "human" else "助手"
            content = str(message.content).strip().replace("\n", " ")
            lines.append(f"[{role}] {content[:500]}")
        return "\n".join(lines)

    @staticmethod
    def _heuristic(question: str, has_tools: bool) -> QueryPlan:
        if _CHAT_PATTERN.match(question.strip()):
            return QueryPlan(query=question, intent="chat", reason="闲聊规则命中")
        if has_tools and any(keyword in question for keyword in _TOOL_KEYWORDS):
            return QueryPlan(query=question, intent="tool", reason="时效/外部信息关键词命中")
        return QueryPlan(query=question, intent="local", reason="默认本地检索")

    def resolve(
        self,
        question: str,
        history: list[BaseMessage] | None = None,
        has_tools: bool = False,
    ) -> QueryPlan:
        question = question.strip()
        if not question:
            return QueryPlan(query="", intent="chat", reason="空问题")
        if not config.query_rewrite_enabled:
            return self._heuristic(question, has_tools)

        dialogue = self._recent_dialogue(history or [])
        # 首轮没有上下文可改写，直接走规则，省掉一次模型调用
        if not dialogue:
            return self._heuristic(question, has_tools)

        prompt = (
            "你是检索查询分析器。请结合对话历史，把当前问题改写成一个不依赖上下文、可独立检索的中文查询，"
            "并判断回答该问题需要哪类信息。\n"
            "只输出一个 JSON 对象，不要输出解释或代码块：\n"
            '{"query": "改写后的查询", "intent": "local|tool|chat"}\n'
            "规则：\n"
            "1. query 控制在 60 字以内，保留法律主体、行为、诉求和条号；如果当前问题本身已经完整，原样返回。\n"
            "2. 追问要补全指代对象，例如“那抢劫罪呢？”应改写为“抢劫罪的量刑规定是什么？”。\n"
            "3. intent 含义：\n"
            "   - local：依据本地法律知识库即可回答；\n"
            "   - tool：明确需要最新法规、实时信息、联网搜索或外部系统数据；\n"
            "   - chat：问候、感谢、闲聊或询问助手身份。\n"
            "4. 不确定是否需要外部信息时选 local，不要过度调用工具。\n"
            "5. 不要编造问题中没有的事实。\n\n"
            f"对话历史：\n{dialogue}\n\n"
            f"当前问题：{question}"
        )
        try:
            response = self._get_model().invoke(prompt)
            content = getattr(response, "content", response)
            text = str(content or "").strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise ValueError("未找到 JSON 输出")
            data = json.loads(match.group(0))
            query = str(data.get("query") or "").strip() or question
            intent = str(data.get("intent") or "local").strip().lower()
            if intent not in {"local", "tool", "chat"}:
                intent = "local"
            if intent == "tool" and not has_tools:
                intent = "local"
            return QueryPlan(query=query[:300], intent=intent, reason="模型改写")
        except Exception:
            return self._heuristic(question, has_tools)
