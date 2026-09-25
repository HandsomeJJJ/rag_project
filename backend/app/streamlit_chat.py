import os
import sys
from datetime import datetime
from html import escape
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import config
from generation.rag_service import RagService
from ingestion.ingest_service import KnowledgeBaseService
from memory.history_store import (
    delete_history,
    get_history,
    is_session_pinned,
    list_session_ids,
    toggle_session_pinned,
)

load_dotenv()


def new_session_id() -> str:
    return f"chat_{uuid4().hex[:12]}"


def default_messages() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": "你好，我是一个法律问答助手，我有什么可以帮助你？"}]

# 聊天记录加载，过滤内部摘要消息
def load_messages_from_history(session_id: str) -> list[dict[str, str]]:
    """从历史存储加载会话消息，并过滤内部摘要消息。

    说明：
    - history_store 会写入 type=system 的内部摘要消息，用于给模型提供长期记忆。
    - UI 展示层不应把这些内部摘要暴露给用户，因此这里做过滤。
    """
    # history: 指定会话的历史对象。
    history = get_history(session_id)
    # messages: Streamlit 前端使用的 role/content 结构列表。
    messages = []
    for msg in history.messages:
        # summary_tag: 用于识别内部摘要消息的前缀。
        summary_tag = config.memory_summary_tag
        if msg.type == "system" and str(msg.content).startswith(summary_tag):
            continue

        # role: LangChain 消息类型映射为前端角色类型。
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})
    return messages or default_messages()


def session_label(session_id: str) -> str:
    history = get_history(session_id)
    first_user_message = ""
    for msg in history.messages:
        if msg.type == "human" and isinstance(msg.content, str):
            first_user_message = msg.content.strip()
            break
    preview = first_user_message[:15] if first_user_message else "新会话"
    return preview

# URL 路由与动作分发
def _consume_query_action() -> None:
    action = st.query_params.get("action")
    sid = st.query_params.get("sid")
    if not action:
        return

    session_ids = list_session_ids()
    active_sid = st.session_state.get("active_session_id")

    if action == "new":
        created_id = new_session_id()
        st.session_state["active_session_id"] = created_id
        st.session_state["message"] = default_messages()
    elif action == "open" and sid and sid in session_ids:
        st.session_state["active_session_id"] = sid
        st.session_state["message"] = load_messages_from_history(sid)
    elif action == "delete" and sid:
        delete_history(sid)
        remaining = list_session_ids()
        if not remaining:
            created_id = new_session_id()
            st.session_state["active_session_id"] = created_id
            st.session_state["message"] = default_messages()
        else:
            if active_sid == sid:
                st.session_state["active_session_id"] = remaining[0]
                st.session_state["message"] = load_messages_from_history(remaining[0])
    elif action == "pin" and sid:
        toggle_session_pinned(sid)

    st.query_params.clear()
    st.rerun()

# 渲染侧边栏历史会话
def _render_sidebar_history(active_session_id: str, session_ids: list[str]) -> None:
    rows: list[str] = []
    for idx, sid in enumerate(session_ids):
        label = escape(session_label(sid))
        is_active = sid == active_session_id
        active_cls = "session-item active" if is_active else "session-item"
        pin_text = "取消置顶" if is_session_pinned(sid) else "置顶"
        pin_icon = "📌" if is_session_pinned(sid) else ""
        pin_mark = f'<span class="pin-icon">{pin_icon}</span> ' if pin_icon else ""
        active_dot = '<span class="active-dot"></span>' if is_active else ""
        rows.append(
            (
                f'<div class="{active_cls}" style="animation-delay: {idx * 0.03}s">'
                f'{active_dot}'
                f'<a class="session-link" href="?action=open&sid={sid}" target="_self">'
                f'{pin_mark}<span class="session-text">{label}</span></a>'
                '<div class="session-actions">'
                f'<a class="action-btn pin-btn" href="?action=pin&sid={sid}" target="_self" title="{pin_text}">'
                f'{"&#128204;" if is_session_pinned(sid) else "&#128204;"}</a>'
                f'<a class="action-btn del-btn" href="?action=delete&sid={sid}" target="_self" title="删除">'
                '&#128465;</a>'
                "</div>"
                "</div>"
            )
        )

    session_count = len(session_ids)
    html = (
        '<div class="history-shell">'
        '<div class="sidebar-brand">'
        '<span class="brand-icon">&#9878;&#65039;</span>'
        '<div class="brand-text">'
        '<div class="brand-title">智能法律问答系统</div>'
        '<div class="brand-sub">劳动法&教育法问答助手</div>'
        '</div>'
        '</div>'
        '<a class="new-chat-btn" href="?action=new" target="_self">'
        '<span class="btn-icon">+</span>'
        '<span>新建会话</span>'
        '</a>'
        '<div class="history-header">'
        '<span class="header-icon">&#128337;</span>'
        '<span>历史会话</span>'
        f'<span class="session-count">{session_count}</span>'
        '</div>'
        '<div class="history-list">'
        + "".join(rows)
        + "</div>"
        "</div>"
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


import markdown
from markupsafe import escape

USER_AVATAR = (
    '<svg class="avatar" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="18" cy="18" r="18" fill="#2563eb"/>'
    '<circle cx="18" cy="14" r="6" fill="#fff"/>'
    '<path d="M6 32c0-6.627 5.373-12 12-12s12 5.373 12 12" fill="#fff"/>'
    '</svg>'
)

ASSISTANT_AVATAR = (
    '<svg class="avatar" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="18" cy="18" r="18" fill="#10b981"/>'
    '<rect x="9" y="10" width="18" height="14" rx="3" fill="#fff"/>'
    '<circle cx="14" cy="17" r="2" fill="#10b981"/>'
    '<circle cx="22" cy="17" r="2" fill="#10b981"/>'
    '<rect x="13" y="22" width="10" height="2" rx="1" fill="#10b981"/>'
    '<rect x="6" y="14" width="3" height="6" rx="1.5" fill="#10b981"/>'
    '<rect x="27" y="14" width="3" height="6" rx="1.5" fill="#10b981"/>'
    '</svg>'
)
# 渲染对话气泡
def render_bubble(role: str, content: str) -> None:
    md_html = markdown.markdown(
        content,
        extensions=[
            "fenced_code",
            "tables",
            "nl2br",
            "sane_lists"
        ]
    )

    cls = "bubble-user" if role == "user" else "bubble-assistant"
    align = "row-user" if role == "user" else "row-assistant"
    avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR

    if role == "user":
        html = (
            f'<div class="chat-row {align}">'
            f'<div class="chat-bubble {cls}">{md_html}</div>'
            f'{avatar}'
            "</div>"
        )
    else:
        html = (
            f'<div class="chat-row {align}">'
            f'{avatar}'
            f'<div class="chat-bubble {cls}">{md_html}</div>'
            "</div>"
        )
    st.markdown(html, unsafe_allow_html=True)

#标签页设置
st.set_page_config(page_title="智能法律问答系统", page_icon="⚖️", layout="centered")
#界面设置
st.markdown(
    """
<style>
    /* 【核心】强制锁定全局背景为纯白，文字为深色 */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    /* 【顶部空白】消除主内容区顶部间距 */
    [data-testid="stMain"] {
        padding-top: 0 !important;
    }
    [data-testid="stMain"] .block-container {
        padding-top: 1rem !important;
    }

    /* ★★★ 主内容区最大宽度 ★★★ */
    /* 默认 centered 模式约 730px，改大可让聊天区域更宽 */
    /* 常用值: 900px / 1000px / 1100px / 100%撑满 */
    .block-container {
        max-width: 1200px !important;
    }

    /* 【标题】强制标题颜色为深蓝色，防止在深色模式下变成白色看不见 */
    h1, h2, h3, [data-testid="stMarkdownContainer"] h1 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }

    /* 【顶部】彻底白底化顶部导航栏 */
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #f1f5f9;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }

    /* 【底部】彻底歼灭底部所有黑色区域 (解决左右黑边) */
    [data-testid="stBottom"], 
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    footer {
        background-color: #ffffff !important;
        background: #ffffff !important;
    }

    /* 【输入框】美化并锁定输入框颜色 */
    [data-testid="stChatInput"] {
        border: 1px solid #cbd5e1 !important;
        background-color: #f8fafc !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #1e293b !important;
    }

     /* 【侧边栏】清爽浅蓝灰风格 */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    /* 1. 强制隐藏 Streamlit 原生的侧边栏占位头部 */
    [data-testid="stSidebarHeader"] {
        display: none !important;
        padding: 0 !important;
        margin: 0 !important;
        height: 0 !important;
    }
    /* 2. 将侧边栏内容区的顶部内边距清零 */
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem !important; /* 留 0.5rem 防止贴得太死，如果想完全贴顶可以改成 0 */
    }

    /* ===== 侧边栏历史记录美化 ===== */
    .history-shell {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* 侧边栏顶部品牌/标题 */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 4px 16px;
        margin-bottom: 12px;
        border-bottom: 1px solid #e2e8f0;
    }
    .brand-icon {
        font-size: 1.6rem;
        line-height: 1;
    }
    .brand-text {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .brand-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        letter-spacing: 0.3px;
    }
    .brand-sub {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 400;
    }

    /* 新建会话按钮 */
    .new-chat-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        width: 100%;
        padding: 10px 16px;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        text-decoration: none !important;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
        transition: all 0.2s ease;
    }
    .new-chat-btn:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        transform: translateY(-1px);
    }
    .new-chat-btn:active {
        transform: translateY(0);
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);
    }
    .btn-icon {
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1;
    }

    /* 历史会话标题 */
    .history-header {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 4px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 8px;
    }
    .header-icon { font-size: 0.85rem; }
    .session-count {
        margin-left: auto;
        background: #e2e8f0;
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 1px 7px;
        border-radius: 10px;
        min-width: 18px;
        text-align: center;
    }

    /* 历史列表容器 */
    .history-list {
        display: flex;
        flex-direction: column;
        gap: 2px;
        max-height: calc(100vh - 220px);
        overflow-y: auto;
        padding-right: 2px;
    }
    .history-list::-webkit-scrollbar { width: 4px; }
    .history-list::-webkit-scrollbar-track { background: transparent; }
    .history-list::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    .history-list::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    /* 会话列表项 */
    .session-item {
        position: relative;
        display: flex;
        align-items: center;
        border-radius: 8px;
        transition: all 0.15s ease;
        animation: fadeSlideIn 0.25s ease both;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateX(-8px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    .session-item:hover {
        background-color: #eef2f7;
    }
    .session-item.active {
        background-color: #e0edff;
    }
    .session-item.active::before {
        content: "";
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 3px;
        height: 60%;
        background: #3b82f6;
        border-radius: 0 3px 3px 0;
    }

    /* 活跃指示点 */
    .active-dot {
        flex-shrink: 0;
        width: 6px;
        height: 6px;
        background: #3b82f6;
        border-radius: 50%;
        margin-left: 8px;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }

    /* 会话链接 */
    .session-link {
        flex: 1;
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 9px 8px;
        color: #334155 !important;
        text-decoration: none !important;
        font-size: 0.85rem;
        line-height: 1.4;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        border-radius: 8px;
        transition: color 0.15s;
    }
    .session-item.active .session-link {
        color: #1e40af !important;
        font-weight: 600;
    }
    .session-text {
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
    }
    .pin-icon {
        flex-shrink: 0;
        font-size: 0.75rem;
        opacity: 0.7;
    }

    /* 操作按钮组 (hover 显示) */
    .session-actions {
        display: flex;
        align-items: center;
        gap: 2px;
        margin-right: 6px;
        opacity: 0;
        transition: opacity 0.15s;
    }
    .session-item:hover .session-actions { opacity: 1; }
    .action-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 6px;
        text-decoration: none !important;
        font-size: 0.75rem;
        transition: all 0.15s;
    }
    .pin-btn {
        color: #64748b !important;
    }
    .pin-btn:hover {
        background: #dbeafe;
        color: #2563eb !important;
    }
    .del-btn {
        color: #94a3b8 !important;
    }
    .del-btn:hover {
        background: #fee2e2;
        color: #dc2626 !important;
    }
    
    /* =====【聊天气泡区域】=====
       下方参数可自行调整气泡外观和宽度 */

    /* 对话行：每条消息的外层容器 */
    /* margin: 上下间距; gap: 头像与气泡之间的距离 */
    .chat-row { display: flex; align-items: flex-start; margin: 1.2rem 0; width: 100%; gap: 0.6rem; }
    /* 用户消息靠右对齐 */
    .row-user { justify-content: flex-end; }
    /* AI消息靠左对齐 */
    .row-assistant { justify-content: flex-start; }

    /* 头像大小 */
    /* 调整 width/height 可改变头像尺寸 */
    .avatar {
        width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    /* 气泡通用样式 */
    /* ★★★ max-width 控制气泡最大宽度占比，改大则气泡更宽 ★★★ */
    /* 例如改成 90% 几乎撑满整行，改成 60% 则更窄 */
    /* padding: 气泡内边距，改大则气泡内部空间更多 */
    .chat-bubble {
        max-width: 80%;
        padding: 1rem 1.4rem;
        border-radius: 16px;
        font-size: 1rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .bubble-user {
        background-color: #2563eb; /* 亮蓝色 */
        color: #ffffff !important;
        border-bottom-right-radius: 4px;
    }
    .bubble-assistant {
        background-color: #f1f5f9; /* 浅灰色 */
        color: #1e293b !important;
        border-bottom-left-radius: 4px;
        border: 1px solid #e2e8f0;
    }
    /* 修正气泡内文字颜色 */
    .chat-bubble p, .chat-bubble li { color: inherit !important; margin-bottom: 0; }

    /* 隐藏 Streamlit 默认的装饰线和状态指示 */
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stStatusWidget"] { display: none !important; }
    .stSpinner { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }

    /* 流式输出区域样式（AI正在逐字输出时的临时气泡） */
    /* ★ max-width 要与 .chat-bubble 保持一致，否则输出时和完成后宽度会跳变 ★ */
    .stream-output {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        border-bottom-left-radius: 4px;
        padding: 1rem 1.4rem;
        max-width: 80%;
        color: #1e293b !important;
        font-size: 1rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .stream-output * { color: #1e293b !important; }

    /* 上传按钮样式由 components.html 内联注入 */

    /* ===== 停止按钮美化 ===== */
    /* 隐藏顶部默认停止按钮 */
    [data-testid="stHeader"] [kind="headerStopSequence"],
    [data-testid="stHeader"] button[kind="headerStopSequence"],
    [data-testid="stHeader"] [data-testid="stHeaderStopSequence"] {
        display: none !important;
    }

    /* 自定义停止按钮 - 融入输入框区域 */
    .custom-stop-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        width: 38px;
        height: 38px;
        padding: 0;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
        transition: all 0.2s ease;
        position: relative;
    }
    .custom-stop-btn:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
        transform: translateY(-1px);
    }
    .custom-stop-btn:active {
        transform: translateY(0);
    }
    .custom-stop-btn svg,
    .custom-stop-btn [data-testid] { display: none !important; }
    .custom-stop-btn::before {
        content: "";
        width: 12px;
        height: 12px;
        background: #ffffff;
        border-radius: 2px;
    }

    /* 停止按钮旋转光环动画 */
    @keyframes stopBtnRingRotate {
        from { transform: translate(-50%, -50%) rotate(0deg); }
        to   { transform: translate(-50%, -50%) rotate(360deg); }
    }
    .custom-stop-btn::after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 52px;
        height: 52px;
        border: 2px solid transparent;
        border-top-color: #ef4444;
        border-right-color: rgba(239, 68, 68, 0.3);
        border-radius: 50%;
        animation: stopBtnRingRotate 1s linear infinite;
        pointer-events: none;
    }

    /* 停止按钮移入输入框区域后的容器 */
    .stop-btn-in-input {
        position: absolute;
        right: 14px;
        bottom: 10px;
        z-index: 100;
    }

    /* ===== 思考中指示器 ===== */
    .thinking-indicator-wrap {
        display: flex;
        justify-content: flex-start;
        margin: 0.6rem 0 0.3rem;
    }
    .thinking-indicator {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 8px 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        font-size: 0.85rem;
        color: #64748b;
    }

    @keyframes ringRotate {
        from { transform: translate(-50%, -50%) rotate(0deg); }
        to   { transform: translate(-50%, -50%) rotate(360deg); }
    }
    @keyframes squarePulse {
        0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        50%      { transform: translate(-50%, -50%) scale(0.75); opacity: 0.7; }
    }
    .thinking-icon {
        position: relative;
        width: 24px;
        height: 24px;
    }
    .thinking-square {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 10px;
        height: 10px;
        background: #3b82f6;
        border-radius: 2px;
        transform: translate(-50%, -50%);
        animation: squarePulse 1.2s ease-in-out infinite;
    }
    .thinking-ring {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 24px;
        height: 24px;
        border: 2px solid transparent;
        border-top-color: #3b82f6;
        border-right-color: rgba(59, 130, 246, 0.25);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        animation: ringRotate 0.9s linear infinite;
    }

    /* 侧边栏内 Streamlit 原生按钮通用样式（如有多余按钮） */
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
        /* =========================================
       【终极修复】强制助手气泡内的所有文字全为深靛蓝
       （彻底解决深色模式下自动变白字导致看不清的问题）
       ========================================= */
    .bubble-assistant h1,
    .bubble-assistant h2,
    .bubble-assistant h3,
    .bubble-assistant h4,
    .bubble-assistant h5,
    .bubble-assistant h6,
    .bubble-assistant strong,
    .bubble-assistant b,
    .bubble-assistant em,
    .bubble-assistant p,
    .bubble-assistant li,
    .bubble-assistant span,
    .bubble-assistant div {
        color: #1e293b !important;
    }
    
    /* 让加粗的字体稍微带一点点深蓝，增加层次感 */
    .bubble-assistant strong,
    .bubble-assistant b {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "session_ids" not in st.session_state:
    sessions = list_session_ids()
    st.session_state["session_ids"] = sessions or [new_session_id()]

if "active_session_id" not in st.session_state:
    st.session_state["active_session_id"] = st.session_state["session_ids"][0]

if st.session_state["active_session_id"] not in st.session_state["session_ids"]:
    st.session_state["session_ids"].append(st.session_state["active_session_id"])

_consume_query_action()

# Refresh sessions list for UI
disk_sessions = list_session_ids() or []
if st.session_state["active_session_id"] not in disk_sessions:
    disk_sessions.insert(0, st.session_state["active_session_id"])
st.session_state["session_ids"] = disk_sessions

_render_sidebar_history(
    active_session_id=st.session_state["active_session_id"],
    session_ids=st.session_state["session_ids"],
)

session_config = config.build_session_config(st.session_state["active_session_id"])

if "message" not in st.session_state:
    st.session_state["message"] = load_messages_from_history(st.session_state["active_session_id"])

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
for message in st.session_state["message"]:
    render_bubble(message["role"], message["content"])
st.markdown("</div>", unsafe_allow_html=True)

# ---- 文件上传区域 ----
if "show_uploader" not in st.session_state:
    st.session_state["show_uploader"] = False

# 处理 + 号按钮点击（通过 query_params）—— 必须在渲染按钮之前处理，确保符号与状态同步
_toggle = st.query_params.get("toggle_upload")
if _toggle:
    st.session_state["show_uploader"] = not st.session_state["show_uploader"]
    st.query_params.clear()
    st.rerun()

# + 号按钮（通过 components.html 注入，移动到聊天输入框左侧）
_btn_symbol = "−" if st.session_state["show_uploader"] else "+"
_btn_title = "收起上传" if st.session_state["show_uploader"] else "上传知识库文件"

_upload_btn_html = (
    '<style>'
    '#upload-toggle-btn{'
    'display:inline-flex;align-items:center;justify-content:center;'
    'width:38px;height:38px;border-radius:50%;margin-right:14px;'
    'background:linear-gradient(135deg,#3b82f6,#2563eb);'
    'color:#ffffff !important;font-size:1.3rem;font-weight:700;'
    'text-decoration:none !important;border:none;outline:none;'
    'box-shadow:0 2px 10px rgba(59,130,246,0.35);'
    'transition:all 0.2s ease;line-height:1;flex-shrink:0;'
    'cursor:pointer;-webkit-user-select:none;'
    '}'
    '#upload-toggle-btn:hover{'
    'transform:scale(1.1);'
    'box-shadow:0 4px 16px rgba(59,130,246,0.5);'
    'text-decoration:none !important;'
    '}'
    '</style>'
    '<div id="upload-btn-anchor">'
    '<a id="upload-toggle-btn" href="?toggle_upload=1" title="' + _btn_title + '">'
    + _btn_symbol +
    '</a></div>'
    '<script>'
    '(function(){'
    'var parentWin=window.parent;'
    'var doc=parentWin.document;'
    'function moveBtn(){'
    'var anchor=document.getElementById("upload-btn-anchor");'
    'if(!anchor)return;'
    'var chatInput=doc.querySelector(\'[data-testid="stChatInput"]\');'
    'if(!chatInput){setTimeout(moveBtn,200);return;}'
    'var container=chatInput.parentElement;'
    'if(!container)return;'
    'var old=doc.querySelector("#upload-btn-anchor");'
    'if(old)old.remove();'
    'var style=document.querySelector("style");'
    'if(style){var s=doc.createElement("style");s.textContent=style.textContent;doc.head.appendChild(s);}'
    'container.style.display="flex";'
    'container.style.alignItems="center";'
    'container.insertBefore(anchor,chatInput);'
    '}'
    'moveBtn();'
    '})();'
    '</script>'
)

components.html(_upload_btn_html, height=0)

# 文件上传器（点击 + 后展开）
if st.session_state["show_uploader"]:
    uploaded_file = st.file_uploader(
        "上传 TXT 文件到知识库",
        type=["txt"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        file_name = uploaded_file.name
        text = uploaded_file.getvalue().decode("utf-8")
        if "kb_service" not in st.session_state:
            st.session_state["kb_service"] = KnowledgeBaseService()
        with st.spinner(f"正在载入知识库：{file_name} ..."):
            result = st.session_state["kb_service"].upload_by_str(text, file_name)
        st.session_state["message"].append({
            "role": "assistant",
            "content": f"📂 **知识库上传结果**\n\n文件：`{file_name}`\n\n{result}",
        })
        st.session_state["show_uploader"] = False
        st.rerun()

prompt = st.chat_input("请输入你的问题")
if prompt:
    st.session_state["message"].append({"role": "user", "content": prompt})
    render_bubble("user", prompt)

    # 思考中指示器（正方形 + 旋转圆圈）
    st.markdown(
        f"""
        <div class="chat-row row-assistant" id="thinking-indicator">
            {ASSISTANT_AVATAR}
            <div class="thinking-indicator">
                <div class="thinking-icon">
                    <div class="thinking-square"></div>
                    <div class="thinking-ring"></div>
                </div>
                <span>思考中...</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # JS: 将停止按钮移入输入框区域
    st.markdown(
        """
        <script>
        (function() {
            var tries = 0;
            function findStopBtn() {
                var btn =
                    document.querySelector('[data-testid="stHeader"] [kind="headerStopSequence"]') ||
                    document.querySelector('[data-testid="stHeader"] button[kind="headerStopSequence"]') ||
                    document.querySelector('[data-testid="stHeaderStopSequence"]') ||
                    document.querySelector('[data-testid="stHeader"] .stButton button');
                if (!btn && ++tries <= 40) {
                    setTimeout(findStopBtn, 150);
                    return;
                }
                if (!btn) return;
                var chatInput = document.querySelector('[data-testid="stChatInput"]');
                if (!chatInput) return;
                var parent = chatInput.parentElement;
                if (!parent) return;
                if (parent.querySelector('.custom-stop-btn')) return;
                btn.style.display = 'none';
                btn.style.position = 'absolute';
                btn.style.right = '14px';
                btn.style.bottom = '10px';
                btn.style.width = '38px';
                btn.style.height = '38px';
                btn.style.padding = '0';
                btn.style.margin = '0';
                btn.style.zIndex = '100';
                btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
                btn.style.color = '#ffffff';
                btn.style.border = 'none';
                btn.style.borderRadius = '12px';
                btn.style.cursor = 'pointer';
                btn.style.boxShadow = '0 2px 8px rgba(239,68,68,0.3)';
                btn.classList.add('custom-stop-btn');
                if (!parent.style.position || parent.style.position === 'static') {
                    parent.style.position = 'relative';
                }
                parent.appendChild(btn);
                btn.style.display = 'inline-flex';
            }
            findStopBtn();
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

    rag_chain = st.session_state["rag"].chain
    stream = rag_chain.stream({"input": prompt}, session_config)

    # 流式输出：逐字显示纯文本，避免 markdown 结构先蹦出来
    placeholder = st.empty()
    collected = []
    for chunk in stream:
        collected.append(chunk)
        placeholder.markdown(
            f'<div class="chat-row row-assistant">'
            f'{ASSISTANT_AVATAR}'
            f'<div class="stream-output">{escape("".join(collected))}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 流式结束：用 markdown 气泡替换纯文本
    final_ai_text = "".join(collected)
    placeholder.empty()
    render_bubble("assistant", final_ai_text)
    st.session_state["message"].append({"role": "assistant", "content": final_ai_text})