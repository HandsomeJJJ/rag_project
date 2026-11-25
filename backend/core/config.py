import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

BASE_DIR = str(PROJECT_ROOT)
data_directory = str(PROJECT_ROOT / "data")

md5_path = str(PROJECT_ROOT / "md5.text")

# Chroma
collection_name = "rag"
persist_directory = str(PROJECT_ROOT / "chroma_db")
chat_history_directory = str(PROJECT_ROOT / "chat_history")

# Spliter
chunk_size = 600
chunk_overlap = 150
chunk_overlap_articles = 0
separators = [
    "\\n第.{1,5}条\\s+",
    "\\n第.{1,5}章",
    "\\n\\n",
    "\\n",
    "。",
]

max_spliter_char_number = 1000

# Retrieval
retrieval_top_k = 5
retrieval_source_filter = ""
hybrid_vector_k = 20
hybrid_bm25_k = 20
hybrid_final_k = 6
hybrid_rrf_k = 60
# RRF 融合后进入重排的候选数
rerank_candidate_k = 20
# 重排开关；模型或网络异常时自动退化为 RRF 顺序
rerank_enabled = os.getenv("RERANK_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
rerank_model_name = os.getenv("RERANK_MODEL_NAME", "gte-rerank-v2")

# 多轮查询改写：仅在存在历史对话时调用轻量模型
query_rewrite_enabled = True
query_rewrite_history_rounds = 2

# Backward compatible alias
similarity_threshold = retrieval_top_k

# 模型服务：默认走 DashScope 原生协议；配置兼容模式地址后走 OpenAI 兼容协议
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL", "").rstrip("/")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
# 主力推理模型，用于生成最终给用户的专业法律回答
chat_model_name = os.getenv("CHAT_MODEL_NAME", "qwen3-max")
# 轻量级推理模型，用于生成对话摘要、意图识别等
light_model_name = os.getenv("LIGHT_MODEL_NAME", "qwen-turbo")

if dashscope_base_url:
    _native_root = dashscope_base_url.removesuffix("/compatible-mode/v1").rstrip("/")
    dashscope_native_api_base = os.getenv(
        "DASHSCOPE_NATIVE_API_BASE",
        f"{_native_root}/api/v1",
    )
else:
    dashscope_native_api_base = os.getenv(
        "DASHSCOPE_NATIVE_API_BASE",
        "https://dashscope.aliyuncs.com/api/v1",
    )

# Memory compression strategy config
# 1) 滑动窗口：保留最近 N 轮对话（1 轮 = 用户 1 条 + 助手 1 条）
memory_keep_recent_rounds = 3
# 2) 摘要触发：当累计轮数超过 M 时触发摘要压缩
memory_summary_trigger_rounds = 5
# 3) Token 预算：压缩后的历史消息估算 token 不超过该值
memory_history_max_tokens = 4000
# 4) 摘要最长字符数：约束摘要体积，防止反向膨胀
memory_summary_max_chars = 1500
# 5) 内部摘要开关：允许按环境快速关闭摘要，仅保留滑动窗口
memory_summary_enabled = True
# 6) 内部摘要消息前缀：用于程序识别，展示层可据此隐藏
memory_summary_tag = "[SESSION_SUMMARY]"
# 7) 压缩调试日志开关：为 True 时在控制台打印压缩过程信息
memory_compression_debug = True

# 文档入库支持格式与注册表（注册表放在挂载目录内，容器重建后仍保留）
supported_document_extensions = [".txt", ".md", ".pdf", ".docx"]
document_registry_path = os.path.join(persist_directory, "document_registry.json")
max_upload_bytes = 30 * 1024 * 1024

# 用户反馈与分享记录，放在 chat_history 挂载目录内
feedback_path = os.path.join(chat_history_directory, "feedback.jsonl")
share_registry_path = os.path.join(chat_history_directory, ".shares.json")
