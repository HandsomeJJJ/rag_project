# 法律智能问答助手（RAG + 混合召回 + 记忆压缩）

本项目是一个面向法律问答场景的本地 RAG 系统，基于 Streamlit 提供上传与对话双页面（已支持在对话页面直接上传），结合 Chroma 向量检索、BM25 关键词检索与 RRF 融合，并引入会话记忆压缩机制，支持较长多轮对话。当前还支持作为 MCP Client 调用外部工具，补足“最新法规”和“外部系统查询”能力。

Embedding 检索本地知识 -> Qwen 判断是否要调用 MCP -> 取外部数据 -> 生成最终答案

## 1. 项目能力

- 文档入库：法律文本预处理、分块、向量化并持久化到 Chroma。
  `ingestion/legal_preprocess.py::preprocess_legal_text`，
  `ingestion/legal_chunker.py::parse_legal_article_units` / `build_chunks_from_article_units`，`ingestion/ingest_service.py::KnowledgeBaseService.upload_by_str`
- 混合召回：向量检索 + BM25 并行召回，通过 RRF 融合后返回证据。
  `retrieval/hybrid_retriever.py::HybridRetrieverService.retrieve` / `rrf_fuse`
- 生成回答：使用通义千问大模型按法律模板输出结构化回复。
  `generation/rag_service.py::RagService._run_agent` / `_build_chain` / `_build_chat_model`
- MCP 外部工具：可连接外部 MCP Server，调用网页搜索、数据库查询等工具。
  `agent/mcp_client.py::MCPClientManager.from_environment` / `list_tools` / `call_tool`，
  `agent/tool_executor.py::build_mcp_tool`
- 会话管理：会话列表、新建、置顶、删除、持久化历史。
  `app/streamlit_chat.py::new_session_id` / `load_messages_from_history` / `_consume_query_action`，`memory/history_store.py::list_session_ids` / `toggle_session_pinned` / `delete_history`
- 记忆压缩：滑动窗口 + 摘要压缩 + Token 预算裁剪。
  `memory/history_store.py::_compress_messages` / `_summarize_messages` / `FileChatMessageHistory.add_messages`
- 可观测性：可开启压缩调试日志，直接在控制台查看压缩过程。
  `memory/history_store.py::_debug_log` / `_compress_messages`

当前模型配置：

- 主回答模型：`qwen3-max`
- 轻量摘要模型：`qwen-turbo`
- 向量模型：`text-embedding-v4`

预览：

![alt text](image/演示01.png)

## 2. 目录结构

```text
KnowledgeBase-RAG-LLM-System/
├── app/
│   ├── streamlit_chat.py            # 对话页面
│   └── streamlit_upload.py          # 文档上传与入库页面
├── core/
│   └── config.py                    # 全局配置（检索/模型/记忆压缩）
├── generation/
│   └── rag_service.py               # RAG + MCP Agent 主链路
├── agent/
│   ├── mcp_client.py                # MCP Client 连接与工具发现
│   └── tool_executor.py             # MCP 工具封装
│   └── rag_service.py               # RAG + MCP Agent 主链路
├── agent/
│   ├── mcp_client.py                # MCP Client 连接与工具发现
│   └── tool_executor.py             # MCP 工具封装
├── retrieval/
│   └── hybrid_retriever.py          # 向量 + BM25 + RRF
├── ingestion/
│   ├── legal_preprocess.py          # 文本预处理
│   ├── legal_chunker.py             # 分块与元数据
│   └── ingest_service.py            # 入库服务
├── infra/
│   └── vector_store.py              # 向量库封装
├── memory/
│   └── history_store.py             # 会话持久化与压缩策略
├── test/
│   ├── test_memory_compression.py   # 记忆压缩单元测试
│   └── validate_memory_compression.py # 压缩行为演示脚本
├── data/                            # 原始法律文本与知识文件
├── doc/                             # 方案、流程图、总结文档
├── chroma_db/                       # 运行后生成：向量库
├── chat_history/                    # 运行后生成：会话历史
├── mcp_servers.json                 # MCP Server 配置样例
├── requirements.txt
└── README.md
```

## 3. 环境准备

建议 Python 3.10 或 3.11。

### 3.1 使用 uv（推荐）

```powershell
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### 3.2 使用 pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. 环境变量

在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=sk-你的真实密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ANONYMIZED_TELEMETRY=False
MCP_SERVER_CONFIG_PATH=./mcp_servers.json
```

如果你不想用环境变量，也可以直接把 `mcp_servers.example.json` 复制为 `mcp_servers.json`，程序会自动读取项目根目录下的 `mcp_servers.json`。

## 5. MCP 外部工具配置

MCP 用于把外部能力接进来，例如网页搜索、MySQL、企业信息查询等。当前实现默认使用 `stdio` 方式连接 MCP Server。

### 5.1 配置文件格式

`mcp_servers.json` 示例：

```json
{
  "servers": [
    {
      "name": "web-search",
      "command": "python",
      "args": ["servers/web_search_mcp.py"],
      "env": {
        "SEARCH_API_KEY": "replace-me"
      },
      "transport": "stdio"
    }
  ]
}
```

### 5.2 可用字段

- `name`: MCP Server 名称。
- `command`: 启动 MCP Server 的命令。
- `args`: 命令参数数组。
- `env`: 进程环境变量。
- `cwd`: 可选工作目录。
- `transport`: 目前使用 `stdio`。

### 5.3 启用方式

- 方式一：把配置文件保存为项目根目录的 `mcp_servers.json`。
- 方式二：在 `.env` 里设置 `MCP_SERVER_CONFIG_PATH` 指向任意配置文件。
- 方式三：将完整 JSON 放进 `MCP_SERVERS_JSON` 环境变量。

如果未配置 MCP，系统会自动退回到“本地检索 + 生成”的模式。

## 6. 启动方式

MCP_SERVER_CONFIG_PATH=./mcp_servers.json

````

如果你不想用环境变量，也可以直接把 `mcp_servers.example.json` 复制为 `mcp_servers.json`，程序会自动读取项目根目录下的 `mcp_servers.json`。

## 5. MCP 外部工具配置

MCP 用于把外部能力接进来，例如网页搜索、MySQL、企业信息查询等。当前实现默认使用 `stdio` 方式连接 MCP Server。

### 5.1 配置文件格式

`mcp_servers.json` 示例：

```json
{
  "servers": [
    {
      "name": "web-search",
      "command": "python",
      "args": ["servers/web_search_mcp.py"],
      "env": {
        "SEARCH_API_KEY": "replace-me"
      },
      "transport": "stdio"
    }
  ]
}
````

### 5.2 可用字段

- `name`: MCP Server 名称。
- `command`: 启动 MCP Server 的命令。
- `args`: 命令参数数组。
- `env`: 进程环境变量。
- `cwd`: 可选工作目录。
- `transport`: 目前使用 `stdio`。

### 5.3 启用方式

- 方式一：把配置文件保存为项目根目录的 `mcp_servers.json`。
- 方式二：在 `.env` 里设置 `MCP_SERVER_CONFIG_PATH` 指向任意配置文件。
- 方式三：将完整 JSON 放进 `MCP_SERVERS_JSON` 环境变量。

如果未配置 MCP，系统会自动退回到“本地检索 + 生成”的模式。

## 6. 启动方式

在项目根目录执行。

### 5.1 启动上传页面（先入库）

```powershell
uv run streamlit run app/streamlit_upload.py
```

### 5.2 启动聊天页面

```powershell
uv run streamlit run app/streamlit_chat.py
```

如果虚拟环境已激活，可简写：

```powershell
streamlit run app/streamlit_upload.py
streamlit run app/streamlit_chat.py
```

## 7. 关键配置说明

## 7. 关键配置说明

配置文件：`core/config.py`

### 7.1 检索相关

### 7.1 检索相关

- `hybrid_vector_k`: 向量召回候选数
- `hybrid_bm25_k`: BM25 召回候选数
- `hybrid_final_k`: 融合后最终证据数
- `hybrid_rrf_k`: RRF 融合参数

### 7.2 记忆压缩相关

### 7.2 记忆压缩相关

- `memory_keep_recent_rounds`: 保留最近轮次（当前 3）
- `memory_summary_trigger_rounds`: 摘要触发轮次（当前 5）
- `memory_history_max_tokens`: 历史 token 预算（当前 4000）
- `memory_summary_max_chars`: 摘要最大字符（当前 1500）
- `memory_summary_enabled`: 是否启用摘要
- `memory_summary_tag`: 内部摘要标识
- `memory_compression_debug`: 是否打印压缩日志（当前开启）

## 8. 记忆压缩策略（当前实现）

## 8. 记忆压缩策略（当前实现）

在 `memory/history_store.py` 中执行：

1. 识别内部摘要消息与普通对话消息。
2. 普通消息按用户发言切分轮次。
3. 优先保留最近 N 轮。
4. 达到触发阈值时，将更早轮次交给 `qwen-turbo` 做增量摘要。
5. 重建为“摘要 + 最近轮次”。
6. 若超 Token 预算，按轮次删除最近窗口中最旧轮，直到满足预算或仅剩 1 轮。

说明：内部摘要对模型可见、对用户界面不可见（聊天页已过滤）。

## 9. 测试与验证

## 9. 测试与验证

### 8.1 单元测试

```powershell
python -m unittest discover -s test -p "test_memory_compression.py" -v
```

### 8.2 行为演示（打印压缩行为）

```powershell
python test/validate_memory_compression.py
```

## 10. 免责声明

本项目输出仅用于学习与技术验证，不构成正式法律意见。涉及真实法律事务请咨询持证律师。
