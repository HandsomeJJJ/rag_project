# 法律智能问答助手（RAG + 混合召回 + 记忆压缩）

本项目是一个面向法律问答场景的本地 RAG 系统。核心仍由 Chroma 向量检索、BM25 关键词检索、RRF 融合、Qwen、Agent/MCP 与会话记忆组成。新增 Vue3 + FastAPI 前后端分离入口；原 Streamlit 页面保留，用作算法 Demo/调试界面。

Embedding 检索本地知识 -> Qwen 判断是否要调用 MCP -> 取外部数据 -> 生成最终答案

## 1.项目功能

- 文档入库：法律文本预处理、分块、向量化并持久化到 Chroma。
  `backend/ingestion/legal_preprocess.py::preprocess_legal_text`，
  `backend/ingestion/legal_chunker.py::parse_legal_article_units` / `build_chunks_from_article_units`，`backend/ingestion/ingest_service.py::KnowledgeBaseService.upload_by_str`
- 混合召回：向量检索 + BM25 并行召回，通过 RRF 融合后返回证据。
  `backend/retrieval/hybrid_retriever.py::HybridRetrieverService.retrieve` / `rrf_fuse`
- 生成回答：使用通义千问大模型按法律模板输出结构化回复。
  `backend/generation/rag_service.py::RagService._run_agent` / `_build_chain` / `_build_chat_model`
- 调用外部 MCP 工具：可连接外部 MCP Server，从而扩展自身能力（如搜索、数据库查询）
  `backend/agent/mcp_client.py::MCPClientManager.from_environment` / `list_tools` / `call_tool`，
  `backend/agent/tool_executor.py::build_mcp_tool`
- 会话管理：Vue 聊天页支持新建、切换、搜索、重命名、置顶和删除会话；标题、置顶状态与对话历史持久化。首次提问生成默认标题，手动标题不会被覆盖。
  `backend/api/main.py`，`backend/app/streamlit_chat.py::new_session_id` / `load_messages_from_history` / `_consume_query_action`，`backend/memory/history_store.py::list_session_ids` / `toggle_session_pinned` / `delete_history`
- 消息操作：支持 SSE 流式回答、停止接收、复制 Markdown 原文和重新生成回答；原 Streamlit 页面仍可用于算法调试。
- 记忆压缩：滑动窗口 + 摘要压缩 + Token 预算裁剪。
  `backend/memory/history_store.py::_compress_messages` / `_summarize_messages` / `FileChatMessageHistory.add_messages`
- 可观测性：可开启压缩调试日志，直接在控制台查看压缩过程。
  `backend/memory/history_store.py::_debug_log` / `_compress_messages`

当前模型配置：

- 主回答模型：`qwen3-max`
- 轻量摘要模型：`qwen-turbo`
- 向量模型：`text-embedding-v4`

前端页面预览：
本地检索成功：
![alt text](image/演示02.png)

## 2. 项目目录结构

```txt
├── backend/
│   ├── api/              # FastAPI 路由及 SSE 响应
│   ├── agent/            # Agent、MCP 客户端与工具
│   ├── app/              # 保留的 Streamlit Demo
│   ├── core/             # 配置与项目根目录数据路径
│   ├── generation/       # RagService
│   ├── infra/            # Chroma 向量库封装
│   ├── ingestion/        # 文档预处理、分块和入库
│   ├── memory/           # 会话历史与记忆压缩
│   ├── retrieval/        # Hybrid Retrieval / RRF
│   ├── servers/          # MCP Server 实现
│   ├── test/             # 测试和调试脚本
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/              # Vue 聊天页、会话列表和 API 调用
│   ├── package.json      # 前端依赖与 dev/build 脚本
│   ├── vite.config.*     # Vite 配置
│   ├── nginx.conf        # Docker 静态服务及 API 代理
│   └── Dockerfile        # Vite 构建 + Nginx 静态托管
├── data/                 # 原始法律文本和知识文件
├── chroma_db/            # Chroma 持久化目录
├── chat_history/         # 会话持久化目录
├── mcp_servers.json      # MCP Server 配置
├── .env                  # 本地环境变量（不提交）
├── .dockerignore
├── .gitignore
├── docker-compose.yml
└── README.md
```

## 3. 环境准备

建议 Python 3.10 或 3.11，前端需要 Node.js 18 或更新版本及 npm。以下命令均从项目根目录执行；Windows 示例使用 PowerShell，macOS/Linux 示例使用 bash/zsh。

### 3.1 使用 uv（推荐）

```powershell
uv venv --python 3.11 backend/.venv
backend\.venv\Scripts\Activate.ps1
uv pip install -r backend/requirements.txt
```

macOS/Linux：

```bash
uv venv --python 3.11 backend/.venv
source backend/.venv/bin/activate
uv pip install -r backend/requirements.txt
```

### 3.2 使用 pip

```powershell
python -m venv backend/.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

macOS/Linux：

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install -r backend/requirements.txt
```

## 4. 环境变量

在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=sk-你的真实密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ANONYMIZED_TELEMETRY=False
MCP_SERVER_CONFIG_PATH=./mcp_servers.json
```

MCP 配置可由 `MCP_SERVER_CONFIG_PATH` 指定；未指定时程序默认读取项目根目录的 `mcp_servers.json`。
请勿将真实密钥提交到 Git。MCP Server 若需要额外密钥，可放在该 Server 配置的 `env` 字段中。

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
      "cwd": "backend",
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

### 6.1 启动 FastAPI 后端

先按第 3 节创建并安装 `backend/.venv`，在项目根目录准备 `.env` 并配置 `DASHSCOPE_API_KEY`。之后在项目根目录执行：

```bash
source backend/.venv/bin/activate
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

PowerShell 可从项目根目录运行 `backend\.venv\Scripts\Activate.ps1`，然后执行 `cd backend` 和上述 `uvicorn` 命令。启动后可访问 `http://localhost:8000/docs` 查看接口文档、访问 `http://localhost:8000/health` 检查服务。

主要接口：`GET /health`、`GET /sessions`、`PATCH /sessions/{session_id}`、`DELETE /sessions/{session_id}`、`POST /sessions/{session_id}/pin`、`POST /chat`（SSE 流式响应）和 `POST /upload`（UTF-8 TXT）。`/chat` 支持通过 `regenerate_message_index` 请求重新生成并替换对应回答。应用启动时初始化一次 `RagService` 和 `KnowledgeBaseService`。可用 `CORS_ORIGINS` 配置允许的前端源。

### 6.2 启动 Vue3 前端

另开终端：

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`。开发服务器默认请求 `http://localhost:8000`；其他后端地址可通过 `VITE_API_BASE_URL` 配置。
首次安装依赖后可用 `npm run build` 检查并生成生产静态文件；开发联调时确保 FastAPI 已在 8000 端口启动。

### 6.3 启动保留的 Streamlit Demo

在仓库根目录激活 `backend/.venv` 后：

```bash
cd backend
streamlit run app/streamlit_upload.py
# 或
streamlit run app/streamlit_chat.py
```

### 6.4 Docker Compose

在仓库根目录准备 `.env`（至少包含 `DASHSCOPE_API_KEY`），然后运行：

```bash
docker compose up --build -d
docker compose logs -f backend
```

Vue 页面访问 `http://localhost:8080`，API 健康检查访问 `http://localhost:8000/health`。后端镜像从 `./backend` 构建，前端镜像从 `./frontend` 构建；Compose 将根目录的 `data/`、`chroma_db/`、`chat_history/`、`mcp_servers.json` 及 `md5.text` 挂载给后端。前端 Nginx 将 `/api` 请求转发到后端。查看前端和后端日志可分别运行 `docker compose logs -f frontend` 与 `docker compose logs -f backend`。关闭服务：

```bash
docker compose down
```

## 7. 关键配置说明

配置文件：`backend/core/config.py`

### 7.1 检索相关

- `hybrid_vector_k`: 向量召回候选数
- `hybrid_bm25_k`: BM25 召回候选数
- `hybrid_final_k`: 融合后最终证据数
- `hybrid_rrf_k`: RRF 融合参数

### 7.2 记忆压缩相关

- `memory_keep_recent_rounds`: 保留最近轮次（当前 3）
- `memory_summary_trigger_rounds`: 摘要触发轮次（当前 5）
- `memory_history_max_tokens`: 历史 token 预算（当前 4000）
- `memory_summary_max_chars`: 摘要最大字符（当前 1500）
- `memory_summary_enabled`: 是否启用摘要
- `memory_summary_tag`: 内部摘要标识
- `memory_compression_debug`: 是否打印压缩日志（当前开启）

## 8. 记忆压缩策略（当前实现）

在 `backend/memory/history_store.py` 中执行：

1. 识别内部摘要消息与普通对话消息。
2. 普通消息按用户发言切分轮次。
3. 优先保留最近 N 轮。
4. 达到触发阈值时，将更早轮次交给 `qwen-turbo` 做增量摘要。
5. 重建为“摘要 + 最近轮次”。
6. 若超 Token 预算，按轮次删除最近窗口中最旧轮，直到满足预算或仅剩 1 轮。

说明：内部摘要对模型可见、对用户界面不可见（聊天页已过滤）。

## 9. 测试与验证

### 9.1 单元测试

```powershell
cd backend
python -m unittest discover -s test -p "test_memory_compression.py" -v
```

### 9.2 行为演示（打印压缩行为）

```powershell
cd backend
python test/validate_memory_compression.py
```

## 10. 免责声明

本项目输出仅用于学习与技术验证，不构成正式法律意见。涉及真实法律事务请咨询持证律师。
