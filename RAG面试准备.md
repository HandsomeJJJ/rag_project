# RAG 面试准备 · 当前项目实现手册

> 校准日期：2026-11-25。以当前仓库代码为准，之前版本中关于 Streamlit、`RagService.chain`、`_run_agent`、MCP 长驻连接、鉴权限流、邻条扩展、63 个单测的描述均已不再适用。
>
> 面试口径：只讲代码里真实存在的东西；未实现的放到“已知缺口/下一步”，不主动包装成已完成。

---

## 0. 30 秒电梯稿

我做的是一个面向劳动法、教育法、刑法场景的本地 RAG 问答系统，后端 FastAPI + LangChain，前端 Vue3，Docker Compose 部署。

链路分四段：

1. **入库**：TXT / Markdown / PDF / DOCX 先做法律文本预处理，按“条”结构化分块，向量化写入 Chroma；同时维护文档注册表，支持删除和整库重建。
2. **检索**：向量 top20 与 BM25 top20 召回，RRF(k=60) 融合出 20 条候选，再用 `qwen3.7-text-rerank` 精排出 top6；问具体条号时优先保留精确命中的法条。
3. **生成**：`qwen3.8-flash` 走 DashScope OpenAI 兼容端点，真正按模型增量流式输出；多轮追问先由轻量模型改写查询并路由“本地 / 工具 / 闲聊”意图；工具按需启用，最多一轮。
4. **记忆**：会话按匿名浏览器用户隔离，本地 JSON 存储；滑动窗口 + 增量摘要 + Token 预算裁剪。

前端是完整聊天产品：SSE 状态与逐字输出、来源卡片、引用校验提示、停止/重新生成/反馈、检索范围、知识库管理、分享与导出。

---

## 1. 项目全景

### 1.1 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 主模型 | `qwen3.8-flash` | 通过 `CHAT_MODEL_NAME` 配置，OpenAI 兼容协议 |
| 轻量模型 | `qwen3.8-flash` | 摘要与查询改写，通过 `LIGHT_MODEL_NAME` 配置 |
| 向量模型 | `text-embedding-v4` | 1024 维，`OpenAIEmbeddings` 接入 |
| 重排模型 | `qwen3.7-text-rerank` | 工作空间原生 rerank 接口 |
| 模型接入 | `langchain-openai` 的 `ChatOpenAI` / `OpenAIEmbeddings` | 配置 `DASHSCOPE_BASE_URL` 时启用 |
| 向量库 | Chroma | 本地持久化，collection `rag` |
| 关键词检索 | `rank-bm25` | 字符级分词，无 jieba 依赖 |
| Agent 协议 | MCP（stdio） | 每次工具调用新建会话，没有长驻连接池 |
| API | FastAPI + SSE | `StreamingResponse`，事件式流 |
| 前端 | Vue 3 + Vite + marked + DOMPurify | 单页聊天应用 |
| 部署 | Docker Compose | 前端 Nginx + 后端 Uvicorn |

> Streamlit 已删除，项目只有 FastAPI + Vue 一套入口。

### 1.2 一次提问的完整链路

```text
Vue App.vue
  └─ POST /chat（携带 rag_uid Cookie）
      └─ FastAPI 校验 session_id、读取用户历史
          └─ 后台线程执行 RagService.stream_answer()
              ├─ QueryResolver：追问改写 + 意图路由
              ├─ HybridRetriever：向量 + BM25 → RRF → rerank
              ├─ 组装 System Prompt（证据 + 工具目录 + 输出规范）
              ├─ 工具轮（仅 tool 意图或本地无证据时启用，最多一轮）
              ├─ 模型 token 增量 → asyncio.Queue → SSE token 事件
              └─ 引用校验 → sources / citation_warning
          └─ 保存 HumanMessage + AIMessage（含 sources）到用户目录
```

SSE 事件：

```text
event: session  会话 ID
event: status   正在理解 / 检索 / 生成 / 调用工具 / 汇总
event: sources  检索证据（用于前端来源卡片）
event: token    模型增量文本
event: done     最终答案、来源、引用警告、耗时
event: error    受控错误提示
```

### 1.3 模块与代码位置

| 模块 | 文件 | 核心内容 |
|---|---|---|
| 配置 | `backend/core/config.py` | 模型、检索、记忆、上传、重排开关 |
| 模型工厂 | `backend/infra/model_factory.py` | `ChatOpenAI` / `OpenAIEmbeddings` 与 DashScope 原生回退 |
| 预处理 | `backend/ingestion/legal_preprocess.py` | 去目录噪声，保留编/章/节/条结构 |
| 分块 | `backend/ingestion/legal_chunker.py` | 按“条”解析，按章节合并，不跨章跨节 |
| 入库 | `backend/ingestion/ingest_service.py` | 多格式解析、分批向量化、注册表、删除/重建、进度回调 |
| 向量层 | `backend/infra/vector_store.py` | Chroma 查询与按来源过滤 |
| 混合召回 | `backend/retrieval/hybrid_retriever.py` | 向量 + BM25 + RRF + 精确条号优先 |
| 精排 | `backend/retrieval/reranker.py` | 工作空间 rerank 接口，失败熔断回退 |
| 法条编号 | `backend/core/legal_numbers.py` | 中文/阿拉伯数字统一归一化 |
| 查询理解 | `backend/generation/query_understanding.py` | 追问改写 + local/tool/chat 路由 |
| 生成 | `backend/generation/rag_service.py` | `stream_answer`、Prompt、工具轮、引用校验 |
| MCP 客户端 | `backend/agent/mcp_client.py` | stdio 连接、工具发现、`python` 解释器纠正 |
| 工具桥接 | `backend/agent/tool_executor.py` | 单个 `mcp_call_tool` 元工具 |
| 记忆 | `backend/memory/history_store.py` | 用户目录、压缩、来源持久化、元数据 |
| 分享 | `backend/memory/share_store.py` | 只读分享 token |
| API | `backend/api/main.py` | 聊天 SSE、会话、文档、任务、反馈、分享、导出 |
| 前端 | `frontend/src/App.vue` / `style.css` | 聊天、来源卡片、知识库抽屉、分享导出 |
| 测试 | `backend/test/` | 14 个单元测试 + 2 个调试脚本 |

### 1.4 关键数字（本机实测）

| 指标 | 值 |
|---|---|
| Chroma 向量数 | 170（`刑法.txt` 128 / `paper.txt` 21 / `Edu_Law.txt` 21） |
| 元数据字段 | 9 个：source / create_time / operator / part / chapter / section / article_no / article_end / chunk_article_end |
| 检索参数 | vector_k=20，bm25_k=20，final_k=6，rrf_k=60，rerank_candidate_k=20 |
| 记忆参数 | 保留最近 3 轮 / 5 轮触发摘要 / 4000 token 预算 / 摘要 ≤1500 字 |
| 单测 | 14 个，约 0.01s，覆盖记忆压缩、分批写入、法律元数据 |
| 本地问答实测 | “盗窃罪是什么”：首 token 约 2.5-3s，总 4-7s，答案约 500-650 字，无工具调用 |
| 工具问答实测 | “最新司法解释”：1 次工具调用，首 token 约 15s，总约 21s（受网络影响） |

> 以上耗时是本机 + 当前模型的实测值，面试时说明“环境相关”，不要当成固定 SLA。

---

## 2. 核心实现

### 2.1 配置与模型接入

- 配置集中在 `core/config.py`，模型名、重排开关、检索参数、记忆参数、上传大小都可用环境变量覆盖。
- 配置 `DASHSCOPE_BASE_URL` 时，统一走 OpenAI 兼容协议：
  - 聊天：`ChatOpenAI(model=CHAT_MODEL_NAME, base_url=..., api_key=...)`
  - Embedding：`OpenAIEmbeddings(model=text-embedding-v4, chunk_size=10)`
- 未配置兼容地址时，回退到 DashScope 原生 `DashScopeEmbeddings` / `ChatTongyi`。
- 重排地址从兼容地址自动推导：`https://.../api/v1/services/rerank/text-rerank/text-rerank`，也可以用 `DASHSCOPE_NATIVE_API_BASE` 手动覆盖。
- 模型名不要硬编码：本账号里 `qwen3-max`、`qwen-turbo` 会 404，当前统一用 `qwen3.8-flash`。

### 2.2 预处理与分块

- `preprocess_legal_text` 去掉目录、重复法律名、空行等噪声；如果文件第一行就是“第一条”，不会误删。
- `parse_legal_article_units` 以“条”为最小单元，记录编、章、节、条号。
- `build_chunks_from_article_units` 按“编/章/节”分组，组内贪心合并到 ≤600 字，**不跨章、不跨节**。
- 非法律结构的 MD/PDF/DOCX 没有“条”时，回退到 `RecursiveCharacterTextSplitter`，保证也能入库。
- 每个 chunk 带 9 个元数据字段，前端来源卡片和法条校验都依赖这些字段。

### 2.3 入库与文档管理

- 支持 TXT / MD / PDF / DOCX；PDF 用 `pypdf`，DOCX 用 `python-docx`。
- `KnowledgeBaseService.upload_by_str`：MD5 去重 → 解析 → 分块 → 分批向量化（每批 10 条，DashScope 限制）→ 写文档注册表。
- 同名文件内容变化时，先删除旧向量，再入库，避免检索到过期法条。
- `GET /documents` 列出注册表与历史 Chroma 文档；`DELETE /documents/{source}` 按 source 删除；`POST /documents/rebuild` 清空索引后重建 `data/`。
- 上传是异步任务：`POST /upload` 返回 `job_id`，前端轮询 `/jobs/{job_id}` 看解析、分块、向量化进度。
- 注册表在 `chroma_db/document_registry.json`，与 Chroma 一起位于挂载目录。

### 2.4 混合召回与精排

链路：

```text
query
 ├─ Chroma 向量 top20（支持 source_filter）
 ├─ BM25 top20（字符级分词，按检索范围缓存）
 └─ RRF(k=60) → 20 条候选
      └─ qwen3.7-text-rerank → top6
           └─ 精确条号优先（可选）
```

- RRF 只看排名，不融合原始分数，免去 BM25 与向量相似度量纲不一致的问题。
- BM25 缓存按 `source_filter` 分桶，文档数量变化时重建；上传/删除/重建后主动失效。
- 问“第X条”时，用 `core/legal_numbers.py` 把中文/阿拉伯条号统一，精确命中的文档排到前面。
- 重排失败会自动熔断 5 分钟，回退到 RRF 顺序，不影响主链路。
- 当前没有邻条扩展、没有 Multi-Query、没有 cross-encoder 本地模型；重排走的是 DashScope 的 rerank 服务。

### 2.5 查询理解与意图路由

- `QueryResolver.resolve(question, history, has_tools)`：
  - 没有历史时走规则，避免首轮多一次模型调用。
  - 有历史时用 `qwen3.8-flash` 输出 JSON：`{"query": "...", "intent": "local|tool|chat"}`。
  - query 限制 60 字，追问必须补全指代，例如“那抢劫罪呢？” → “抢劫罪的基本量刑档次是什么？”。
- 意图：
  - `local`：本地知识库可答；
  - `tool`：明确要最新法规、实时信息、联网或外部系统；
  - `chat`：问候、感谢、闲聊。
- 不确定时默认 `local`，避免过度调用工具。
- 模型异常、JSON 解析失败、超时都回退原问题，不阻断检索。

### 2.6 生成与工具策略

- `RagService.stream_answer` 是同步生成器，FastAPI 用 `asyncio.to_thread` 把它放进工作线程，再通过 `asyncio.Queue` 把事件交回事件循环。
- 没有 `RunnableWithMessageHistory`、没有 `chain`、没有 `_run_agent`；历史由 API 层读取后显式传入。
- 工具启用条件：
  - `intent == "tool"`，或
  - 本地检索完全为空。
- 工具只允许一轮：第一轮用带工具的模型，执行完工具后，后续轮次强制切回不带工具的模型收尾，避免重复联网。
- 工具失败不抛出，错误文本作为 `ToolMessage` 回喂给模型，让它改写答案或说明无法查询。
- Prompt 结构：
  - 结论先行（2-4 句）
  - 法律依据（中文条号 + `[n]` 引用）
  - 分析（2-4 点，必要时举例）
  - 风险与建议（可选）
  - 一句免责声明
- 回答长度：简单概念题 200-400 字，复杂问题 400-800 字；只有对比两罪/两方案才用表格。
- 不输出“我先检索”“根据你提供的证据”等过程说明。

### 2.7 答案风格与引用校验

- 系统提示要求：法条编号统一用中文数字（如“第二百六十四条”），不要混用“第264条”。
- 引用编号必须对应该句实际依据，不能所有内容都标 `[1]`。
- `_citation_warning` 在生成后做一次正则校验：
  - 提取回答中的“第X条”；
  - 与检索证据的元数据和正文对齐；
  - 中文/阿拉伯数字统一归一化；
  - 没出现在证据里的条号给出黄色提示，不阻断回答。
- 来源以 `sources` 事件回给前端，并写入 AI 消息的 `additional_kwargs`，刷新页面仍可查看。

### 2.8 记忆与会话隔离

- 存储：`chat_history/<user_id>/<session_id>`，每个会话一个 JSON 文件；`<user_id>/.session_meta.json` 存标题、置顶、更新时间、检索范围、消息数。
- 用户标识：浏览器首次访问写入 HttpOnly `rag_uid` Cookie，所有会话、分享、反馈按该匿名用户隔离。
- 写入：临时文件 + `os.replace` 原子替换，避免读到半截 JSON。
- 压缩：
  1. 识别内部摘要消息与普通消息；
  2. 按用户发言切轮次；
  3. 保留最近 3 轮；
  4. 达到 5 轮时，把更早轮次交给轻量模型做增量摘要；
  5. 重建“摘要 + 最近轮次”；
  6. 超过 4000 token 预算时，从最近窗口最旧轮开始删。
- 摘要消息对模型可见，对前端隐藏。
- 会话列表只返回元数据，打开会话时再通过 `GET /sessions/{id}` 加载消息，避免列表一次拉全量历史。

### 2.9 API

| 接口 | 作用 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /sessions` | 会话元数据列表 |
| `GET /sessions/{id}` | 会话详情与消息 |
| `PATCH /sessions/{id}` | 重命名 |
| `DELETE /sessions/{id}` | 删除会话 |
| `POST /sessions/{id}/pin` | 置顶/取消置顶 |
| `POST /sessions/{id}/source-filter` | 设置检索范围 |
| `POST /sessions/{id}/share` | 生成只读分享 token |
| `GET /share/{token}` | 查看分享会话 |
| `GET /sessions/{id}/export` | 导出 Markdown |
| `POST /chat` | SSE 流式问答 |
| `POST /upload` | 上传文档，返回 job_id |
| `GET /jobs/{id}` | 查询入库/重建进度 |
| `GET /documents` | 文档列表 |
| `DELETE /documents/{source}` | 删除文档 |
| `POST /documents/rebuild` | 重建索引 |
| `POST /feedback` | 反馈（有用/没用/引用有误） |

- `/chat` 支持 `regenerate_message_index` 重新生成指定回答。
- 匿名 Cookie 隔离不是登录鉴权；没有权限模型、限流和审计。

### 2.10 前端

- SSE 解析：`session` / `status` / `sources` / `token` / `done` / `error`。
- 生成中显示“正在理解 / 检索 / 生成 / 调用工具 / 汇总”状态。
- 回答下方展示来源 chip，点击展开原文并高亮起始条号。
- 引用校验有缺失条号时显示黄色提示。
- 消息操作：复制、重新生成、👍、👎、引用有误。
- 检索范围下拉：全部知识库或单个文件，随会话保存。
- 知识库抽屉：上传、文档列表、删除、重建、任务进度。
- 分享：生成只读链接；导出：Markdown 下载、PDF 浏览器打印。
- 会话列表懒加载，打开时再拉消息。

### 2.11 部署

- 前端 Dockerfile：Vite build → Nginx 静态托管。
- 后端 Dockerfile：Python 3.11 slim + Uvicorn。
- `docker-compose.yml`：前端 8080，后端 8000；挂载 `data/`、`chroma_db/`、`chat_history/`、`mcp_servers.json`、`md5.text`。
- 前端默认请求当前主机的 8000 端口，也可以用 `VITE_API_BASE_URL` 覆盖。

---

## 3. 高频追问（20 题）

1. **为什么不能只用向量检索？**
   法律条号、罪名、术语需要精确匹配，BM25 对“第263条”“盗窃罪”这类关键词更稳；向量负责语义泛化，两者互补。

2. **RRF 怎么算？**
   `score(d)=Σ 1/(k+rank_i(d))`，k=60。只融合排名，不做分数归一化。

3. **为什么 RRF 后还要 Rerank？**
   RRF 解决“多路召回怎么合并”，Rerank 解决“候选内部谁更相关”；精排模型看 query-doc 对，精度更高但更慢，所以只对 20 条候选做。

4. **重排失败怎么办？**
   捕获异常后熔断 5 分钟，直接返回 RRF 顺序，保证问答不中断。

5. **为什么要做查询改写？**
   多轮追问里有大量指代和省略，直接用原问题检索会丢上下文；改写只用于检索，最终回答仍基于用户原话和历史。

6. **工具为什么不是每轮都开？**
   普通法条问题本地证据足够，开工具会增加延迟和误调用；只有明确要最新/外部信息，或本地为空时才挂工具。

7. **这是标准 ReAct 吗？**
   不是。它是基于模型原生 Tool Calling 的循环，没有显式 Thought/Action/Observation 文本轨迹；而且当前最多一轮工具。

8. **是真正的 Token Streaming 吗？**
   是。`stream_answer` 直接迭代 `model.stream()`，每个增量通过队列转成 SSE `token` 事件，不再等全文生成后切片。

9. **为什么不再用 `RunnableWithMessageHistory`？**
   历史读写、压缩、用户隔离都在自己的 `history_store` 里；显式传入 history 更可控，也少了链式封装的维护成本。

10. **记忆压缩怎么做？**
    滑动窗口保留最近 3 轮，超过 5 轮触发增量摘要，最后按 4000 token 预算裁剪最旧轮；摘要对模型可见、对 UI 隐藏。

11. **多用户怎么隔离？**
    首次访问写 HttpOnly `rag_uid` Cookie，会话目录按 user_id 分桶；会话、分享、反馈都按该 ID 查询。

12. **上传链路是怎样的？**
    上传后创建后台 job，解析文件 → 分块 → 分批 embedding → 写 Chroma + 注册表；前端轮询 job 进度。

13. **引用怎么校验？**
    生成后提取“第X条”，与检索证据的元数据/正文对齐；中文和阿拉伯数字统一归一化，缺失的条号提示用户核实。

14. **为什么用 qwen3.8-flash？**
    当前工作空间可用、速度较快，支持 OpenAI 兼容协议和流式；主模型和摘要/改写共用它，减少模型切换成本。

15. **为什么走 OpenAI 兼容端点？**
    工作空间只提供 compatible-mode 地址，原生 SDK 的默认域名不适用；统一用 `ChatOpenAI`/`OpenAIEmbeddings` 后，聊天和向量都能正常工作。

16. **MCP 失败会怎样？**
    工具异常会作为 ToolMessage 回喂给模型；模型可以改用本地证据回答并说明未获取到外部信息，不会中断会话。

17. **文档删除会删什么？**
    按 `source` 删除 Chroma 向量和注册表记录，同时移除对应的 MD5；不会删除 `data/` 里的原始文件。

18. **现在有哪些没做？**
    没有登录鉴权/限流、没有 MCP 长驻连接池、没有逐工具 schema、没有邻条扩展/Multi-Query、没有离线评测集；单测只有 14 个。

19. **怎么证明优化有效？**
    用固定法律问题对比优化前后的首 token、总耗时、工具调用次数、来源数量和引用校验结果；目前是手工实测，还没有自动化评测。

20. **下一步优先做什么？**
    先做离线评测集和 Recall@K/引用准确率，再做鉴权限流、MCP 连接池和逐工具 schema，最后考虑邻条扩展和缓存。

---

## 4. 已知缺口（自用，不主动背）

- 匿名 Cookie 不是登录鉴权，不能防冒用；没有权限、限流、审计。
- MCP 每次调用新建 stdio 会话，冷启动成本高；没有连接池。
- 工具只暴露一个 `mcp_call_tool` 元工具，没有逐工具 schema。
- 没有邻条/同节扩展，没有 Multi-Query。
- 没有离线评测集和自动化指标；引用校验是正则对齐，不是语义级。
- 单测只有 14 个，主要覆盖记忆、分批和元数据；没有 API/Agent/MCP 集成测试。
- 检索没有暴露相关性分数，精确条号优先是启发式。

## 5. 下一步优化排序

1. 法律 QA 评测集：Recall@K、MRR、引用准确率、答案正确性。
2. 鉴权与限流：登录、用户权限、API 限流、审计日志。
3. MCP 连接池：长驻 stdio，降低工具冷启动。
4. 逐工具 schema 绑定：减少参数错误和误调用。
5. 邻条扩展/同节扩展：法律场景常用。
6. 缓存与可观测性：查询缓存、重排缓存、首 token/总耗时看板。

## 6. 简历表述

可写：

> Vue3 + FastAPI 法律 RAG 问答系统：混合召回（Chroma + BM25 + RRF）、qwen3.7 精排、多轮查询改写与意图路由、SSE token 级流式、引用溯源与法条校验、记忆压缩、MCP 工具、知识库管理与匿名用户隔离。

不要写：

- “标准 ReAct Agent”（实际是 Tool Calling）
- “MCP 长驻连接池”（未实现）
- “登录鉴权/限流”（未实现）
- “63 个单测”（实际 14 个）
- “邻条扩展 / Multi-Query”（未实现）

## 7. 可复跑命令

```powershell
cd backend

# 单元测试（预期 14 个 OK）
python -m unittest discover -s test -p "test_*.py" -v

# 记忆压缩行为演示
python test/validate_memory_compression.py

# 混合召回预览（会真实调用 embedding/rerank）
python test/preview_hybrid_recall.py

# 启动后端
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

```powershell
cd frontend
npm install
npm run dev
```

```bash
docker compose up --build -d
docker compose down
```
