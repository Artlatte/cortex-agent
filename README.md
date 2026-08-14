# 🧠 Cortex Agent

**企业级多智能体 + RAG 知识库问答平台** — 一个覆盖「多模型网关 → RAG 全链路 → Agent 运行时 → 多 Agent 编排 → 生产级可靠性」的完整 LLM 应用项目。

- 🚫 **零 Key 可跑**：内置离线 DemoProvider 与 HashEmbedder，克隆即可体验全部功能，无需任何 API Key
- 🔌 **多模型接入**：OpenAI 兼容（OpenAI / DeepSeek / GLM / Moonshot）、Anthropic Claude、Google Gemini，统一抽象 + 结构化输出
- 📚 **RAG 全链路**：文档解析（txt/md/pdf/docx）→ 递归切分 → Embedding → **BM25 + 向量混合检索（RRF）** → 重排序 → 带引用生成
- 🤖 **Agent 架构**：ReAct 工具调用循环、任务规划、短期记忆（窗口 + 摘要）、长期记忆（向量检索）、多 Agent 编排（规划 → 并行研究 → 核查 → 撰写）
- 🛡️ **生产级可靠性**：并发控制、超时、重试（指数退避 + 抖动）、**熔断器**、多模型降级链、幂等、会话持久化
- 📈 **可观测性**：Trace ID 全链路追踪、结构化 JSON 日志、Prometheus 指标
- 🔗 **MCP 支持**：stdio JSON-RPC MCP 客户端，外部工具一键桥接为 Agent 工具

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/Artlatte/cortex-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Artlatte/cortex-agent/actions)

---

## 📖 目录

- [快速开始](#-快速开始)
- [离线演示（无需 API Key）](#-离线演示无需-api-key)
- [功能详解](#-功能详解)
  - [LLM 网关](#1-llm-网关多模型--可靠性)
  - [RAG 知识库](#2-rag-知识库)
  - [Agent 运行时](#3-agent-运行时)
  - [多 Agent 编排](#4-多-agent-编排)
  - [MCP 集成](#5-mcp-集成)
  - [可观测性](#6-可观测性)
- [HTTP API](#-http-api)
- [配置说明](#-配置说明)
- [项目结构](#-项目结构)
- [测试与代码质量](#-测试与代码质量)
- [架构与设计文档](#-架构与设计文档)
- [Roadmap](#-roadmap)

## 🚀 快速开始

要求：Python 3.10+

```bash
git clone https://github.com/Artlatte/cortex-agent.git
cd cortex-agent

# 安装（可选：创建虚拟环境 python -m venv .venv && 激活）
pip install -e ".[dev]"
```

### 启动 HTTP 服务

```bash
cortex serve                     # 默认 127.0.0.1:8000，内置离线 Demo 模型
```

打开 http://127.0.0.1:8000/docs 查看 Swagger 文档，或直接调用：

```bash
# 对话（Agent + RAG 工具）
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是混合检索？"}'

# 知识库入库
curl -X POST http://127.0.0.1:8000/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths": ["examples/data"]}'

# 混合检索
curl -X POST http://127.0.0.1:8000/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "RRF 融合", "top_k": 5}'

# 异步提交 Agent 任务（支持幂等键）
curl -X POST http://127.0.0.1:8000/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task": "调研 RAG 的混合检索", "idempotency_key": "req-001"}'
curl http://127.0.0.1:8000/v1/agents/{session_id}   # 查询状态

# Prometheus 指标
curl http://127.0.0.1:8000/metrics
```

### Docker 部署

```bash
cd docker && docker compose up --build
```

## 🎬 离线演示（无需 API Key）

三个开箱即用的演示，分别对应 Agent、RAG、多 Agent 编排：

```bash
cortex demo agent      # ReAct Agent：调用计算器/时钟工具
cortex demo rag        # RAG：文档入库 → 混合检索 → 带引用问答
cortex demo multi      # 多 Agent：规划 → 并行研究 → 核查 → 撰写
```

> 演示基于内置 `DemoProvider`（模拟模型行为），全程离线、确定性运行，适合 CI 验证、开发调试和向面试官现场演示。

## ✨ 功能详解

### 1. LLM 网关（多模型 + 可靠性）

```python
from cortex.llm import LLMGateway
from cortex.llm.providers import build_provider
from cortex.config import ProviderConfig

gateway = LLMGateway([
    build_provider(ProviderConfig(name="deepseek", kind="openai",
        base_url="https://api.deepseek.com/v1", api_key_env="DEEPSEEK_API_KEY",
        model="deepseek-chat")),
    build_provider(ProviderConfig(name="claude", kind="anthropic",
        api_key_env="ANTHROPIC_API_KEY", model="claude-3-5-sonnet-latest")),
])
```

- **统一抽象**：一套消息/工具/结构化输出格式，自动映射到各协议（OpenAI Chat Completions、Anthropic Messages、Gemini generateContent）
- **重试**：仅重试 429/5xx/超时，指数退避 + 随机抖动
- **熔断器**：连续失败达阈值自动打开，冷却后放行半开探测，避免雪崩
- **降级链**：主模型失败自动切换备用模型，全部失败抛出聚合错误
- **结构化输出**：JSON Schema 约束生成，解析/校验失败自动把错误回传模型修复重试

```python
class ExtractedInfo(BaseModel):
    name: str
    age: int

info = await generate_structured(gateway, messages, ExtractedInfo)  # 类型安全
```

### 2. RAG 知识库

完整管线：`解析 → 切分 → Embedding → 混合检索 → 重排序 → 生成`

```python
from cortex.config import CortexConfig
from cortex.rag import RAGPipeline

pipeline = RAGPipeline(CortexConfig.default())          # 默认 HashEmbedder 离线可用
report = await pipeline.ingest(["examples/data"])        # 支持文件/目录，失败不中断
result = await pipeline.search("什么是混合检索？", top_k=5)
for hit in result.hits:
    print(hit.score, hit.sources, hit.doc.page_content[:100])
```

- **文档解析**：txt / markdown / PDF（pypdf）/ DOCX（python-docx），坏文件报错但不断流程
- **递归切分**：中文句读感知（。！？）+ 重叠窗口，保留来源与切片序号元数据
- **混合检索**：向量（余弦）与 BM25 各取候选 → **RRF 加权融合**（`α/(k+rank_v) + (1-α)/(k+rank_b)`）
- **重排序**：默认规则重排（查询词命中率），可选 Cross-Encoder（`sentence-transformers`，缺失时优雅降级）
- **持久化**：向量 + 元数据落盘，重启秒级加载；索引可替换为 FAISS/Milvus/Qdrant

### 3. Agent 运行时

```python
from cortex.agent.react import ReActAgent
from cortex.agent.tools import ToolRegistry, tool

@tool(description="搜索知识库")
async def kb_search(query: str) -> str:
    ...

agent = ReActAgent(gateway, ToolRegistry([kb_search]))
result = await agent.run("帮我查一下年假政策")
```

- **ReAct 循环**：推理 → 工具调用 → 观察 → 再推理，工具异常自动回传模型自纠错，迭代次数硬上限防失控
- **工具系统**：`@tool` 装饰器从类型标注（含 Pydantic 模型）自动生成 JSON Schema，一套定义输出 OpenAI/Anthropic/Gemini 三种格式
- **上下文工程**：系统提示 + 工具清单按需组装，Token 预算自动裁剪旧消息
- **记忆**：短期滚动窗口 + LLM 摘要压缩；长期记忆 SQLite 持久化 + 向量相似检索，跨会话可用
- **运行时**：会话状态机（queued/running/done/failed/cancelled）、信号量并发控制、`idempotency_key` 幂等去重、SQLite 持久化与重启恢复

### 4. 多 Agent 编排

```
任务 → Planner（结构化规划）→ 多个 Researcher 并行执行（RAG 工具）
     → Critic（核查证据，pass/revise）→ Writer（带引用成文）
```

```python
from cortex.agent.planner import Planner
from cortex.agent.orchestrator import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator(
    planner=Planner(gateway),
    research_agents=[ReActAgent(gateway, rag_tools), ReActAgent(gateway, rag_tools)],
    gateway=gateway, max_parallel=2,
)
result = await orchestrator.run("调研：为什么 RAG 需要混合检索和重排序？")
print(result.report())   # 完整过程：计划、各步骤结果、核查意见、最终答案
```

- **Planner**：结构化输出生成 3–6 步可执行计划，标注依赖关系
- **并行研究**：`asyncio.gather` + 信号量限流，多个 Researcher 同时工作
- **Critic**：基于证据核查结论（pass / revise + 问题清单），核查失败优雅降级
- **Writer**：综合证据撰写最终答案，必须回应核查意见
- **计划执行器**：`PlannedExecutor` 支持失败自动重规划剩余步骤

### 5. MCP 集成

```python
from cortex.mcp import MCPClient, mcp_to_agent_tools

async with MCPClient([sys.executable, "my_mcp_server.py"]) as client:
    tools = await mcp_to_agent_tools(client)   # 外部工具 → Agent 工具
    registry = ToolRegistry(tools)             # 直接注册进 Agent
```

- stdio 传输、JSON-RPC 2.0、`initialize` 握手、请求超时、优雅关闭
- MCP 工具自动转换为 Agent 工具（含 JSON Schema），命名冲突自动加前缀

### 6. 可观测性

- **Trace ID**：每个 HTTP 请求分配 `X-Request-ID`，贯穿 LLM 调用 / 工具执行 / 检索的所有 Span
- **结构化日志**：JSON 格式（含 trace_id / span_id / duration_ms），直接对接 ELK/Loki
- **Prometheus 指标**（`GET /metrics`）：

| 指标 | 含义 |
| --- | --- |
| `llm_requests_total{provider,status}` | 各模型请求量（ok/http_429/circuit_open...） |
| `llm_retries_total{provider}` | 重试次数 |
| `llm_circuit_state{provider}` | 熔断状态（0/1） |
| `llm_latency_ms{provider}` | 模型延迟 |
| `tool_calls_total{tool,error}` | 工具调用量/失败量 |
| `agent_runs_total{agent,status}` | Agent 运行结果 |
| `runtime_sessions_total{status}` | 会话状态分布 |
| `rag_queries_total`、`http_requests_total{path,status}` | 检索与 HTTP 流量 |

## 🌐 HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthz` | 健康检查（组件状态、版本） |
| POST | `/v1/chat` | 对话（支持 `stream: true` SSE 流式） |
| POST | `/v1/rag/ingest` | 知识库入库（文件/目录） |
| POST | `/v1/rag/search` | 混合检索 |
| POST | `/v1/agents/run` | 异步提交 Agent 任务（支持幂等键） |
| GET | `/v1/agents/{id}` | 查询会话状态与结果 |
| GET | `/v1/agents` | 会话列表 |
| DELETE | `/v1/agents/{id}` | 取消会话 |
| GET | `/metrics` | Prometheus 指标 |

## ⚙️ 配置说明

配置通过 JSON 文件（环境变量 `CORTEX_CONFIG` 指定）或代码加载；**API Key 只存环境变量，不进配置文件**。

参考 [`examples/config.example.json`](examples/config.example.json)：配置了 DeepSeek / OpenAI / Claude / Gemini 四个 Provider + OpenAI Embedding + 熔断/重试/RAG 参数。

```bash
export DEEPSEEK_API_KEY=sk-xxx
export CORTEX_CONFIG=examples/config.example.json
cortex serve
```

不配置时使用内置离线默认值（DemoProvider + HashEmbedder），所有功能照常运行。

## 📁 项目结构

```
cortex-agent/
├── src/cortex/
│   ├── llm/            # 多模型网关：适配器、重试/熔断/降级、结构化输出
│   │   ├── gateway.py      # LLMGateway + CircuitBreaker + 重试
│   │   ├── providers.py    # OpenAI 兼容 / Anthropic / Gemini 适配器
│   │   ├── structured.py   # JSON Schema 结构化输出 + 修复重试
│   │   └── mock.py         # MockProvider（测试）/ DemoProvider（离线演示）
│   ├── rag/            # RAG 管线
│   │   ├── loaders.py      # txt/md/pdf/docx 解析
│   │   ├── chunking.py     # 递归切分（中文感知）
│   │   ├── embeddings.py   # OpenAI / Hash 兜底
│   │   ├── bm25.py         # Okapi BM25（CJK 分词）
│   │   ├── vector_store.py # numpy 向量存储 + 持久化
│   │   ├── retriever.py    # 混合检索 + RRF + 重排序
│   │   └── pipeline.py     # 入库/检索/保存/加载
│   ├── agent/          # Agent 内核
│   │   ├── tools.py        # 工具注册与 Schema 生成
│   │   ├── react.py        # ReAct 循环
│   │   ├── planner.py      # 任务规划 + 计划执行器
│   │   ├── orchestrator.py # 多 Agent 编排
│   │   ├── memory.py       # 短期/长期记忆
│   │   ├── runtime.py      # 会话运行时（并发/幂等/持久化）
│   │   └── builtin_tools.py# 内置工具（计算器/时钟/RAG/记忆）
│   ├── mcp/            # MCP 客户端与工具桥接
│   ├── api/            # FastAPI 应用、路由、中间件
│   ├── config.py       # 配置模型
│   ├── logging.py      # 结构化日志 + Trace/Span
│   ├── metrics.py      # 指标注册表 + Prometheus 导出
│   ├── demo.py         # 离线演示
│   └── cli.py          # 命令行入口
├── tests/              # 70 个单元/集成测试（含 MCP 模拟服务器，离线运行）
├── examples/           # 演示脚本 + 示例知识库 + 配置样例
├── docs/               # 需求分析、架构设计文档
├── docker/             # Dockerfile + compose
└── .github/workflows/  # CI（lint + 测试，Python 3.10/3.11）
```

## 🧪 测试与代码质量

```bash
pip install -e ".[dev]"
pytest            # 全量测试（离线运行，无需网络）
ruff check src tests   # 静态检查
```

测试覆盖：网关重试/熔断/降级、结构化输出修复、ReAct 循环（含工具报错自恢复）、规划器重规划、
多 Agent 编排、运行时幂等/并发/持久化、记忆、上下文裁剪、混合检索、RAG 入库、MCP 协议、API 集成。

CI（GitHub Actions）：Python 3.10 / 3.11 双版本，ruff + pytest 全绿。

## 📚 架构与设计文档

- [`docs/DESIGN.md`](docs/DESIGN.md) — 需求分析、技术选型、关键方案、里程碑与风险（完整的技术设计过程）
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 架构图、模块职责、请求链路、可替换点

## 🗺️ Roadmap

- [x] 多模型网关 + 重试/熔断/降级 + 结构化输出
- [x] RAG 全链路（解析/切分/混合检索/重排序）
- [x] ReAct Agent + 规划 + 记忆 + 多 Agent 编排
- [x] MCP 客户端与工具桥接
- [x] 运行时幂等/并发/持久化 + 可观测性
- [x] 离线演示、Docker、CI
- [ ] 模型原生流式输出（SSE token 级）
- [ ] 语义缓存（降低重复问题成本）
- [ ] 多模态文档解析（OCR / 表格）
- [ ] 语音对话前端（WebSocket + ASR/TTS）
- [ ] 分布式向量库与消息队列

## 📄 License

[MIT](LICENSE)

---

⭐ 如果这个项目对你有帮助，欢迎 Star！问题与建议请提 [Issue](https://github.com/Artlatte/cortex-agent/issues)。
