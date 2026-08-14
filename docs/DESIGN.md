# 需求分析与技术方案设计

> 本文档演示从业务需求到技术设计的完整过程，对应"独立完成业务分析、技术设计、开发、测试和上线"的能力要求。

## 1. 需求分析

### 1.1 背景与痛点

企业接入大模型时普遍遇到三类问题：

| 痛点 | 具体表现 |
| --- | --- |
| **幻觉与知识脱节** | 模型不知道企业内部知识，答案无法追溯，用户不信任 |
| **单模型绑定** | 依赖单一供应商，限流/宕机时业务不可用，成本不可控 |
| **Demo 级工程** | 无重试、无超时、无熔断、无监控，无法上生产；Agent 无限循环、上下文爆炸 |

### 1.2 目标用户与场景

- **场景 A：企业知识库问答**。员工问"年假政策是什么"，系统检索内部制度文档并给出带引用的答案。
- **场景 B：数据/调研助手**。给定一个调研目标，多个 Agent 并行检索、核查、汇总成报告。
- **场景 C：业务系统集成**。通过 MCP 把企业内部工具（CRM、工单系统）接入 Agent，自然语言驱动业务操作。

### 1.3 需求清单（P0 / P1）

**P0（核心）**

1. RAG 全链路：文档解析（txt/md/pdf/docx）→ 切分 → Embedding → 向量检索；
2. 混合检索（BM25 + 向量 + RRF 融合）+ 重排序；
3. Agent 运行时：ReAct 工具调用循环、任务规划、短期/长期记忆；
4. 多模型网关：OpenAI 兼容 / Anthropic / Gemini，结构化输出；
5. 生产可靠性：超时、重试、熔断、降级、幂等、并发控制；
6. 可观测性：Trace ID、结构化日志、Prometheus 指标；
7. HTTP API：对话、RAG 入库/检索、Agent 任务提交与状态查询。

**P1（增强）**

8. 多 Agent 编排（规划 → 并行研究 → 核查 → 撰写）；
9. MCP 客户端（动态接入外部工具）；
10. 离线演示模式（无 API Key 可跑通全流程，方便验证与演示）；
11. Docker 部署、CI、代码质量门禁。

### 1.4 非目标（本期不做）

- 多租户与权限体系；分布式向量库（接口预留，默认本地实现）；
- 语音、OCR、图片理解（架构上通过工具/多模态模型扩展）。

## 2. 技术选型

| 维度 | 选型 | 理由 |
| --- | --- | --- |
| 语言 | Python 3.10+ | AI 生态最丰富；asyncio 支撑高并发 IO |
| Web 框架 | FastAPI + uvicorn | 异步原生、Pydantic 校验、自动文档 |
| 模型接入 | httpx（自研适配层） | 不绑定 SDK 版本；三种协议统一抽象；可用 MockTransport 测试 |
| 结构化输出 | Pydantic v2 + JSON Schema | 校验 + 修复重试闭环 |
| 向量存储 | numpy 本地实现（接口可替换） | 零依赖可跑通；生产替换 FAISS/Milvus/Qdrant |
| 全文检索 | 自研 BM25（Okapi） | 中文友好分词（CJK 单字 + ASCII 词） |
| 记忆 | SQLite | 单机零运维；向量库化时平滑迁移 |
| 测试 | pytest + pytest-asyncio + ruff | 标准工具链 |
| 部署 | Docker + GitHub Actions | 一键构建、CI 门禁 |

## 3. 关键技术方案

### 3.1 多模型网关的可靠性设计

```
请求 → 主模型 Provider
        ├─ 熔断器：连续失败 N 次 → 打开 → 冷却后放行半开探测
        ├─ 重试：仅对 429/5xx/超时 重试，指数退避 + 随机抖动
        └─ 失败 → 按权重降级到备用 Provider → 全失败则聚合报错
```

### 3.2 混合检索与 RRF

- 向量检索与 BM25 各取 `max(top_k×4, 20)` 个候选；
- RRF 加权融合：`score = α/(k+rank_v) + (1-α)/(k+rank_b)`，k=60；
- 规则重排（查询词命中率加分）默认开启，Cross-Encoder 可选（缺失时优雅降级）。

### 3.3 Agent 循环与防失控

- 最大迭代次数硬上限；
- 工具调用参数由 JSON Schema 约束，执行异常作为观察结果回传模型自纠错；
- Token 预算裁剪 + 历史摘要压缩。

### 3.4 幂等与状态持久化

- `idempotency_key → session_id` 唯一映射（SQLite），重复提交返回同一会话；
- 会话状态机 `queued → running → done/failed/cancelled`，每步落库，重启可恢复。

### 3.5 可观测性

- 每请求一个 Trace ID，贯穿 LLM 调用/工具执行/检索 Span；
- 指标：`llm_requests_total{provider,status}`、`llm_retries_total`、`llm_circuit_state`、
  `tool_calls_total{tool,error}`、`runtime_sessions_total{status}`、`http_requests_total{path,status}` 等。

## 4. 里程碑与验收标准

| 里程碑 | 验收标准 |
| --- | --- |
| M1 网关 + Agent 内核 | 单测覆盖重试/熔断/降级；ReAct 循环可离线跑通 |
| M2 RAG 全链路 | 混合检索命中示例文档；入库/检索 API 可用 |
| M3 多 Agent + MCP | 编排结果包含计划/研究/核查/成文；MCP 工具桥接测试通过 |
| M4 上线化 | ruff 零告警、pytest 全绿、Docker 可启动、CI 通过 |

## 5. 风险与对策

| 风险 | 对策 |
| --- | --- |
| Embedding 模型不可用 | 内置确定性 HashEmbedder 兜底，离线可跑 |
| 重排序模型体积大 | 默认规则重排，Cross-Encoder 懒加载 |
| 单模型供应商故障 | 多 Provider 降级链 + 熔断 |
| Agent 失控（循环/幻觉） | 迭代上限、工具参数校验、Critic 核查 |
