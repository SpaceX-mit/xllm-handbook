# xLLM 开发者手册 - 总览

## 文档体系架构

本手册按照 **IPD (Integrated Product Development) 集成产品开发流程** 结合 **AI Native 开发理念** 构建，旨在为开发者提供：

1. **AI可复刻验收** - 每个文档都包含结构化的验收标准，AI工具可直接解析并执行验证
2. **人类可读理解** - 保持技术深度与可读性的平衡，使用代码片段和图表辅助理解

## 文档层级结构

```
handbook/
├── [00] 概念与架构层 (Concept & Architecture)
│   ├── 00_OVERVIEW.md                    # 本文档 - 文档体系总览
│   ├── 01_GLOSSARY.md                    # 核心概念术语表
│   ├── 02_ARCHITECTURE.md                # 系统架构设计
│   └── 03_DESIGN_PRINCIPLES.md           # 设计原则与哲学
│
├── [01] 需求与规格层 (Requirements & Specifications)  
│   ├── 01_PRODUCT_REQUIREMENTS.md        # 产品需求规格
│   ├── 02_FUNCTIONAL_SPEC.md             # 功能规格说明书
│   └── 03_NON_FUNCTIONAL_SPEC.md         # 非功能需求规格
│
├── [02] 设计与建模层 (Design & Modeling)
│   ├── 01_DOMAIN_MODEL.md                # 领域模型设计
│   ├── 02_SEQUENCE_DIAGRAMS.md           # 核心时序图
│   ├── 03_COMPONENT_DESIGN.md            # 组件设计
│   └── 04_DATA_FLOW.md                   # 数据流设计
│
├── [03] 接口与协议层 (Interface & Protocol)
│   ├── 01_API_DESIGN.md                  # API设计规范
│   ├── 02_PROTO_DEFINITIONS.md           # 通信协议定义
│   └── 03_CONFIG_SCHEMA.md                # 配置schema
│
├── [04] 实现与代码层 (Implementation)
│   ├── 01_MODULE_IMPLEMENTATIONS/        # 各模块实现详解
│   ├── 02_CODE_PATTERNS.md               # 核心代码模式
│   └── 03_ALGORITHM_REFERENCES.md         # 算法实现参考
│
├── [05] 测试与验证层 (Testing & Verification)
│   ├── 01_TEST_STRATEGY.md               # 测试策略
│   ├── 02_UNIT_TEST_GUIDE.md             # 单元测试指南
│   ├── 03_INTEGRATION_TEST.md            # 集成测试
│   └── 04_BENCHMARK_SPEC.md              # 性能基准测试
│
├── [06] 部署与运维层 (Deployment & Operations)
│   ├── 01_DEPLOYMENT_GUIDE.md            # 部署指南
│   ├── 02_OPS_MANUAL.md                  # 运维手册
│   └── 03_TROUBLESHOOTING.md             # 故障排查
│
└── [07] AI Native 开发规范
    ├── 01_AI_COPILOT_CONVENTIONS.md       # AI辅助编程规范
    ├── 02_CODE_GENERATION_STANDARDS.md    # 代码生成标准
    ├── 03_AUTOMATED_REFACTORING.md        # 自动重构规范
    └── 04_VERIFICATION_CRITERIA.md        # AI验收标准
```

## 文档验收机制

每个文档包含以下元信息，供AI工具解析：

```yaml
---
document_id: ARCH-001
version: 1.0.0
category: architecture
owner: xllm-team
last_updated: 2026-07-23
verification_level: AI_PARSEABLE | HUMAN_READABLE | BOTH
ai_acceptance_criteria:
  - criterion: "文档可被Markdown解析器完整解析"
  - criterion: "代码示例可直接复制执行"
  - criterion: "图表描述可被Graphviz/Mermaid渲染"
status: APPROVED
---
```

## 核心设计理念

### 1. 领域驱动设计 (DDD)

xLLM 采用 DDD 原则组织代码：

| 限界上下文 | 核心领域 | 代码位置 |
|-----------|---------|---------|
| **调度域** | 请求调度、批处理、优先级 | `core/scheduler/` |
| **推理域** | 模型执行、前向计算 | `core/runtime/` |
| **缓存域** | KV Cache、Block管理 | `core/framework/block/` |
| **模型域** | 模型加载、权重管理 | `core/framework/model/` |
| **服务域** | API服务、协议解析 | `api_service/` |

### 2. 智能体协作模式

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Native 开发流程                        │
├─────────────────────────────────────────────────────────────┤
│  需求 → 规格 → 设计 → 实现 → 测试 → 部署 → 监控              │
│    ↑        ↑        ↑       ↑       ↑       ↑              │
│  [Agent-R] [Agent-S] [Agent-D] [Agent-I] [Agent-T] [Agent-O]│
└─────────────────────────────────────────────────────────────┘
```

每个阶段由专门的AI Agent负责，形成协作闭环。

## 快速导航

| 模块 | 核心类 | 职责 | 入口文档 |
|-----|-------|------|---------|
| **Scheduler** | `Scheduler`, `ContinuousScheduler` | 请求调度与批处理 | [Scheduler设计](./02_DESIGN_AND_MODELING/02_SCHEDULER_DESIGN.md) |
| **Worker** | `Worker`, `WorkerImpl` | 模型执行引擎 | [Worker设计](./02_DESIGN_AND_MODELING/03_WORKER_DESIGN.md) |
| **Batch** | `Batch`, `BatchManager` | 动态批处理管理 | [Batch设计](./02_DESIGN_AND_MODELING/04_BATCH_DESIGN.md) |
| **KVCache** | `KVCache`, `BlockManager` | KV缓存管理 | [Cache设计](./02_DESIGN_AND_MODELING/05_KV_CACHE_DESIGN.md) |
| **Model** | `CausalLM`, `ModelLoader` | 模型加载与推理 | [Model设计](./02_DESIGN_AND_MODELING/06_MODEL_DESIGN.md) |

## 版本历史

| 版本 | 日期 | 更新内容 | 作者 |
|-----|------|---------|------|
| 1.0.0 | 2026-07-23 | 初始版本 | xLLM Team |
