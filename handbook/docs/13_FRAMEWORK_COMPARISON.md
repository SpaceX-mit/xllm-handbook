# xLLM vs llama.cpp 深度对比分析

## 文档信息

```yaml
---
document_id: COMP-001
version: 1.0.0
category: comparison
owner: xllm-team
verification_level: HUMAN

abstract: |
  本文档对 xLLM 和 llama.cpp 两个 LLM 推理框架进行深度对比分析，
  从架构设计、模块组织、API 设计、测试策略、编码规范等多个维度
  进行系统性的对比，旨在帮助开发者理解不同设计哲学下的实现差异。
---
```

---

## 1. 概述

### 1.1 项目定位

| 维度 | xLLM | llama.cpp |
|------|------|-----------|
| **项目性质** | 企业级分布式推理框架 | 开源轻量级推理引擎 |
| **代码规模** | 大型复杂系统 | 简洁核心实现 |
| **目标平台** | 多 GPU 集群、NPU、MLU、DCU | PC、服务器、各类硬件 |
| **用户群体** | 企业内部/商业部署 | 开源社区/个人开发者 |

### 1.2 设计哲学差异

**xLLM 哲学宣言**：

```
"性能是功能，可观测是必需，可维护是责任。"
```

| 核心价值 | 落地实践 |
|---------|---------|
| **性能即功能** | Continuous Batching、Prefix Cache、投机解码 |
| **可观测必需** | Metrics、Tracing、Logging |
| **可维护是责任** | DDD、Clean Architecture、文档 |

**llama.cpp 哲学宣言**：

```
"简单性是可靠性的先决条件" — Edsger Dijkstra
```

| 核心原则 | 说明 |
|---------|------|
| **简单性优先** | 除非有性能数据证明，复杂性是不必要的 |
| **零成本抽象** | 高级 API 不应产生运行时开销 |
| **显式优于隐式** | 避免魔法般的隐式转换 |
| **渐进式复杂度** | 基础功能简单，高级功能可组合 |

---

## 2. 文档体系对比

### 2.1 目录结构

**xLLM Handbook** — 按功能域组织：

```
handbook/docs/
├── 00_OVERVIEW.md                    # 文档总览
├── 01_GLOSSARY.md                    # 术语表
├── 02_ARCHITECTURE.md                # 架构设计
├── 03_DESIGN_PRINCIPLES.md           # 设计原则
├── 04_DOMAIN_MODEL.md                # DDD 领域模型
├── 05_SCHEDULER_DESIGN.md            # 调度器设计
├── 06_WORKER_DESIGN.md               # Worker 设计
├── 07_KV_CACHE_DESIGN.md             # KV Cache 设计
├── 08_AI_NATIVE_DEVELOPMENT.md       # AI Native 开发
├── 09_API_DESIGN.md                  # API 设计
├── 10_CONFIG_SCHEMA.md               # 配置 Schema
├── 11_TEST_STRATEGY.md               # 测试策略
└── 12_DEPLOYMENT.md                  # 部署指南
```

**llama.cpp Handbook** — 按 IPD 阶段组织：

```
handbook/
├── 00-overview/                      # 概念阶段
│   ├── 01-product-vision.md
│   ├── 02-technical-vision.md
│   ├── 03-architecture-overview.md
│   └── 04-design-principles.md
├── 01-requirements/                  # 需求阶段
│   ├── 01-functional-spec.md
│   ├── 02-performance-req.md
│   └── 03-api-contract.md
├── 02-design/                        # 设计阶段
│   ├── 01-system-design.md
│   └── 02-module-design/
│       ├── 01-model-loader.md
│       ├── 02-context-manager.md
│       ├── 03-kv-cache.md
│       ├── 04-vocab.md
│       ├── 05-sampler.md
│       ├── 06-graph-executor.md
│       └── 07-memory-manager.md
├── 03-implementation/                # 实现阶段
├── 04-test/                         # 测试阶段
└── 05-release/                      # 发布阶段
```

### 2.2 文档风格差异

| 维度 | xLLM | llama.cpp |
|------|------|-----------|
| **语言** | 中文为主 | 英文为主 |
| **组织** | 功能域划分 | IPD 流程阶段划分 |
| **侧重点** | 架构设计、领域模型 | 模块实现、贡献指南 |
| **目标读者** | 企业内部开发者 | 开源社区贡献者 |

---

## 3. 架构设计对比

### 3.1 架构层次

**xLLM 5 层架构**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           xLLM Architecture                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      Service Layer                                 │  │
│  │     OpenAI API │ Anthropic API │ Internal API                      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Scheduler Layer                                  │  │
│  │  Continuous │ DisaggPD │ FixedSteps │ ZeroEvict                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      Engine Layer                                  │  │
│  │                   WorkerPool (TP×CP)                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Runtime Layer                                  │  │
│  │      CausalLM │ CausalVLM │ DitModel │ RecCausalLM               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Backend Layer                                  │  │
│  │                   CUDA │ NPU │ MLU │ DCU                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**llama.cpp 4 层架构**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            llama.cpp                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │            llama-cli │ llama-server │ libllama                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              llama_model + llama_context                          │  │
│  │     vocab │ arch │ hparams │ kv_cache │ sampler                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    ggml (Compute Engine)                          │  │
│  │              ops │ tensor │ alloc │ compute                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                  ggml-backend (Device)                            │  │
│  │              CPU │ CUDA │ Metal │ Vulkan │ SYCL                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心差异分析

| 架构维度 | xLLM | llama.cpp | 设计取舍 |
|---------|------|-----------|---------|
| **并行策略** | TP/PP/DP/CP/EP 完整支持 | 主要依赖 GGML 后端 | xLLM 支持大规模分布式 |
| **调度策略** | 多种策略可选 | ggml_backend_sched | xLLM 强调 SLO 驱动 |
| **模型类型** | 多类型 (LLM/VLM/Rec/Dit) | 主要 CausalLM | xLLM 通用性更强 |
| **后端抽象** | 多硬件后端适配层 | ggml-backend 统一抽象 | llama.cpp 更统一简洁 |
| **分布式** | 原生支持多节点 | 无原生支持 | 定位不同 |

---

## 4. 模块设计对比

### 4.1 KV Cache 设计

**xLLM KV Cache 架构**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KV Cache Block Management                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      BlockManager                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐   │  │
│  │  │                   Free Blocks Pool                        │   │  │
│  │  │   ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐    │   │  │
│  │  │   │ B0 │ │ B1 │ │ B2 │ │ B3 │ │ B4 │ │ B5 │ │ B6 │ ... │   │  │
│  │  │   └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘       │   │  │
│  │  └─────────────────────────────────────────────────────────┘   │  │
│  │                            │                                     │  │
│  │         ┌──────────────────┼──────────────────┐                │  │
│  │         ▼                  ▼                  ▼                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │  │
│  │  │ Sequence A  │  │ Sequence B  │  │ Sequence C  │            │  │
│  │  │ [B0,B1,B2] │  │ [B3,B4]    │  │ [B5,B6,B7] │            │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      Prefix Cache                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐   │  │
│  │  │ Hash(Key) │ Tokens │ BlockIds │ RefCount │ TTL │ Status │   │  │
│  │  ├───────────┼────────┼──────────┼──────────┼─────┼─────────┤   │  │
│  │  │ 0x1234... │[1,2,3] │ [B0,B1] │    3    │ ∞   │ ACTIVE  │   │  │
│  │  │ 0x5678... │[1,2]   │ [B2,B3] │    1    │ 60s │ COLD    │   │  │
│  │  └─────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**llama.cpp KV Cache 操作**：

```cpp
// KV 缓存核心操作
llama_kv_cache_seq_rm(ctx, seq_id, start, end);    // 删除序列
llama_kv_cache_seq_cp(ctx, src_seq, dst_seq, ...);  // 复制序列
llama_kv_cache_seq_keep(ctx, seq_id);                // 保留序列
llama_kv_cache_seq_shift(ctx, seq_id, ...);          // 平移位置
```

| KV Cache 特性 | xLLM | llama.cpp |
|--------------|------|-----------|
| **Block 管理** | BlockManager 抽象 | 直接结构体 |
| **Prefix Cache** | 完整实现 | 无原生支持 |
| **多序列支持** | SequencesGroup | 多 seq_id |
| **异构打包** | 复杂策略 | 基础 batch |
| **引用计数** | 支持 | 不支持 |

### 4.2 Scheduler 设计

**xLLM Scheduler 策略**：

```cpp
// 调度策略工厂
class SchedulerFactory {
 public:
    static std::unique_ptr<Scheduler> Create(SchedulerType type) {
        switch (type) {
            case SchedulerType::CONTINUOUS:
                return std::make_unique<ContinuousScheduler>();
            case SchedulerType::DISAGG_PD:
                return std::make_unique<DisaggPDScheduler>();
            case SchedulerType::FIXED_STEPS:
                return std::make_unique<FixedStepsScheduler>();
            case SchedulerType::ZERO_EVICT:
                return std::make_unique<ZeroEvictScheduler>();
        }
    }
};

// SLO 驱动优先级
Urgency calculate_urgency(const Request& request) {
    int32_t remaining_time = request.get_remaining_time();
    if (remaining_time <= request.ttft_slo_ms() * 0.5) {
        return Urgency::STARVED;  // 接近 TTFT 截止
    }
    if (remaining_time <= request.tpot_slo_ms() * 2) {
        return Urgency::URGENT;   // 接近 TPOT 截止
    }
    return Urgency::NORMAL;
}
```

**llama.cpp Scheduler**：

```cpp
// 基于 ggml_backend_sched 的调度
ggml_backend_sched_t sched = ggml_backend_sched_new(backends, n_backends);

// 自动批处理
ggml_backend_sched_graph_compute(sched, graph);
```

| Scheduler 特性 | xLLM | llama.cpp |
|---------------|------|-----------|
| **调度策略** | 多种策略可选 | ggml_backend_sched |
| **优先级** | SLO 驱动 | 基础调度 |
| **Continuous Batching** | 原生支持 | 基础支持 |
| **P/D 分离** | DisaggPDScheduler | 无 |
| **Prefix Cache 调度** | 支持 | 无 |

### 4.3 Worker 设计

**xLLM Worker 架构**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Worker Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                         Worker                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                  Public Interface                            │  │  │
│  │  │  init_model │ sleep │ wakeup │ step │ allocate_kv_cache     │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                            │                                       │  │
│  │  ┌─────────────────────────┴─────────────────────────────────┐  │  │
│  │  │                    WorkerImpl (PIMPL)                       │  │  │
│  │  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │  │  │
│  │  │  │   Executor     │  │  KVCacheMgr    │  │ Model        │  │  │  │
│  │  │  │ (ACL/CUDA/...)│  │               │  │ (CausalLM)   │  │  │  │
│  │  │  └────────────────┘  └────────────────┘  └──────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Worker 类型层次:                                                         │
│  ┌─────────────────────┐                                                │
│  │ WorkerImpl          │  (基类)                                         │
│  │ ├─ LLMWorkerImpl    │  (语言模型)                                      │
│  │ ├─ VLMWorkerImpl    │  (视觉语言模型)                                  │
│  │ ├─ RecWorkerImpl    │  (推荐模型)                                      │
│  │ └─ DitWorkerImpl    │  (扩散模型)                                      │
│  └─────────────────────┘                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**llama.cpp 结构**：

```cpp
// 简洁的两核心结构
struct llama_model {
    llm_arch arch;
    struct llama_hparams hparams;
    struct llama_vocab vocab;
    std::vector<llama_layer> layers;
    // ...
};

class llama_context {
    const llama_model & model;
    std::unique_ptr<llama_kv_cache> kv_self;
    ggml_backend_sched_t sched;
    std::unique_ptr<llama_sampler_chain> sampler;
    // ...
};
```

| Worker 特性 | xLLM | llama.cpp |
|------------|------|-----------|
| **生命周期** | Init→Ready→Sleep↔Wakeup→Exit | 加载/释放 |
| **异步接口** | folly::SemiFuture | 同步为主 |
| **多模型类型** | LLM/VLM/Embedding/Rec/Dit | 主要 CausalLM |
| **PIMPL 模式** | 使用 | 不使用 |
| **状态管理** | 完整状态机 | 简单状态 |

---

## 5. API 设计对比

### 5.1 xLLM OpenAI 兼容 API

```yaml
# 扩展字段
"extra_body": {
    "request_id": "string",       # 请求 ID
    "priority": "integer",        # 优先级 0-9
    "ttft_slo_ms": "integer",   # TTFT SLO (ms)
    "tpot_slo_ms": "integer",    # TPOT SLO (ms)
    "ttlt_slo_ms": "integer",    # TTLT SLO (ms)
    "prefix_cache": "boolean"    # 前缀缓存
}
```

### 5.2 llama.cpp C API

```c
// 创建/销毁
struct llama_model * llama_model_load_from_file(const char *, struct llama_model_params);
void llama_model_free(struct llama_model *);

struct llama_context * llama_new_context_with_model(struct llama_model *, struct llama_context_params);
void llama_free(struct llama_context *);

// 推理
int llama_decode(struct llama_context *, struct llama_batch);
int llama_encode(struct llama_context *, struct llama_batch);

// 采样
llama_token llama_sampler_sample(llama_sampler *, struct llama_context *, int);
```

| API 维度 | xLLM | llama.cpp |
|---------|------|-----------|
| **协议** | REST/JSON + Protocol Buffers | C API + HTTP Server |
| **兼容性** | OpenAI v1.x + Anthropic | OpenAI 兼容 Server |
| **扩展性** | 企业级扩展 (SLO/Priority) | 基础扩展 |
| **类型安全** | Protobuf 定义 | C struct |

---

## 6. 测试策略对比

### 6.1 xLLM 测试金字塔

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Testing Pyramid                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                                    ▲                                      │
│                                   /E2E\                                   │
│                                  /     \                                  │
│                                 /  集成  \                                │
│                                /   测试   \                               │
│                               /────────────\                              │
│                              /    组件      \                             │
│                             /     测试       \                            │
│                            /──────────────────\                           │
│                           /      单元          \                          │
│                          /       测试           \                         │
│                         /────────────────────────\                        │
│                                                                          │
│   比例:  单元测试 70% | 组件测试 20% | 集成测试 8% | E2E测试 2%          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 llama.cpp 测试

```cpp
#define DOCTEST_CONFIG_SUPER_EXCLUSIVE_STATES
#include <doctest/doctest.h>

TEST_CASE("Inference Pipeline") {
    auto * model = load_test_model();
    auto * ctx = llama_new_context_with_model(model, params);
    auto * sampler = llama_sampler_chain_from_str(ctx, "temp:0.7,top_k:40");
    // ...
}
```

| 测试维度 | xLLM | llama.cpp |
|---------|------|-----------|
| **测试框架** | Google Test + Mock | doctest |
| **Mock 框架** | 完善 (MockScheduler 等) | 无 |
| **性能测试** | SLO 合规测试 | 基础 benchmark |
| **并发测试** | 100+ 并发请求 | 基础测试 |
| **CI/CD** | 完整流水线 | 基础 CI |

---

## 7. 编码规范对比

### 7.1 命名规范

**xLLM 命名**：

```cpp
// 文件: snake_case
llama_model.cpp
llama_kv_cache.cpp

// 类型: camelCase
class ContinuousScheduler;
struct BatchManager;

// 函数: snake_case
void add_request(std::shared_ptr<Request>& request);
ForwardInput prepare_inputs(Batch& batch);
```

**llama.cpp 命名**：

```c
// 文件: snake_case
llama-model.cpp
llama-kv-cache.cpp

// 类型: llama_<module>_<name>
struct llama_model_params;
enum llama_ftype;

// 成员: trailing_underscore
struct llama_context {
    const llama_model & model_;
    uint32_t n_ctx_;
    bool is_sampling_;
};

// 常量: SCREAMING_SNAKE
#define LLAMA_MAX_N_TOKENS 2048
```

### 7.2 AI 开发约束

**xLLM AI Native**：

```cpp
// @ai_action: 生成单元测试
// @ai_verification: 验证 KV Cache 正确分配
// @brief: 分配指定数量的 Block
// @param: num_blocks - 需要的 Block 数量
// @return: 分配的 Block ID 列表
class BlockManager {
 public:
    virtual std::vector<uint32_t> allocate_blocks(uint32_t num_blocks) = 0;
};
```

**llama.cpp AI 约束**：

```
禁止:
✗ AI 写 PR 描述
✗ AI 提交 commit
✗ 不理解代码就提交
✗ 自动化提交/PR

允许:
✓ 学习、探索、理解代码库
✓ 人类代码的建议
✓ 机械性任务 (格式化、完成模式)
✓ 文档草稿 (贡献者已理解组件)
```

---

## 8. 部署模式对比

### 8.1 部署架构

**xLLM 部署模式**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    P/D Separation Deployment                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                      Load Balancer                               │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                    │                          │                          │
│                    ▼                          ▼                          │
│  ┌─────────────────────────┐    ┌─────────────────────────┐           │
│  │    Prefill Cluster      │    │     Decode Cluster       │           │
│  │  ┌─────────────────┐   │    │  ┌─────────────────┐    │           │
│  │  │ Prefill Node 1  │   │    │  │  Decode Node 1  │    │           │
│  │  │  • High Memory  │   │    │  │  • Low Latency  │    │           │
│  │  └─────────────────┘   │    │  └─────────────────┘    │           │
│  └─────────────────────────┘    └─────────────────────────┘           │
│                 │                                │                       │
│                 │         KV Cache Transfer      │                       │
│                 └────────────────────────┬────────┘                       │
│                                          ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    RDMA Network / Mooncake                          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**llama.cpp 部署模式**：

```bash
# CLI 模式
./llama-cli -m model.gguf -i

# Server 模式
./llama-server -m model.gguf --host 0.0.0.0 --port 8080

# 库模式
#include <llama.h>
auto * model = llama_load_model_from_file("model.gguf", params);
```

| 部署维度 | xLLM | llama.cpp |
|---------|------|-----------|
| **单机部署** | 多 GPU Worker Pool | 单/多 GPU |
| **P/D 分离** | 原生支持 | 不支持 |
| **多节点** | Coordinator + NCCL/GLOO | 不支持 |
| **容器化** | Docker + K8s | 基础 Docker |
| **复杂度** | 高 | 低 |

---

## 9. 设计哲学总结

### 9.1 核心差异

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    设计哲学对比                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   xLLM                              llama.cpp                           │
│   ─────                              ────────                            │
│                                                                          │
│   企业级 + 分布式                    轻量级 + 本地                         │
│       │                                  │                                │
│       ├── 强调可扩展性                      ├── 强调简单性                  │
│       ├── 强调可观测性                      ├── 零成本抽象                  │
│       ├── 强调 SLO 合规                     ├── 显式优于隐式                │
│       ├── DDD 领域边界                      └── 渐进式复杂度                │
│       └── 多层抽象                                                      │
│                                                                          │
│   适用场景:                          适用场景:                            │
│   • 大规模集群推理                     • 本地/边缘部署                    │
│   • 企业级服务                         • 资源受限环境                      │
│   • 多模型多租户                       • 快速原型开发                      │
│   • SLO 严格保障                       • 简单集成                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 互补价值

| 学习方向 | xLLM 提供 | llama.cpp 提供 |
|---------|----------|---------------|
| **架构设计** | 企业级分布式架构 | 简洁核心架构 |
| **设计模式** | DDD、PIMPL、工厂模式 | 零成本抽象、渐进式复杂度 |
| **工程实践** | 大型项目组织 | 小型项目简洁 |
| **代码风格** | 企业级规范 | 开源社区风格 |
| **测试策略** | 完善测试体系 | 轻量测试框架 |

---

## 10. 相关文档

| 文档 | 说明 |
|-----|------|
| [架构设计](./02_ARCHITECTURE.md) | xLLM 系统架构 |
| [设计原则](./03_DESIGN_PRINCIPLES.md) | xLLM 设计哲学 |
| [测试策略](./11_TEST_STRATEGY.md) | xLLM 测试规范 |
| [llama.cpp Handbook](../../llama.cpp-handbook/handbook/README.md) | llama.cpp 文档体系 |

---

## 附录 A: 关键术语对照

| 术语 | xLLM | llama.cpp |
|-----|------|-----------|
| 推理引擎 | LLMEngine + Worker | llama_context |
| 模型加载 | Worker::init_model | llama_model_load_from_file |
| 批处理 | Batch + Scheduler | ggml_backend_sched |
| KV 缓存 | BlockManager + KVCache | llama_kv_cache |
| 采样器 | 内置 | llama_sampler_chain |
| 调度器 | ContinuousScheduler | ggml_backend_sched |
| 位置编码 | RoPE (内置) | RoPE (内置) |
| 量化支持 | 多种 | GGUF 量化格式 |
