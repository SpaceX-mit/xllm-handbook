# xLLM vs llama.cpp vs vLLM 深度对比分析

## 文档信息

```yaml
---
document_id: COMP-002
version: 1.0.0
category: comparison
owner: xllm-team
verification_level: HUMAN

abstract: |
  本文档对 xLLM、llama.cpp 和 vLLM 三个主流 LLM 推理框架进行系统性对比，
  从架构设计、核心特性、性能优化、社区生态等多个维度进行分析，
  旨在帮助开发者理解不同框架的设计哲学和技术选择。
---
```

---

## 1. 概述

### 1.1 项目定位

| 维度 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **项目性质** | 企业级分布式推理框架 | 开源轻量级推理引擎 | 开源高性能推理服务框架 |
| **开发主体** | 内部团队 | GGML 社区 | UC Berkeley / vLLM 团队 |
| **代码规模** | 大型复杂系统 | 轻量核心 (~200K 行 C/C++) | 中大型 (~500K 行 Python/C++) |
| **主要语言** | C++ | C/C++ | Python/C++ |
| **目标平台** | 多 GPU 集群、NPU、MLU、DCU | PC、服务器、各类硬件 | GPU 服务器集群 |
| **发布时间** | 内部项目 | 2023 | 2023 |

### 1.2 核心特性矩阵

| 特性 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **Continuous Batching** | ✅ 原生支持 | ⚠️ 有限支持 | ✅ 原生支持 |
| **PagedAttention** | ✅ 实现 | ❌ 无 | ✅ 核心特性 |
| **Prefix Caching** | ✅ 支持 | ❌ 无 | ✅ 支持 |
| **Speculative Decoding** | ✅ 支持 | ❌ 无 | ✅ 支持 |
| **张量并行 (TP)** | ✅ 支持 | ❌ 无 | ✅ 支持 |
| **流水线并行 (PP)** | ✅ 支持 | ❌ 无 | ⚠️ 有限支持 |
| **数据并行 (DP)** | ✅ 支持 | ⚠️ 多实例 | ⚠️ 多实例 |
| **多模态支持** | ✅ VLM | ⚠️ 有限 | ✅ 支持 |
| **MoE 支持** | ✅ 支持 | ✅ 有限 | ✅ 支持 |

### 1.3 设计哲学差异

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         设计哲学三角                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                              简单性                                       │
│                           (llama.cpp)                                    │
│                                ▲                                          │
│                               /│\                                         │
│                              / │ \                                        │
│                             /  │  \                                       │
│                            /   │   \                                      │
│                           /    │    \                                     │
│                          /     │     \                                    │
│                         /      │      \                                   │
│                        ▼       ▼       ▼                                  │
│               性能优先  ◄────────┼────────►  可扩展性                      │
│               (vLLM)             │        (xLLM)                         │
│                                                                          │
│  • llama.cpp: 简单性优先, 零成本抽象, 渐进式复杂度                         │
│  • vLLM: 性能优先, PagedAttention, 连续批处理                             │
│  • xLLM: 可扩展性优先, DDD 设计, 多硬件支持                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 文档体系对比

### 2.1 文档结构

**xLLM Handbook** — 按功能域组织 (13+ 文档):

```
handbook/docs/
├── 00_OVERVIEW.md              # 文档总览
├── 01_GLOSSARY.md              # 术语表
├── 02_ARCHITECTURE.md          # 架构设计
├── 03_DESIGN_PRINCIPLES.md     # 设计原则
├── 04_DOMAIN_MODEL.md          # DDD 领域模型
├── 05_SCHEDULER_DESIGN.md      # 调度器设计
├── 06_WORKER_DESIGN.md         # Worker 设计
├── 07_KV_CACHE_DESIGN.md       # KV Cache 设计
├── 08_AI_NATIVE_DEVELOPMENT.md # AI Native 开发
├── 09_API_DESIGN.md            # API 设计
├── 10_CONFIG_SCHEMA.md         # 配置 Schema
├── 11_TEST_STRATEGY.md         # 测试策略
└── 12_DEPLOYMENT.md           # 部署指南
```

**llama.cpp Handbook** — 按 IPD 阶段组织 (20+ 文档):

```
handbook/
├── 00-overview/                 # 概念阶段
│   ├── 01-product-vision.md
│   ├── 03-architecture-overview.md
│   └── 04-design-principles.md
├── 01-requirements/             # 需求阶段
├── 02-design/                  # 设计阶段
│   └── 02-module-design/       # 7 个模块详细设计
├── 03-implementation/          # 实现阶段
├── 04-test/                    # 测试阶段
└── 05-release/                 # 发布阶段
```

**vLLM Handbook** — 按功能+阶段混合组织 (30+ 文档):

```
handbook/
├── 00_Overview/               # 总览
├── 01_Concept/                # 概念阶段
│   ├── product-vision.md
│   └── features/               # 3 个核心特性
│       ├── paged-attention.md
│       ├── continuous-batching.md
│       └── speculative-decoding.md
├── 02_Plan/                   # 计划阶段
├── 03_Develop/                # 开发阶段
│   ├── core-modules/           # 核心模块
│   ├── inference/              # 推理优化
│   ├── distributed/           # 分布式推理
│   └── api-layer/             # API 层
├── 04_Verify/                 # 验证阶段
├── 05_Release/                # 发布阶段
├── 06_Lifecycle/             # 生命周期
├── AI_Native/                # AI Native 开发
└── Code_Analysis/            # 代码分析
```

### 2.2 文档风格对比

| 维度 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **语言** | 中文 | 英文 | 中文 (部分英文) |
| **组织** | 功能域划分 | IPD 流程阶段 | 混合组织 |
| **侧重点** | 架构设计、DDD | 模块实现 | 功能特性、代码分析 |
| **AI 可复刻** | 高 | 中等 | 高 |
| **目标读者** | 企业内部 | 开源社区 | 开发者+用户 |

---

## 3. 架构设计对比

### 3.1 架构层次

**xLLM 5 层架构**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Service Layer: OpenAI API │ Anthropic API │ Internal API              │
├─────────────────────────────────────────────────────────────────────────┤
│  Scheduler Layer: Continuous │ DisaggPD │ FixedSteps │ ZeroEvict      │
├─────────────────────────────────────────────────────────────────────────┤
│  Engine Layer: WorkerPool (TP×CP)                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Runtime Layer: CausalLM │ CausalVLM │ DitModel │ RecCausalLM          │
├─────────────────────────────────────────────────────────────────────────┤
│  Backend Layer: CUDA │ NPU │ MLU │ DCU                               │
└─────────────────────────────────────────────────────────────────────────┘
```

**llama.cpp 4 层架构**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  llama-cli │ llama-server │ libllama                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  llama_model + llama_context                                          │
│  (vocab │ arch │ hparams │ kv_cache │ sampler)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ggml (Compute Engine): ops │ tensor │ alloc │ compute                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ggml-backend: CPU │ CUDA │ Metal │ Vulkan │ SYCL                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**vLLM 6 层架构**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  API Layer: OpenAI Server │ gRPC │ CLI │ Python API                   │
├─────────────────────────────────────────────────────────────────────────┤
│  Engine Core: Coordinator │ InputProcessor │ OutputProcessor           │
├─────────────────────────────────────────────────────────────────────────┤
│  Scheduler: Scheduler │ RequestQueue │ Scheduling Policy               │
├─────────────────────────────────────────────────────────────────────────┤
│  Worker Layer: GPU Model Runner │ CPU Worker │ XPU Worker              │
├─────────────────────────────────────────────────────────────────────────┤
│  KV Cache Layer: BlockPool │ KVCacheManager │ PrefixCaching           │
├─────────────────────────────────────────────────────────────────────────┤
│  Kernel Layer: FlashAttention │ FlashMLA │ Triton Kernels             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 代码规模对比

| 模块 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **核心引擎** | Worker (~10K) | llama.cpp (~50K) | core.py (~98K) |
| **调度器** | Scheduler (~15K) | ggml_backend_sched | scheduler.py (~130K) |
| **模型运行** | LLMWorker (~20K) | llama.cpp 内核 | gpu_model_runner.py (~348K) |
| **KV Cache** | BlockManager (~8K) | llama_kv_cache (~5K) | kv_cache_manager.py (~81K) |
| **总代码量** | ~500K C++ | ~200K C/C++ | ~500K Python/C++ |

---

## 4. 核心特性对比

### 4.1 Continuous Batching

**xLLM 实现**:
```cpp
// 调度策略工厂
class SchedulerFactory {
    static std::unique_ptr<Scheduler> Create(SchedulerType type);
};

// SLO 驱动调度
Urgency calculate_urgency(const Request& request) {
    if (remaining_time <= request.ttft_slo_ms() * 0.5) {
        return Urgency::STARVED;
    }
    return Urgency::NORMAL;
}
```

**vLLM 实现**:
```python
def schedule(self) -> SchedulerOutput:
    # 1. 释放已完成的请求
    finished = self._release_finished_requests()
    
    # 2. 调度新的 Prefill
    prefill_scheduled = self._schedule_prefill()
    
    # 3. 调度 Decode
    decode_scheduled = self._schedule_decode()
    
    return SchedulerOutput(...)
```

**llama.cpp 实现**:
```c
// 基础的批处理支持
llama_decode(ctx, batch);
// 无原生 Continuous Batching，需用户自行管理
```

| 维度 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **实现** | 完整 | 基础 | 完整 |
| **策略** | SLO 驱动 | 无 | 贪心 |
| **优化** | 多种调度策略 | 无 | Chunked Prefill |
| **优先级** | TTFT/TPOT/TTLT | 无 | 短 prompt 优先 |

### 4.2 PagedAttention

**vLLM PagedAttention**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PagedAttention 原理                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  逻辑视图:                                                              │
│  Seq A: [tok0][tok1][tok2]... → Block[0,1,2]                          │
│  Seq B: [tok0][tok1]...       → Block[3]                              │
│                                                                          │
│  物理视图:                                                              │
│  Block 0: [A:0-15]  Block 1: [A:16-31]  Block 2: [A:32-47]           │
│  Block 3: [B:0-15]  Block 4: [B:16-31]  ...                           │
│                                                                          │
│  与 OS 虚拟内存类比:                                                    │
│  虚拟地址 → 分页 → 物理内存                                             │
│  逻辑token → Block → GPU 显存                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**xLLM Block Management**:

```cpp
// 与 vLLM 类似的设计
class BlockManager {
    std::vector<Block*> free_blocks_;
    std::unordered_map<int64_t, Block*> allocated_blocks_;
    
    std::vector<uint32_t> allocate(uint32_t num_blocks);
    void free(const std::vector<uint32_t>& block_ids);
};
```

**llama.cpp**:
```c
// 无 PagedAttention，使用连续分配
struct llama_kv_cache {
    // 连续的 KV 存储
    // 内存碎片化问题
};
```

| 维度 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **实现** | ✅ Block 管理 | ❌ 无 | ✅ 核心特性 |
| **Block 大小** | 16 | N/A | 16 (默认) |
| **内存效率** | 高 | 低 | 高 |
| **前缀缓存** | ✅ | ❌ | ✅ |

### 4.3 KV Cache 管理对比

**xLLM KV Cache 架构**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BlockManager 架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Free Blocks Pool                                                │   │
│  │ [B0] [B1] [B2] [B3] [B4] [B5] [B6] [B7] ...                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│        ┌─────────────────────┼─────────────────────┐                    │
│        ▼                     ▼                     ▼                     │
│  ┌───────────┐        ┌───────────┐        ┌───────────┐              │
│  │ Sequence A │       │ Sequence B │       │ Sequence C │              │
│  │[B0,B1,B2] │        │ [B3,B4]   │        │[B5,B6,B7] │              │
│  └───────────┘        └───────────┘        └───────────┘              │
│                                                                          │
│  Prefix Cache Table:                                                     │
│  Hash │ Tokens │ BlockIds │ RefCount │ TTL                           │
│  ─────┼────────┼──────────┼──────────┼─────                          │
│  0x123│ [1,2,3]│ [B0,B1] │    3     │ ∞                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**vLLM KV Cache Manager**:

```python
class KVCacheManager:
    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self._free_blocks: Set[int] = set(range(num_blocks))
        self._allocated_blocks: Dict[int, Block] = {}
        self._req_to_blocks: Dict[str, List[int]] = {}
        self._hash_to_block: Dict[int, int] = {}  # 前缀缓存
    
    def allocate(self, req_id: str, num_tokens: int) -> List[int]:
        # O(1) amortized 分配
    
    def fork(self, parent_req_id, child_req_id, num_common_tokens):
        # Beam Search 共享
```

**llama.cpp KV Cache**:

```c
// 简单连续存储
struct llama_kv_cache {
    struct ggml_tensor * k;  // [n_ctx, n_layer, n_heads, head_dim]
    struct ggml_tensor * v;  // [n_ctx, n_layer, n_heads, head_dim]
    
    // 操作
    llama_kv_cache_seq_rm(ctx, seq_id, start, end);
    llama_kv_cache_seq_cp(ctx, src, dst, ...);
    llama_kv_cache_seq_keep(ctx, seq_id);
};
```

### 4.4 Speculative Decoding

| 维度 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **实现** | ✅ 支持 | ❌ 无 | ✅ 支持 |
| **策略** | 可配置 | N/A | 多种策略 |
| **优化** | 投机解码 | N/A | 验证优化 |

---

## 5. 分布式支持对比

### 5.1 并行策略

**xLLM 完整并行支持**:

```cpp
// 配置示例
runtime::Options options;
options.tp_size = 8;           // 张量并行度
options.pp_size = 2;           // 流水线并行度
options.dp_size = 4;            // 数据并行度
options.cp_size = 2;            // 上下文并行度
options.ep_size = 8;            // 专家并行度
```

**vLLM 并行支持**:

```python
# Tensor Parallelism
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # TP
    pipeline_parallel_size=2,  # PP (实验性)
)

# Data Parallelism (多实例)
# 需要外部负载均衡
```

**llama.cpp**:
```bash
# 无原生分布式，需使用外部工具
# 例如：使用 llama.cpp 多实例 + nginx 负载均衡
```

### 5.2 P/D 分离

| 维度 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **P/D 分离** | ✅ DisaggPDScheduler | ❌ 无 | ❌ 无 |
| **RDMA 支持** | ✅ 支持 | ❌ 无 | ❌ 无 |
| **多节点协调** | ✅ Coordinator | ❌ 无 | ❌ 无 |

---

## 6. API 设计对比

### 6.1 OpenAI 兼容 API

**xLLM**:
```yaml
# 扩展字段
"extra_body": {
    "request_id": "string",
    "priority": "integer",        # 0-9
    "ttft_slo_ms": "integer",
    "tpot_slo_ms": "integer",
    "ttlt_slo_ms": "integer",
    "prefix_cache": "boolean"
}
```

**vLLM**:
```python
# Python API
from vllm import LLM
llm = LLM(model="meta-llama/Llama-2-7b-hf")
output = llm.generate("Hello world")
```

**llama.cpp**:
```c
// C API
struct llama_model * llama_model_load_from_file(path, params);
struct llama_context * llama_new_context_with_model(model, params);
int llama_decode(ctx, batch);
llama_token llama_sampler_sample(sampler, ctx, -1);
```

### 6.2 API 风格对比

| 维度 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **协议** | REST + Protobuf | C API | REST + Python |
| **类型安全** | Protobuf 定义 | C struct | Python class |
| **同步/异步** | 混合 | 同步 | 异步优先 |
| **流式输出** | ✅ 支持 | ✅ 支持 | ✅ 支持 |

---

## 7. 性能特性对比

### 7.1 性能指标

| 维度 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **显存利用率** | 高 | 中 | 高 (PagedAttention) |
| **吞吐量** | 高 | 中 | 高 |
| **延迟** | 可配置 SLO | 固定 | 低 |
| **首 token 延迟** | SLO 保障 | 依赖 batch | 优化 (Chunked Prefill) |
| **GPU 利用率** | 高 | 中 | 高 (Continuous Batching) |

### 7.2 量化支持

| 量化类型 | xLLM | llama.cpp | vLLM |
|---------|------|-----------|------|
| **FP16** | ✅ | ✅ | ✅ |
| **INT8** | ✅ | ✅ | ✅ |
| **FP8** | ✅ | ✅ | ✅ |
| **GPTQ** | ✅ | ✅ | ✅ |
| **AWQ** | ✅ | ✅ | ✅ |
| **GGUF** | ❌ | ✅ 核心格式 | ❌ |

---

## 8. 部署模式对比

### 8.1 部署架构

**xLLM 部署**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    P/D Separation Deployment                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Load Balancer ──┬──▶ Prefill Cluster ──┐                             │
│                  │                       │ KV Cache Transfer            │
│                  └──▶ Decode Cluster ◀───┘                             │
│                                          │                             │
│                                    RDMA Network                          │
└─────────────────────────────────────────────────────────────────────────┘

模式: 单机 │ P/D 分离 │ 多节点
```

**vLLM 部署**:

```python
# 单 GPU
llm = LLM(model="llama-2-7b", tensor_parallel_size=1)

# 多 GPU
llm = LLM(model="llama-2-70b", tensor_parallel_size=4)

# Docker
# docker run --gpus all vllm/vllm-openai:latest ...
```

**llama.cpp 部署**:

```bash
# CLI 模式
./llama-cli -m model.gguf -i

# Server 模式
./llama-server -m model.gguf --host 0.0.0.0 --port 8080
```

### 8.2 资源需求

| 模型规模 | xLLM | llama.cpp | vLLM |
|---------|------|-----------|------|
| **1B** | 4 GB | 2 GB | 4 GB |
| **7B** | 16 GB | 8 GB | 16 GB |
| **13B** | 24 GB | 16 GB | 24 GB |
| **70B** | 64 GB | 48 GB | 64 GB |

---

## 9. 社区生态对比

### 9.1 开源生态

| 维度 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **GitHub Stars** | N/A (内部) | 70K+ | 45K+ |
| **社区活跃度** | 低 (内部) | 高 | 高 |
| **贡献者** | 内部团队 | 500+ | 300+ |
| **第三方集成** | 有限 | 广泛 | 广泛 |
| **文档完善度** | 高 | 中 | 高 |

### 9.2 硬件支持

| 硬件 | xLLM | llama.cpp | vLLM |
|------|------|-----------|------|
| **NVIDIA GPU** | ✅ | ✅ | ✅ |
| **AMD GPU** | ⚠️ | ✅ | ⚠️ |
| **Intel GPU** | ⚠️ | ✅ | ❌ |
| **Apple Silicon** | ⚠️ | ✅ | ⚠️ |
| **NPU (昇腾)** | ✅ | ❌ | ❌ |
| **MLU (寒武纪)** | ✅ | ❌ | ❌ |
| **DCU (AMD)** | ✅ | ❌ | ❌ |

---

## 10. 测试策略对比

### 10.1 测试框架

**xLLM**:
```cpp
// Google Test + Mock
class MockScheduler : public Scheduler {
    MOCK_METHOD(bool, add_request, ...);
};

TEST_F(BlockManagerTest, Allocate) {
    auto blocks = manager_->allocate(1);
    EXPECT_EQ(blocks.size(), 1);
}
```

**vLLM**:
```python
# pytest
def test_basic_scheduling():
    scheduler = Scheduler(...)
    scheduler.add_request(Request("req1", "Hello"))
    output = scheduler.schedule()
    assert len(output.scheduled_requests) == 1
```

**llama.cpp**:
```cpp
// doctest
TEST_CASE("Inference Pipeline") {
    auto * model = load_test_model();
    auto * ctx = llama_new_context_with_model(model, params);
    // ...
}
```

### 10.2 测试金字塔

| 测试类型 | xLLM | vLLM | llama.cpp |
|---------|------|------|-----------|
| **单元测试** | ✅ 完善 | ✅ 完善 | ✅ 基础 |
| **集成测试** | ✅ 完善 | ✅ 完善 | ⚠️ 有限 |
| **性能测试** | ✅ SLO 测试 | ✅ Benchmark | ⚠️ 基础 |
| **Mock 框架** | ✅ Google Mock | ✅ unittest.mock | ❌ 无 |

---

## 11. AI 开发规范对比

### 11.1 AI 代码生成约束

**xLLM AI Native**:

```cpp
// 结构化注释，便于 AI 解析
// @ai_action: 生成单元测试
// @ai_verification: 验证 KV Cache 正确分配
class BlockManager {
    // @brief: 分配指定数量的 Block
    // @param: num_blocks - 需要的 Block 数量
    // @return: 分配的 Block ID 列表
    virtual std::vector<uint32_t> allocate_blocks(uint32_t num_blocks) = 0;
};
```

**vLLM AI Guidelines**:

```python
# Google-style docstrings
def schedule(self) -> SchedulerOutput:
    """
    调度主方法。
    
    Args:
        throttle_prefills: 是否限制 prefill 数量
        
    Returns:
        SchedulerOutput: 调度输出
    """
```

**llama.cpp AI 约束**:

```
禁止:
✗ AI 写 PR 描述
✗ AI 提交 commit
✗ 不理解代码就提交
✓ 学习、建议、机械任务
```

---

## 12. 总结对比

### 12.1 选择指南

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         框架选择指南                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  选择 xLLM 如果:                                                        │
│  ├── 需要企业级分布式推理                                               │
│  ├── 需要多硬件支持 (NPU, MLU, DCU)                                     │
│  ├── 需要 SLO 保障                                                     │
│  ├── 需要 P/D 分离部署                                                 │
│  └── 需要 DDD 架构的企业项目                                            │
│                                                                          │
│  选择 vLLM 如果:                                                        │
│  ├── 需要高性能 GPU 推理服务                                            │
│  ├── 需要 PagedAttention 优化显存                                       │
│  ├── 需要 Continuous Batching 高吞吐                                   │
│  ├── 需要张量并行支持大模型                                            │
│  └── 需要良好的 Python 集成                                            │
│                                                                          │
│  选择 llama.cpp 如果:                                                  │
│  ├── 需要轻量级本地推理                                                │
│  ├── 需要广泛的硬件支持 (包括嵌入式)                                    │
│  ├── 需要 GGUF 模型格式                                                │
│  ├── 资源受限环境                                                      │
│  └── 偏好简单 C/C++ 代码库                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 12.2 核心差异汇总

| 维度 | xLLM | vLLM | llama.cpp |
|------|------|------|-----------|
| **定位** | 企业级分布式 | 高性能服务 | 轻量级引擎 |
| **架构复杂度** | 高 | 中 | 低 |
| **性能优化** | 全面 | 显存+吞吐 | 基础 |
| **分布式** | 完整支持 | 有限 | 无 |
| **硬件支持** | 多硬件 | GPU 为主 | 广泛 |
| **学习曲线** | 陡峭 | 中等 | 平缓 |
| **适用场景** | 大规模集群 | GPU 服务 | 本地/边缘 |

### 12.3 技术演进路径

```
llama.cpp ──────────────▶ xLLM
(简单核心)              (企业扩展)
    │                        │
    │                        │
    └────────┬───────────────┘
             │
             ▼
           vLLM
        (性能优化)
```

**演进关系**:
- **llama.cpp**: 提供基础推理能力，简单但功能有限
- **vLLM**: 在 llama.cpp 基础上增加 PagedAttention 和 Continuous Batching
- **xLLM**: 在 vLLM 基础上增加分布式、多硬件、SLO 支持

---

## 附录 A: 术语对照表

| 术语 | xLLM | vLLM | llama.cpp |
|-----|------|------|-----------|
| 推理引擎 | LLMEngine + Worker | LLMEngine | llama_context |
| 模型加载 | Worker::init_model | LLM class | llama_model_load_from_file |
| 批处理 | Batch + Scheduler | Scheduler | ggml_backend_sched |
| KV 缓存 | BlockManager | KVCacheManager | llama_kv_cache |
| 采样器 | 内置 | 内置 | llama_sampler_chain |
| 调度器 | ContinuousScheduler | Scheduler | ggml_backend_sched |
| 分块处理 | Chunked Prefill | Chunked Prefill | 无 |
| 前缀缓存 | Prefix Cache | PrefixCaching | 无 |
| 显存分页 | Block Manager | PagedAttention | 无 |

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [xLLM vs llama.cpp](./13_FRAMEWORK_COMPARISON.md) | xLLM 与 llama.cpp 深度对比 |
| [xLLM 架构设计](./02_ARCHITECTURE.md) | xLLM 系统架构 |
| [xLLM 设计原则](./03_DESIGN_PRINCIPLES.md) | xLLM 设计哲学 |
| [vLLM Handbook](../../vllm-handbook/handbook/README.md) | vLLM 文档体系 |
| [llama.cpp Handbook](../../llama.cpp-handbook/handbook/README.md) | llama.cpp 文档体系 |
