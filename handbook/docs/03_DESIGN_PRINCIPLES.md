# xLLM 设计原则与哲学

## 文档信息

```yaml
---
document_id: ARCH-002
version: 1.0.0
category: design_principles
owner: xllm-team
verification_level: BOTH

ai_acceptance_criteria:
  - 每条原则必须包含"为什么"和"怎么做"
  - 原则需与代码示例对应
  - 违反原则需在代码审查时标注
---
```

---

## 1. 核心设计哲学

### 1.1 哲学宣言

> **"性能是功能，可观测是必需，可维护是责任。"**

xLLM 的设计围绕三个核心价值：

| 价值 | 含义 | 落地 |
|-----|------|------|
| **性能即功能** | 推理性能是产品的核心竞争力 | Continuous Batching、Prefix Cache、投机解码 |
| **可观测必需** | 无法测量的就无法优化 | Metrics、Tracing、Logging |
| **可维护是责任** | 代码被阅读的次数远多于编写 | DDD、Clean Architecture、文档 |

### 1.2 设计决策树

```
                              ┌─────────────────────────────┐
                              │     遇到设计决策时           │
                              └─────────────────────────────┘
                                          │
                                          ▼
                        ┌───────────────────────────────────┐
                        │  1. 这个决策会影响推理性能吗？     │
                        └───────────────────────────────────┘
                                    │               │
                                   Yes              No
                                    │               │
                                    ▼               ▼
                        ┌───────────────────┐   ┌───────────────────┐
                        │  优先考虑性能     │   │  优先考虑可维护性  │
                        │  (Benchmark验证) │   │  (遵循DDD原则)    │
                        └───────────────────┘   └───────────────────┘
                                    │               │
                                    ▼               ▼
                        ┌───────────────────┐   ┌───────────────────┐
                        │  决策: 抽象是否    │   │  决策: 接口是否   │
                        │  引入性能开销？   │   │  足够清晰？       │
                        └───────────────────┘   └───────────────────┘
                                    │               │
                                   Yes              No
                                    │               │
                                    ▼               ▼
                        ┌───────────────────┐   ┌───────────────────┐
                        │  使用PIMPL模式    │   │  重构接口设计     │
                        │  隐藏实现细节     │   │  提高抽象层次     │
                        └───────────────────┘   └───────────────────┘
```

---

## 2. 架构设计原则

### 2.1 层级依赖原则 (Layer Dependency Rule)

**原则**: **上层模块依赖下层抽象，下层模块不依赖上层实现**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          依赖方向 (正确)                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  Application Layer (应用层)    - 依赖 Domain Layer                 │    │
│   │  API Service → Scheduler → Engine → Model → Layers → Backend     │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                        ▲                                     │
│                                        │                                     │
│                              依赖方向是单向的，向下依赖抽象                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          依赖方向 (错误)                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ✗ 禁止: API Service 直接引用 Backend 实现                                   │
│   ✗ 禁止: Scheduler 包含 CUDA Graph 代码                                      │
│   ✗ 禁止: Model 层包含 brpc 协议代码                                          │
│                                                                              │
│   原因: 破坏抽象边界，增加耦合，难以替换实现                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**代码验证**:

```cpp
// ✓ 正确: 上层依赖下层抽象
// xllm/core/framework/model/causal_lm.h - 抽象接口
class CausalLM {
 public:
  virtual ForwardOutput forward(
      const ModelInputParams& params,
      const ModelArgs& args,
      const ForwardInput& inputs,
      int32_t cp_size = 1) = 0;
  virtual ~CausalLM() = default;
};

// xllm/core/runtime/llm_worker_impl.h - 依赖抽象
class LLMWorkerImpl {
  std::unique_ptr<CausalLM> model_;  // 依赖抽象，不依赖实现
};

// ✗ 错误: 违反层级依赖
// class LLMWorkerImpl {
//   // 禁止直接包含CUDA相关头
//   #include <cuda_runtime.h>  // 错误！
// };
```

### 2.2 稳定抽象原则 (Stable Abstractions Principle)

**原则**: **稳定的模块应该是抽象的，抽象的模块应该是稳定的**

| 模块稳定性 | 抽象程度 | 示例 |
|-----------|---------|------|
| 高稳定 | 高抽象 | `Scheduler`, `Model` 基类 |
| 中稳定 | 中抽象 | `Batch`, `Request` |
| 低稳定 | 低抽象 | 具体实现 `CUDAGraphExecutorImpl` |

### 2.3 PIMPL 原则 (Pointer to Implementation)

**原则**: **公共接口隐藏实现细节，使用PIMPL模式**

**适用场景**:
1. 公共API需要稳定二进制接口
2. 实现可能频繁变化
3. 减少编译依赖

```cpp
// ✓ 正确: Worker使用PIMPL隐藏实现
// worker.h - 公开头文件
class Worker {
 public:
  Worker(const ParallelArgs& args, const torch::Device& device, 
        const runtime::Options& options, WorkerType type);
  ~Worker();
  
  ForwardInput prepare_inputs(Batch& batch);
  std::optional<ForwardOutput> step(const ForwardInput& inputs);
  
 private:
  WorkerImpl* impl_;  // Pimpl - 隐藏实现细节
};

// worker_impl.h - 私有实现
class WorkerImpl {
  // 包含所有实现细节
  std::unique_ptr<Executor> executor_;
  std::unique_ptr<KVCacheManager> kv_cache_manager_;
  // ... 大量私有成员
};
```

---

## 3. 模块设计原则

### 3.1 Scheduler 设计原则

#### 原则 S1: 调度策略与执行解耦

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    调度策略与执行解耦                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐              ┌─────────────────┐                      │
│   │    Scheduler    │              │      Worker     │                      │
│   │  (What to run)   │    ────▶    │  (How to run)   │                      │
│   │                 │              │                 │                      │
│   │  • 决定哪些请求  │              │  • 执行前向计算  │                      │
│   │  • 决定批处理策略│              │  • 管理KV Cache │                      │
│   │  • 管理优先级    │              │  • 处理采样     │                      │
│   └─────────────────┘              └─────────────────┘                      │
│                                                                              │
│   Scheduler 输出: Batch (待执行的序列集合)                                     │
│   Worker 输入: ForwardInput (前向计算所需数据)                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**代码示例**:

```cpp
// scheduler.h - 只负责调度决策
class Scheduler {
 public:
  // 调度器不知道Worker如何执行
  virtual void step(const absl::Duration& timeout) = 0;
  virtual bool add_request(std::shared_ptr<Request>& request) = 0;
  virtual uint32_t get_waiting_requests_num() const = 0;
};

// worker.h - 只负责执行
class Worker {
 public:
  // Worker不知道调度策略
  ForwardInput prepare_inputs(Batch& batch);
  std::optional<ForwardOutput> step(const ForwardInput& inputs);
};
```

#### 原则 S2: 优先级感知调度

**原则**: **SLO驱动调度，根据TTFT/TPOT/TTLT优先级分配资源**

```cpp
// 调度优先级计算
Urgency calculate_urgency(const Request& request) {
    int32_t remaining_time = request.get_remaining_time();
    int32_t ttft_slo = request.ttft_slo_ms();
    int32_t tpot_slo = request.tpot_slo_ms();
    
    // 接近TTFT截止 → STARVED
    if (remaining_time <= ttft_slo * 0.5) {
        return Urgency::STARVED;
    }
    // 接近TPOT截止 → URGENT  
    if (remaining_time <= tpot_slo * 2) {
        return Urgency::URGENT;
    }
    return Urgency::NORMAL;
}
```

### 3.2 Batch 设计原则

#### 原则 B1: 动态批处理

**原则**: **批大小和组成应动态调整，而非固定**

```cpp
// 动态决定批大小
BatchComposition decide_batch_composition(const std::vector<Request>& candidates) {
    size_t prefill_tokens = sum_prefill_tokens(candidates);
    size_t decode_tokens = sum_decode_tokens(candidates);
    
    // Prefill密集型: 优先打包prefill请求
    if (prefill_tokens > decode_tokens * 2) {
        return BatchType::PREFILL_HEAVY;
    }
    // Decode密集型: 批量decode
    return BatchType::DECODE_HEAVY;
}
```

#### 原则 B2: 异构序列打包

**原则**: **不同长度的序列应智能打包，最大化GPU利用率**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       异构序列打包策略                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   候选序列:                                                                    │
│   Seq A: 100 tokens (即将结束)                                                │
│   Seq B: 2000 tokens (Prefill)                                               │
│   Seq C: 50 tokens (Decode)                                                  │
│   Seq D: 1500 tokens (Prefill)                                               │
│                                                                              │
│   策略1 (Naive): [A, C] + [B] + [D]                                          │
│   → 3个batch，小batch多，利用率低                                              │
│                                                                              │
│   策略2 (Packing): [A, C, B.part1] + [B.part2, D.part1] + [D.part2]          │
│   → 更均衡的token分布，GPU利用率更高                                          │
│                                                                              │
│   策略3 (Chunked Prefill): [A, C] + [B.chunk1] + [B.chunk2] + [D.chunk1]    │
│   → 长序列分块处理，短序列优先完成                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Worker 设计原则

#### 原则 W1: 生命周期管理

**原则**: **Worker支持完整的生命周期：Init → Ready → Sleep ↔ Wakeup → Exit**

```cpp
// Worker 状态机
enum class WorkerStatus {
    UNINITIALIZED,    // 未初始化
    INITIALIZING,     // 初始化中
    READY,            // 就绪
    RUNNING,          // 执行中
    SLEEPING,         // 深度休眠 (RL场景)
    WAKING_UP,        // 唤醒中
    ERROR,            // 错误状态
    EXITING           // 退出中
};

// 生命周期接口
class Worker {
 public:
    // 初始化: 加载模型、分配KV Cache
    bool init_model(const std::string& weights_path, int32_t seed, 
                    MasterStatus status);
    
    // 休眠: 保存状态、释放内存
    bool sleep(MasterStatus status);
    
    // 唤醒: 恢复状态、重新分配内存
    bool wakeup(const WakeupOptions& options);
    
    // 执行: 单步推理
    ForwardInput prepare_inputs(Batch& batch);
    std::optional<ForwardOutput> step(const ForwardInput& inputs);
};
```

#### 原则 W2: 异步优先

**原则**: **耗时操作提供异步接口，避免阻塞**

```cpp
class Worker {
 public:
    // 同步接口 - 短操作
    bool init_model(const std::string& weights_path, int32_t seed,
                   MasterStatus status);
                   
    // 异步接口 - 长操作
    folly::SemiFuture<bool> init_model_async(const std::string& weights_path,
                                            int32_t seed,
                                            MasterStatus status);
    
    // 批量异步
    folly::SemiFuture<std::optional<ForwardOutput>> step_async(
        const ForwardInput& inputs);
        
    // 获取结果
    folly::SemiFuture<std::optional<ForwardOutput>> get_last_step_result_async();
};
```

---

## 4. 数据设计原则

### 4.1 不可变性原则 (Immutability)

**原则**: **优先使用不可变数据结构，避免隐式状态修改**

```cpp
// ✓ 推荐: 使用const和不可变输入
ForwardOutput CausalLM::forward(const ModelInputParams& params,
                               const ModelArgs& args,
                               const ForwardInput& inputs,
                               int32_t cp_size) {
    // inputs 是只读的
    // 返回新的 ForwardOutput，不修改输入
    return output;
}

// ✗ 避免: 隐式修改输入
// void process_input(ForwardInput& inputs) {
//     inputs.hidden_states[0] = ...;  // 隐式修改
// }
```

### 4.2 零拷贝原则 (Zero-Copy)

**原则**: **数据流经系统时尽量减少拷贝**

```cpp
// ✓ 推荐: 引用传递，避免不必要的拷贝
void Batch::process_sample_output(const SampleOutput& output,
                                 bool replace_fake_token) {
    // 使用 const& 接收参数
    // 内部操作尽量使用引用
}

// ✗ 避免: 不必要的拷贝
// void process_sample_output(SampleOutput output, ...) {
//     // 拷贝构造，除非必要
// }
```

### 4.3 数据局部性原则 (Data Locality)

**原则**: **相关数据应放在一起，提高缓存命中率**

```cpp
// 好的数据布局: 相关数据集中
struct Sequence {
    std::vector<int> token_ids;      // Token序列
    std::vector<int> position_ids;  // 位置ID (与token_ids相邻，缓存友好)
    std::vector<Block*> blocks;      // KV Cache块
    SequenceStage stage;             // 状态
};

// 不好的数据布局: 分散的数据
// struct BadSequence {
//     std::vector<int> token_ids;
//     SequenceStage stage;
//     std::vector<Block*> blocks;  // 与token_ids不相邻
//     std::vector<int> position_ids;
// };
```

---

## 5. 接口设计原则

### 5.1 接口单一职责

**原则**: **每个接口只做一件事**

```cpp
// ✓ 推荐: 单一职责接口
class Batch {
 public:
    void add(Sequence* sequence);                    // 添加序列
    void add(const std::vector<Sequence*>& seqs);   // 批量添加
    ForwardInput prepare_forward_input(...);         // 准备输入
    void process_sample_output(const SampleOutput&,
                               bool replace_fake_token);  // 处理输出
};

// ✗ 避免: 职责不单一的接口
// class Batch {
// public:
//     void add_and_prepare_and_forward(Sequence* seq);  // 太多职责
// };
```

### 5.2 接口稳定性

**原则**: **公共API必须稳定，不稳定的接口标记为experimental**

```cpp
// 稳定接口
class Scheduler {
 public:
    virtual ~Scheduler() = default;
    virtual void step(const absl::Duration& timeout) = 0;
    virtual bool add_request(std::shared_ptr<Request>& request) = 0;
};

// 实验性接口 (标记明确)
class ExperimentalFeature {
 public:
    // 实验性接口必须有文档说明
    // Experimental: This API may change in future releases
    virtual folly::SemiFuture<bool> experimental_api() = 0;
};
```

### 5.3 错误处理原则

**原则**: **使用异常处理错误情况，返回值表示成功/失败**

```cpp
// ✓ 推荐: 明确的错误处理
class Worker {
 public:
    // 返回true表示成功，false表示失败
    bool init_model(const std::string& weights_path, int32_t seed,
                   MasterStatus status) {
        if (!load_weights(weights_path)) {
            LOG(ERROR) << "Failed to load weights from " << weights_path;
            return false;
        }
        return true;
    }
    
    // 异步版本使用 folly::Try
    folly::SemiFuture<bool> init_model_async(...) {
        return folly::makeFuture().then([](auto&&) {
            if (!init_model(...)) {
                return folly::makeUnexpected(Status::LOAD_WEIGHTS_FAILED);
            }
            return true;
        });
    }
};
```

---

## 6. 性能设计原则

### 6.1 延迟-吞吐平衡

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Latency vs Throughput Tradeoff                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Throughput                                                                    │
│      ▲                                                                        │
│      │                    ★ Maximum Throughput                                │
│      │                  ╱╲                                                     │
│      │                ╱    ╲                                                   │
│      │              ╱        ╲                                                 │
│      │            ╱            ╲                                               │
│      │          ╱                ╲                                             │
│      │        ╱                    ╲ ★ Maximum Latency (Batch Size = 1)       │
│      │      ╱                        ╲                                        │
│      │    ╱                            ╲                                      │
│      │  ╱                                ╲                                    │
│      └──────────────────────────────────────────────▶ Latency                  │
│         ↑                    ↑                    ↑                           │
│        BS=1                  BS=8               BS=64                       │
│                                                                              │
│   原则: 根据SLO选择最优Batch Size                                              │
│   • 在线服务: 优先延迟 (小Batch)                                                │
│   • 离线批处理: 优先吞吐 (大Batch)                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 内存效率原则

**原则**: **最小化内存占用，最大化KV Cache利用率**

```cpp
// ✓ 推荐: 动态计算内存需求
KVCacheCapacity calculate_capacity(const ModelArgs& args,
                                   const runtime::Options& options) {
    int64_t total_memory = get_device_memory();
    int64_t reserved = options.kv_cache_reserved_memory_mb * 1024 * 1024;
    int64_t available = total_memory - reserved;
    
    // 计算每个Block的大小
    int64_t block_size = args.num_kv_heads * args.head_dim * 
                          args.block_seq_len * sizeof(dtype);
    
    // 计算可分配的Block数
    int64_t num_blocks = available / block_size;
    
    return {num_blocks, block_size};
}

// ✗ 避免: 固定分配
// const int64_t KV_CACHE_BLOCKS = 10000;  // 硬编码
```

### 6.3 算子融合原则

**原则**: **高频小算子融合为大算子，减少Kernel Launch开销**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Operator Fusion Examples                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Before Fusion:                    After Fusion:                             │
│   ┌────────┐   ┌────────┐          ┌────────────────────────┐                │
│   │ LayerNorm│ → │  Add   │   →     │   FusedLayerNormAdd   │                │
│   └────────┘   └────────┘          └────────────────────────┘                │
│                                                                              │
│   Before:                           After:                                    │
│   ┌──────┐ ┌──────┐ ┌─────────┐    ┌─────────────────────┐                   │
│   │Rotary │ → │QKVProj│ → │ScaledDotProduct│    │ FusedAttention    │                   │
│   │Emb   │   │      │   │    Attention    │    │ (FlashAttention)  │                   │
│   └──────┘ └──────┘ └─────────┘    └─────────────────────┘                   │
│                                                                              │
│   Kernel Launch: 3次                    Kernel Launch: 1次                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 可维护性原则

### 7.1 DDD 领域边界

**原则**: **清晰的领域边界，避免跨域直接依赖**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DDD Bounded Contexts                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────────┐ │
│   │                     Scheduling Context                                 │ │
│   │  Request, Sequence, Batch, Scheduler, Policy                          │ │
│   │  边界: 只处理请求调度，不涉及模型计算                                    │ │
│   └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    │ 使用Interface通信                         │
│                                    ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────────┐ │
│   │                     Inference Context                                  │ │
│   │  Worker, Executor, Model, KVCache, BlockManager                       │ │
│   │  边界: 只处理模型推理，不涉及业务调度                                    │ │
│   └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    │ 使用Interface通信                         │
│                                    ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────────┐ │
│   │                     Service Context                                   │ │
│   │  API Service, Protocol, Auth, RateLimiter                            │ │
│   │  边界: 只处理外部协议，不涉及内部逻辑                                    │ │
│   └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 最小知识原则 (Law of Demeter)

**原则**: **只与直接朋友通信，不和陌生人说话**

```cpp
// ✗ 违反: 链式调用
scheduler->get_batch_manager()->get_sequence(0)->get_token(0);

// ✓ 正确: 只与直接依赖通信
scheduler->get_sequence_first_token(request_id);

// ✗ 违反: 过深的依赖
batch->model_->executor_->graph_executor_->kernel_;

// ✓ 正确: 清晰的层次调用
worker->step(forward_input);
```

### 7.3 SOLID 原则

| 原则 | 说明 | 在xLLM中的应用 |
|-----|------|--------------|
| **S**ingle Responsibility | 单一职责 | Scheduler只负责调度 |
| **O**pen/Closed | 开闭原则 | 通过扩展而非修改实现新功能 |
| **L**iskov Substitution | 里氏替换 | 所有Scheduler可相互替换 |
| **I**nterface Segregation | 接口隔离 | Worker提供精简接口 |
| **D**ependency Inversion | 依赖倒置 | 依赖抽象如CausalLM而非实现 |

---

## 8. AI Native 开发原则

### 8.1 代码自解释性

**原则**: **代码应该能够自我解释，减少对外部文档的依赖**

```cpp
// ✗ 缺乏解释
auto x = y->z();

// ✓ 自解释代码
int32_t num_available_blocks = block_manager_->available_block_count();
uint32_t max_decode_tokens = calculate_max_decode_tokens(
    num_available_blocks,
    sequences_per_request,
    SLO_SAFETY_MARGIN
);
```

### 8.2 机器友好格式

**原则**: **代码结构应便于AI解析和生成**

```cpp
// ✓ AI友好: 清晰的注释标记
// @ai_action: 生成单元测试
// @ai_verification: 验证KV Cache正确分配
class BlockManager {
 public:
    // @brief: 分配指定数量的Block
    // @param: num_blocks - 需要的Block数量
    // @return: 分配的Block ID列表
    // @throws: std::runtime_error 如果Block不足
    virtual std::vector<uint32_t> allocate_blocks(uint32_t num_blocks) = 0;
};

// ✓ AI友好: 结构化的日志
XLLM_LOG(INFO, "Batch", {
    {"batch_id", batch.batch_id()},
    {"size", batch.size()},
    {"prefill_tokens", prefill_tokens},
    {"decode_tokens", decode_tokens},
    {"estimated_time_ms", estimated_time}
});
```

### 8.3 可验证性

**原则**: **所有关键路径必须可验证**

```cpp
// ✓ 推荐: 验证点明确
class Scheduler {
 public:
    // 关键方法必须有验证
    bool add_request(std::shared_ptr<Request>& request) {
        // 前置条件验证
        XLLM_DCHECK(request != nullptr);
        XLLM_DCHECK(!request->finished());
        XLLM_DCHECK(request->sequences().size() > 0);
        
        // 业务逻辑
        bool added = add_impl(request);
        
        // 后置条件验证
        XLLM_DCHECK(request_queue_.size() <= max_queue_size_);
        
        return added;
    }
};
```

---

## 9. 违反原则的处理

### 9.1 代码审查检查点

```yaml
code_review_checklist:
  architecture:
    - "是否引入跨层依赖？"
    - "是否使用了PIMPL模式隐藏实现？"
    - "新接口是否与现有设计一致？"
    
  performance:
    - "是否有不必要的内存分配？"
    - "是否有不必要的拷贝？"
    - "循环中是否有重复计算？"
    
  maintainability:
    - "是否有清晰的命名？"
    - "是否有必要的注释？"
    - "是否遵循SOLID原则？"
    
  ai_friendly:
    - "代码是否自解释？"
    - "关键路径是否可验证？"
    - "日志格式是否结构化？"
```

### 9.2 技术债务处理

```cpp
// TODO: 技术债务标记
// @technical_debt: 临时解决方案，需要重构
// @reason: 由于接口兼容性原因暂时保留
// @ticket: JIRA-12345
// @priority: medium
// @estimated_hours: 8
char* legacy_api_buffer = new char[1024];  // 临时方案
```

---

## 10. 相关文档

| 文档 | 说明 |
|-----|------|
| [架构设计](./02_ARCHITECTURE.md) | 系统整体架构 |
| [领域模型](./02_DOMAIN_MODEL.md) | DDD领域划分 |
| [代码规范](../05_CODE/01_CODING_STANDARDS.md) | 详细编码规范 |
| [测试规范](../06_TESTING/01_TEST_STRATEGY.md) | 测试策略 |
