# xLLM 架构设计文档 (ARCHITECTURE)

> **文档版本**: v1.0  
> **项目**: xLLM 大模型推理框架  
> **日期**: 2026-07-23

---

## 1. 设计目标与原则

### 1.1 设计目标

1. **高性能**: 高吞吐、低延迟的推理服务
2. **多硬件**: 统一抽象，适配多种国产 AI 加速卡
3. **可扩展**: 支持新模型、新硬件、新调度策略
4. **可观测**: 完善的监控和调优能力
5. **高可用**: 企业级部署能力

### 1.2 设计原则

| 原则 | 描述 |
|------|------|
| **分层解耦** | Service、Engine、Worker、Executor 四层分离 |
| **平台抽象** | 通过 Platform 层统一硬件接口 |
| **数据驱动** | 以 Request、Batch 为核心组织数据流 |
| **异步优先** | 大量使用 async/await 和 Future |
| **零拷贝** | 避免不必要的数据拷贝 |

---

## 2. 架构分层

### 2.1 四层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         四层架构                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Service Layer (服务层)                        │   │
│  │                                                                     │   │
│  │  职责: 协议解析、API 处理、响应格式化                                 │   │
│  │  依赖: brpc (HTTP/gRPC)                                           │   │
│  │                                                                     │   │
│  │  核心类: ChatServiceImpl, CompletionServiceImpl, ...               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Engine Layer (引擎层)                         │   │
│  │                                                                     │   │
│  │  职责: 请求调度、批次管理、Engine 生命周期                           │   │
│  │  依赖: Scheduler, Batch, Request                                    │   │
│  │                                                                     │   │
│  │  核心类: LLMEngine, VLMEngine, RecEngine, Scheduler                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Worker Layer (Worker 层)                       │   │
│  │                                                                     │   │
│  │  职责: 模型执行、KV Cache 管理、多级缓存                             │   │
│  │  依赖: Executor, KVCache, BlockManager                            │   │
│  │                                                                     │   │
│  │  核心类: Worker, WorkerImpl, KVCacheManager                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Platform Layer (平台层)                          │   │
│  │                                                                     │   │
│  │  职责: 硬件抽象、Kernel 实现、图优化                                 │   │
│  │  依赖: CUDA/NPU/MLU SDK                                            │   │
│  │                                                                     │   │
│  │  核心类: Platform, ExecutorImpl, Kernels                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 各层职责边界

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         职责边界                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Service Layer                                                          │
│  ├── ✅ 协议解析 (HTTP/gRPC → Protobuf)                                  │
│  ├── ✅ 请求验证 (参数检查、认证)                                          │
│  ├── ✅ 响应格式化 (Protobuf → JSON/SSE)                                 │
│  ├── ❌ 调度决策 (交给 Engine 层)                                         │
│  └── ❌ 模型推理 (交给 Worker 层)                                         │
│                                                                         │
│  Engine Layer                                                           │
│  ├── ✅ 请求调度 (添加、移除、优先级)                                      │
│  ├── ✅ 批次组装 (合并请求、分配资源)                                      │
│  ├── ✅ SLO 追踪 (TTFT、TPOT、TTLT)                                      │
│  ├── ❌ 底层计算 (交给 Worker 层)                                         │
│  └── ❌ 硬件操作 (交给 Platform 层)                                        │
│                                                                         │
│  Worker Layer                                                           │
│  ├── ✅ 前向传播 (调用 Executor)                                          │
│  ├── ✅ KV Cache 管理 (分配、回收、交换)                                   │
│  ├── ✅ 序列状态管理 (生成 Token、状态更新)                                 │
│  ├── ❌ Kernel 实现 (交给 Platform 层)                                    │
│  └── ❌ 硬件驱动 (交给 Platform 层)                                        │
│                                                                         │
│  Platform Layer                                                         │
│  ├── ✅ Tensor 操作 (创建、内存管理)                                       │
│  ├── ✅ Kernel 执行 (Attention、Linear...)                               │
│  ├── ✅ 图优化 (CUDA Graph、ACL Graph)                                   │
│  ├── ❌ 业务逻辑 (交给上层)                                               │
│  └── ❌ 调度决策 (交给上层)                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件设计

### 3.1 Request (请求) 组件

#### 设计意图

Request 是用户请求的抽象，包含完整的请求生命周期管理。

#### 类结构

```cpp
// core/framework/request/request.h

class Request : public RequestBase {
 public:
  // ===== 构造函数 =====
  Request(const std::string& request_id,
          const std::string& x_request_id,
          const std::string& x_request_time,
          const RequestState& state,
          const std::string& service_request_id = "",
          const std::string& source_xservice_addr = "");

  // ===== 核心属性 =====
  std::unique_ptr<SequencesGroup> sequences_group_;  // 序列组 (支持 beam search)
  std::atomic<bool> cancelled_;                       // 取消标志
  
  // ===== SLO 调度 =====
  int32_t deadline_ms_;         // 截止时间 (毫秒)
  Urgency urgency_;             // 紧急程度: {STARVED=2, URGENT=1, NORMAL=0}
  bool starved_;                // 是否处于饥饿状态
  
  // ===== SLO 参数 =====
  int32_t ttft_slo_ms_;         // Time To First Token SLO
  int32_t tpot_slo_ms_;         // Time Per Output Token SLO  
  int32_t ttlt_slo_ms_;         // Time To Last Token SLO
  
  int32_t ttft_priority_weight_;    // TTFT 优先级权重
  int32_t tpot_priority_weight_;    // TPOT 优先级权重
  int32_t ttlt_priority_weight_;   // TTLT 优先级权重
  
  // ===== 方法 =====
  bool finished() const;                        // 是否完成
  void set_cancel();                            // 设置取消
  bool cancelled() const;                       // 是否已取消
  void set_deadline_ms();                      // 计算截止时间
  int32_t get_remaining_time() const;          // 剩余时间
  
  RequestOutput generate_output(...);            // 生成输出
};
```

#### 状态机

```
                    ┌──────────────┐
                    │   PENDING    │  等待调度
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
          ┌────────►│  WAITING    │◄─┐  被抢占
          │         └──────┬───────┘  │
          │                │          │
          │                ▼          │
          │         ┌──────────────┐  │
          │         │  RUNNING     │──┘  抢占恢复
          │         └──────┬───────┘
          │                │
          │       ┌────────┴────────┐
          │       │                 │
          │       ▼                 ▼
          │┌──────────────┐  ┌──────────────┐
          ││   PREFILL    │  │   DECODE     │
          │└──────┬───────┘  └──────┬───────┘
          │       │                 │
          │       └────────┬────────┘
          │                │
          │                ▼
          │         ┌──────────────┐
          │         │  FINISHED   │  完成
          │         └──────────────┘
          │
          │(取消路径)
          └───────────────────────► CANCELLED
```

### 3.2 Batch (批次) 组件

#### 设计意图

Batch 聚合多个请求的序列，一次前向传播处理多个序列，提高 GPU 利用率。

#### 类结构

```cpp
// core/framework/batch/batch.h

class Batch {
 public:
  // ===== 序列管理 =====
  std::vector<Sequence*> sequences_;              // 直接包含的序列
  std::vector<SequencesGroup*> sequence_groups_;   // 序列组 (Rec 模型)
  
  // ===== 前向类型 =====
  BatchForwardType batch_forward_type_;            // 前向传播类型
  //   - PREFILL: 预填充阶段 (处理输入 token)
  //   - DECODE: 解码阶段 (逐 token 生成)
  //   - PREFILL_DECODE: 混合阶段
  //   - RECURRENT: 循环阶段 (Rec 模型)
  
  // ===== KV Cache 交换 =====
  std::vector<BlockTransferInfo> swap_block_transfer_infos_;
  // 用于 P/D 分离时，Decode 节点从 Prefill 节点获取 KV Cache
  
  // ===== 多模态数据 =====
  std::vector<MMData> mm_data_vec_;              // 图像/视频数据
  
  // ===== 方法 =====
  void add(Sequence* sequence);                   // 添加序列
  ForwardInput prepare_forward_input(...);        // 准备前向输入
  void process_sample_output(...);               // 处理采样输出
  void finish();                                 // 标记所有序列完成
};
```

#### Batch 组装流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Batch 组装流程                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Scheduler:                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. 从 Waiting Queue 获取候选请求                                   │   │
│  │ 2. 根据优先级和 SLO 选择序列                                        │   │
│  │ 3. 检查 Batch 容量约束 (max_tokens, max_batch_size)                │   │
│  │ 4. 分配 KV Cache Block                                            │   │
│  │ 5. 创建 Batch 对象                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         Batch                                     │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                    │   │
│  │  │ Seq 1  │ │ Seq 2  │ │ Seq 3  │ │ Seq 4  │  ...              │   │
│  │  │[1,2,3] │ │[4,5]   │ │[6,7,8,9]│ │[10]    │                    │   │
│  │  │Tokens: │ │Tokens: │ │Tokens: │ │Tokens: │                    │   │
│  │  │ prefill│ │ decode │ │mixed    │ │ decode │                    │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘                    │   │
│  │                                                                   │   │
│  │  batch_forward_type: DECODE (假设都处于 decode 阶段)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Scheduler (调度器) 组件

#### 设计意图

Scheduler 负责请求的排队、批次组装、SLO 追踪和调度决策。

#### 核心接口

```cpp
// core/scheduler/scheduler.h

class Scheduler : public SchedulerBase {
 public:
  // ===== 请求管理 =====
  virtual bool add_request(std::shared_ptr<Request>& request) = 0;
  virtual uint32_t get_waiting_requests_num() const = 0;
  
  // ===== 调度执行 =====
  virtual void step(const absl::Duration& timeout) = 0;
  
  // ===== 指标 =====
  virtual void get_latency_metrics(
      std::vector<int64_t>& ttft,
      std::vector<int64_t>& tbt) = 0;
  
  // ===== 前缀缓存 =====
  virtual void reset_prefix_cache() {}
};
```

#### 调度策略实现

**Continuous Batching 策略**:

```
时间 ──────────────────────────────────────────────────────────────────────►

  Step 1                    Step 2                    Step 3
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ Batch: [A, B, C]    │  │ Batch: [A, B, D, E] │  │ Batch: [A, D, E, F] │
│                     │  │                     │  │                     │
│ A: ████████░░░░░░░░ │  │ A: ████████████░░░░ │  │ A: ████████████████ │
│ B: ██████████████░░░ │  │ B: █████████████████ │  │ D: ██████░░░░░░░░░ │
│ C: ████████████████░ │  │ D: ████░░░░░░░░░░░░ │  │ E: ████████████░░░ │
│                     │  │ E: █████░░░░░░░░░░░░ │  │ F: ███░░░░░░░░░░░░ │
│ Actions:            │  │ Actions:            │  │ Actions:            │
│ - C finished        │  │ - B finished        │  │ - A finished        │
│ - Add D, E         │  │ - Add F             │  │                     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

### 3.4 Worker (工作器) 组件

#### 设计意图

Worker 是模型执行的基本单元，负责 KV Cache 管理和前向传播。

#### 类结构

```cpp
// core/runtime/worker.h

class Worker {
 public:
  // ===== 生命周期 =====
  bool init_model(const std::string& weights_path, ...);
  bool sleep(MasterStatus status);         // 深度睡眠 (释放显存)
  bool wakeup(const WakeupOptions& options);
  bool update_weights(const std::string& weights_path);
  
  // ===== KV Cache 管理 =====
  bool allocate_kv_cache(const KVCacheShape& shape);
  
  // P/D 分离: 从其他节点拉取 KV Block
  folly::SemiFuture<bool> pull_kv_blocks_async(...);
  uint32_t transfer_kv_blocks(...);
  
  // ===== 执行 =====
  ForwardInput prepare_inputs(Batch& batch);
  std::optional<ForwardOutput> step(const ForwardInput& inputs);
  
  // ===== Profile =====
  bool start_profile();
  bool stop_profile();
};
```

### 3.5 Executor (执行器) 组件

#### 设计意图

Executor 封装平台特定的执行优化，如 CUDA Graph、ACL Graph 等。

#### 类结构

```cpp
// core/runtime/executor.h

class Executor {
 public:
  Executor(CausalLM* model,
           const ModelArgs& args,
           const torch::Device& device,
           const runtime::Options& options);
  
  // ===== 输入准备 =====
  ForwardInput prepare_inputs(Batch& batch);
  
  // ===== 前向传播 =====
  ModelOutput forward(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      std::vector<KVCache>& kv_caches,
      const ModelInputParams& params);
  
  // ===== 图优化 =====
  void prepare_graph_input(...);  // 准备 CUDA Graph 输入
  
 private:
  std::unique_ptr<ExecutorImpl> impl_;  // 平台相关实现
};
```

#### 图优化机制

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CUDA Graph 优化                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Without CUDA Graph:                                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Kernel1 │─►│ Kernel2 │─►│ Kernel3 │─►│ Kernel4 │─►│ Kernel5 │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
│      │            │            │            │            │             │
│      ▼            ▼            ▼            ▼            ▼             │
│   CPU Launch Overhead (5x)                                            │
│                                                                         │
│  With CUDA Graph:                                                      │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │                     cudaGraph_t                                │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │ Kernel1 → Kernel2 → Kernel3 → Kernel4 → Kernel5        │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│                       cudaGraphExec_t                                   │
│                              │                                          │
│                              ▼                                          │
│                    ┌──────────────────┐                                │
│                    │ graphLaunch()    │  Single CPU Launch              │
│                    └──────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.6 KV Cache (键值缓存) 组件

#### 设计意图

KV Cache 存储注意力层的 Key 和 Value，实现 token 生成的高效复用。

#### 类结构

```cpp
// core/framework/kv_cache/kv_cache.h

class KVCache {
 public:
  // ===== 缓存访问 =====
  torch::Tensor get_k_cache() const;
  torch::Tensor get_v_cache() const;
  
  // MLA 特有
  torch::Tensor get_index_cache() const;
  
  // 量化支持
  std::optional<torch::Tensor> get_k_cache_scale() const;
  std::optional<torch::Tensor> get_v_cache_scale() const;
  
  // ===== 缓存操作 =====
  void swap_blocks(torch::Tensor& src, torch::Tensor& dst);
  
 private:
  std::unique_ptr<KVCacheImpl> impl_;
};

// 多级 KV Cache (基于 Mooncake)
struct KVCacheTensors {
  torch::Tensor gpu_k;    // L1: GPU 显存 (最低延迟)
  torch::Tensor gpu_v;
  torch::Tensor cpu_k;    // L2: 主机内存 (中等延迟)
  torch::Tensor cpu_v;
  torch::Tensor disk_k;   // L3: SSD (高延迟, 大容量)
  torch::Tensor disk_v;
};
```

#### Paged Attention 机制

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Paged Attention + KV Cache                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  传统方式 (Contiguous):                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ KV Cache Block 0  │ KV Cache Block 1  │ KV Cache Block 2  │ ... │   │
│  │ [K,V for tokens   │ [K,V for tokens   │ [K,V for tokens   │     │   │
│  │  0-15]            │  16-31]            │  32-47]           │     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  问题: 内存碎片、预分配浪费                                              │
│                                                                         │
│  Paged 方式:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Request A: Block 0 → Block 2 → Block 5                          │   │
│  │ Request B: Block 1 → Block 3                                    │   │
│  │ Request C: Block 4                                              │   │
│  │                                                                 │   │
│  │ Physical Blocks:                                                │   │
│  │ [Block 0] [Block 1] [Block 2] [Block 3] [Block 4] [Block 5] ...│   │
│  │   A         B         A         B         C         A          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  优势: 内存利用率高、支持灵活分配/回收                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 分布式架构

### 4.1 P/D 分离架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Prefill-Decode 分离架构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────┐     KV Transfer     ┌─────────────────┐  │
│  │      Prefill Node(s)     │ ──────────────────► │   Decode Node(s)│  │
│  │                         │                      │                  │  │
│  │  ┌───────────────────┐  │                      │  ┌────────────┐ │  │
│  │  │ Prefill Scheduler │  │                      │  │ Decode     │ │  │
│  │  └─────────┬─────────┘  │                      │  │ Scheduler  │ │  │
│  │            │              │                      │  └─────┬──────┘ │  │
│  │            ▼              │                      │        │        │  │
│  │  ┌───────────────────┐    │                      │        ▼        │  │
│  │  │ Prefill Worker    │    │                      │  ┌──────────┐  │  │
│  │  │                   │    │                      │  │ Decode   │  │  │
│  │  │ [Layer 0]         │    │                      │  │ Worker   │  │  │
│  │  │ [Layer 1]         │    │                      │  │          │  │  │
│  │  │ ...               │    │                      │  │ [Layer 0]│  │  │
│  │  │ [Layer N]         │    │                      │  │ [Layer 1]│  │  │
│  │  │                   │    │                      │  │ ...      │  │  │
│  │  └───────────────────┘    │                      │  │ [Layer N]│  │  │
│  │            │              │                      │  └──────────┘  │  │
│  │            │ KV Cache     │                      │        │        │  │
│  └────────────┼──────────────┴──────────────────────┼────────┘        │
│               │                                      │                  │
│               ▼                                      ▼                  │
│        ┌────────────┐                        ┌────────────┐             │
│        │ KV Cache   │                        │ KV Cache   │             │
│        │ (Prefill)  │                        │ (Decode)   │             │
│        └────────────┘                        └────────────┘             │
│                                                                         │
│  通信协议: disagg_pd.proto                                              │
│  传输方式: RDMA / gRPC                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 多级缓存架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      多级 KV Cache 架构 (基于 Mooncake)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        LLM Inference                             │   │
│  │                                                                   │   │
│  │                        ┌─────────────┐                           │   │
│  │                        │   Prefill   │                           │   │
│  │                        │   Compute   │                           │   │
│  │                        └──────┬──────┘                           │   │
│  │                               │                                   │   │
│  │              ┌────────────────┼────────────────┐                 │   │
│  │              ▼                ▼                ▼                 │   │
│  │       ┌──────────┐     ┌──────────┐     ┌──────────┐              │   │
│  │       │ L1 Cache │     │ L2 Cache │     │ L3 Cache │              │   │
│  │       │ (GPU HBM)│     │ (CPU DDR)│     │  (SSD)   │              │   │
│  │       │ ~80GB    │     │ ~512GB   │     │ ~10TB    │              │   │
│  │       │ ~1μs     │     │ ~100μs   │     │ ~1ms     │              │   │
│  │       └──────────┘     └──────────┘     └──────────┘              │   │
│  │             │                │                │                     │   │
│  └─────────────┼────────────────┼────────────────┼─────────────────────┘   │
│                │                │                │                        │
│                ▼                ▼                ▼                        │
│         ┌───────────┐     ┌───────────┐     ┌───────────┐                │
│         │  NVIDIA   │     │  CPU +    │     │   NVMe   │                │
│         │   HBM     │     │  RDMA     │     │   SSD    │                │
│         └───────────┘     └───────────┘     └───────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 数据流设计

### 5.1 请求处理数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        请求处理数据流                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 请求接收 (HTTP/gRPC)                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ POST /v1/chat/completions                                       │   │
│  │ {                                                              │   │
│  │   "model": "Qwen2.5-7B",                                      │   │
│  │   "messages": [{"role": "user", "content": "Hello!"}]        │   │
│  │ }                                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  2. 协议解析 (Service Layer)                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ChatServiceImpl::Chat()                                        │   │
│  │   - 解析 JSON → ChatRequest                                    │   │
│  │   - 应用 Chat Template                                          │   │
│  │   - Tokenize 输入                                               │   │
│  │   - 创建 Request 对象                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  3. 调度 (Engine Layer)                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ LLMEngine::AddRequest()                                        │   │
│  │   - Request → Scheduler                                         │   │
│  │   - Scheduler::step() 周期性调度                                │   │
│  │   - Batch 组装                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  4. 执行 (Worker Layer)                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Worker::step()                                                 │   │
│  │   - prepare_inputs() → ForwardInput                            │   │
│  │   - Executor::forward() → ModelOutput                          │   │
│  │   - process_sample_output() → Token                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  5. 响应返回 (Service Layer)                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ChatServiceImpl::StreamResponse()                               │   │
│  │   - Token → JSON/SSE 格式化                                     │   │
│  │   - 流式发送                                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  6. HTTP Response (Stream)                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ data: {"choices":[{"delta":{"content":"Hello"}}]}              │   │
│  │ data: {"choices":[{"delta":{"content":" how"}}]}               │   │
│  │ data: {"choices":[{"delta":{"content":" are"}}]}              │   │
│  │ ...                                                            │   │
│  │ data: [DONE]                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 模型推理数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        模型推理数据流                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: [token_ids]                                                    │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────┐                                                    │
│  │   Embedding     │  Token → Hidden States                             │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Transformer Layers (N)                         │   │
│  │                                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │                    Layer i                               │    │   │
│  │  │                                                         │    │   │
│  │  │  ┌───────────────┐                                      │    │   │
│  │  │  │ Attention     │ ← Input: hidden_states, position_ids  │    │   │
│  │  │  │               │ ← KV Cache (for decode)              │    │   │
│  │  │  │ • QKV Proj    │                                      │    │   │
│  │  │  │ • Rotary PE   │                                      │    │   │
│  │  │  │ • Attention   │                                      │    │   │
│  │  │  │ • O Proj      │                                      │    │   │
│  │  │  │ → Output      │ → hidden_states                      │    │   │
│  │  │  └───────────────┘                                      │    │   │
│  │  │         │                                               │    │   │
│  │  │         ▼                                               │    │   │
│  │  │  ┌───────────────┐                                      │    │   │
│  │  │  │ FFN           │                                      │    │   │
│  │  │  │               │                                      │    │   │
│  │  │  │ • Gate Proj   │                                      │    │   │
│  │  │  │ • Up Proj     │                                      │    │   │
│  │  │  │ • SiLU × Up   │  (or MoE)                           │    │   │
│  │  │  │ • Down Proj   │                                      │    │   │
│  │  │  │ → Output      │ → hidden_states                      │    │   │
│  │  │  └───────────────┘                                      │    │   │
│  │  │         │                                               │    │   │
│  │  │         ▼                                               │    │   │
│  │  │  ┌───────────────┐                                      │    │   │
│  │  │  │ RMSNorm       │  → Layer Output                      │    │   │
│  │  │  └───────────────┘                                      │    │   │
│  │  │                                                         │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │   RMSNorm       │  Final Layer Norm                                 │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │   LM Head      │  Hidden → Vocab Scores                             │
│  └────────┬────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │   Sampler      │  Scores → Token                                     │
│  │                 │                                                    │
│  │ • Greedy       │  ← output_token                                     │
│  │ • Random       │                                                    │
│  │ • Beam Search  │                                                    │
│  └─────────────────┘                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 扩展点设计

### 6.1 模型扩展

```
添加新模型的步骤:

1. 创建模型目录
   xllm/models/llm/my_model/

2. 实现模型类
   ┌─────────────────────────────────────────────────────────┐
   │ class MyModel : public CausalLM {                      │
   │  public:                                                │
   │   MyModel(const ModelArgs& args, ...);                 │
   │                                                          │
   │  private:                                                │
   │   std::unique_ptr<MyAttentionLayer> attention_;         │
   │   std::unique_ptr<MyFFN> ffn_;                         │
   │   // ...                                                │
   │ };                                                       │
   └─────────────────────────────────────────────────────────┘

3. 注册模型
   ┌─────────────────────────────────────────────────────────┐
   │ // models/model_registry.cpp                           │
   │ ModelRegistry::Register("my_model",                    │
   │     [](const ModelArgs& args, ...) {                   │
   │         return std::make_unique<MyModel>(args, ...);    │
   │     });                                                 │
   └─────────────────────────────────────────────────────────┘

4. 添加配置解析
   ┌─────────────────────────────────────────────────────────┐
   │ // core/framework/config/model_config.cpp                │
   │ if (model_type == "my_model") {                        │
   │     return MyModelConfig::FromJSON(config);            │
   │ }                                                      │
   └─────────────────────────────────────────────────────────┘
```

### 6.2 硬件扩展

```
添加新硬件支持的步骤:

1. 创建平台抽象
   ┌─────────────────────────────────────────────────────────┐
   │ // core/platform/my_platform.h                          │
   │ class MyPlatform : public PlatformBase {               │
   │  public:                                                │
   │   static Platform* Create() { return new MyPlatform(); } │
   │                                                          │
   │   DeviceType GetDeviceType() override;                  │
   │   bool Initialize(const PlatformOptions&) override;     │
   │   void* Alloc(size_t size) override;                    │
   │   // ...                                                │
   │ };                                                      │
   └─────────────────────────────────────────────────────────┘

2. 实现 Executor
   ┌─────────────────────────────────────────────────────────┐
   │ // core/runtime/my_executor_impl.h                      │
   │ class MyExecutorImpl : public ExecutorImpl {           │
   │  public:                                                │
   │   ModelOutput forward(...) override;                   │
   │   void prepare_graph(...) override;                    │
   │ };                                                      │
   └─────────────────────────────────────────────────────────┘

3. 实现 Kernel
   ┌─────────────────────────────────────────────────────────┐
   │ // core/kernels/my/attention.cpp                        │
   │ void AttentionForward(                                  │
   │     const Tensor& query,                               │
   │     const Tensor& key,                                 │
   │     const Tensor& value,                               │
   │     Tensor& output,                                    │
   │     const AttentionParams& params) {                  │
   │     // MyPlatform specific implementation               │
   │ }                                                      │
   └─────────────────────────────────────────────────────────┘

4. 注册到 Factory
   ┌─────────────────────────────────────────────────────────┐
   │ // core/runtime/executor_impl_factory.cpp               │
   │ ExecutorImpl* CreateExecutorImpl(                       │
   │     const DeviceType& type) {                          │
   │   switch (type) {                                      │
   │     case DeviceType::MY_PLATFORM:                       │
   │       return new MyExecutorImpl();                     │
   │     // ...                                             │
   │   }                                                    │
   │ }                                                      │
   └─────────────────────────────────────────────────────────┘
```

### 6.3 调度器扩展

```
添加新调度策略的步骤:

1. 继承 Scheduler 基类
   ┌─────────────────────────────────────────────────────────┐
   │ class MyScheduler : public Scheduler {                 │
   │  public:                                                │
   │   bool add_request(...) override;                       │
   │   void step(...) override;                             │
   │                                                          │
   │  private:                                                │
   │   // 自定义调度逻辑                                       │
   │   std::vector<Sequence*> select_sequences(...);         │
   │ };                                                      │
   └─────────────────────────────────────────────────────────┘

2. 实现调度算法
   ┌─────────────────────────────────────────────────────────┐
   │ void MyScheduler::step(...) {                          │
   │   // 1. 收集等待中的请求                                  │
   │   // 2. 根据自定义策略选择序列                            │
   │   // 3. 分配资源                                        │
   │   // 4. 组建 Batch                                      │
   │   // 5. 提交执行                                        │
   │ }                                                      │
   └─────────────────────────────────────────────────────────┘

3. 注册到 Factory
   ┌─────────────────────────────────────────────────────────┐
   │ // core/scheduler/scheduler_factory.cpp                 │
   │ Scheduler* CreateScheduler(const std::string& type) {  │
   │   if (type == "my_scheduler") {                       │
   │     return new MyScheduler();                          │
   │   }                                                    │
   │ }                                                      │
   └─────────────────────────────────────────────────────────┘
```

---

## 7. 设计决策记录

| ID | 决策 | 理由 | 影响 |
|----|------|------|------|
| ADR-001 | 使用四层架构 | 分离关注点，便于测试和扩展 | 模块化程度高 |
| ADR-002 | Batch 作为核心抽象 | 支持 Continuous Batching | GPU 利用率高 |
| ADR-003 | KV Cache 多级设计 | 平衡延迟和容量 | 支持长序列 |
| ADR-004 | P/D 分离调度 | 优化不同阶段资源使用 | 降低 TTFT |
| ADR-005 | 平台抽象统一 Kernel | 屏蔽硬件差异 | 多硬件支持 |
| ADR-006 | 使用 brpc 作为 RPC | 高性能、企业级 | 服务稳定性 |

---

**文档结束**
