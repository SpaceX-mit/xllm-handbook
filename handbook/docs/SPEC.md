# xLLM 技术规格说明书 (SPEC)

> **文档版本**: v1.0  
> **项目**: xLLM 大模型推理框架  
> **日期**: 2026-07-23  
> **状态**: 发布

---

## 1. 概述

### 1.1 项目定位

xLLM 是一个**高效的开源大模型推理框架**，专为国产 AI 加速卡优化设计，提供企业级的服务部署能力。

### 1.2 核心能力

| 能力 | 描述 |
|------|------|
| **模型推理** | 支持 LLM、VLM、DiT、REC 等多种模型类型 |
| **国产硬件** | 深度适配 NPU、MLU、DCU、MUSA、ILU 等国产芯片 |
| **高性能调度** | Continuous Batching、P/D 分离调度 |
| **企业级部署** | 高可用、弹性扩缩容、SLO 保障 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           xLLM 系统架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Service Layer (服务层)                        │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │   │
│  │  │  Chat   │ │Completion│ │Embedding│ │Anthropic│ │  Rerank │  │   │
│  │  │         │ │         │ │         │ │         │ │         │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────┼─────────────────────────────────┐   │
│  │                         Engine Layer (引擎层)                     │   │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐                │   │
│  │  │ LLMEngine │◄──►│ VLMEngine │◄──►│ RecEngine │                │   │
│  │  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘                │   │
│  │        └────────────────┼────────────────┘                       │   │
│  │                         ▼                                         │   │
│  │              ┌──────────────────┐                                 │   │
│  │              │    Scheduler     │                                 │   │
│  │              │  (调度器层)       │                                 │   │
│  │              └────────┬─────────┘                                 │   │
│  └───────────────────────┼───────────────────────────────────────────┘   │
│                          │                                               │
│  ┌───────────────────────┼───────────────────────────────────────────┐   │
│  │                  Worker Layer (Worker 层)                         │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                      │   │
│  │  │  Prefill  │◄─┤   Batch   │◄─┤  Decode   │                      │   │
│  │  │  Worker   │  │  Manager  │  │  Worker   │                      │   │
│  │  └───────────┘  └───────────┘  └───────────┘                      │   │
│  │                                                                  │   │
│  │  ┌─────────────────────────────────────────────┐                │   │
│  │  │              KV Cache Manager               │                │   │
│  │  │          (基于 Mooncake 多级缓存)            │                │   │
│  │  └─────────────────────────────────────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Platform Layer (平台层)                       │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │   │
│  │  │  CUDA  │ │  NPU   │ │  MLU   │ │  DCU   │ │  MUSA  │ ...    │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │   │
│  │                                                                  │   │
│  │  ┌────────────────────────────────────────────────────────┐    │   │
│  │  │                    Kernels (算子)                       │    │   │
│  │  │  Attention │ Linear │ RMSNorm │ RoPE │ MoE │ ...      │    │   │
│  │  └────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         核心组件关系图                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    ┌──────────────┐                                                     │
│    │   Request    │  用户请求                                            │
│    └──────┬───────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│    ┌──────────────┐                                                     │
│    │   Scheduler  │  调度器: 请求排队、批次组织                           │
│    └──────┬───────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│    ┌──────────────┐                                                     │
│    │    Batch     │  批次: 聚合多个请求                                   │
│    └──────┬───────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│    ┌──────────────┐                                                     │
│    │   Forward    │  前向传播: 准备输入、执行推理                        │
│    │    Input     │                                                     │
│    └──────┬───────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│    ┌──────────────┐                                                     │
│    │    Worker    │  Worker: 执行计算                                    │
│    └──────┬───────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│    ┌──────────────┐                                                     │
│    │  Executor    │  执行器: 图优化、Kernel 调用                        │
│    └──────┬───────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│    ┌──────────────┐                                                     │
│    │   Platform   │  平台抽象: 适配不同硬件                            │
│    └──────────────┘                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块规格

### 3.1 Request (请求)

**文件**: `core/framework/request/request.h`

```cpp
class Request : public RequestBase {
 public:
  // 核心属性
  std::unique_ptr<SequencesGroup> sequences_group_;  // 序列组
  std::atomic<bool> cancelled_;                       // 取消标志
  
  // SLO 调度
  int32_t deadline_ms_;                               // 截止时间
  Urgency urgency_;                                    // 紧急程度
  bool starved_;                                       // 饥饿状态
  
  // 统计
  int32_t ttft_slo_ms_;    // Time To First Token SLO
  int32_t tpot_slo_ms_;    // Time Per Output Token SLO
  int32_t ttlt_slo_ms_;    // Time To Last Token SLO
};
```

**规格说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `request_id` | string | 全局唯一请求 ID |
| `sequences_group_` | SequencesGroup | 管理多个候选序列 |
| `cancelled_` | atomic_bool | 原子操作，支持并发取消 |
| `urgency_` | Urgency | {STARVED=2, URGENT=1, NORMAL=0} |
| `ttft_slo_ms_` | int32 | 首 Token 延迟 SLO |
| `tpot_slo_ms_` | int32 | 每输出 Token 延迟 SLO |

### 3.2 Batch (批次)

**文件**: `core/framework/batch/batch.h`

```cpp
class Batch {
 public:
  // 序列管理
  std::vector<Sequence*> sequences_;              // 批次中的序列
  std::vector<SequencesGroup*> sequence_groups_;  // 序列组
  
  // 前向传播类型
  BatchForwardType batch_forward_type_;            // PREFILL / DECODE / PREFILL_DECODE
  
  // KV Cache 交换信息
  std::vector<BlockTransferInfo> swap_block_transfer_infos_;
  
  // 多模态数据
  std::vector<MMData> mm_data_vec_;
  
  // 核心方法
  void add(Sequence* sequence);                    // 添加序列
  ForwardInput prepare_forward_input(...);          // 准备输入
  void process_sample_output(...);                 // 处理输出
};
```

**批次类型枚举**:

```cpp
enum class BatchForwardType {
  PREFILL = 0,              // 预填充阶段
  DECODE = 1,               // 解码阶段
  PREFILL_DECODE = 2,       // 混合阶段
  RECURRENT = 3,            // 循环阶段 (Rec 模型)
};
```

### 3.3 Scheduler (调度器)

**文件**: `core/scheduler/scheduler.h`

```cpp
class Scheduler : public SchedulerBase {
 public:
  // 调度接口
  virtual bool add_request(std::shared_ptr<Request>& request) = 0;
  virtual void step(const absl::Duration& timeout) = 0;
  virtual uint32_t get_waiting_requests_num() const = 0;
  
  // 延迟指标
  virtual void get_latency_metrics(
      std::vector<int64_t>& ttft,
      std::vector<int64_t>& tbt) = 0;
};
```

**调度器实现**:

| 调度器 | 文件 | 用途 |
|--------|------|------|
| `ContinuousScheduler` | `continuous_scheduler.cpp` | 连续批处理 |
| `DisaggPDScheduler` | `disagg_pd_scheduler.cpp` | P/D 分离调度 |
| `PDOOCScheduler` | `pd_ooc_scheduler.cpp` | P/D 乱序 |
| `DitScheduler` | `dit_scheduler.cpp` | DiT 调度 |

### 3.4 Worker (工作器)

**文件**: `core/runtime/worker.h`

```cpp
class Worker {
 public:
  // 生命周期
  bool init_model(const std::string& weights_path, ...);
  bool sleep(MasterStatus status);
  bool wakeup(const WakeupOptions& options);
  
  // KV Cache 管理
  bool allocate_kv_cache(const KVCacheShape& shape);
  folly::SemiFuture<bool> pull_kv_blocks_async(...);
  uint32_t transfer_kv_blocks(...);
  
  // 执行
  ForwardInput prepare_inputs(Batch& batch);
  std::optional<ForwardOutput> step(const ForwardInput& inputs);
  
  // Profile
  bool start_profile();
  bool stop_profile();
};
```

**Worker 类型**:

```cpp
enum class WorkerType {
  LLM = 0,              // 标准 LLM Worker
  VLM = 1,              // 视觉语言模型 Worker
  REC = 2,              // 推荐模型 Worker
  EMBED = 3,            // 嵌入模型 Worker
};
```

### 3.5 Executor (执行器)

**文件**: `core/runtime/executor.h`

```cpp
class Executor {
 public:
  // 执行接口
  ForwardInput prepare_inputs(Batch& batch);
  ModelOutput forward(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      std::vector<KVCache>& kv_caches,
      const ModelInputParams& params);
      
 private:
  std::unique_ptr<ExecutorImpl> impl_;  // 平台相关实现
};
```

**Executor 实现**:

| 平台 | 实现类 | 文件 |
|------|--------|------|
| CUDA | `CUDAGraphExecutorImpl` | `cuda_graph_executor_impl.cpp` |
| NPU | `ACLGraphExecutorImpl` | `acl_graph_executor_impl.cpp` |
| MLU | `MLUGraphExecutorImpl` | `mlu_graph_executor_impl.cpp` |
| DCU | `DCUGraphExecutorImpl` | `dcu_graph_executor_impl.cpp` |

### 3.6 KV Cache (键值缓存)

**文件**: `core/framework/kv_cache/kv_cache.h`

```cpp
class KVCache {
 public:
  // 缓存访问
  torch::Tensor get_k_cache() const;
  torch::Tensor get_v_cache() const;
  
  // 缓存交换
  void swap_blocks(torch::Tensor& src, torch::Tensor& dst);
  
 private:
  std::unique_ptr<KVCacheImpl> impl_;  // 多级缓存实现
};

// 多级 KV Cache (基于 Mooncake)
struct KVCacheTensors {
  torch::Tensor gpu_k;    // L1: GPU 显存
  torch::Tensor gpu_v;
  torch::Tensor cpu_k;    // L2: 主机内存
  torch::Tensor cpu_v;
  torch::Tensor disk_k;   // L3: SSD (可选)
  torch::Tensor disk_v;
};
```

### 3.7 ModelArgs (模型参数)

**文件**: `core/framework/model/model_args.h`

**核心模型配置**:

```cpp
struct ModelArgs {
  // 模型结构
  PROPERTY(std::string, model_type);        // "llama", "qwen", "glm"...
  PROPERTY(int64_t, hidden_size);            // 隐藏层维度
  PROPERTY(int64_t, n_layers);              // 层数
  PROPERTY(int64_t, n_heads);                // 注意力头数
  PROPERTY(int64_t, n_kv_heads);            // KV 头数 (GQA)
  PROPERTY(int64_t, head_dim);              // 头维度
  PROPERTY(int64_t, vocab_size);            // 词表大小
  
  // 位置编码
  PROPERTY(float, rope_theta);              // RoPE 基础频率
  PROPERTY(int64_t, max_position_embeddings); // 最大位置
  
  // MoE 配置
  PROPERTY(bool, use_moe);                  // 是否使用 MoE
  PROPERTY(int32_t, n_routed_experts);      // 路由专家数
  PROPERTY(int32_t, n_activated_experts);   // 激活专家数
  
  // MLA 配置 (DeepSeek V2/V3)
  PROPERTY(bool, enable_mla);               // 多头潜在注意力
  PROPERTY(int32_t, q_lora_rank);           // Q LoRA 秩
  PROPERTY(int32_t, kv_lora_rank);          // KV LoRA 秩
  
  // MTP 配置 (DeepSeek V3)
  PROPERTY(int32_t, num_nextn_predict_layers); // MTP 层数
};
```

---

## 4. 接口规格

### 4.1 Python API

**文件**: `pybind/llm.py`

```python
class LLM:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",           # "cuda", "npu", "mlu", "dcu"
        tensor_parallel: int = 1,
        dtype: str = "float16",         # "float16", "bfloat16", "int8"
        **kwargs
    ):
        """初始化 LLM 推理引擎"""
        
    def chat(
        self,
        messages: List[Dict],
        sampling_params: Optional[SamplingParams] = None
    ) -> ChatResponse:
        """对话补全 (同步)"""
        
    def stream_chat(
        self,
        messages: List[Dict],
        sampling_params: Optional[SamplingParams] = None
    ) -> Generator[str, None, None]:
        """流式对话"""
        
    def generate(
        self,
        prompts: Union[str, List[str]],
        **kwargs
    ) -> List[str]:
        """通用补全"""
```

### 4.2 C++ API

**文件**: `c_api/xllm.h`

```cpp
// 核心结构
typedef struct XLLMHandle* XLLMHandle;
typedef struct XLLMRequest* XLLMRequest;

// API 函数
XLLMHandle xllm_create_handle(const char* model_path, ...);
void xllm_destroy_handle(XLLMHandle handle);

XLLMRequest xllm_create_request(const char* prompt, ...);
void xllm_add_request(XLLMHandle handle, XLLMRequest request);

int xllm_step(XLLMHandle handle);
int xllm_finish_request(XLLMHandle handle, XLLMRequest request);

char* xllm_get_output(XLLMRequest request);
```

### 4.3 gRPC API

**文件**: `proto/chat.proto`

```protobuf
service ChatService {
  // 流式对话
  rpc Chat(ChatRequest) returns (stream ChatResponse);
  
  // 非流式对话
  rpc ChatComplete(ChatRequest) returns (ChatCompleteResponse);
}

message ChatRequest {
  string model = 1;
  repeated Message messages = 2;
  SamplingParams sampling_params = 3;
  bool stream = 4;
}

message ChatResponse {
  string content = 1;
  bool finish = 2;
  Usage usage = 3;
}
```

---

## 5. 硬件支持规格

### 5.1 平台抽象

```cpp
class PlatformBase {
 public:
  virtual DeviceType GetDeviceType() = 0;
  virtual bool Initialize(const PlatformOptions& options) = 0;
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* ptr) = 0;
  virtual void Memcpy(void* dst, const void* src, size_t size) = 0;
};
```

### 5.2 支持矩阵

| 硬件 | 简称 | 驱动要求 | Kernel 路径 |
|------|------|----------|-------------|
| NVIDIA GPU | CUDA | CUDA 11.8+ | `kernels/cuda/` |
| 华为 Ascend | NPU | HDK 25.2.0+ | `kernels/npu/` |
| 寒武纪 MLU | MLU | - | `kernels/mlu/` |
| 海光 DCU | DCU | - | `kernels/dcu/` |
| 摩尔线程 MUSA | MUSA | - | `kernels/musa/` |
| 芯动科技 ILU | ILU | - | `kernels/ilu/` |

---

## 6. 性能规格

### 6.1 延迟指标

| 指标 | 定义 | 目标 |
|------|------|------|
| TTFT | Time To First Token (首 Token 延迟) | < 100ms |
| TPOT | Time Per Output Token (每 Token 延迟) | < 20ms |
| TTLT | Time To Last Token (总延迟) | < 10s / 1K tokens |

### 6.2 吞吐指标

| 场景 | Batch Size | 吞吐目标 |
|------|------------|----------|
| 小批量 | 16 | > 500 tokens/s |
| 中批量 | 64 | > 2000 tokens/s |
| 大批量 | 256 | > 5000 tokens/s |

---

## 7. 验收标准

### 7.1 功能验收

| 功能 | 验收条件 |
|------|----------|
| 模型加载 | 成功加载 HuggingFace 格式模型 |
| 对话推理 | 正确返回对话结果 |
| 流式输出 | 支持 SSE 流式响应 |
| 采样控制 | 支持 temperature, top_p, top_k |
| 多模态 | 支持图像输入和理解 |

### 7.2 性能验收

| 指标 | 验收阈值 |
|------|----------|
| 冷启动时间 | < 30s (7B 模型) |
| 内存占用 | < 模型大小 * 1.5 |
| GPU 利用率 | > 80% (满负载) |
| 错误率 | < 0.01% |

### 7.3 兼容性验收

| 测试项 | 验收条件 |
|--------|----------|
| CUDA | 通过 NVIDIA GPU 测试 |
| NPU | 通过 Ascend NPU 测试 |
| MLU | 通过 Cambricon MLU 测试 |
| 模型兼容性 | 支持主流开源模型 |

---

## 8. 附录

### 8.1 术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| TTFT | Time To First Token | 首 Token 延迟 |
| TPOT | Time Per Output Token | 每输出 Token 延迟 |
| TTLT | Time To Last Token | 端到端延迟 |
| KV Cache | Key-Value Cache | 注意力键值缓存 |
| GQA | Grouped Query Attention | 分组查询注意力 |
| MoE | Mixture of Experts | 混合专家 |
| MLA | Multi-head Latent Attention | 多头潜在注意力 |
| MTP | Multi-Token Prediction | 多 Token 预测 |
| P/D | Prefill/Decode | 预填充/解码分离 |

### 8.2 参考文档

- [架构设计](./ARCHITECTURE.md)
- [接口定义](./INTERFACE.md)
- [代码规范](../.agents/skills/code-review/references/custom-code-style.md)

---

**文档结束**
