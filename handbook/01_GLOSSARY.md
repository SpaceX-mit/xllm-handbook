# xLLM 核心概念术语表

## 文档信息

```yaml
---
document_id: CONCEPT-001
version: 1.0.0
category: glossary
owner: xllm-team
verification_level: BOTH
---

# AI验收标准
ai_acceptance_criteria:
  - 所有术语定义必须包含: 名称、定义、代码位置、相关概念
  - 复杂概念需提供代码示例
  - 术语关系需形成网状结构
```

---

## A

### Attention (注意力机制)
**定义**: Transformer模型中的核心计算单元，用于建模序列中token之间的依赖关系。

**代码位置**: `xllm/core/layers/attention.h`

**核心公式**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**xLLM实现要点**:
- 支持Multi-Head Attention (MHA)
- 支持Multi-Query Attention (MQA)
- 支持Grouped-Query Attention (GQA)
- 支持Flash Attention融合

---

## B

### Batch (批处理)
**定义**: 将多个请求合并为一个推理单元进行处理，以提升GPU利用率。

**代码位置**: `xllm/core/framework/batch/batch.h`

**核心数据结构**:
```cpp
class Batch {
    std::vector<Sequence*> sequences_;        // 批次中的序列
    std::vector<SequencesGroup*> sequence_groups_;  // Beam Search组
    BatchForwardType batch_forward_type_;     // 前向类型
};
```

**关键操作**:
| 操作 | 说明 | 方法 |
|-----|------|-----|
| `add()` | 添加序列到批次 | Batch::add() |
| `prepare_forward_input()` | 准备前向输入 | Batch::prepare_forward_input() |
| `process_sample_output()` | 处理采样输出 | Batch::process_sample_output() |

---

### Beam Search (束搜索)
**定义**: 一种启发式搜索算法，在生成过程中维护多个候选序列。

**代码位置**: `xllm/core/framework/batch/batch.h`

**参数**:
```cpp
state_.sampling_param.beam_width  // 束宽度
```

---

## C

### CausalLM (因果语言模型)
**定义**: 基于自回归方式工作的语言模型，每个token只能看到之前的token。

**代码位置**: `xllm/core/framework/model/causal_lm.h`

**类继承关系**:
```cpp
class CausalLM : public ModelBase {
    // 核心推理接口
    virtual ForwardOutput forward(
        const ModelInputParams& params,
        const ModelArgs& args,
        const ForwardInput& inputs,
        int32_t cp_size = 1
    ) = 0;
};
```

---

### Chunked Prefill (分块预填充)
**定义**: 将长序列的预填充阶段拆分为多个小块处理，避免长请求阻塞。

**代码位置**: `xllm/core/scheduler/disagg_pd_scheduler.h`

**触发条件**:
```cpp
// 当序列长度超过阈值时启用分块
if (sequence->num_tokens() > chunk_threshold) {
    sequence_group->set_chunked_prefill_stage();
}
```

---

### Continuous Batching (连续批处理)
**定义**: 在批次中某个序列生成完成后，立即插入新请求，无需等待所有序列完成。

**代码位置**: `xllm/core/scheduler/continuous_scheduler.h`

**核心优势**: GPU利用率提升 2-3x

---

## D

### DP (Data Parallelism, 数据并行)
**定义**: 将模型复制到多个设备，每个设备处理不同的请求数据。

**代码位置**: `xllm/core/framework/parallel_state/`

**配置参数**:
```cpp
options_.dp_size    // 数据并行度
dp_local_size_      // 本地DP组大小
```

---

### DIT (Diffusion Transformer, 扩散Transformer)
**定义**: 基于Transformer架构的扩散模型，用于文生图等任务。

**代码位置**: `xllm/core/runtime/dit_executor.h`

---

## E

### EPLB (Expert Parallel Load Balance, 专家并行负载均衡)
**定义**: 在MoE架构中平衡不同专家的计算负载。

**代码位置**: `xllm/core/framework/eplb/eplb_manager.h`

**核心策略**:
```cpp
class EplbManager {
    std::unique_ptr<EplbPolicy> policy_;  // 负载均衡策略
};
```

---

## F

### Forward (前向传播)
**定义**: 输入数据通过神经网络计算得到输出的过程。

**代码位置**: `xllm/core/runtime/forward_params.h`

**参数结构**:
```cpp
struct ForwardInput {
    std::vector<torch::Tensor> hidden_states;  // 输入hidden states
    std::vector<torch::Tensor> position_ids;   // 位置ID
    std::vector<torch::Tensor> attention_mask;  // 注意力掩码
};
```

---

## G

### GQA (Grouped-Query Attention, 分组查询注意力)
**定义**: K/V头数少于Q头数的注意力机制，在质量和效率间取得平衡。

**代码位置**: `xllm/core/layers/attention.h`

**配置**:
```yaml
num_q_heads: 32      # Query头数
num_kv_heads: 8     # Key/Value头数 (GQA)
```

---

## H

### HuggingFace Model Loader
**定义**: 加载HuggingFace格式模型权重的组件。

**代码位置**: `xllm/core/framework/hf_model_loader.h`

**支持格式**:
- `safetensors` - 安全张量格式
- `pytorch_model.bin` - PyTorch checkpoint
- `config.json` - 模型配置

---

## K

### KV Cache (键值缓存)
**定义**: 缓存自注意力计算中的Key和Value张量，避免重复计算。

**代码位置**: `xllm/core/framework/kv_cache/kv_cache.h`

**核心结构**:
```cpp
class KVCache {
    torch::Tensor k_cache_;  // Key缓存 [num_heads, seq_len, head_dim]
    torch::Tensor v_cache_;  // Value缓存 [num_heads, seq_len, head_dim]
};
```

**Block管理**:
- `BlockManager` - 块分配器
- `Block` - 最小分配单元
- 支持滑动窗口、Prefix Cache等优化

---

## M

### MTP (Multi-Token Prediction, 多token预测)
**定义**: 在单次前向传播中预测多个未来token，加速推理。

**代码位置**: `xllm/core/runtime/mtp_worker_impl.h`

---

### MoE (Mixture of Experts, 专家混合)
**定义**: 由多个专家网络组成的架构，每次只激活部分专家。

**代码位置**: `xllm/core/layers/moe_layer.h`

**配置**:
```yaml
num_experts: 8           # 专家总数
num_active_experts: 2    # 每次激活的专家数
```

---

### MP (Model Parallelism, 模型并行)
**定义**: 将模型拆分到多个设备。

**子类型**:
| 类型 | 说明 | 代码位置 |
|-----|------|---------|
| TP (Tensor Parallel) | 张量并行，按层拆分 | `parallel_state/tensor_parallel_state.h` |
| PP (Pipeline Parallel) | 流水线并行，按层拆分 | `distributed_runtime/pipeline.h` |
| EP (Expert Parallel) | 专家并行，MoE专用 | `eplb/` |

---

### MQA (Multi-Query Attention, 多查询注意力)
**定义**: 所有Query头共享一个Key/Value头。

**配置**:
```yaml
num_q_heads: 32
num_kv_heads: 1  # MQA
```

---

## P

### Prefill (预填充阶段)
**定义**: 处理输入prompt，计算第一个token的阶段。

**代码位置**: `xllm/core/framework/request/sequence.h`

**特征**:
- 计算密集型
- 处理全部输入token
- 生成第一个输出token

---

### Prefix Cache (前缀缓存)
**定义**: 复用多个请求共享的系统提示词部分。

**代码位置**: `xllm/core/framework/prefix_cache/`

**命中条件**:
```cpp
// 检查前缀是否可缓存
bool can_cache_prefix(const std::vector<int>& tokens) {
    return is_static_prefix(tokens) && has_available_blocks();
}
```

---

## R

### Request (请求)
**定义**: 用户发起的一次完整推理请求。

**代码位置**: `xllm/core/framework/request/request.h`

**核心状态**:
```cpp
class Request {
    RequestState state_;                    // 请求状态
    std::unique_ptr<SequencesGroup> sequences_group_;  // 生成序列组
    std::atomic<bool> cancelled_;           // 取消标志
};
```

**SLO参数**:
| 参数 | 说明 | 单位 |
|-----|------|-----|
| TTFT | Time To First Token | ms |
| TPOT | Time Per Output Token | ms |
| TTLT | Time To Last Token | ms |

---

### RL (Reinforcement Learning, 强化学习)
**定义**: 通过与环境交互学习最优策略的机器学习方法。

**代码位置**: `xllm/core/runtime/rl_worker_impl.h`

**Sleep模式**: 支持RL场景下的深度休眠与唤醒

---

## S

### Sampling (采样)
**定义**: 从模型输出的logits分布中选择下一个token的过程。

**代码位置**: `xllm/core/framework/sampling/sampler.h`

**支持的采样策略**:
| 策略 | 类 | 参数 |
|-----|-----|-----|
| Greedy | `GreedySampler` | temperature=0 |
| Random | `RandomSampler` | temperature>0 |
| Beam Search | `BeamSearchSampler` | beam_width>1 |
| Top-K | `TopKSampler` | top_k |
| Top-P | `TopPSampler` | top_p |

---

### Scheduler (调度器)
**定义**: 负责管理请求队列、决定批处理策略的核心组件。

**代码位置**: `xllm/core/scheduler/scheduler.h`

**类型**:
| 调度器 | 适用场景 | 特点 |
|-------|---------|------|
| `ContinuousScheduler` | 在线服务 | 高吞吐、低延迟 |
| `DisaggPDScheduler` | P/D分离部署 | Prefill/Decode分离 |
| `ZeroEvictionScheduler` | 资源受限 | 防止驱逐 |
| `FixedStepsScheduler` | 离线批量 | 固定步数 |

---

### Sequence (序列)
**定义**: 单次生成任务对应的token序列。

**代码位置**: `xllm/core/framework/request/sequence.h`

**阶段状态**:
```cpp
enum class SequenceStage {
    PREFILL,           // 预填充阶段
    DECODE,            // 解码阶段
    FINISHED,          // 已完成
};
```

---

### Speculative Decoding (投机解码)
**定义**: 使用小模型快速生成候选token，大模型验证。

**代码位置**: `xllm/core/runtime/speculative_worker_impl.h`

**加速比**: 通常 2-4x

---

### SSM (State Space Model, 状态空间模型)
**定义**: 如Mamba等新一代RNN-like模型。

**代码位置**: `xllm/core/layers/ssm_layer.h`

---

## T

### TP (Tensor Parallelism, 张量并行)
**定义**: 将模型权重按张量维度拆分到多个设备。

**代码位置**: `xllm/core/framework/parallel_state/tensor_parallel_state.h`

**通信**: AllReduce 用于聚合分片计算结果

---

## V

### VLM (Vision-Language Model, 视觉语言模型)
**定义**: 同时理解图像和文本的多模态模型。

**代码位置**: `xllm/core/framework/multimodal/vlm_processor.h`

**处理流程**:
```
Image → Encoder → Vision Hidden States → Concatenate with Text → LLM
```

---

## W

### Worker (工作器)
**定义**: 执行模型推理的核心执行单元。

**代码位置**: `xllm/core/runtime/worker.h`

**生命周期**:
```cpp
class Worker {
    bool init_model(...);       // 模型初始化
    bool allocate_kv_cache(...); // KV Cache分配
    ForwardOutput step(...);     // 单步推理
    bool sleep(...);            // 深度休眠
    bool wakeup(...);           // 唤醒
};
```

**类型**:
| Worker类型 | 说明 |
|-----------|------|
| `LLMWorkerImpl` | 标准LLM推理 |
| `VLMWorkerImpl` | 视觉语言模型 |
| `EmbeddingWorkerImpl` | Embedding生成 |
| `RecWorkerImpl` | 推荐模型 |

---

## X

### XTensor (扩展张量)
**定义**: 支持跨设备、跨进程的统一张量寻址机制。

**代码位置**: `xllm/core/distributed_runtime/xtensor.h`

**特性**:
- 统一的虚拟地址空间
- 支持远程内存访问
- 支持动态迁移

---

## 术语关系图

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                      Request                            │
                    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
                    │  │   Sequence   │  │ SequencesGrp │  │    Batch     │  │
                    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
                    │         │                 │                 │          │
                    │         ▼                 ▼                 ▼          │
                    │  ┌─────────────────────────────────────────────────┐   │
                    │  │                  Scheduler                       │   │
                    │  │  ContinuousScheduler │ DisaggPDScheduler │ ... │   │
                    │  └─────────────────────────────────────────────────┘   │
                    │                         │                             │
                    │                         ▼                             │
                    │  ┌─────────────────────────────────────────────────┐   │
                    │  │                    Worker                        │   │
                    │  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │   │
                    │  │  │ Forward │  │ KVCache │  │  Model (CausalLM)│ │   │
                    │  │  └─────────┘  └─────────┘  └─────────────────┘ │   │
                    │  └─────────────────────────────────────────────────┘   │
                    │                         │                             │
                    │                         ▼                             │
                    │  ┌─────────────────────────────────────────────────┐   │
                    │  │               Execution Backend                 │   │
                    │  │   CUDA   │   NPU   │   MLU   │   DCU   │  ...  │   │
                    │  └─────────────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────────────┘
```

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [架构设计](./02_ARCHITECTURE.md) | 系统整体架构 |
| [领域模型](./02_DOMAIN_MODEL.md) | DDD领域划分 |
| [调度器设计](./02_SCHEDULER_DESIGN.md) | 调度器实现详解 |
| [Worker设计](./03_WORKER_DESIGN.md) | Worker实现详解 |
