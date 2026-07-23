# xLLM 核心代码深度分析 (CODE_ANALYSIS)

> **文档版本**: v1.0  
> **项目**: xLLM 大模型推理框架  
> **日期**: 2026-07-23

---

## 1. 核心执行路径分析

### 1.1 请求到 Token 的完整路径

```
┌─────────────────────────────────────────────────────────────────────────┐
│              请求 → Token 完整执行路径                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Layer 1: API 入口                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ pybind/llm.py::LLM.chat()                                      │   │
│  │   │                                                             │   │
│  │   ▼                                                             │   │
│  │ pybind/bind.cpp::chat_binding()                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Layer 2: Service                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ api_service/chat_service_impl.cpp::ChatServiceImpl::Chat()     │   │
│  │   │                                                             │   │
│  │   ├── 1. 解析 ChatRequest                                       │   │
│  │   ├── 2. 构建内部 Request                                       │   │
│  │   ├── 3. 调用 engine->AddRequest()                            │   │
│  │   └── 4. 返回 ChatResponse                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Layer 3: Engine                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ distributed_runtime/llm_engine.cpp::LLMEngine                  │   │
│  │   │                                                             │   │
│  │   ├── AddRequest(): 提交请求到 Scheduler                        │   │
│  │   ├── step(): 执行一次调度循环                                  │   │
│  │   └── prepare_inputs(): 准备 Batch 输入                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Layer 4: Scheduler                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ scheduler/continuous_scheduler.cpp::ContinuousScheduler          │   │
│  │   │                                                             │   │
│  │   ├── add_request(): 添加到等待队列                             │   │
│  │   ├── schedule(): 批次组装                                       │   │
│  │   └── forward_runner(): 提交执行                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Layer 5: Worker                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ runtime/worker_impl.cpp::WorkerImpl                             │   │
│  │   │                                                             │   │
│  │   ├── prepare_inputs(): Batch → ForwardInput                   │   │
│  │   ├── forward(): 执行推理                                       │   │
│  │   └── process_output(): 处理采样结果                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Layer 6: Executor                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ runtime/*_graph_executor_impl.cpp                               │   │
│  │   │                                                             │   │
│  │   ├── forward(): CUDA Graph 执行                                │   │
│  │   └── model->forward(): 模型前向                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Layer 7: Model                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ models/llm/*.cpp::CausalLM::forward()                           │   │
│  │   │                                                             │   │
│  │   ├── embedding(): Token → Hidden                               │   │
│  │   ├── for each layer:                                           │   │
│  │   │   ├── attention(): 自注意力                                 │   │
│  │   │   └── ffn(): 前馈网络                                       │   │
│  │   └── lm_head(): Hidden → Logits                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Layer 8: Kernels                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ kernels/*/ops_api.cpp                                          │   │
│  │   │                                                             │   │
│  │   ├── attention_forward()                                       │   │
│  │   ├── linear_forward()                                          │   │
│  │   ├── rms_norm_forward()                                        │   │
│  │   └── rope_forward()                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Layer 9: Device                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ CUDA/NPU/MLU Runtime                                           │   │
│  │   │                                                             │   │
│  │   ├── cudaLaunchKernel()                                       │   │
│  │   └── aclopExecute()                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 关键代码解析

### 2.1 WorkerImpl 执行循环

**文件**: `runtime/worker_impl.cpp`

```cpp
// 核心执行循环 - 简化版

namespace xllm {

class WorkerImpl {
 public:
    // 同步执行一步
    std::optional<ForwardOutput> step(const ForwardInput& inputs) {
        // ========== Step 1: 准备输入 ==========
        auto input_tensor = PrepareInputTensor(inputs);
        
        // ========== Step 2: 前向传播 ==========
        ModelOutput model_output;
        if (use_cuda_graph_) {
            // 使用 CUDA Graph 优化
            model_output = cuda_graph_forward(input_tensor);
        } else {
            // 普通前向传播
            model_output = model_->forward(input_tensor);
        }
        
        // ========== Step 3: 采样 ==========
        auto sample_output = Sample(model_output.logits, 
                                   inputs.sampling_params);
        
        // ========== Step 4: 更新状态 ==========
        UpdateKVCache(inputs.kv_caches);
        
        return ConstructOutput(sample_output);
    }
    
 private:
    // 准备输入 Tensor
    Tensor PrepareInputTensor(const ForwardInput& inputs) {
        // 1. 合并 token 和 position
        // 2. 处理 RoPE (如果需要)
        // 3. 处理多模态输入
        return MakeMergedInput(inputs);
    }
    
    // CUDA Graph 前向传播
    ModelOutput cuda_graph_forward(const Tensor& input) {
        // 关键优化: 使用预捕获的图执行
        // 避免多次 CPU-GPU 同步
        
        // 更新图输入
        UpdateGraphInput(input);
        
        // 单次图执行
        cudaGraphLaunch(graph_exec_, stream_);
        
        // 读取输出
        return ReadGraphOutput();
    }
    
    // 采样
    SampleOutput Sample(const Tensor& logits, 
                        const SamplingParams& params) {
        if (params.temperature == 0.0) {
            return GreedySampler::Sample(logits);
        } else {
            return RandomSampler::Sample(logits, params);
        }
    }
};

}  // namespace xllm
```

### 2.2 Scheduler 调度逻辑

**文件**: `scheduler/continuous_scheduler.cpp`

```cpp
// Continuous Batching 调度器实现

namespace xllm {

class ContinuousScheduler::Impl {
 public:
    void Schedule(const absl::Duration& timeout) {
        auto deadline = absl::Now() + timeout;
        
        // ========== Phase 1: 准备新请求 ==========
        PrepareNewRequests();
        
        // ========== Phase 2: 选择序列 ==========
        auto sequences = SelectSequences();
        
        if (sequences.empty()) {
            return;  // 没有可调度的序列
        }
        
        // ========== Phase 3: 组装批次 ==========
        auto batch = BuildBatch(sequences);
        
        // ========== Phase 4: 执行 ==========
        ExecuteBatch(batch);
        
        // ========== Phase 5: 后处理 ==========
        PostProcessBatch(batch);
    }
    
 private:
    // 选择要调度的序列
    std::vector<Sequence*> SelectSequences() {
        std::vector<Sequence*> candidates;
        
        // 1. 从 PENDING 状态获取
        for (auto& req : pending_requests_) {
            candidates.push_back(req->GetNextSequence());
        }
        
        // 2. 从 RUNNING 状态获取
        for (auto& req : running_requests_) {
            candidates.push_back(req->GetCurrentSequence());
        }
        
        // 3. 优先级排序
        std::sort(candidates.begin(), candidates.end(),
            [this](Sequence* a, Sequence* b) {
                return CompareByUrgency(a, b);
            });
        
        // 4. 截取到批次容量
        return TruncateToBatchSize(candidates);
    }
    
    // 按紧急程度比较
    bool CompareByUrgency(Sequence* a, Sequence* b) {
        auto req_a = a->request();
        auto req_b = b->request();
        
        // 检查饥饿状态
        if (req_a->is_starved() && !req_b->is_starved()) {
            return true;
        }
        if (!req_a->is_starved() && req_b->is_starved()) {
            return false;
        }
        
        // 检查紧急程度
        if (req_a->urgency() != req_b->urgency()) {
            return req_a->urgency() > req_b->urgency();
        }
        
        // 检查剩余时间
        return req_a->get_remaining_time() < req_b->get_remaining_time();
    }
    
    // 组装批次
    BatchPtr BuildBatch(const std::vector<Sequence*>& sequences) {
        auto batch = std::make_unique<Batch>();
        
        // 1. 添加序列
        for (auto* seq : sequences) {
            batch->add(seq);
        }
        
        // 2. 确定前向类型
        batch->refresh_forward_type();
        
        // 3. 分配 KV Cache Block
        AllocateKVCacheBlocks(batch.get());
        
        // 4. 准备输入
        batch->prepare_forward_input(...);
        
        return batch;
    }
    
    // 执行批次
    void ExecuteBatch(BatchPtr& batch) {
        // 1. 分发到 Worker
        auto inputs = batch->prepare_forward_input(...);
        
        // 2. 异步执行
        auto future = worker_->step_async(inputs);
        
        // 3. 等待完成
        auto output = future.get();
        
        // 4. 处理输出
        batch->process_sample_output(output);
    }
};
```

### 2.3 KV Cache 管理

**文件**: `framework/kv_cache/kv_cache.h`

```cpp
// KV Cache 实现

namespace xllm {

class KVCache {
 public:
    // 获取 K Cache
    torch::Tensor get_k_cache() const {
        return impl_->GetTensor(KVCacheTensorType::K);
    }
    
    // 获取 V Cache
    torch::Tensor get_v_cache() const {
        return impl_->GetTensor(KVCacheTensorType::V);
    }
    
    // 块交换 (用于 P/D 分离)
    void swap_blocks(torch::Tensor& src, torch::Tensor& dst) {
        // 将 src 的内容复制到 dst
        // 用于在不同的 KV Cache 区域之间移动数据
        impl_->CopyBlocks(src, dst);
    }
    
 private:
    std::unique_ptr<KVCacheImpl> impl_;
};

// KV Cache 实现 - 支持多级缓存
class KVCacheImpl {
 public:
    // 多级存储
    struct Level {
        torch::Tensor data;      // 存储数据
        MemoryType type;         // GPU/CPU/DISK
        size_t capacity;         // 容量
        size_t used;             // 已使用
    };
    
    std::vector<Level> levels_;  // L1, L2, L3
    
    // 获取张量
    torch::Tensor GetTensor(KVCacheTensorType type) {
        // 优先从 L1 获取
        for (auto& level : levels_) {
            if (level.type == MemoryType::GPU) {
                return level.data;
            }
        }
        return {};
    }
    
    // 块交换
    void CopyBlocks(torch::Tensor& src, torch::Tensor& dst) {
        // 根据源和目标的位置决定复制方式
        if (src.device().is_gpu() && dst.device().is_gpu()) {
            // GPU 内部复制
            dst.copy_(src);
        } else if (src.device().is_gpu() && dst.device().is_cpu()) {
            // GPU -> CPU
            dst.copy_(src);  // 或使用异步复制
        } else if (src.device().is_cpu() && dst.device().is_gpu()) {
            // CPU -> GPU
            dst.copy_(src);
        }
        // DISK 存储需要额外的 I/O 操作
    }
};

// 分配 KV Cache
void allocate_kv_caches(
    std::vector<KVCache>& kv_caches,
    const KVCacheShape& shape,
    const KVCacheCreateOptions& options) {
    
    for (int i = 0; i < shape.num_layers; ++i) {
        // 为每一层分配 KV Cache
        kv_caches[i] = KVCache(shape, options, i);
    }
}

}  // namespace xllm
```

### 2.4 Model 前向传播

**文件**: `models/llm/llama.cpp` (简化模型实现)

```cpp
// LLaMA 模型实现

namespace xllm {

class LlamaModel : public CausalLM {
 public:
    // 前向传播
    ModelOutput forward(const ModelInput& input) override {
        // ========== 1. Embedding ==========
        auto hidden_states = embedding_->forward(input.input_tokens);
        
        // ========== 2. Transformer Layers ==========
        for (int layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
            auto& layer = layers_[layer_idx];
            
            // Residual connection 准备
            auto layer_input = hidden_states;
            
            // ========== 2.1 Self Attention ==========
            // a. RMSNorm
            auto attn_input = rms_norm_(hidden_states);
            
            // b. QKV 投影
            auto [q, k, v] = qkv_proj_(attn_input);
            
            // c. RoPE
            apply_rotary_pos_emb(q, k, cos_cached_, sin_cached_);
            
            // d. Attention
            auto attn_output = attention_(q, k, v, kv_cache_[layer_idx]);
            
            // e. O 投影
            attn_output = o_proj_(attn_output);
            
            // f. 残差连接
            hidden_states = layer_input + attn_output;
            
            // ========== 2.2 FFN ==========
            // a. RMSNorm
            auto ffn_input = rms_norm_(hidden_states);
            
            // b. SwiGLU
            auto ffn_output = ffn_(ffn_input);
            
            // c. 残差连接
            hidden_states = hidden_states + ffn_output;
        }
        
        // ========== 3. Final RMSNorm ==========
        hidden_states = final_rms_norm_(hidden_states);
        
        // ========== 4. LM Head ==========
        auto logits = lm_head_(hidden_states);
        
        return ModelOutput{.logits = logits};
    }
    
 private:
    std::unique_ptr<Embedding> embedding_;
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    std::unique_ptr<RMSNorm> final_rms_norm_;
    std::unique_ptr<Linear> lm_head_;
    
    // KV Cache
    std::vector<KVCache> kv_cache_;
    
    // RoPE 缓存
    torch::Tensor cos_cached_;
    torch::Tensor sin_cached_;
};

// Transformer 层
class TransformerLayer {
 public:
    // Self Attention
    std::unique_ptr<RMSNorm> input_norm_;
    std::unique_ptr<Linear> qkv_proj_;
    std::unique_ptr<Attention> attention_;
    std::unique_ptr<Linear> o_proj_;
    
    // FFN
    std::unique_ptr<RMSNorm> post_attention_norm_;
    std::unique_ptr<FFN> ffn_;
};

}  // namespace xllm
```

### 2.5 Attention Kernel

**文件**: `kernels/cuda/attention.cu` (核心实现)

```cpp
// Flash Attention CUDA 实现

namespace xllm {

// 模板参数说明:
// - THREAD_BLOCK_SIZE: 线程块大小
// - HEAD_DIM: 头维度
// - IS_CAUSAL: 是否因果
template <int THREAD_BLOCK_SIZE, int HEAD_DIM, bool IS_CAUSAL>
__global__ void attention_kernel(
    const float* __restrict__ q,      // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ k,      // [batch, num_kv_heads, seq_len, head_dim]
    const float* __restrict__ v,      // [batch, num_kv_heads, seq_len, head_dim]
    float* __restrict__ output,        // [batch, num_heads, seq_len, head_dim]
    const float scale,
    const int seq_len,
    const int num_kv_heads) {
    
    // ========== 1. 线程和内存分配 ==========
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = threadIdx.y;  // 当前 token
    
    // ========== 2. 加载 Q ==========
    // 每个线程负责 Q 的一个元素
    float q_val = q[GET_INDEX(batch_idx, head_idx, seq_idx, threadIdx.x)];
    
    // ========== 3. Flash Attention 主循环 ==========
    // 块级tiling: 将 K, V 分块加载到 Shared Memory
    float acc = 0.0f;  // 累加器
    float max_val = -INFINITY;
    
    for (int block_start = 0; block_start < seq_len; 
         block_start += BLOCK_SIZE) {
        
        // a. 加载 K, V 块到 Shared Memory
        __syncthreads();
        load_k_block(block_start);
        load_v_block(block_start);
        __syncthreads();
        
        // b. 计算 Q @ K^T
        float block_max = -INFINITY;
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            float k_val = k_shared[j * HEAD_DIM + threadIdx.x];
            float score = q_val * k_val * scale;
            
            // Causal mask
            if (IS_CAUSAL && (seq_idx * BLOCK_SIZE + j > seq_idx)) {
                score = -INFINITY;
            }
            
            block_max = max(block_max, score);
        }
        
        // c. Softmax 数值稳定化
        float prev_max = max_val;
        max_val = max(prev_max, block_max);
        float exp_sum = exp(block_max - max_val);
        
        // d. 累加到输出
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            float k_val = k_shared[j * HEAD_DIM + threadIdx.x];
            float score = q_val * k_val * scale - max_val;
            float weight = exp(score);
            
            float v_val = v_shared[j * HEAD_DIM + threadIdx.x];
            acc *= exp(prev_max - max_val);  // rescale
            acc += weight * v_val;
        }
    }
    
    // ========== 4. 写入输出 ==========
    output[GET_INDEX(batch_idx, head_idx, seq_idx, threadIdx.x)] = acc;
}

}  // namespace xllm
```

---

## 3. 数据流详解

### 3.1 Request 到 Sequence 的转换

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Request → SequencesGroup → Sequence                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Request (用户请求)                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ request_id: "req-123"                                           │   │
│  │ messages: [                                                     │   │
│  │   {"role": "user", "content": "Hello!"}                        │   │
│  │ ]                                                               │   │
│  │ sampling_params: { temperature: 0.7 }                           │   │
│  │ best_of: 1  // 生成候选数                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  SequencesGroup (序列组 - 支持多候选)                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ sequences_: [                                                   │   │
│  │   Sequence 0: "Hello!" → ["Hi", "How"] → ...                   │   │
│  │ ]  // best_of = 1 时只有一个序列                                 │   │
│  │                                                                  │   │
│  │ beam_width > 1 时:                                               │   │
│  │ sequences_: [                                                   │   │
│  │   Sequence 0: beam 0                                             │   │
│  │   Sequence 1: beam 1                                             │   │
│  │   ...                                                           │   │
│  │ ]                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  Sequence (单个序列)                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ tokens_: [15426, 2420, 124, 5342]  // tokenized "Hello world!" │   │
│  │                                                                  │   │
│  │ kv_cache_blocks_: [block_0, block_1, ...]                       │   │
│  │                                                                  │   │
│  │ status_: PREFILL | DECODE | FINISHED                           │   │
│  │                                                                  │   │
│  │ prompt_tokens: 4          // 初始 prompt token 数               │   │
│  │ num_generated_tokens: 0   // 已生成 token 数                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Batch 的组装过程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Batch 组装过程                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  候选序列:                                                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ Seq A  │ │ Seq B  │ │ Seq C  │ │ Seq D  │ │ Seq E  │           │
│  │ prefill│ │ decode │ │ decode │ │ decode │ │ decode │           │
│  │ 16 tok │ │ 10 tok │ │ 8 tok  │ │ 12 tok │ │ 5 tok  │           │
│  │ 256 max│ │ 256 max│ │ 256 max│ │ 256 max│ │ 256 max│           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│                                                                         │
│  Batch 容量限制:                                                         │
│  - max_batch_size: 4                                                   │
│  - max_tokens_per_batch: 64                                            │
│                                                                         │
│  选择过程:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: 按紧急程度排序                                           │   │
│  │   [Seq A (starved), Seq B, Seq C, Seq D, Seq E]                │   │
│  │                                                                   │   │
│  │ Step 2: 选择前 4 个 (max_batch_size)                            │   │
│  │   [Seq A, Seq B, Seq C, Seq D]                                 │   │
│  │                                                                   │   │
│  │ Step 3: 检查 token 数量                                          │   │
│  │   16 + 10 + 8 + 12 = 46 ≤ 64 ✓                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  生成的 Batch:                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Batch {                                                         │   │
│  │   batch_id: 42,                                                │   │
│  │   sequences: [Seq A, Seq B, Seq C, Seq D],                     │   │
│  │   batch_forward_type: PREFILL_DECODE,  // 混合                   │   │
│  │   total_tokens: 46,                                            │   │
│  │   allowed_max_tokens: [240, 246, 248, 244],                    │   │
│  │   swap_block_infos: [...],                                     │   │
│  │   mm_data: [...]                                                │   │
│  │ }                                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 关键算法分析

### 4.1 SLO 感知调度

```cpp
// 调度优先级计算

namespace xllm {

class SLOScheduler {
 public:
    // 计算请求优先级分数
    double CalculatePriority(Request* req) {
        auto seq = req->sequences()[0];
        int64_t remaining_time = req->get_remaining_time();
        
        // 基础分数
        double score = 0.0;
        
        // ========== 因素 1: 剩余时间 ==========
        // 时间越少，分数越高
        if (remaining_time < 0) {
            score += 1000.0;  // 已超时，极高优先级
        } else if (remaining_time < 1000) {  // < 1s
            score += 500.0;
        } else if (remaining_time < 5000) {  // < 5s
            score += 200.0;
        }
        
        // ========== 因素 2: TTFT 紧迫度 ==========
        if (seq->is_prefill_stage()) {
            int64_t ttft_deadline = req->ttft_slo_ms();
            int64_t ttft_elapsed = req->get_elapsed_time_ms();
            int64_t ttft_remaining = ttft_deadline - ttft_elapsed;
            
            if (ttft_remaining < 0) {
                score += 300.0;  // TTFT 已超时
            } else if (ttft_remaining < ttft_deadline * 0.5) {
                score += 100.0;  // TTFT 紧迫
            }
        }
        
        // ========== 因素 3: TPOT 紧迫度 ==========
        if (!seq->is_prefill_stage()) {
            int64_t tpot_target = req->tpot_slo_ms();
            int64_t current_tpot = CalculateCurrentTPOT(seq);
            
            if (current_tpot > tpot_target) {
                double ratio = (double)current_tpot / tpot_target;
                score += ratio * 50.0;
            }
        }
        
        // ========== 因素 4: 优先级权重 ==========
        score *= req->tpot_priority_weight();
        
        // ========== 因素 5: 饥饿惩罚 ==========
        if (req->is_starved()) {
            score *= 2.0;  // 饥饿请求加权
        }
        
        return score;
    }
    
 private:
    // 计算当前 TPOT
    int64_t CalculateCurrentTPOT(Sequence* seq) {
        // 简单计算: 最近 N 个 token 的平均延迟
        const int WINDOW = 5;
        auto recent_latencies = seq->GetRecentLatencies(WINDOW);
        return Average(recent_latencies);
    }
};

}  // namespace xllm
```

### 4.2 CUDA Graph 优化

```cpp
// CUDA Graph 捕获和执行

namespace xllm {

class CUDAGraphExecutor {
 public:
    // 捕获计算图
    void CaptureGraph() {
        // 1. 开始捕获
        CUDA_CHECK(cudaStreamBeginCapture(
            stream_, 
            cudaStreamCaptureModeThreadLocal
        ));
        
        // 2. 执行前向传播 (自动记录所有 CUDA 调用)
        ExecuteForwardInternal();
        
        // 3. 结束捕获
        CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
        
        // 4. 实例化图
        CUDA_CHECK(cudaGraphInstantiate(
            &graph_exec_,
            graph_,
            NULL,
            NULL,
            0
        ));
        
        // 5. 更新状态
        graph_captured_ = true;
    }
    
    // 使用图执行
    void GraphForward() {
        if (!graph_captured_) {
            CaptureGraph();
        }
        
        if (input_changed_) {
            // 输入改变了，需要重新捕获
            cudaGraphExecDestroy(graph_exec_);
            CaptureGraph();
            input_changed_ = false;
        }
        
        // 单次图执行
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
    }
    
 private:
    void ExecuteForwardInternal() {
        // 前向传播的所有 CUDA 调用都会被捕获
        // 包括:
        // - gemm (矩阵乘法)
        // - attention kernel
        // - RMSNorm kernel
        // - element-wise operations
        
        for (auto& layer : layers_) {
            layer->Forward();
        }
    }
    
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaStream_t stream_;
    bool graph_captured_ = false;
    bool input_changed_ = false;
};

}  // namespace xllm
```

---

## 5. 内存管理分析

### 5.1 KV Cache 内存布局

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 内存布局 (Paged Attention)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  逻辑视图 (按序列):                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Seq A: [Block 0] → [Block 2] → [Block 5]                      │   │
│  │ Seq B: [Block 1] → [Block 3]                                  │   │
│  │ Seq C: [Block 4]                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  物理视图 (按 Block):                                                    │
│  ┌───────┬───────┬───────┬───────┬───────┬───────┐                  │
│  │Blk 0  │Blk 1  │Blk 2  │Blk 3  │Blk 4  │Blk 5  │  ...           │
│  │ seq A │ seq B │ seq A │ seq B │ seq C │ seq A │                  │
│  │ layer0│ layer0│ layer0│ layer0│ layer0│ layer0│                  │
│  │ layer1│ layer1│ layer1│ layer1│ layer1│ layer1│                  │
│  │  ...  │  ...  │  ...  │  ...  │  ...  │  ...  │                  │
│  │ layerN│ layerN│ layerN│ layerN│ layerN│ layerN│                  │
│  └───────┴───────┴───────┴───────┴───────┴───────┘                  │
│                                                                         │
│  Block 内部布局 (单个 Block):                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ [K Cache]  shape: [num_kv_heads, block_size, head_dim]         │   │
│  │ [V Cache]  shape: [num_kv_heads, block_size, head_dim]         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  计算时访问 (Paged Attention):                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ for each sequence:                                             │   │
│  │   blocks = sequence.kv_cache_blocks                            │   │
│  │   for each token_idx in range(seq_len):                        │   │
│  │     block_id = blocks[token_idx / block_size]                  │   │
│  │     offset = token_idx % block_size                            │   │
│  │     k_block = all_k_blocks[block_id]                           │   │
│  │     k = k_block[:, offset, :]                                  │   │
│  │     v = v_block[:, offset, :]                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 内存分配策略

```cpp
// Block 分配策略

namespace xllm {

class BlockManager {
 public:
    // 分配 Block
    int AllocateBlock() {
        // 1. 尝试从空闲列表获取
        if (!free_blocks_.empty()) {
            int block_id = free_blocks_.back();
            free_blocks_.pop_back();
            allocated_blocks_.insert(block_id);
            return block_id;
        }
        
        // 2. 尝试驱逐低优先级序列的 Block
        for (auto* seq : GetEvictableSequences()) {
            auto blocks = seq->ReleaseBlocks();
            for (int block_id : blocks) {
                free_blocks_.push_back(block_id);
            }
            if (!free_blocks_.empty()) {
                return AllocateBlock();  // 递归
            }
        }
        
        // 3. 扩展 Block 池 (如果有更多 GPU 内存)
        if (CanExpand()) {
            ExpandBlockPool();
            return AllocateBlock();  // 递归
        }
        
        // 4. 无法分配
        return -1;
    }
    
    // 驱逐策略
    std::vector<Sequence*> GetEvictableSequences() {
        std::vector<Sequence*> candidates;
        
        // 1. 优先驱逐完成的请求
        for (auto* seq : allocated_sequences_) {
            if (seq->IsFinished()) {
                candidates.push_back(seq);
            }
        }
        
        // 2. 然后驱逐低优先级请求
        if (candidates.empty()) {
            for (auto* seq : allocated_sequences_) {
                if (seq->priority() < HIGH_PRIORITY) {
                    candidates.push_back(seq);
                }
            }
        }
        
        // 3. 按 LRU 排序
        std::sort(candidates.begin(), candidates.end(),
            [](Sequence* a, Sequence* b) {
                return a->last_access_time() < b->last_access_time();
            });
        
        return candidates;
    }
};

}  // namespace xllm
```

---

## 6. 性能优化点总结

### 6.1 关键优化清单

| 优化点 | 实现位置 | 效果 |
|--------|----------|------|
| Continuous Batching | `scheduler/` | GPU 利用率提升 3-5x |
| CUDA Graph | `runtime/*_executor_impl.cpp` | CPU 开销降低 50% |
| Paged Attention | `framework/kv_cache/` | 内存利用率提升 2x |
| 平台 Kernel | `kernels/*/` | 算子性能提升 20-50% |
| 异步执行 | `runtime/worker.h` | 延迟降低 30% |
| 量化推理 | `framework/quant/` | 吞吐提升 2x |
| P/D 分离 | `distributed_runtime/` | TTFT 降低 50% |

### 6.2 瓶颈分析矩阵

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         性能瓶颈分析矩阵                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  小批量 (< 8)           中批量 (8-64)       大批量 (> 64)               │
│  ├────────────────────┬────────────────────┬────────────────────┤     │
│  │ CPU-GPU 同步       │ Batch 组装开销     │ 内存带宽           │     │
│  │ (主要瓶颈)         │ (主要瓶颈)         │ (主要瓶颈)         │     │
│  │                    │                    │                    │     │
│  │ 解决方案:          │ 解决方案:          │ 解决方案:          │     │
│  │ - CUDA Graph      │ - 智能预取         │ - 量化             │     │
│  │ - 异步执行        │ - 批次优化         │ - 算子融合         │     │
│  └────────────────────┴────────────────────┴────────────────────┘     │
│                                                                         │
│  长序列 (> 4K)          多模态                 MoE 模型                │
│  ├────────────────────┬────────────────────┬────────────────────┤     │
│  │ KV Cache 容量      │ 图像预处理         │ 路由开销           │     │
│  │ (主要瓶颈)         │ (主要瓶颈)         │ (主要瓶颈)         │     │
│  │                    │                    │                    │     │
│  │ 解决方案:          │ 解决方案:          │ 解决方案:          │     │
│  │ - 多级缓存         │ - GPU 预处理       │ - 专家并行         │     │
│  │ - 选择性缓存       │ - 批处理图像       │ - 动态路由         │     │
│  └────────────────────┴────────────────────┴────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**文档结束**
