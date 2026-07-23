# xLLM Worker 设计详解

## 文档信息

```yaml
---
document_id: DESIGN-WORKER-001
version: 1.0.0
category: component_design
owner: xllm-team
verification_level: BOTH
depends_on:
  - DESIGN-001  # Domain Model
  - DESIGN-SCHEDULER-001  # Scheduler
---
```

---

## 1. Worker 架构概览

### 1.1 Worker家族

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Worker Family                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────────┐                                 │
│                         │   Worker        │                                 │
│                         │  (公开接口)      │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                           │
│                                  ▼                                           │
│                    ┌─────────────────────────┐                            │
│                    │      WorkerImpl          │                            │
│                    │    (PIMPL实现)          │                            │
│                    └────────┬────────────────┘                            │
│                             │                                                │
│     ┌───────────────────────┼───────────────────────┐                      │
│     │                       │                       │                      │
│     ▼                       ▼                       ▼                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐          │
│  │LLMWorkerImpl│     │VLMWorkerImpl│     │RecWorkerImpl        │          │
│  │             │     │             │     │                     │          │
│  │ 纯文本推理  │     │多模态推理   │     │推荐模型推理          │          │
│  └─────────────┘     └─────────────┘     └─────────────────────┘          │
│     │                       │                       │                      │
│     ▼                       ▼                       ▼                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                 │
│  │EmbedWorker  │     │DitWorkerImpl│     │Speculative  │                 │
│  │             │     │             │     │WorkerImpl   │                 │
│  │Embedding生成│     │扩散模型推理 │     │投机解码     │                 │
│  └─────────────┘     └─────────────┘     └─────────────┘                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Worker 接口设计

```cpp
// 代码位置: xllm/core/runtime/worker.h

/**
 * @class Worker
 * @brief Worker公共接口 - 推理执行单元
 * 
 * @design 使用PIMPL模式隐藏实现细节
 * @thread_safety 公共方法是线程安全的
 */
class Worker {
 public:
    /**
     * @brief 构造函数
     * @param parallel_args 并行配置
     * @param device 设备
     * @param options 运行时选项
     * @param worker_type Worker类型
     */
    Worker(const ParallelArgs& parallel_args,
          const torch::Device& device,
          const runtime::Options& options,
          WorkerType worker_type);
    
    ~Worker();
    
    // ========== 不可拷贝 ==========
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    Worker(Worker&&) = delete;
    Worker& operator=(Worker&&) = delete;

    // ========== 生命周期管理 ==========
    
    /**
     * @brief 初始化模型 (同步)
     * @param model_weights_path 权重路径
     * @param random_seed 随机种子
     * @param master_status 主节点状态
     * @return 初始化是否成功
     */
    bool init_model(const std::string& model_weights_path,
                   int32_t random_seed,
                   MasterStatus master_status);
    
    /**
     * @brief 初始化模型 (异步)
     */
    folly::SemiFuture<bool> init_model_async(
        const std::string& model_weights_path,
        int32_t random_seed,
        MasterStatus master_status);
    
    /**
     * @brief 深度休眠 (用于RL)
     */
    bool sleep(MasterStatus master_status);
    
    /**
     * @brief 唤醒
     */
    bool wakeup(const WakeupOptions& options);
    
    /**
     * @brief 异步唤醒
     */
    folly::SemiFuture<bool> wakeup_async(const WakeupOptions& options);
    
    /**
     * @brief 更新权重 (热更新)
     */
    bool update_weights(const std::string& weights_path);

    // ========== 推理执行 ==========
    
    /**
     * @brief 准备前向输入
     * @param batch 输入批次
     * @return 前向输入
     */
    ForwardInput prepare_inputs(Batch& batch);
    
    /**
     * @brief 执行一步推理 (同步)
     */
    std::optional<ForwardOutput> step(const ForwardInput& inputs);
    
    /**
     * @brief 执行一步推理 (异步)
     */
    folly::SemiFuture<std::optional<ForwardOutput>> step_async(
        const ForwardInput& inputs);
    
    /**
     * @brief 获取上一步结果 (异步)
     */
    folly::SemiFuture<std::optional<ForwardOutput>> get_last_step_result_async();

    // ========== KV Cache管理 ==========
    
    /**
     * @brief 估算KV Cache容量
     */
    std::tuple<int64_t, int64_t> estimate_kv_cache_capacity();
    
    /**
     * @brief 分配KV Cache
     */
    bool allocate_kv_cache(const KVCacheShape& kv_cache_shape);
    
    /**
     * @brief 异步分配KV Cache
     */
    folly::SemiFuture<bool> allocate_kv_cache_async(
        const KVCacheShape& kv_cache_shape);
    
    /**
     * @brief 异步分配KV Cache (带传输)
     */
    virtual folly::SemiFuture<bool> allocate_kv_cache_with_transfer_async(
        const KVCacheShape& kv_cache_shape);

    // ========== KV传输 ==========
    
    /**
     * @brief 从远程拉取KV Block
     */
    virtual folly::SemiFuture<bool> pull_kv_blocks_async(
        const uint64_t src_cluster_id,
        const std::string& src_addr,
        const std::vector<uint64_t>& src_blocks,
        const std::vector<uint64_t>& dst_blocks,
        const std::vector<uint64_t>& src_linear_state_ids = {},
        const std::vector<uint64_t>& dst_linear_state_ids = {});
    
    /**
     * @brief 传输KV Block
     */
    virtual uint32_t transfer_kv_blocks(
        const uint64_t batch_id,
        const std::vector<BlockTransferInfo>& block_transfer_info);
    
    virtual uint32_t transfer_kv_blocks(
        const uint64_t batch_id,
        Slice<BlockTransferInfo>& block_transfer_info);

    // ========== 集群管理 ==========
    
    /**
     * @brief 获取缓存信息
     */
    void get_cache_info(uint64_t& cluster_id, std::string& addr, uint16_t& port);
    
    /**
     * @brief 链接远程集群
     */
    bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                     const std::vector<std::string>& addrs,
                     const std::vector<uint16_t>& ports);
    
    /**
     * @brief 取消链接远程集群
     */
    bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                       const std::vector<std::string>& addrs,
                       const std::vector<uint16_t>& ports);
    
    /**
     * @brief P2P链接 (权重传输)
     */
    bool link_p2p(const std::string& remote_addr);
    bool unlink_p2p(const std::string& remote_addr);

    // ========== Profiling ==========
    
    bool start_profile();
    bool stop_profile();
    folly::SemiFuture<bool> start_profile_async();
    folly::SemiFuture<bool> stop_profile_async();
    
    /**
     * @brief 获取活跃激活内存
     */
    int64_t get_active_activation_memory();
    folly::SemiFuture<int64_t> get_active_activation_memory_async();

    // ========== 辅助方法 ==========
    
    const bool is_driver();
    const torch::Device& device() const;
    folly::SemiFuture<folly::Unit> process_group_test_async();

 private:
    WorkerImpl* impl_ = nullptr;  // PIMPL
    ThreadPool threadpool_{/*num_threads=*/1, /*cpu_binding=*/false,
                          /*pool_name=*/"Worker.async"};
};
```

---

## 2. WorkerImpl 详解

### 2.1 实现层次

```cpp
// 代码位置: xllm/core/runtime/worker_impl.h

/**
 * @class WorkerImpl
 * @brief Worker实现基类
 * 
 * @design 所有Worker类型的基类，包含通用实现
 */
class WorkerImpl {
 public:
    WorkerImpl(const ParallelArgs& parallel_args,
              const torch::Device& device,
              const runtime::Options& options,
              WorkerType worker_type);
    
    virtual ~WorkerImpl() = default;
    
    // ========== 子类需要实现 ==========
    
    /**
     * @brief 初始化模型 (子类实现)
     */
    virtual bool init_model_impl(const std::string& model_weights_path,
                                int32_t random_seed,
                                MasterStatus master_status) = 0;
    
    /**
     * @brief 执行前向计算 (子类实现)
     */
    virtual std::optional<ForwardOutput> forward_impl(
        const ForwardInput& inputs) = 0;
    
    /**
     * @brief 准备模型输入 (子类实现)
     */
    virtual ForwardInput prepare_inputs_impl(Batch& batch) = 0;
    
    // ========== 通用实现 ==========
    
    /**
     * @brief 休眠
     */
    virtual bool sleep_impl(MasterStatus status);
    
    /**
     * @brief 唤醒
     */
    virtual bool wakeup_impl(const WakeupOptions& options);
    
    /**
     * @brief 分配KV Cache
     */
    virtual bool allocate_kv_cache_impl(const KVCacheShape& shape);
    
    // ========== 公共方法 ==========
    
    bool init_model(const std::string& model_weights_path,
                   int32_t random_seed,
                   MasterStatus master_status);
    
    ForwardInput prepare_inputs(Batch& batch);
    std::optional<ForwardOutput> step(const ForwardInput& inputs);
    
    bool sleep(MasterStatus status);
    bool wakeup(const WakeupOptions& options);

 protected:
    // ========== 受保护成员 ==========
    
    torch::Device device_;
    runtime::Options options_;
    WorkerType worker_type_;
    
    // 模型
    std::unique_ptr<CausalLM> model_;
    
    // 执行器
    std::unique_ptr<Executor> executor_;
    
    // KV Cache管理器
    std::unique_ptr<KVCacheManager> kv_cache_manager_;
    
    // Block管理器
    std::unique_ptr<BlockManagerPool> block_manager_pool_;
    
    // 并行状态
    std::unique_ptr<ParallelState> parallel_state_;
    
    // 分词器
    std::unique_ptr<Tokenizer> tokenizer_;
    
    // 统计
    WorkerStats stats_;
    
 private:
    // 初始化辅助
    bool load_model_weights(const std::string& path);
    bool initialize_kv_cache();
    bool initialize_executor();
};
```

### 2.2 LLMWorkerImpl 实现

```cpp
// 代码位置: xllm/core/runtime/llm_worker_impl.h

/**
 * @class LLMWorkerImpl
 * @brief 标准LLM Worker实现
 */
class LLMWorkerImpl : public WorkerImpl {
 public:
    LLMWorkerImpl(const ParallelArgs& parallel_args,
                 const torch::Device& device,
                 const runtime::Options& options);
    
    // ========== WorkerImpl 实现 ==========
    
    bool init_model_impl(const std::string& model_weights_path,
                        int32_t random_seed,
                        MasterStatus master_status) override;
    
    std::optional<ForwardOutput> forward_impl(
        const ForwardInput& inputs) override;
    
    ForwardInput prepare_inputs_impl(Batch& batch) override;

 private:
    // ========== 前向计算 ==========
    
    /**
     * @brief 执行Prefill阶段
     */
    ForwardOutput forward_prefill(const ForwardInput& inputs,
                                  const ForwardParams& params);
    
    /**
     * @brief 执行Decode阶段
     */
    ForwardOutput forward_decode(const ForwardInput& inputs,
                                const ForwardParams& params);
    
    /**
     * @brief 执行混合阶段
     */
    ForwardOutput forward_mixed(const ForwardInput& inputs,
                                const ForwardParams& params);
    
    // ========== 输入准备 ==========
    
    /**
     * @brief 准备Prefill输入
     */
    ForwardInput prepare_prefill_input(Batch& batch);
    
    /**
     * @brief 准备Decode输入
     */
    ForwardInput prepare_decode_input(Batch& batch);
    
    // ========== 模型相关 ==========
    
    std::unique_ptr<CausalLM> model_;
    ModelArgs model_args_;
    
    // 采样器
    std::unique_ptr<Sampler> sampler_;
    
    // 统计
    size_t num_prefill_tokens_ = 0;
    size_t num_decode_tokens_ = 0;
};
```

### 2.3 执行流程

```cpp
// LLMWorkerImpl::step 实现
std::optional<ForwardOutput> LLMWorkerImpl::step(const ForwardInput& inputs) {
    // 1. 构建前向参数
    ForwardParams params;
    params.batch_forward_type = inputs.batch_forward_type();
    params.num_decoding_tokens = inputs.num_decoding_tokens();
    
    // 2. 执行前向计算
    auto output = model_->forward(params, model_args_, inputs, cp_size_);
    
    // 3. 执行采样
    auto sample_output = sampler_->sample(output);
    
    // 4. 返回结果
    return sample_output;
}

// prepare_inputs_impl 实现
ForwardInput LLMWorkerImpl::prepare_inputs_impl(Batch& batch) {
    // 根据批次类型选择准备策略
    switch (batch.batch_forward_type()) {
        case BatchForwardType::PREFILL:
            return prepare_prefill_input(batch);
        case BatchForwardType::DECODE:
            return prepare_decode_input(batch);
        case BatchForwardType::MIXED:
            return prepare_mixed_input(batch);
        default:
            XLLM_LOG(FATAL) << "Unknown batch forward type";
    }
}

// prepare_decode_input 实现
ForwardInput LLMWorkerImpl::prepare_decode_input(Batch& batch) {
    ForwardInput input;
    
    // 1. 获取最后一个token的embedding
    std::vector<torch::Tensor> embeddings;
    for (auto* seq : batch) {
        int last_token = seq->get_last_token_id();
        auto embed = embedding_layer_->forward(last_token);
        embeddings.push_back(embed);
    }
    input.set_input_embeddings(torch::stack(embeddings));
    
    // 2. 准备position_ids
    std::vector<int64_t> position_ids;
    for (auto* seq : batch) {
        position_ids.push_back(seq->num_tokens() - 1);
    }
    input.set_position_ids(torch::tensor(position_ids, torch::kLong));
    
    // 3. 准备attention_mask (decode阶段通常是因果 mask)
    // 对于纯decode，attention_mask可以省略（使用causal mask）
    
    return input;
}
```

---

## 3. VLMWorkerImpl 详解

### 3.1 视觉语言模型处理

```cpp
// 代码位置: xllm/core/runtime/vlm_worker_impl.h

/**
 * @class VLMWorkerImpl
 * @brief 视觉语言模型Worker实现
 */
class VLMWorkerImpl : public WorkerImpl {
 public:
    VLMWorkerImpl(const ParallelArgs& parallel_args,
                 const torch::Device& device,
                 const runtime::Options& options);
    
    // ========== WorkerImpl 实现 ==========
    
    ForwardInput prepare_inputs_impl(Batch& batch) override;
    std::optional<ForwardOutput> forward_impl(
        const ForwardInput& inputs) override;

 private:
    // ========== 多模态处理 ==========
    
    /**
     * @brief 预处理图像
     */
    torch::Tensor preprocess_images(const std::vector<ImageData>& images);
    
    /**
     * @brief 执行视觉编码
     */
    torch::Tensor encode_vision(const torch::Tensor& images);
    
    /**
     * @brief 融合视觉和文本特征
     */
    ForwardInput fuse_multimodal(const torch::Tensor& vision_hidden,
                                const ForwardInput& text_input);

    // ========== 组件 ==========
    
    std::unique_ptr<VisionEncoder> vision_encoder_;
    std::unique_ptr<MultimodalProjector> projector_;
    
    // 图像处理配置
    VLMConfig vlm_config_;
};
```

### 3.2 多模态输入流程

```cpp
ForwardInput VLMWorkerImpl::prepare_inputs_impl(Batch& batch) {
    // 1. 分离文本和多模态数据
    auto text_seqs = extract_text_sequences(batch);
    auto mm_data = extract_multimodal_data(batch);
    
    // 2. 处理文本输入 (与LLM相同)
    auto text_input = LLMWorkerImpl::prepare_inputs_impl(text_seqs);
    
    // 3. 处理图像
    if (!mm_data.empty()) {
        // 视觉编码
        auto vision_output = vision_encoder_->forward(mm_data.images);
        
        // 投影到语言模型空间
        auto vision_hidden = projector_->forward(vision_output);
        
        // 融合
        return fuse_multimodal(vision_hidden, text_input);
    }
    
    return text_input;
}
```

---

## 4. 生命周期管理

### 4.1 状态机

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Worker Lifecycle State Machine                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌────────────────┐                                                       │
│    │  UNINITIALIZED │                                                       │
│    └───────┬────────┘                                                       │
│            │ init_model()                                                   │
│            ▼                                                                │
│    ┌────────────────┐     ┌────────────────┐                              │
│    │ INITIALIZING   │────▶│     READY      │                              │
│    └────────────────┘     └───────┬────────┘                              │
│                                   │                                         │
│              ┌─────────────────────┼─────────────────────┐                   │
│              │                     │                     │                   │
│              ▼                     ▼                     ▼                   │
│    ┌────────────────┐     ┌────────────────┐     ┌────────────────┐       │
│    │    RUNNING     │     │    SLEEPING    │     │     ERROR      │       │
│    │                │     │                │     │                │       │
│    │ step()正在执行  │     │ 模型卸载/休眠   │     │ 异常状态       │       │
│    └────────────────┘     └───────┬────────┘     └───────┬────────┘       │
│              │                     │                     │                   │
│              │                     │ wakeup()            │ reset()          │
│              │                     ▼                     │                   │
│              │            ┌────────────────┐              │                   │
│              │            │   WAKING_UP   │              │                   │
│              │            └───────┬────────┘              │                   │
│              │                    │                      │                   │
│              └────────────────────┴──────────────────────┘                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 睡眠/唤醒实现

```cpp
// sleep 实现
bool WorkerImpl::sleep_impl(MasterStatus status) {
    XLLM_LOG(INFO) << "Worker entering sleep mode, status=" << static_cast<int>(status);
    
    // 1. 保存当前状态
    save_checkpoint();
    
    // 2. 释放模型权重 (可选)
    if (status == MasterStatus::DEEP_SLEEP) {
        // 深度休眠：完全释放权重
        model_->unload_weights();
    }
    
    // 3. 释放KV Cache (深度休眠)
    if (status == MasterStatus::DEEP_SLEEP) {
        kv_cache_manager_->free_all();
    }
    
    // 4. 释放激活内存
    executor_->free_activation_memory();
    
    // 5. 释放GPU资源
    if (status == MasterStatus::DEEP_SLEEP) {
        executor_->release_device_memory();
    }
    
    state_ = WorkerState::SLEEPING;
    XLLM_LOG(INFO) << "Worker entered sleep mode";
    return true;
}

// wakeup 实现
bool WorkerImpl::wakeup_impl(const WakeupOptions& options) {
    XLLM_LOG(INFO) << "Worker waking up from sleep mode";
    
    state_ = WorkerState::WAKING_UP;
    
    // 1. 重新分配设备内存
    executor_->allocate_device_memory();
    
    // 2. 重新加载模型权重 (如果是深度休眠)
    if (options.reload_weights) {
        model_->load_weights(options.weights_path);
    }
    
    // 3. 重新分配KV Cache
    if (options.restore_kv_cache) {
        kv_cache_manager_->allocate(options.kv_cache_shape);
    }
    
    // 4. 恢复执行器状态
    executor_->restore_state(options.checkpoint_path);
    
    state_ = WorkerState::READY;
    XLLM_LOG(INFO) << "Worker woke up successfully";
    return true;
}
```

---

## 5. Executor 架构

### 5.1 Executor 层次

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Executor Hierarchy                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────────┐                                 │
│                         │   Executor      │                                 │
│                         │  (抽象基类)      │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                           │
│                                  ▼                                           │
│                         ┌─────────────────┐                                 │
│                         │BaseExecutorImpl │                                 │
│                         │  (基础实现)      │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                           │
│     ┌───────────────────────────┼───────────────────────────┐              │
│     │                           │                           │              │
│     ▼                           ▼                           ▼              │
│  ┌─────────────┐         ┌─────────────┐           ┌─────────────┐         │
│  │   CUDA      │         │    NPU      │           │    MLU      │         │
│  │   Graph     │         │    Graph    │           │    Graph    │         │
│  │ Executor    │         │ Executor    │           │ Executor    │         │
│  │ Impl        │         │ Impl        │           │ Impl        │         │
│  └─────────────┘         └─────────────┘           └─────────────┘         │
│       │                        │                          │                 │
│       └────────────────────────┼──────────────────────────┘                 │
│                                ▼                                             │
│                    ┌─────────────────────────┐                             │
│                    │  GraphExecutorImpl      │                             │
│                    │  (通用图执行器)          │                             │
│                    │                         │                             │
│                    │  • CUDA Graph捕获       │                             │
│                    │  • 图重放优化            │                             │
│                    │  • 动态shape处理        │                             │
│                    └─────────────────────────┘                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 CUDA Graph 实现

```cpp
// 代码位置: xllm/core/runtime/cuda_graph_executor_impl.h

/**
 * @class CUDAGraphExecutorImpl
 * @brief CUDA Graph执行器 - 使用CUDA Graph优化执行
 */
class CUDAGraphExecutorImpl : public BaseExecutorImpl {
 public:
    CUDAGraphExecutorImpl(const torch::Device& device,
                         const ExecutorOptions& options);
    
    // ========== ExecutorImpl 实现 ==========
    
    void capture() override;
    void replay() override;
    bool can_capture(const ForwardParams& params) override;
    
 private:
    // ========== Graph管理 ==========
    
    /**
     * @brief 捕获当前执行的计算图
     */
    void capture_graph();
    
    /**
     * @brief 重放捕获的图
     */
    void replay_graph();
    
    /**
     * @brief 更新图参数
     */
    void update_graph();
    
    // ========== 状态 ==========
    
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    
    // 捕获状态
    bool is_captured_ = false;
    bool needs_update_ = false;
    
    // 输入输出映射
    std::vector<void*> input_ptrs_;
    std::vector<void*> output_ptrs_;
    
    // 参数缓存
    ForwardParams cached_params_;
};
```

---

## 6. WorkerClient (分布式)

### 6.1 远程调用封装

```cpp
// 代码位置: xllm/core/runtime/worker_client.h

/**
 * @class WorkerClient
 * @brief Worker客户端 - 封装远程Worker调用
 * 
 * @design 统一本地和远程Worker调用接口
 */
class WorkerClient {
 public:
    WorkerClient(const std::string& address,
                std::shared_ptr<brpc::Channel> channel);
    
    // ========== 生命周期 ==========
    
    folly::SemiFuture<bool> init_model_async(...);
    folly::SemiFuture<bool> sleep_async(...);
    folly::SemiFuture<bool> wakeup_async(...);
    
    // ========== 推理执行 ==========
    
    /**
     * @brief 异步执行一步推理
     */
    folly::SemiFuture<std::optional<ForwardOutput>> step_async(
        const ForwardInput& inputs);
    
    /**
     * @brief 获取最后一步结果
     */
    folly::SemiFuture<std::optional<ForwardOutput>> get_last_step_result_async();

    // ========== KV Cache管理 ==========
    
    folly::SemiFuture<bool> allocate_kv_cache_async(...);
    folly::SemiFuture<bool> pull_kv_blocks_async(...);

 private:
    std::string address_;
    std::shared_ptr<brpc::Channel> channel_;
    std::unique_ptr<WorkerService_Stub> stub_;
};
```

### 6.2 使用示例

```cpp
// 创建远程Worker客户端
auto channel = std::make_shared<brpc::Channel>();
channel->Init("127.0.0.1:8000", &options);

auto worker_client = std::make_shared<WorkerClient>("127.0.0.1:8000", channel);

// 异步调用
auto future = worker_client->step_async(forward_input);
future.then([](auto&& result) {
    if (result.hasValue()) {
        process_output(result.value());
    }
});
```

---

## 7. 并行状态管理

### 7.1 ParallelState

```cpp
// 代码位置: xllm/core/framework/parallel_state/parallel_state.h

/**
 * @class ParallelState
 * @brief 并行状态管理 - 管理TP/PP/DP/CP/EP并行配置
 */
class ParallelState {
 public:
    ParallelState(int tp_size, int pp_size, int dp_size, 
                  int cp_size, int ep_size);
    
    // ========== 秩查询 ==========
    
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    
    // ========== TP相关 ==========
    
    int tp_rank() const { return tp_rank_; }
    int tp_size() const { return tp_size_; }
    int get_tensor_model_parallel_rank() const { return tp_rank_; }
    bool is_first_tensor_parallel_rank() const { return tp_rank_ == 0; }
    bool is_last_tensor_parallel_rank() const { return tp_rank_ == tp_size_ - 1; }
    
    // ========== PP相关 ==========
    
    int pp_rank() const { return pp_rank_; }
    int pp_size() const { return pp_size_; }
    bool is_first_pipeline_stage() const { return pp_rank_ == 0; }
    bool is_last_pipeline_stage() const { return pp_rank_ == pp_size_ - 1; }
    
    // ========== DP相关 ==========
    
    int dp_rank() const { return dp_rank_; }
    int dp_size() const { return dp_size_; }
    
    // ========== CP相关 ==========
    
    int cp_rank() const { return cp_rank_; }
    int cp_size() const { return cp_size_; }
    
    // ========== EP相关 ==========
    
    int ep_rank() const { return ep_rank_; }
    int ep_size() const { return ep_size_; }

 private:
    int rank_;
    int world_size_;
    
    int tp_rank_ = 0, tp_size_ = 1;
    int pp_rank_ = 0, pp_size_ = 1;
    int dp_rank_ = 0, dp_size_ = 1;
    int cp_rank_ = 0, cp_size_ = 1;
    int ep_rank_ = 0, ep_size_ = 1;
    
    // 进程组
    std::unique_ptr<ProcessGroup> tp_group_;
    std::unique_ptr<ProcessGroup> pp_group_;
    std::unique_ptr<ProcessGroup> dp_group_;
    std::unique_ptr<ProcessGroup> cp_group_;
    std::unique_ptr<ProcessGroup> ep_group_;
};
```

---

## 8. AI验收标准

### 8.1 接口验证

```yaml
ai_verification:
  worker:
    - name: "接口完整性"
      check: |
        1. Worker公共接口必须包含所有生命周期方法
        2. 所有方法必须有实现或标记为纯虚
        3. PIMPL模式正确使用
        
    - name: "线程安全"
      check: |
        1. 公共方法使用适当的同步机制
        2. 异步方法返回 folly::SemiFuture
        
    - name: "错误处理"
      check: |
        1. 所有可能失败的操作返回bool或使用Try
        2. 错误情况有日志记录
```

### 8.2 集成验证

```cpp
// 集成测试用例
TEST(WorkerIntegration, EndToEndInference) {
    // 1. 创建Worker
    Worker worker(parallel_args, device, options, WorkerType::LLM);
    
    // 2. 初始化
    ASSERT_TRUE(worker.init_model(model_path, /*seed=*/42, MasterStatus::WAKEUP));
    
    // 3. 分配KV Cache
    auto [num_blocks, block_size] = worker.estimate_kv_cache_capacity();
    ASSERT_TRUE(worker.allocate_kv_cache(KVCacheShape{...}));
    
    // 4. 准备输入
    auto batch = create_test_batch();
    auto input = worker.prepare_inputs(batch);
    
    // 5. 执行推理
    auto output = worker.step(input);
    ASSERT_TRUE(output.has_value());
    
    // 6. 验证输出
    EXPECT_EQ(output->get_tokens().size(), batch.size());
}
```

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [领域模型](./04_DOMAIN_MODEL.md) | DDD领域划分 |
| [Scheduler设计](./05_SCHEDULER_DESIGN.md) | 调度器实现详解 |
| [Batch设计](./06_BATCH_DESIGN.md) | Batch实现详解 |
| [KV Cache设计](./07_KV_CACHE_DESIGN.md) | KV Cache管理 |
| [API设计](./09_API_DESIGN.md) | Worker API |
