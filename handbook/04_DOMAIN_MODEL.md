# xLLM 领域模型设计

## 文档信息

```yaml
---
document_id: DESIGN-001
version: 1.0.0
category: domain_model
owner: xllm-team
verification_level: BOTH

ai_acceptance_criteria:
  - 所有领域对象包含完整的状态定义
  - 聚合边界清晰明确
  - 实体标识符定义明确
  - 领域事件定义完整
---
```

---

## 1. 领域概览

### 1.1 限界上下文划分

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    xLLM Bounded Contexts & Subdomains                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Scheduling Context                                   │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │                     Scheduling Subdomain                         │  │ │
│  │  │                                                                  │  │ │
│  │  │  实体:          聚合:              值对象:                       │  │ │
│  │  │  • Request      • RequestAggregate  • RequestId                   │  │ │
│  │  │  • Sequence     • SequenceGroupAgg  • SequenceStage                │  │ │
│  │  │  • Batch                            • SLOConfig                   │  │ │
│  │  │  • Scheduler                       • Urgency                     │  │ │
│  │  │                                                                  │  │ │
│  │  │  领域服务:      仓储:             领域事件:                       │  │ │
│  │  │  • SchedulingSvc • RequestRepository • RequestScheduled           │  │ │
│  │  │  • SLOTracker    • BatchRepository  • BatchFormed                 │  │ │
│  │  │                                                  • RequestFinished │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    │ 跨上下文通信 (通过接口)                    │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Inference Context                                   │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │                     Inference Subdomain                          │  │ │
│  │  │                                                                  │  │ │
│  │  │  实体:          聚合:              值对象:                       │  │ │
│  │  │  • Worker       • WorkerAggregate   • WorkerId                   │  │ │
│  │  │  • Block        • KVCacheAggregate  • BlockId                   │  │ │
│  │  │  • Model                           • KVCacheShape              │  │ │
│  │  │  • Executor                         • ParallelConfig            │  │ │
│  │  │                                                                  │  │ │
│  │  │  领域服务:      仓储:             领域事件:                       │  │ │
│  │  │  • InferenceSvc  • BlockRepository  • BlockAllocated             │  │ │
│  │  │  • CacheManager  • ModelRegistry    • ModelLoaded                │  │ │
│  │  │                                                  • WorkerReady   │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    │ 跨上下文通信 (通过接口)                    │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Service Context                                     │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │                     Service Subdomain                             │  │ │
│  │  │                                                                  │  │ │
│  │  │  实体:          聚合:              值对象:                       │  │ │
│  │  │  • Session      • SessionAggregate  • SessionId                   │  │ │
│  │  │  • APIKey                            • APIKeyConfig              │  │ │
│  │  │                                                                  │  │ │
│  │  │  领域服务:      仓储:             领域事件:                       │  │ │
│  │  │  • AuthService   • SessionRepository • SessionCreated             │  │ │
│  │  │  • RateLimiter   • APIKeyRepository  • RateLimitExceeded        │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 调度上下文 (Scheduling Context)

### 2.1 RequestAggregate (请求聚合)

**聚合根**: `Request`

```cpp
// 代码位置: xllm/core/framework/request/request.h

/**
 * @class Request
 * @brief Request聚合根 - 代表用户发起的一次完整推理请求
 * 
 * @invariant Request必须包含至少一个Sequence
 * @invariant Request的SLO配置在创建后不可修改
 * @invariant Request在完成后不能被修改
 */
class Request : public RequestBase {
 public:
    // ========== 构造函数 ==========
    Request(const std::string& request_id,
            const std::string& x_request_id,
            const std::string& x_request_time,
            const RequestState& state,
            const std::string& service_request_id = "",
            const std::string& source_xservice_addr = "");
    
    // ========== 身份标识 ==========
    const std::string& request_id() const { return request_id_; }
    
    // ========== 聚合关系 ==========
    /**
     * @brief 获取请求包含的所有序列
     * @details 返回值是唯一真相来源
     */
    std::vector<std::unique_ptr<Sequence>>& sequences() {
        return sequences_group_->sequences();
    }
    
    SequencesGroup* sequence_group() { return sequences_group_.get(); }
    
    // ========== 业务方法 ==========
    
    /**
     * @brief 扩展序列以支持Beam Search
     * @param share_prefix 是否共享前缀
     */
    bool expand_sequences(bool share_prefix = true);
    
    /**
     * @brief 标记请求已取消
     */
    void set_cancel() { cancelled_.store(true, std::memory_order_relaxed); }
    
    /**
     * @brief 检查请求是否已完成
     */
    bool finished() const;
    
    /**
     * @brief 生成输出
     * @param tokenizer 分词器
     * @param thread_pool 线程池（可选）
     */
    RequestOutput generate_output(const Tokenizer& tokenizer,
                                 ThreadPool* thread_pool = nullptr);
    
    // ========== SLO管理 ==========
    
    /**
     * @brief 获取TTFT SLO (Time To First Token)
     */
    int32_t ttft_slo_ms() const {
        return state_.scheduler_param.ttft_slo_ms;
    }
    
    /**
     * @brief 获取TPOT SLO (Time Per Output Token)
     */
    int32_t tpot_slo_ms() const {
        return state_.scheduler_param.tpot_slo_ms;
    }
    
    /**
     * @brief 获取TTLT SLO (Time To Last Token)
     */
    int32_t ttlt_slo_ms() const {
        return state_.scheduler_param.ttlt_slo_ms;
    }
    
    /**
     * @brief 计算剩余时间
     */
    int32_t get_remaining_time() const {
        return get_deadline_ms() - get_elapsed_time_ms();
    }
    
    // ========== 优先级管理 ==========
    
    /**
     * @brief 设置请求紧急度
     */
    void set_urgency(Urgency urgency) { urgency_ = urgency; }
    
    Urgency urgency() const { return urgency_; }
    
    void set_starved(bool starved) { starved_ = starved; }
    bool is_starved() const { return starved_; }

 private:
    // ========== 聚合内部状态 ==========
    
    RequestState state_;  // 请求状态
    std::unique_ptr<SequencesGroup> sequences_group_;  // 序列组
    std::atomic<bool> cancelled_{false};  // 取消标志
    
    // SLO追踪
    int32_t elapsed_time_ms_ = 0;
    int32_t deadline_ms_ = 0;
    
    // 优先级状态
    Urgency urgency_ = Urgency::NORMAL;
    bool starved_ = false;
    
    // 前缀缓存统计
    size_t num_prefix_cache_tokens_ = 0;
    
    // 创建序列组（私有）
    void create_sequences_group();
};
```

### 2.2 Sequence (序列)

**实体**: `Sequence`

```cpp
// 代码位置: xllm/core/framework/request/sequence.h

/**
 * @class Sequence
 * @brief 序列 - 代表一次生成任务对应的token序列
 * 
 * @identity 由 (request_id, sequence_index) 唯一标识
 * @invariant token_ids_ 和 position_ids_ 长度必须一致
 */
class Sequence {
 public:
    // ========== 构造函数 ==========
    Sequence(int64_t request_id,
             int64_t parent_sequence_id,
             int64_t sample_index,
             size_t max_tokens,
             const SamplingParams& sampling_params,
             const std::vector<int>& token_ids);
    
    // ========== 身份标识 ==========
    int64_t request_id() const { return request_id_; }
    int64_t sample_index() const { return sample_index_; }
    
    // ========== Token操作 ==========
    
    /**
     * @brief 添加单个token
     */
    void append_token(int token_id, const Token& token);
    
    /**
     * @brief 批量添加tokens
     */
    void append_tokens(const std::vector<int>& token_ids,
                      const std::vector<Token>& tokens);
    
    /**
     * @brief 获取所有token IDs
     */
    const std::vector<int>& get_token_ids() const { return token_ids_; }
    
    // ========== 状态查询 ==========
    
    /**
     * @brief 获取当前序列长度
     */
    size_t num_tokens() const { return token_ids_.size(); }
    
    /**
     * @brief 获取prompt长度
     */
    size_t num_prompt_tokens() const { return num_prompt_tokens_; }
    
    /**
     * @brief 获取decode token数量
     */
    size_t num_decode_tokens() const { 
        return num_tokens() - num_prompt_tokens_; 
    }
    
    /**
     * @brief 检查是否在预填充阶段
     */
    bool is_prefill_stage() const {
        return stage_ == SequenceStage::PREFILL;
    }
    
    /**
     * @brief 检查是否完成
     */
    bool is_finished() const { return finished_; }
    
    SequenceStage stage() const { return stage_; }
    
    // ========== 时间统计 ==========
    
    /**
     * @brief 记录TTFT时间
     */
    void record_first_token_time();
    
    /**
     * @brief 获取TTFT延迟（秒）
     */
    double time_to_first_token_latency_seconds() const;
    
    /**
     * @brief 获取首token时间点
     */
    absl::Time first_token_time() const { return first_token_time_; }
    
 private:
    // ========== 身份 ==========
    int64_t request_id_;
    int64_t parent_sequence_id_;
    int64_t sample_index_;
    
    // ========== Token数据 ==========
    std::vector<int> token_ids_;
    size_t num_prompt_tokens_;
    
    // ========== 状态 ==========
    SequenceStage stage_ = SequenceStage::PREFILL;
    std::atomic<bool> finished_{false};
    
    // ========== 时间统计 ==========
    absl::Time created_time_;
    absl::Time first_token_time_;
    absl::Time last_token_time_;
};
```

### 2.3 SequencesGroup (序列组)

**聚合**: `SequencesGroup`

```cpp
// 代码位置: xllm/core/framework/request/sequences_group.h

/**
 * @class SequencesGroup
 * @brief 序列组 - 管理一组相关的序列（如Beam Search候选）
 * 
 * @aggregation 一个Request包含一个SequencesGroup
 * @invariant 至少包含一个Sequence
 * @invariant 所有Sequence共享相同的prompt
 */
class SequencesGroup {
 public:
    // ========== 构造函数 ==========
    explicit SequencesGroup(int64_t request_id);
    
    // ========== 序列管理 ==========
    
    /**
     * @brief 获取所有序列
     */
    std::vector<Sequence*>& sequences() { return sequences_; }
    const std::vector<Sequence*>& sequences() const { return sequences_; }
    
    /**
     * @brief 添加候选序列（Beam Search）
     */
    void add_candidate(const std::vector<Sequence*>& candidates);
    
    /**
     * @brief 获取最佳序列
     */
    Sequence* best_sequence() const;
    
    // ========== 状态查询 ==========
    
    /**
     * @brief 检查是否所有序列都完成
     */
    bool all_finished() const;
    
    /**
     * @brief 获取完成的序列数
     */
    size_t num_finished() const;
    
    /**
     * @brief 获取最长的序列
     */
    size_t max_num_tokens() const;
    
    // ========== 分块预填充 ==========
    
    /**
     * @brief 是否处于分块预填充阶段
     */
    bool is_chunked_prefill_stage() const {
        return chunked_prefill_;
    }
    
    void set_chunked_prefill_stage(bool enable) {
        chunked_prefill_ = enable;
    }

 private:
    int64_t request_id_;
    std::vector<Sequence*> sequences_;
    bool chunked_prefill_ = false;
};
```

### 2.4 Batch (批处理)

**聚合**: `Batch`

```cpp
// 代码位置: xllm/core/framework/batch/batch.h

/**
 * @class Batch
 * @brief 批处理 - 代表一次推理调度的序列集合
 * 
 * @purpose 批处理是将多个Request/Sequence打包后的一次执行单元
 * @lifetime 单次Scheduler::step()周期
 * @identity 由全局递增的batch_id标识
 */
class Batch {
 public:
    // ========== 构造函数 ==========
    Batch() = default;
    Batch(Sequence* sequence);
    Batch(const std::vector<Sequence*>& sequences);
    
    // ========== 序列管理 ==========
    
    /**
     * @brief 添加单个序列
     * @param sequence 要添加的序列
     * @param allowed_max_token 该序列允许的最大token数
     */
    void add(Sequence* sequence,
             uint32_t allowed_max_token = std::numeric_limits<uint32_t>::max());
    
    /**
     * @brief 批量添加序列
     */
    void add(const std::vector<Sequence*>& sequences);
    
    /**
     * @brief 获取批次大小
     */
    size_t size() const { return sequences_.size(); }
    bool empty() const { return sequences_.empty() && sequence_groups_.empty(); }
    
    /**
     * @brief 随机访问序列
     */
    Sequence* operator[](size_t i) { return sequences_[i]; }
    
    // ========== 输入准备 ==========
    
    /**
     * @brief 准备前向输入
     * @param num_decoding_tokens 允许的decode token数量
     * @param min_decoding_batch_size 最小的decode批大小
     * @param args 模型参数
     * @param cp_size 上下文并行度
     */
    ForwardInput prepare_forward_input(uint32_t num_decoding_tokens,
                                      uint32_t min_decoding_batch_size,
                                      const ModelArgs& args,
                                      int32_t cp_size = 1);
    
    // ========== 输出处理 ==========
    
    /**
     * @brief 处理采样输出
     * @param replace_fake_token 是否替换假token（用于流水线并行）
     */
    void process_sample_output(const SampleOutput& sample_output,
                             bool replace_fake_token,
                             bool force_requested_beam_result_size = false);
    
    /**
     * @brief 处理Beam Search输出
     */
    void process_beam_search_output(const RawForwardOutput& raw_output,
                                   bool replace_fake_token);
    
    // ========== 批处理优化 ==========
    
    /**
     * @brief 计算序列交换索引（用于优化KV Cache布局）
     */
    std::unordered_map<uint32_t, uint32_t> cal_seq_exchange_index(
        std::vector<uint32_t>& kv_cache_tokens_num);
    
    /**
     * @brief 更新前向计算类型
     */
    void update_forward_type(Sequence* sequence);
    void refresh_forward_type();
    
    // ========== 标识 ==========
    void set_batch_id() {
        if (batch_id_ == UNINITIALIZED_BATCH_ID) {
            batch_id_ = batch_counter_++;
        }
    }
    uint64_t batch_id() const { return batch_id_; }

 private:
    std::vector<Sequence*> sequences_;
    std::vector<SequencesGroup*> sequence_groups_;
    std::vector<BlockTransferInfo> swap_block_transfer_infos_;
    std::vector<uint32_t> allowed_max_tokens_;
    std::vector<torch::Tensor> input_embeddings_vec_;
    std::vector<MMData> mm_data_vec_;
    std::vector<OutputTarget> output_targets_;
    BatchForwardType batch_forward_type_;
    uint64_t batch_id_ = UNINITIALIZED_BATCH_ID;
    
    static std::atomic<uint64_t> batch_counter_;
};
```

---

## 3. 推理上下文 (Inference Context)

### 3.1 WorkerAggregate (Worker聚合)

**聚合根**: `Worker`

```cpp
// 代码位置: xllm/core/runtime/worker.h

/**
 * @class Worker
 * @brief Worker聚合根 - 代表一个执行推理的工作节点
 * 
 * @identity 由 (device_id, worker_type) 唯一标识
 * @invariant Worker在生命周期内保持单一职责
 * @invariant Worker与一个物理设备绑定
 */
class Worker {
 public:
    // ========== 构造函数 ==========
    Worker(const ParallelArgs& parallel_args,
          const torch::Device& device,
          const runtime::Options& options,
          WorkerType worker_type);
    
    ~Worker();
    
    // ========== 生命周期 ==========
    
    /**
     * @brief 初始化模型
     * @param model_weights_path 权重路径
     * @param random_seed 随机种子
     * @param master_status 主节点状态
     * @return 初始化是否成功
     */
    bool init_model(const std::string& model_weights_path,
                   int32_t random_seed,
                   MasterStatus master_status);
    
    /**
     * @brief 进入深度休眠（RL场景）
     */
    bool sleep(MasterStatus master_status);
    
    /**
     * @brief 从休眠唤醒
     */
    bool wakeup(const WakeupOptions& options);
    
    /**
     * @brief 更新权重（热更新）
     */
    bool update_weights(const std::string& weights_path);
    
    // ========== 推理执行 ==========
    
    /**
     * @brief 准备前向输入
     */
    ForwardInput prepare_inputs(Batch& batch);
    
    /**
     * @brief 执行单步推理
     */
    std::optional<ForwardOutput> step(const ForwardInput& inputs);
    
    /**
     * @brief 异步执行单步推理
     */
    folly::SemiFuture<std::optional<ForwardOutput>> step_async(
        const ForwardInput& inputs);
    
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
    
    // ========== 集群管理 ==========
    
    /**
     * @brief 链接远程集群
     */
    bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                     const std::vector<std::string>& addrs,
                     const std::vector<uint16_t>& ports);
    
    /**
     * @brief 获取缓存信息
     */
    void get_cache_info(uint64_t& cluster_id, std::string& addr, uint16_t& port);
    
    // ========== Profiling ==========
    
    bool start_profile();
    bool stop_profile();
    
    folly::SemiFuture<bool> start_profile_async();
    folly::SemiFuture<bool> stop_profile_async();

 private:
    WorkerImpl* impl_ = nullptr;
    ThreadPool threadpool_{/*num_threads=*/1, /*cpu_binding=*/false, 
                          /*pool_name=*/"Worker.async"};
};
```

### 3.2 KVCacheAggregate (KV Cache聚合)

**聚合根**: `KVCache`

```cpp
// 代码位置: xllm/core/framework/kv_cache/kv_cache.h

/**
 * @class KVCache
 * @brief KV Cache聚合根 - 管理单个注意力层的Key-Value缓存
 * 
 * @purpose 缓存自注意力计算中的K和V张量，避免重复计算
 * @identity 由 (layer_id, block_id) 标识
 */
class KVCache final {
 public:
    // ========== 构造函数 ==========
    KVCache();
    explicit KVCache(const KVCacheTensors& tensors);
    explicit KVCache(const IndexedKVCacheTensors& tensors);
    explicit KVCache(const LinearAttentionKVCacheTensors& tensors);
    explicit KVCache(const QuantizedKVCacheTensors& tensors);
    
    /**
     * @brief 创建新的KV Cache
     */
    KVCache(const KVCacheShape& kv_cache_shape,
           const KVCacheCreateOptions& create_options,
           int64_t layer_id);
    
    // ========== 缓存访问 ==========
    
    /**
     * @brief 获取Key缓存张量
     */
    torch::Tensor get_k_cache() const;
    
    /**
     * @brief 获取Value缓存张量
     */
    torch::Tensor get_v_cache() const;
    
    /**
     * @brief 获取索引缓存（用于索引注意力）
     */
    torch::Tensor get_index_cache() const;
    
    // ========== 量化缓存 ==========
    
    std::optional<torch::Tensor> get_k_cache_scale() const;
    std::optional<torch::Tensor> get_v_cache_scale() const;
    
    // ========== 块交换 ==========
    
    /**
     * @brief 与另一个KVCache交换Block
     */
    void swap_blocks(torch::Tensor& src_tensor, torch::Tensor& dst_tensor);
    
    // ========== 状态查询 ==========
    
    /**
     * @brief 检查缓存是否为空
     */
    bool empty() const;
    
    /**
     * @brief 获取缓存形状
     */
    std::vector<std::vector<int64_t>> get_shapes();

 private:
    std::unique_ptr<KVCacheImpl> impl_;
};

/**
 * @brief 批量分配KV Cache
 */
void allocate_kv_caches(std::vector<KVCache>& kv_caches,
                        const KVCacheShape& kv_cache_shape,
                        const KVCacheCreateOptions& create_options);
```

### 3.3 Block (Block实体)

**实体**: `Block`

```cpp
// 代码位置: xllm/core/framework/block/block.h

/**
 * @class Block
 * @brief Block - KV Cache的最小分配单元
 * 
 * @identity 由全局唯一的block_id标识
 * @invariant Block大小固定，由KVCacheShape决定
 * @invariant Block在同一时刻只能被一个Sequence使用
 */
class Block {
 public:
    // ========== 类型定义 ==========
    enum class BlockType {
        UNKNOWN = 0,
        PREFILL = 1,    // 预填充Block
        DECODE = 2,     // 解码Block
        PREFILL_DECODE = 3,  // 混合Block
    };
    
    // ========== 构造函数 ==========
    Block(int64_t block_id, 
          const std::vector<int64_t>& shape,
          torch::Device device);
    
    // ========== 身份 ==========
    int64_t block_id() const { return block_id_; }
    
    // ========== Block操作 ==========
    
    /**
     * @brief 获取Block的KV Cache张量
     */
    std::vector<torch::Tensor>& get_tensors() { return tensors_; }
    const std::vector<torch::Tensor>& get_tensors() const { return tensors_; }
    
    /**
     * @brief 获取单个张量
     */
    torch::Tensor get_tensor(size_t idx) const { 
        return tensors_.at(idx); 
    }
    
    // ========== 状态管理 ==========
    
    /**
     * @brief 获取Block类型
     */
    BlockType block_type() const { return block_type_; }
    
    /**
     * @brief 设置Block类型
     */
    void set_block_type(BlockType type) { block_type_ = type; }
    
    /**
     * @brief 获取已使用的token数
     */
    int num_tokens_used() const { return num_tokens_used_; }
    
    /**
     * @brief 设置已使用的token数
     */
    void set_num_tokens_used(int num_tokens) { num_tokens_used_ = num_tokens; }
    
    /**
     * @brief 获取该Block能存储的最大token数
     */
    int max_num_tokens() const { return max_num_tokens_; }
    
    /**
     * @brief 标记为已分配
     */
    void mark_allocated(int64_t owner_id) {
        allocated_.store(true, std::memory_order_relaxed);
        owner_id_ = owner_id;
    }
    
    /**
     * @brief 标记为已释放
     */
    void mark_free() {
        allocated_.store(false, std::memory_order_relaxed);
        owner_id_ = -1;
    }
    
    /**
     * @brief 检查是否已分配
     */
    bool is_allocated() const { return allocated_.load(std::memory_order_relaxed); }
    
    /**
     * @brief 获取所有者ID
     */
    int64_t owner_id() const { return owner_id_; }
    
    /**
     * @brief 引用计数管理
     */
    void add_ref() { ref_count_++; }
    void release_ref() { 
        if (--ref_count_ == 0) {
            mark_free();
        }
    }
    int ref_count() const { return ref_count_; }

 private:
    int64_t block_id_;
    std::vector<torch::Tensor> tensors_;  // K/V张量对
    BlockType block_type_ = BlockType::UNKNOWN;
    int num_tokens_used_ = 0;
    int max_num_tokens_;
    std::atomic<bool> allocated_{false};
    int64_t owner_id_ = -1;
    int ref_count_ = 1;  // Prefix Cache引用计数
};
```

### 3.4 BlockManager (Block管理器)

**领域服务**: `BlockManager`

```cpp
// 代码位置: xllm/core/framework/block/block_manager.h

/**
 * @class BlockManager
 * @brief Block管理器 - 负责Block的分配和回收
 * 
 * @responsibility 
 *  - 管理Block池
 *  - 处理Block分配请求
 *  - 处理Block释放
 *  - 支持Block交换
 */
class BlockManager {
 public:
    // ========== 构造函数 ==========
    BlockManager(int64_t num_blocks,
                const KVCacheShape& shape,
                torch::Device device);
    
    // ========== Block管理 ==========
    
    /**
     * @brief 分配指定数量的Block
     * @param num_blocks 需要的Block数量
     * @return 分配的Block ID列表
     * @throws 如果Block不足
     */
    virtual std::vector<int64_t> allocate(int64_t num_blocks);
    
    /**
     * @brief 释放Block
     */
    virtual void free(const std::vector<int64_t>& block_ids);
    
    /**
     * @brief 释放某个Sequence的所有Block
     */
    virtual void free_sequence(int64_t sequence_id);
    
    /**
     * @brief 获取Block指针
     */
    virtual Block* get_block(int64_t block_id);
    
    /**
     * @brief 获取可用Block数
     */
    virtual int64_t get_num_available_blocks() const;
    
    // ========== 序列Block查询 ==========
    
    /**
     * @brief 获取Sequence占用的Block列表
     */
    virtual std::vector<int64_t> get_sequence_blocks(int64_t sequence_id) const;
    
    /**
     * @brief 检查Sequence是否有足够的Block
     */
    virtual bool can_allocate(int64_t sequence_id, 
                             int64_t num_needed_blocks) const;

 private:
    int64_t num_blocks_;
    KVCacheShape shape_;
    torch::Device device_;
    std::vector<std::unique_ptr<Block>> blocks_;
    std::unordered_set<int64_t> free_blocks_;  // 空闲Block集合
    std::unordered_map<int64_t, std::vector<int64_t>> sequence_to_blocks_;  // Sequence -> Blocks映射
};
```

---

## 4. 领域服务

### 4.1 SchedulerService

```cpp
// 代码位置: xllm/core/scheduler/scheduler.h

/**
 * @class Scheduler
 * @brief 调度服务 - 核心调度逻辑
 * 
 * @responsibility
 *  - 管理请求队列
 *  - 决定批处理组成
 *  - 触发请求执行
 */
class Scheduler : public SchedulerBase {
 public:
    virtual ~Scheduler() = default;
    
    // ========== 请求管理 ==========
    
    /**
     * @brief 添加新请求到调度队列
     * @return 是否添加成功
     */
    virtual bool add_request(std::shared_ptr<Request>& request) = 0;
    
    /**
     * @brief 获取等待中的请求数
     */
    virtual uint32_t get_waiting_requests_num() const = 0;
    
    // ========== 执行 ==========
    
    /**
     * @brief 执行调度步骤
     * @param timeout 超时时间
     */
    virtual void step(const absl::Duration& timeout) = 0;
    
    // ========== 指标 ==========
    
    /**
     * @brief 获取延迟指标
     */
    virtual void get_latency_metrics(std::vector<int64_t>& ttft,
                                    std::vector<int64_t>& tbt) = 0;
    
    /**
     * @brief 获取实例信息
     */
    virtual const InstanceInfo& get_instance_info() = 0;
    
    // ========== 前缀缓存 ==========
    
    /**
     * @brief 重置前缀缓存
     * @details 用于RL唤醒后清除废弃的KV Cache
     */
    virtual void reset_prefix_cache() {}
};

/**
 * @class SchedulerBase
 * @brief 调度器基类
 */
class SchedulerBase {
 public:
    virtual ~SchedulerBase() = default;
    
    virtual void step(const absl::Duration& timeout) = 0;
    virtual void generate() = 0;  // 离线生成
    
    virtual void incr_pending_requests(size_t count) {}
    virtual void decr_pending_requests() {}
    virtual size_t num_pending_requests() { return 0; }
};
```

### 4.2 InferenceService

```cpp
// 代码位置: xllm/core/distributed_runtime/llm_engine.h

/**
 * @class LLMEngine
 * @brief 推理引擎 - 执行模型推理的核心服务
 * 
 * @responsibility
 *  - 管理Worker池
 *  - 协调分布式推理
 *  - 处理KV Cache传输
 */
class LLMEngine : public Engine {
 public:
    LLMEngine(const runtime::Options& options,
             std::shared_ptr<DistManager> dist_manager = nullptr);
    
    virtual ~LLMEngine() = default;
    
    // ========== 推理执行 ==========
    
    /**
     * @brief 执行一步推理
     */
    ForwardOutput step(std::vector<Batch>& batch) override;
    
    // ========== 初始化 ==========
    
    /**
     * @brief 初始化引擎
     */
    bool init(MasterStatus master_status) override;
    
    // ========== KV Cache传输 ==========
    
    /**
     * @brief 从远程拉取KV Block
     */
    bool pull_kv_blocks(...) override;
    
    /**
     * @brief 传输KV Block到远程
     */
    std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(...) override;
    
    // ========== 生命周期 ==========
    
    bool sleep(MasterStatus master_status) override;
    bool wakeup(const WakeupOptions& options) override;
    
    // ========== 权重更新 ==========
    
    /**
     * @brief 热更新模型权重
     */
    bool update_weights(const std::string& weights_path) override;
    
    // ========== Profiling ==========
    
    bool start_profile() override;
    bool stop_profile() override;

 private:
    // Worker池
    std::vector<std::shared_ptr<WorkerClient>> worker_clients_;
    
    // 并行配置
    uint32_t dp_size_ = 1;
    uint32_t cp_size_ = 1;
    uint32_t dp_local_tp_size_;
    uint32_t dp_local_size_;
    
    // 分布式管理
    std::shared_ptr<DistManager> dist_manager_ = nullptr;
    
    // EPLB (Expert Parallel Load Balance)
    torch::Tensor expert_load_data_;
    std::unique_ptr<EplbManager> eplb_manager_ = nullptr;
};
```

---

## 5. 值对象

### 5.1 KVCacheShape

```cpp
// 代码位置: xllm/core/framework/kv_cache/kv_cache_shape.h

/**
 * @struct KVCacheShape
 * @brief KV Cache形状定义
 * 
 * @immutable 创建后不可修改
 */
struct KVCacheShape {
    int64_t num_heads;           // 注意力头数
    int64_t num_kv_heads;        // KV头数 (MHA/MQA/GQA)
    int64_t head_dim;            // 头维度
    int64_t block_seq_len;       // 每个Block的序列长度
    int64_t num_layers;          // 层数
    
    /**
     * @brief 计算单个Block的token数
     */
    int64_t num_tokens_per_block() const {
        return block_seq_len;
    }
    
    /**
     * @brief 计算单个Block的内存大小
     */
    int64_t bytes_per_block(const torch::ScalarType& dtype) const {
        // K + V 两个张量
        return 2 * num_kv_heads * head_dim * block_seq_len * 
               torch::elementSize(to_c10_dtype(dtype));
    }
};
```

### 5.2 ParallelArgs

```cpp
// 代码位置: xllm/core/common/types.h

/**
 * @struct ParallelArgs
 * @brief 并行配置参数
 * 
 * @immutable 创建后不可修改
 */
struct ParallelArgs {
    int tp_size = 1;              // 张量并行度
    int pp_size = 1;              // 流水线并行度
    int dp_size = 1;              // 数据并行度
    int cp_size = 1;              // 上下文并行度
    int ep_size = 1;              // 专家并行度
    
    /**
     * @brief 计算总Worker数
     */
    int total_workers() const {
        return tp_size * pp_size * dp_size * cp_size;
    }
    
    /**
     * @brief 验证配置有效性
     */
    bool is_valid() const {
        return tp_size >= 1 && pp_size >= 1 && dp_size >= 1 && 
               cp_size >= 1 && ep_size >= 1;
    }
};
```

### 5.3 SLOConfig

```cpp
// 代码位置: xllm/core/framework/request/request_state.h

/**
 * @struct SchedulerParam
 * @brief 调度参数，包含SLO配置
 * 
 * @immutable 部分字段可运行时修改
 */
struct SchedulerParam {
    // 优先级
    RequestPriority priority = RequestPriority::NORMAL;
    int priority_weight = 1;
    
    // SLO配置 (毫秒)
    int32_t ttft_slo_ms = 1000;       // Time To First Token
    int32_t tpot_slo_ms = 100;        // Time Per Output Token
    int32_t ttl_t_slo_ms = 30000;     // Time To Last Token
    
    // SLO权重
    int32_t ttft_priority_weight = 1;
    int32_t tpot_priority_weight = 1;
    int32_t ttl_t_priority_weight = 1;
    
    // 离线模式
    bool offline = false;
};
```

---

## 6. 领域事件

### 6.1 事件定义

```cpp
// 代码位置: xllm/core/common/event.h

namespace xllm {

/**
 * @enum DomainEventType
 * @brief 领域事件类型
 */
enum class DomainEventType {
    // Request相关事件
    REQUEST_ARRIVED,
    REQUEST_SCHEDULED,
    REQUEST_STARTED,
    REQUEST_FINISHED,
    REQUEST_CANCELLED,
    REQUEST_TIMEOUT,
    
    // Batch相关事件
    BATCH_FORMED,
    BATCH_EXECUTED,
    
    // Worker相关事件
    WORKER_READY,
    WORKER_SLEEPING,
    WORKER_WAKING,
    
    // Block相关事件
    BLOCK_ALLOCATED,
    BLOCK_RELEASED,
    BLOCK_SWAPPED,
    
    // 模型相关事件
    MODEL_LOADED,
    MODEL_UPDATED,
    
    // SLO相关事件
    SLO_BREACHED_TTFT,
    SLO_BREACHED_TPOT,
    SLO_BREACHED_TTLT,
};

/**
 * @struct DomainEvent
 * @brief 领域事件基类
 */
struct DomainEvent {
    DomainEventType type;
    int64_t timestamp_ns;
    std::string trace_id;
    std::unordered_map<std::string, std::string> metadata;
};

/**
 * @struct RequestScheduledEvent
 * @brief 请求被调度事件
 */
struct RequestScheduledEvent : DomainEvent {
    std::string request_id;
    uint32_t batch_size;
    uint32_t prefill_tokens;
    uint32_t decode_tokens;
};

/**
 * @struct SLOBreachedEvent
 * @brief SLO违规事件
 */
struct SLOBreachedEvent : DomainEvent {
    std::string request_id;
    std::string slo_type;  // "TTFT", "TPOT", "TTLT"
    int64_t target_ms;
    int64_t actual_ms;
    int64_t breach_ms;
};

}  // namespace xllm
```

### 6.2 事件发布

```cpp
// 事件发布示例
void Scheduler::publish_request_scheduled(const std::string& request_id,
                                          const Batch& batch) {
    RequestScheduledEvent event;
    event.type = DomainEventType::REQUEST_SCHEDULED;
    event.timestamp_ns = get_current_time_ns();
    event.request_id = request_id;
    event.batch_size = batch.size();
    event.prefill_tokens = calculate_prefill_tokens(batch);
    event.decode_tokens = calculate_decode_tokens(batch);
    
    event_bus_.publish(event);
}
```

---

## 7. 聚合关系图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Aggregate Relationships                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     RequestAggregate                                  │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                      SequencesGroup                          │    │    │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │    │    │
│  │  │  │Sequence 0 │ │Sequence 1 │ │Sequence 2 │ │Sequence 3 │   │    │    │
│  │  │  │(Beam 0)   │ │(Beam 1)   │ │(Beam 2)   │ │(Beam 3)   │   │    │    │
│  │  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘   │    │    │
│  │  │        └─────────────┼─────────────┼─────────────┘         │    │    │
│  │  └──────────────────────┼─────────────┼──────────────────────────┘    │
│  │                         │             │                                │
│  │                         ▼             ▼                                │
│  │  ┌───────────────────────────────────────────────────────────────┐      │
│  │  │                      Batch                                    │      │
│  │  │  ┌─────────────────────────────────────────────────────────┐ │      │
│  │  │  │                    Blocks[]                              │ │      │
│  │  │  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │ │      │
│  │  │  │  │ B0 │ │ B1 │ │ B2 │ │ B3 │ │ B4 │ │ B5 │ │ B6 │ ...  │ │      │
│  │  │  │  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘       │ │      │
│  │  │  └─────────────────────────────────────────────────────────┘ │      │
│  │  └───────────────────────────────────────────────────────────────┘      │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    │ 被调度执行                               │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     WorkerAggregate                                  │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                     Executor                                  │    │    │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │    │    │
│  │  │  │ Layer 0 │ │ Layer 1 │ │ Layer 2 │ │ Layer N │          │    │    │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │    │    │
│  │  │          │           │           │           │          │    │    │
│  │  │          └───────────┴───────────┴───────────┘          │    │    │
│  │  │                         │                                 │    │    │
│  │  │                         ▼                                 │    │    │
│  │  │  ┌─────────────────────────────────────────────────────┐  │    │    │
│  │  │  │                 KVCacheManager                      │  │    │    │
│  │  │  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐          │  │    │    │
│  │  │  │  │Layer 0│ │Layer 1│ │Layer 2│ │Layer N│          │  │    │    │
│  │  │  │  │[Blocks]│ │[Blocks]│ │[Blocks]│ │[Blocks]│         │  │    │    │
│  │  │  │  └───────┘ └───────┘ └───────┘ └───────┘          │  │    │    │
│  │  │  └─────────────────────────────────────────────────────┘  │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. AI验收标准

### 8.1 领域模型验证

```yaml
ai_verification:
  domain_model:
    - name: "聚合根完整性"
      description: "所有聚合必须包含明确的聚合根"
      check: |
        grep -r "class.*Aggregate" xllm/core | grep -v "//.*class"
      
    - name: "实体标识符"
      description: "所有实体必须有明确的身份标识方法"
      check: |
        对于每个 Entity 类:
        1. 检查是否有 *id() 或 get_id() 方法
        2. 检查标识符类型是否一致
        
    - name: "值对象不可变性"
      description: "值对象必须不可变"
      check: |
        对于每个 ValueObject:
        1. 检查是否有 const 方法
        2. 检查是否缺少 setter 方法
        3. 检查成员变量是否都是 const 或值类型
        
    - name: "聚合边界"
      description: "聚合内部引用应强于外部引用"
      check: |
        1. 检查聚合根是否包含所有子实体
        2. 检查外部引用是否通过ID而非直接引用
```

### 8.2 代码位置映射

| 领域对象 | 代码位置 | 类型 |
|---------|---------|------|
| `Request` | `xllm/core/framework/request/request.h` | 聚合根 |
| `Sequence` | `xllm/core/framework/request/sequence.h` | 实体 |
| `SequencesGroup` | `xllm/core/framework/request/sequences_group.h` | 聚合 |
| `Batch` | `xllm/core/framework/batch/batch.h` | 聚合 |
| `Worker` | `xllm/core/runtime/worker.h` | 聚合根 |
| `KVCache` | `xllm/core/framework/kv_cache/kv_cache.h` | 聚合根 |
| `Block` | `xllm/core/framework/block/block.h` | 实体 |
| `BlockManager` | `xllm/core/framework/block/block_manager.h` | 领域服务 |
| `Scheduler` | `xllm/core/scheduler/scheduler.h` | 领域服务 |
| `LLMEngine` | `xllm/core/distributed_runtime/llm_engine.h` | 领域服务 |

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [架构设计](./02_ARCHITECTURE.md) | 系统整体架构 |
| [设计原则](./03_DESIGN_PRINCIPLES.md) | 设计原则与哲学 |
| [Scheduler设计](./04_SCHEDULER_DESIGN.md) | 调度器实现详解 |
| [Worker设计](./05_WORKER_DESIGN.md) | Worker实现详解 |
