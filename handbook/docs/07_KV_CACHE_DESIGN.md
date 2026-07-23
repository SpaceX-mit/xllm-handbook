# xLLM KV Cache 设计详解

## 文档信息

```yaml
---
document_id: DESIGN-KVCACHE-001
version: 1.0.0
category: component_design
owner: xllm-team
verification_level: BOTH
depends_on:
  - DESIGN-001  # Domain Model
---
```

---

## 1. KV Cache 架构概览

### 1.1 设计理念

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     KV Cache Design Philosophy                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  问题: 自回归生成需要缓存所有历史token的Key和Value                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Transformer Layer                              │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Attention Score                               │  │   │
│  │  │                                                                  │  │   │
│  │  │    Q = x * W_q        K = x * W_k        V = x * W_v          │  │   │
│  │  │      │                  │                  │                     │  │   │
│  │  │      ▼                  ▼                  ▼                     │  │   │
│  │  │   [q_1]   [q_2]     [k_1]   [k_2]      [v_1]   [v_2]         │  │   │
│  │  │                     [k_3]   [k_4]      [v_3]   [v_4]         │  │   │
│  │  │                     [k_5]              [v_5]                 │  │   │
│  │  │                       │                    │                     │  │   │
│  │  │                       └────────────────────┘                     │  │   │
│  │  │                               │                                 │  │   │
│  │  │                               ▼                                 │  │   │
│  │  │                    Attention(Q, K, V)                           │  │   │
│  │  │                                                                  │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  问题: 每个新token都需要与所有历史K/V计算注意力                                  │
│  解决: 缓存K和V，避免重复计算                                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         KV Cache                                      │   │
│  │                                                                        │   │
│  │    Time Step 1:    [k_1, v_1]  ────────────────────────────────────┐  │   │
│  │    Time Step 2:    [k_1, v_1], [k_2, v_2]  ──────────────────────┐ │  │   │
│  │    Time Step 3:    [k_1, v_1], [k_2, v_2], [k_3, v_3]  ──────┐ │ │  │   │
│  │    Time Step N:    [k_1, v_1], ..., [k_n, v_n]  ──────────┐ │ │ │ │  │   │
│  │                                                    │ │ │ │ │ │ │ │ │ │ │  │   │
│  │                                                    │ │ │ │ │ │ │ │ │ │ │ │  │   │
│  │                                                    ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼  │   │
│  │                                                New Token计算            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          KV Cache Components                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      BlockManagerPool                                 │   │
│   │  ┌───────────────────────────────────────────────────────────────┐  │   │
│   │  │                    BlockManager                                │  │   │
│   │  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐         │  │   │
│   │  │  │Block 0│ │Block 1│ │Block 2│ │Block 3│ │Block 4│  ...    │  │   │
│   │  │  │[k,v]  │ │[k,v]  │ │[k,v]  │ │[k,v]  │ │[k,v]  │         │  │   │
│   │  │  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘         │  │   │
│   │  │       │                                                         │  │   │
│   │  │       └───▶ 每个Block存储固定数量的KV对                           │  │   │
│   │  └───────────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                         │
│                                     ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       KVCacheManager                                  │   │
│   │                                                                        │   │
│   │  ┌────────────────────────────────────────────────────────────────┐  │   │
│   │  │                    Layer KV Caches                               │  │   │
│   │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │  │   │
│   │  │  │Layer 0   │  │Layer 1   │  │Layer 2   │  │Layer N   │      │  │   │
│   │  │  │[Blocks]  │  │[Blocks]  │  │[Blocks]  │  │[Blocks]  │      │  │   │
│   │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │  │   │
│   │  └────────────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                         │
│                                     ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         Block                                         │   │
│   │                                                                        │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │                        KVBlock                                │   │   │
│   │   │                                                             │   │   │
│   │   │   k_cache: [num_kv_heads, block_seq_len, head_dim]         │   │   │
│   │   │   v_cache: [num_kv_heads, block_seq_len, head_dim]         │   │   │
│   │   │                                                             │   │   │
│   │   │   seq_len_offset: 该Block在序列中的起始位置                    │   │   │
│   │   │   num_used_tokens: 该Block已使用的token数                     │   │   │
│   │   │                                                             │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. KVCache 实现

### 2.1 KVCache 类

```cpp
// 代码位置: xllm/core/framework/kv_cache/kv_cache.h

/**
 * @class KVCache
 * @brief KV Cache聚合根 - 管理单个注意力层的Key-Value缓存
 * 
 * @design 支持多种缓存格式：标准、MQA、GQA、量化等
 */
class KVCache final {
 public:
    // ========== 构造函数 ==========
    
    /** 默认构造 */
    KVCache();
    
    /** 从张量构造 */
    explicit KVCache(const KVCacheTensors& tensors);
    explicit KVCache(const IndexedKVCacheTensors& tensors);
    explicit KVCache(const LinearAttentionKVCacheTensors& tensors);
    explicit KVCache(const QuantizedKVCacheTensors& tensors);
    explicit KVCache(const DeepSeekV4KVCacheTensors& tensors);
    
    /** 创建新缓存 */
    KVCache(const KVCacheShape& kv_cache_shape,
           const KVCacheCreateOptions& create_options,
           int64_t layer_id);
    
    KVCache(const KVCacheShape& kv_cache_shape,
           const KVCacheCreateOptions& create_options,
           BlockType type,
           int64_t layer_count);
    
    // ========== 不可拷贝 ==========
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    
    // 可移动
    KVCache(KVCache&&) noexcept = default;
    KVCache& operator=(KVCache&&) noexcept = default;

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
     * @brief 获取索引缓存 (用于索引注意力)
     */
    torch::Tensor get_index_cache() const;
    
    // ========== 量化 ==========
    
    /**
     * @brief 获取量化缩放因子
     */
    std::optional<torch::Tensor> get_k_cache_scale() const;
    std::optional<torch::Tensor> get_v_cache_scale() const;
    std::optional<torch::Tensor> get_indexer_cache_scale() const;
    
    // ========== 特殊缓存 (Mamba/SSM) ==========
    
    torch::Tensor get_conv_cache() const;           // 卷积缓存
    torch::Tensor get_ssm_cache() const;             // SSM缓存
    torch::Tensor get_swa_cache() const;             // 滑动窗口缓存
    
    // ========== 压缩缓存 ==========
    
    torch::Tensor get_compress_kv_state() const;
    torch::Tensor get_compress_score_state() const;
    torch::Tensor get_compress_index_kv_state() const;
    torch::Tensor get_compress_index_score_state() const;
    torch::Tensor get_compress_state() const;
    torch::Tensor get_compress_index_state() const;
    
    // ========== 块类型特定张量 ==========
    
    BlockTypeTensorMap get_block_type_tensors(BlockType type) const;
    
    // ========== 状态查询 ==========
    
    bool empty() const;
    std::vector<std::vector<int64_t>> get_shapes();
    
    // ========== 块操作 ==========
    
    /**
     * @brief 与另一个KVCache交换Block
     */
    void swap_blocks(torch::Tensor& src_tensor, torch::Tensor& dst_tensor);
    
    /**
     * @brief 获取缓存张量列表
     */
    std::vector<KVCacheTensor> get_cache_tensors() const;

 private:
    std::unique_ptr<KVCacheImpl> impl_;  // PIMPL
};

/**
 * @brief 批量分配KV Cache
 */
void allocate_kv_caches(std::vector<KVCache>& kv_caches,
                        const KVCacheShape& kv_cache_shape,
                        const KVCacheCreateOptions& create_options);
```

### 2.2 KVCacheImpl 抽象

```cpp
// 代码位置: xllm/core/framework/kv_cache/kv_cache_impl.h

/**
 * @class KVCacheImpl
 * @brief KV Cache实现基类
 */
class KVCacheImpl {
 public:
    virtual ~KVCacheImpl() = default;
    
    // 获取张量
    virtual torch::Tensor get_k_cache() const = 0;
    virtual torch::Tensor get_v_cache() const = 0;
    
    // 块交换
    virtual void swap_blocks(torch::Tensor& src, torch::Tensor& dst) = 0;
    
    // 状态
    virtual bool empty() const = 0;
    
 protected:
    KVCacheShape shape_;
    torch::Device device_;
    torch::ScalarType dtype_;
};

// 标准实现
class StandardKVCacheImpl : public KVCacheImpl {
 public:
    StandardKVCacheImpl(const KVCacheShape& shape,
                       const KVCacheCreateOptions& options);
    
    torch::Tensor get_k_cache() const override { return k_cache_; }
    torch::Tensor get_v_cache() const override { return v_cache_; }
    
    void swap_blocks(torch::Tensor& src, torch::Tensor& dst) override {
        auto tmp = src.clone();
        src.copy_(dst);
        dst.copy_(tmp);
    }
    
    bool empty() const override { 
        return !k_cache_.defined() || !v_cache_.defined(); 
    }

 private:
    torch::Tensor k_cache_;  // [num_kv_heads, max_seq_len, head_dim]
    torch::Tensor v_cache_;  // [num_kv_heads, max_seq_len, head_dim]
};
```

---

## 3. Block 管理

### 3.1 Block 类

```cpp
// 代码位置: xllm/core/framework/block/block.h

/**
 * @class Block
 * @brief Block - KV Cache的最小分配单元
 * 
 * @design Block是不可变的容器，包含固定大小的KV缓存
 * @identity 由全局唯一的block_id标识
 */
class Block {
 public:
    // ========== 类型定义 ==========
    
    enum class BlockType {
        UNKNOWN = 0,
        PREFILL = 1,           // 预填充Block (新计算)
        DECODE = 2,            // 解码Block (逐token追加)
        PREFILL_DECODE = 3,    // 混合Block
    };
    
    // ========== 构造函数 ==========
    
    /**
     * @brief 构造Block
     * @param block_id Block唯一标识
     * @param shape Block形状
     * @param device 设备
     */
    Block(int64_t block_id,
         const std::vector<int64_t>& shape,
         torch::Device device);
    
    // ========== 身份 ==========
    
    int64_t block_id() const { return block_id_; }
    
    // ========== 数据访问 ==========
    
    /**
     * @brief 获取所有张量
     */
    std::vector<torch::Tensor>& get_tensors() { return tensors_; }
    const std::vector<torch::Tensor>& get_tensors() const { return tensors_; }
    
    /**
     * @brief 获取指定张量
     */
    torch::Tensor get_tensor(size_t idx) const { 
        return tensors_.at(idx); 
    }
    
    // ========== 类型管理 ==========
    
    BlockType block_type() const { return block_type_; }
    void set_block_type(BlockType type) { block_type_ = type; }
    
    // ========== 空间管理 ==========
    
    /**
     * @brief 获取已使用的token数
     */
    int num_tokens_used() const { return num_tokens_used_; }
    
    /**
     * @brief 设置已使用的token数
     */
    void set_num_tokens_used(int num_tokens) { num_tokens_used_ = num_tokens; }
    
    /**
     * @brief 获取最大token容量
     */
    int max_num_tokens() const { return max_num_tokens_; }
    
    /**
     * @brief 是否有空闲空间
     */
    bool has_free_space() const { 
        return num_tokens_used_ < max_num_tokens_; 
    }
    
    // ========== 生命周期 ==========
    
    /**
     * @brief 分配给指定序列
     */
    void mark_allocated(int64_t owner_id) {
        allocated_.store(true, std::memory_order_relaxed);
        owner_id_ = owner_id;
    }
    
    /**
     * @brief 释放
     */
    void mark_free() {
        allocated_.store(false, std::memory_order_relaxed);
        owner_id_ = -1;
        num_tokens_used_ = 0;
    }
    
    /**
     * @brief 是否已分配
     */
    bool is_allocated() const { 
        return allocated_.load(std::memory_order_relaxed); 
    }
    
    /**
     * @brief 获取所有者ID
     */
    int64_t owner_id() const { return owner_id_; }
    
    // ========== 引用计数 (Prefix Cache) ==========
    
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
    
    // 空间管理
    int num_tokens_used_ = 0;
    int max_num_tokens_;
    
    // 分配状态
    std::atomic<bool> allocated_{false};
    int64_t owner_id_ = -1;
    
    // Prefix Cache引用计数
    int ref_count_ = 1;
};
```

### 3.2 BlockManager

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
    /**
     * @brief 构造BlockManager
     * @param num_blocks 总Block数
     * @param shape Block形状
     * @param device 设备
     */
    BlockManager(int64_t num_blocks,
                const KVCacheShape& shape,
                torch::Device device);
    
    // ========== 分配/释放 ==========
    
    /**
     * @brief 分配指定数量的Block
     * @param num_blocks 需要的Block数量
     * @return 分配的Block ID列表
     * @throws std::runtime_error 如果Block不足
     */
    virtual std::vector<int64_t> allocate(int64_t num_blocks);
    
    /**
     * @brief 释放Block
     */
    virtual void free(const std::vector<int64_t>& block_ids);
    
    /**
     * @brief 释放某个序列的所有Block
     */
    virtual void free_sequence(int64_t sequence_id);
    
    /**
     * @brief 释放所有Block
     */
    virtual void free_all();
    
    // ========== 查询 ==========
    
    /**
     * @brief 获取Block指针
     */
    virtual Block* get_block(int64_t block_id);
    
    /**
     * @brief 获取可用Block数
     */
    virtual int64_t get_num_available_blocks() const;
    
    /**
     * @brief 获取总Block数
     */
    int64_t num_blocks() const { return num_blocks_; }
    
    // ========== 序列Block查询 ==========
    
    /**
     * @brief 获取Sequence占用的Block列表
     */
    virtual std::vector<int64_t> get_sequence_blocks(int64_t sequence_id) const;
    
    /**
     * @brief 检查是否可以分配
     */
    virtual bool can_allocate(int64_t sequence_id, 
                             int64_t num_needed_blocks) const;
    
    // ========== 交换 ==========
    
    /**
     * @brief 添加交换请求
     */
    virtual void add_swap_request(const SwapRequest& request);
    
    /**
     * @brief 获取Block的KV张量
     */
    virtual std::pair<torch::Tensor, torch::Tensor> get_block_kv_tensors(
        int64_t block_id) const;

 protected:
    // ========== 内部方法 ==========
    
    /**
     * @brief 获取空闲Block
     */
    virtual int64_t get_free_block();
    
    /**
     * @brief 检查序列是否有足够的空间
     */
    bool has_sufficient_space(int64_t sequence_id, int64_t num_needed) const;

 private:
    int64_t num_blocks_;
    KVCacheShape shape_;
    torch::Device device_;
    
    // Block池
    std::vector<std::unique_ptr<Block>> blocks_;
    
    // 空闲Block集合
    std::unordered_set<int64_t> free_blocks_;
    
    // Sequence -> Blocks映射
    std::unordered_map<int64_t, std::vector<int64_t>> sequence_to_blocks_;
    
    // Block -> Sequence反向索引
    std::unordered_map<int64_t, int64_t> block_to_sequence_;
    
    // 锁
    mutable std::mutex mu_;
};
```

### 3.3 BlockManager 派生类

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     BlockManager Hierarchy                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────────┐                                 │
│                         │  BlockManager   │                                 │
│                         │  (基类)          │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                           │
│         ┌────────────────────────┼────────────────────────┐                 │
│         │                        │                        │                   │
│         ▼                        ▼                        ▼                   │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐            │
│  │  Single     │         │ Concurrent  │         │Composite    │            │
│  │  Block      │         │ Block       │         │ Block       │            │
│  │  Manager    │         │ Manager     │         │ Manager     │            │
│  │             │         │             │         │             │            │
│  │ 简单场景    │         │ 并发写入     │         │ 多Block类型  │            │
│  │ 单Sequence  │         │ 支持         │         │ 组合         │            │
│  └─────────────┘         └─────────────┘         └──────┬──────┘            │
│                                                          │                    │
│                                                          ▼                    │
│                                                 ┌─────────────────┐          │
│                                                 │  Hierarchy      │          │
│                                                 │  Block          │          │
│                                                 │  Manager        │          │
│                                                 │                 │          │
│                                                 │ 分层Block管理    │          │
│                                                 └─────────────────┘          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                      Specialized Managers                             │     │
│  │                                                                        │     │
│  │  SlidingWindowBlockManager - 滑动窗口Block                             │     │
│  │  LinearStateBlockManager   - 线性注意力State Block                     │     │
│  │                                                                        │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```cpp
// 单Sequence BlockManager
class SingleBlockManager : public BlockManager {
    // 每个Sequence独占整个BlockManager
    // 用于简单场景
};

// 并发BlockManager
class ConcurrentBlockManager : public BlockManager {
 public:
    /**
     * @brief 并发分配
     */
    std::vector<int64_t> allocate_concurrent(
        int64_t sequence_id, 
        int64_t num_blocks,
        std::function<bool(Block*)> filter);
    
    /**
     * @brief 乐观分配
     */
    bool try_allocate_optimistic(
        int64_t sequence_id,
        int64_t num_blocks);
};

// 滑动窗口BlockManager
class SlidingWindowBlockManager : public BlockManager {
 public:
    /**
     * @brief 获取滑动窗口范围内的Block
     */
    std::vector<int64_t> get_window_blocks(
        int64_t sequence_id,
        int64_t window_start,
        int64_t window_end);
    
    /**
     * @brief 驱逐旧Block
     */
    void evict_oldest_blocks(int64_t count);
};

// 分层BlockManager
class HierarchyBlockManager : public BlockManager {
 public:
    HierarchyBlockManager(
        std::vector<std::unique_ptr<BlockManager>> managers,
        std::vector<int> level_sizes);
    
    /**
     * @brief 获取Block所属层级
     */
    int get_block_level(int64_t block_id) const;
    
    /**
     * @brief 分配到指定层级
     */
    std::vector<int64_t> allocate_at_level(
        int level,
        int64_t num_blocks);
    
 private:
    std::vector<std::unique_ptr<BlockManager>> managers_;
    std::vector<int> level_sizes_;
    std::unordered_map<int64_t, int> block_to_level_;
};
```

---

## 4. Prefix Cache

### 4.1 设计理念

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Prefix Cache Design                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  场景: 多个请求共享相同的系统提示词                                              │
│                                                                              │
│  Request 1: [System: 你是一个助手] [User: 什么是AI?]                           │
│  Request 2: [System: 你是一个助手] [User: 解释机器学习]                        │
│  Request 3: [System: 你是一个助手] [User: 什么是深度学习]                      │
│                                                                              │
│              System部分可以复用！                                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                      Prefix Cache Table                              │     │
│  │                                                                        │     │
│  │  Hash("你是一个助手")  →  Block[0, 1, 2]  →  RefCount: 3            │     │
│  │                                                                        │     │
│  │  [Block 0] [Block 1] [Block 2] ──────────────────────────────────┐   │     │
│  │      │         │         │                                        │   │     │
│  │      ▼         ▼         ▼                                        │   │     │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                             │   │     │
│  │  │ Request │ │ Request │ │ Request │                             │   │     │
│  │  │    1    │ │    2    │ │    3    │                             │   │     │
│  │  └─────────┘ └─────────┘ └─────────┘                             │   │     │
│  │      ▲         ▲         ▲                                        │   │     │
│  │      └─────────┴─────────┴────────────────────────────────────────┘   │     │
│  │                       RefCount = 3                                      │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  效果:                                                                        │
│  • 第一个请求计算完整的System + User                                          │
│  • 后续请求跳过System部分计算，直接使用缓存                                     │
│  • 节省 50-80% 的预填充计算                                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Prefix Cache 实现

```cpp
// 代码位置: xllm/core/framework/prefix_cache/prefix_cache.h

/**
 * @class PrefixCache
 * @brief 前缀缓存 - 复用共享前缀
 */
class PrefixCache {
 public:
    PrefixCache(std::shared_ptr<BlockManager> block_manager);
    
    // ========== 查询 ==========
    
    /**
     * @brief 尝试查找前缀缓存
     * @param tokens 输入token序列
     * @return 如果命中，返回缓存的Block列表；否则返回空
     */
    std::optional<std::vector<int64_t>> lookup(const std::vector<int>& tokens);
    
    /**
     * @brief 检查是否可以缓存
     */
    bool can_cache(const std::vector<int>& tokens) const;
    
    // ========== 管理 ==========
    
    /**
     * @brief 添加前缀到缓存
     * @param tokens 前缀token序列
     * @param blocks 对应的Block列表
     */
    void add(const std::vector<int>& tokens, 
            const std::vector<int64_t>& blocks);
    
    /**
     * @brief 移除前缀
     */
    void remove(const std::vector<int>& tokens);
    
    /**
     * @brief 清除所有缓存
     */
    void clear();
    
    /**
     * @brief 获取命中率统计
     */
    CacheStats get_stats() const;

 private:
    /**
     * @brief 计算token序列的哈希
     */
    std::string compute_hash(const std::vector<int>& tokens) const;
    
    /**
     * @brief 递归查找最长匹配前缀
     */
    std::optional<std::vector<int64_t>> find_longest_prefix(
        const std::vector<int>& tokens);

 private:
    std::shared_ptr<BlockManager> block_manager_;
    
    // 哈希表: token_hash → {blocks, ref_count, ttl}
    struct CacheEntry {
        std::vector<int64_t> blocks;
        int ref_count;
        absl::Time created;
        absl::Duration ttl;
    };
    std::unordered_map<std::string, CacheEntry> cache_table_;
    
    // Token序列到哈希的映射 (用于快速查找)
    std::unordered_map<std::string, std::string> token_to_hash_;
    
    // 统计
    size_t hits_ = 0;
    size_t misses_ = 0;
};
```

### 4.3 Prefix Cache 使用流程

```cpp
// 在Scheduler中使用
bool ContinuousScheduler::try_prefix_cache(std::shared_ptr<Request>& request) {
    // 1. 获取请求的token序列
    auto& tokens = request->get_tokens();
    
    // 2. 查询前缀缓存
    auto cached_blocks = prefix_cache_->lookup(tokens);
    
    if (cached_blocks.has_value()) {
        // 命中！直接使用缓存的Blocks
        XLLM_LOG(INFO) << "Prefix cache hit for request " << request->request_id();
        
        // 3. 更新引用计数
        for (auto block_id : *cached_blocks) {
            block_manager_->get_block(block_id)->add_ref();
        }
        
        // 4. 设置请求的prefix cache信息
        request->record_num_prefix_cache_tokens(cached_blocks->size() * block_size);
        
        // 5. 跳过前缀计算
        return true;
    }
    
    return false;
}

// 在请求完成后释放引用
void on_request_finished(std::shared_ptr<Request>& request) {
    auto prefix_blocks = request->get_prefix_cache_blocks();
    
    for (auto block_id : prefix_blocks) {
        auto* block = block_manager_->get_block(block_id);
        block->release_ref();  // 引用计数-1
        
        if (block->ref_count() == 0) {
            // Block可被其他请求复用
            // 根据LRU策略决定是否保留
        }
    }
}
```

---

## 5. KV Cache 传输 (P/D 分离)

### 5.1 传输架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    KV Cache Transfer Architecture                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Prefill Node                                                         Decode │
│   ┌────────────────────────────────────────────────────────────────────────┐ │
│   │                                                                         │ │
│   │    ┌──────────────────────────────────────────────────────────────┐  │ │
│   │    │                    KV Cache Manager                            │  │ │
│   │    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │  │ │
│   │    │  │Layer 0 │ │Layer 1 │ │Layer 2 │ │Layer 3 │ │ ...    │      │  │ │
│   │    │  │[Blocks]│ │[Blocks]│ │[Blocks]│ │[Blocks]│ │        │      │  │ │
│   │    │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘      │  │ │
│   │    └──────────────────────────────────────────────────────────────┘  │ │
│   │                              │                                           │ │
│   │                              │ BlockTransferInfo                        │ │
│   │                              ▼                                           │ │
│   │    ┌──────────────────────────────────────────────────────────────┐  │ │
│   │    │               Transfer Engine (Mooncake)                     │  │ │
│   │    │                                                              │  │ │
│   │    │   ┌─────────────────────────────────────────────────────┐  │  │ │
│   │    │   │              RDMA / TCP Transfer                     │  │  │ │
│   │    │   │                                                              │  │ │
│   │    │   │   [Layer 0 Blocks] ────────▶  [Layer 0 Blocks]   │  │  │ │
│   │    │   │   [Layer 1 Blocks] ────────▶  [Layer 1 Blocks]   │  │  │ │
│   │    │   │   [Layer 2 Blocks] ────────▶  [Layer 2 Blocks]   │  │  │ │
│   │    │   │                                                              │  │ │
│   │    │   └─────────────────────────────────────────────────────┘  │  │ │
│   │    └──────────────────────────────────────────────────────────────┘  │ │
│   │                              │                                           │ │
│   └──────────────────────────────┼───────────────────────────────────────────┘ │
│                                  │                                            │
│                                  │ RDMA/NIC                                   │
│                                  ▼                                            │
│   ┌────────────────────────────────────────────────────────────────────────┐ │
│   │                        Decode Node                                    │ │
│   │    ┌──────────────────────────────────────────────────────────────┐  │ │
│   │    │               Transfer Engine (Mooncake)                     │  │ │
│   │    │                                                              │  │ │
│   │    │   接收并写入对应的KV Cache位置                                  │  │ │
│   │    │                                                              │  │ │
│   │    └──────────────────────────────────────────────────────────────┘  │ │
│   │                              │                                           │ │
│   │                              ▼                                           │ │
│   │    ┌──────────────────────────────────────────────────────────────┐  │ │
│   │    │                    KV Cache Manager                            │  │ │
│   │    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │  │ │
│   │    │  │Layer 0 │ │Layer 1 │ │Layer 2 │ │Layer 3 │ │ ...    │      │  │ │
│   │    │  │[Blocks]│ │[Blocks]│ │[Blocks]│ │[Blocks]│ │        │      │  │ │
│   │    │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘      │  │ │
│   │    └──────────────────────────────────────────────────────────────┘  │ │
│   └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 传输接口

```cpp
// 代码位置: xllm/core/framework/kv_cache_transfer/transfer_engine.h

/**
 * @class TransferEngine
 * @brief KV Cache传输引擎
 */
class TransferEngine {
 public:
    virtual ~TransferEngine() = default;
    
    /**
     * @brief 异步传输Block
     * @param src/src_addr 源地址
     * @param dst/dst_addr 目标地址
     * @param blocks 要传输的Block列表
     */
    virtual folly::SemiFuture<TransferResult> transfer_async(
        const std::string& src_addr,
        const std::vector<uint64_t>& src_blocks,
        const std::string& dst_addr,
        const std::vector<uint64_t>& dst_blocks,
        const TransferOptions& options) = 0;
    
    /**
     * @brief 批量传输
     */
    virtual std::vector<folly::SemiFuture<TransferResult>> transfer_batch(
        const std::vector<TransferRequest>& requests) = 0;
    
    /**
     * @brief 预取Blocks
     */
    virtual void prefetch(const std::vector<TransferRequest>& requests) = 0;
};

/**
 * @struct BlockTransferInfo
 * @brief Block传输信息
 */
struct BlockTransferInfo {
    uint64_t batch_id;
    std::string src_addr;
    std::string dst_addr;
    std::vector<uint64_t> src_blocks;
    std::vector<uint64_t> dst_blocks;
    std::vector<uint64_t> src_linear_state_ids;
    std::vector<uint64_t> dst_linear_state_ids;
    int priority = 0;
};

/**
 * @struct TransferResult
 * @brief 传输结果
 */
struct TransferResult {
    bool success;
    size_t bytes_transferred;
    absl::Duration duration;
    std::string error_message;
};
```

### 5.3 传输流程

```cpp
// 在Worker中触发传输
folly::SemiFuture<bool> Worker::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
    
    // 1. 构建传输请求
    TransferRequest request;
    request.src_addr = src_addr;
    request.src_blocks = src_blocks;
    request.dst_blocks = dst_blocks;
    request.src_linear_state_ids = src_linear_state_ids;
    request.dst_linear_state_ids = dst_linear_state_ids;
    
    // 2. 获取传输引擎
    auto engine = TransferEngineFactory::get(src_cluster_id);
    
    // 3. 异步传输
    return engine->transfer_async(
        request.src_addr,
        request.src_blocks,
        "",  // 本地
        request.dst_blocks,
        TransferOptions{}
    ).toStdFuture().then([](auto&& result) {
        return result.success;
    });
}

// 调度器触发传输
void DisaggPDScheduler::trigger_kv_transfer(
    const Request& request,
    const PrefillOutput& prefill_result) {
    
    // 1. 提取KV信息
    auto kv_blocks = prefill_result.get_kv_blocks();
    auto kv_shape = prefill_result.get_kv_shape();
    
    // 2. 发送到Decode节点
    auto future = worker_client_->pull_kv_blocks_async(
        /*src_cluster*/ 0,
        /*src_addr*/ prefill_node_addr_,
        /*src_blocks*/ kv_blocks,
        /*dst_blocks*/ allocate_dst_blocks(kv_blocks.size()),
        /*linear_state_ids*/ {}
    );
    
    // 3. 监控传输进度
    future.then([this, &request](bool success) {
        if (success) {
            // 传输完成，通知Decode节点开始处理
            notify_decode_node_ready(request);
        } else {
            // 传输失败，回退策略
            handle_transfer_failure(request);
        }
    });
}
```

---

## 6. AI验收标准

### 6.1 功能验证

```yaml
ai_verification:
  kv_cache:
    - name: "Block分配正确性"
      test: |
        1. 分配num_blocks个Block，验证都成功
        2. 验证Block ID不重复
        3. 验证已分配的Block不再出现在空闲列表
        
    - name: "Block释放正确性"
      test: |
        1. 释放后Block回到空闲列表
        2. 释放后Block的引用计数正确
        3. 释放后Block数据被清空
        
    - name: "Prefix Cache正确性"
      test: |
        1. 相同前缀的请求能命中缓存
        2. 引用计数在请求完成时正确递减
        3. 无引用时Block被正确回收
        
    - name: "传输正确性"
      test: |
        1. 传输后Decode节点的数据与Prefill节点一致
        2. 传输过程不丢失数据
        3. 失败时能正确回退
```

### 6.2 性能验证

```cpp
// 性能测试
TEST(KVCachePerformance, BlockAllocationThroughput) {
    BlockManager manager(/*num_blocks=*/10000, shape, device);
    
    const int iterations = 10000;
    auto start = absl::Now();
    
    for (int i = 0; i < iterations; ++i) {
        auto blocks = manager.allocate(/*num=*/8);
        manager.free(blocks);
    }
    
    auto duration = absl::Now() - start;
    double throughput = iterations / absl::ToDoubleSeconds(duration);
    
    EXPECT_GT(throughput, 100000);  // 10万次/秒
}

TEST(KVCachePerformance, PrefixCacheHitRate) {
    // 模拟1000个请求，80%共享相同系统提示
    // 验证缓存命中率 > 70%
}
```

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [领域模型](./04_DOMAIN_MODEL.md) | DDD领域划分 |
| [Worker设计](./06_WORKER_DESIGN.md) | Worker实现详解 |
| [Batch设计](./06_BATCH_DESIGN.md) | Batch实现详解 |
| [配置Schema](./09_CONFIG_SCHEMA.md) | KV Cache配置 |
