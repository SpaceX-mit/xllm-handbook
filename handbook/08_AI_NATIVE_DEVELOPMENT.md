# xLLM AI Native 开发规范

## 文档信息

```yaml
---
document_id: AI-NATIVE-001
version: 1.0.0
category: ai_native
owner: xllm-team
verification_level: BOTH
purpose: "定义AI辅助开发规范，使代码可被AI工具解析、生成、验证"
---
```

---

## 1. AI Native 开发理念

### 1.1 核心理念

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     AI Native Development Philosophy                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                       │   │
│   │    传统开发:        人 → 代码 → 机器                                   │   │
│   │                                                                       │   │
│   │    AI辅助开发:      人 ←→ AI ←→ 代码 ←→ 机器                          │   │
│   │                     ↑         ↑         ↑                             │   │
│   │                     └─────────┴─────────┘                             │   │
│   │                         协作闭环                                       │   │
│   │                                                                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   核心原则:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  1. 代码自解释 (Self-Documenting Code)                               │   │
│   │     • 命名即文档，注释即规范                                          │   │
│   │     • 结构化注释，便于AI解析                                          │   │
│   │                                                                       │   │
│   │  2. 机器友好格式 (Machine-Friendly Format)                            │   │
│   │     • 标准化注释标签                                                  │   │
│   │     • 结构化输出格式                                                  │   │
│   │     • 正则可解析的模式                                                │   │
│   │                                                                       │   │
│   │  3. 可验证设计 (Verifiable Design)                                    │   │
│   │     • 关键路径有断言                                                  │   │
│   │     • 测试用例覆盖                                                   │   │
│   │     • 性能基准明确                                                   │   │
│   │                                                                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 开发流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      AI Native Development Workflow                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  需求分析   │───▶│  规格生成   │───▶│  代码生成   │───▶│  验证测试   │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘ │
│       │                  │                  │                     │         │
│       │     ┌───────────┴───────────┐       │               ┌────┴────┐    │
│       │     │                     │       │               │         │    │
│       ▼     ▼                     ▼       ▼               ▼         ▼    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  Human      │    │  AI Agent   │    │  AI Agent   │    │  Human +   │ │
│  │  (Review)   │◀──▶│  (Spec)     │◀──▶│  (Code)     │◀──▶│  AI Agent  │ │
│  │             │    │             │    │             │    │  (Review)  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│       │                  │                  │                     │         │
│       │                  │                  │                     │         │
│       ▼                  ▼                  ▼                     ▼         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  文档更新   │    │  结构化规格 │    │  可执行代码 │    │  CI/CD验证 │ │
│  │  人工review │    │  AI可解析   │    │  AI可理解   │    │  AI可运行  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         反馈循环                                       │   │
│  │   测试失败 ──▶ AI分析原因 ──▶ AI修复 ──▶ 重新测试 ──▶ 循环直到通过    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 代码自解释规范

### 2.1 命名规范

```cpp
// 代码位置: 任意

// ========== 好的命名: 自解释 ==========

// ✗ 差: 模糊
int process(int x);
void handle(Event* e);
void compute();

// ✓ 好: 明确意图
int calculate_batch_size_based_on_memory_limit(int available_memory_bytes);
void handle_request_timeout(Event* timeout_event);
void compute_attention_score_with_causal_mask(
    const Tensor& query,    // [batch, num_heads, seq_len, head_dim]
    const Tensor& key,      // [batch, num_kv_heads, seq_len, head_dim]
    const Tensor& value,    // [batch, num_kv_heads, seq_len, head_dim]
    bool causal_mask = true  // 是否应用因果掩码
);

// ========== 好的命名: 包含类型信息 ==========

// 类名使用 PascalCase
class RequestScheduler;
class KVCacheBlockManager;

// 接口使用 I 或 Interface 后缀
class IWorkerInterface;
class SchedulerPolicyInterface;

// 枚举值使用 UPPER_SNAKE_CASE
enum class Urgency { STARVED, URGENT, NORMAL };
enum class BlockType { UNKNOWN, PREFILL, DECODE, PREFILL_DECODE };

// 成员变量使用 m_ 前缀或 _ 后缀
class Worker {
 private:
    int m_num_active_requests_;
    std::unique_ptr<WorkerImpl> impl_;  // PIMPL
};

// 常量使用 k 前缀
static constexpr int kDefaultBatchSize = 32;
static constexpr double kDefaultTemperature = 0.7;

// ========== 好的命名: 包含单位 ==========

// ✗ 差
int timeout;
int size;

// ✓ 好
absl::Duration request_timeout_ms;
size_t memory_size_bytes;
int64_t num_tokens_per_block;
```

### 2.2 注释规范

```cpp
// 代码位置: 任意

// ========== 文件头注释 ==========

/**
 * @file scheduler.h
 * @brief 请求调度器核心实现
 * 
 * @details 
 * 调度器负责管理请求队列、决定批处理策略、优化GPU利用率。
 * 支持多种调度策略: ContinuousBatching、DisaggPD、ZeroEviction等。
 * 
 * @author xllm-team
 * @date 2026-07-23
 * @version 1.0.0
 * 
 * @sa https://docs.xllm-ai.com/scheduler
 */

// ========== 类/结构体注释 ==========

/**
 * @class Batch
 * @brief 批处理聚合根
 * 
 * @details 
 * Batch代表一次推理调度的序列集合。一个Batch包含多个Sequence，
 * 在一次前向传播中处理完成。
 * 
 * @invariant 
 * - size() >= 1 (非空)
 * - 所有Sequence必须属于同一模型实例
 * 
 * @thread_safety 
 * Batch对象由单个线程拥有，不共享
 * 
 * @example
 * @code
 * Batch batch;
 * batch.add(sequence1);
 * batch.add(sequence2);
 * auto input = batch.prepare_forward_input(...);
 * @endcode
 */
class Batch {
    // ...
};

// ========== 方法注释 ==========

/**
 * @brief 添加序列到批次
 * 
 * @param sequence 要添加的序列指针
 * @param allowed_max_token 该序列允许的最大token数
 * 
 * @return 添加是否成功
 * 
 * @throws None 此方法不会抛出异常
 * 
 * @pre sequence != nullptr
 * @pre !sequence->is_finished()
 * 
 * @post batch.size() 增加1
 * 
 * @complexity O(1)
 * 
 * @note 
 * 如果KV Cache不足，此方法可能失败
 */
bool add(Sequence* sequence, 
         uint32_t allowed_max_token = std::numeric_limits<uint32_t>::max());

// ========== 行内注释 ==========

// 计算注意力分数
// 注意: 这里使用scaled dot-product attention，缩放因子为 1/sqrt(d_k)
auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(head_dim_);

// TODO: 优化性能 - 当前实现为O(n^2)，可考虑使用Flash Attention降至O(n)
// @ai_action: 评估Flash Attention集成
```

### 2.3 文档化约束

```cpp
// ========== 使用宏定义文档化约束 ==========

/**
 * @brief 分配KV Cache Block
 * 
 * @param num_blocks 需要的Block数量
 * @return 分配的Block ID列表
 * 
 * @requires num_blocks > 0
 * @requires num_blocks <= available_blocks()
 * @ensures result.size() == num_blocks
 * @ensures all_of(result, [](id){ return get_block(id) != nullptr; })
 */
std::vector<int64_t> allocate(int64_t num_blocks);

// ========== 使用结构化注释标记 ==========

// @ai_context: 性能关键路径
// @ai_verification: 需要性能测试
// @ai_optimization: 可考虑使用对象池减少分配
void* allocate_critical_path() {
    // ...
}

// @ai_boundary: 边界条件
// @ai_test_cases: 
//   - num_blocks = 0
//   - num_blocks = available_blocks()
//   - num_blocks > available_blocks() (应抛出异常)
std::vector<int64_t> allocate(int64_t num_blocks) {
    if (num_blocks > available_blocks()) {
        throw std::runtime_error("Insufficient blocks");
    }
    // ...
}
```

---

## 3. 机器友好格式

### 3.1 AI注释标签

```cpp
// 代码位置: 任意

// ========== AI动作标签 ==========

// @ai_action: 生成单元测试
// @ai_verification: 验证返回值范围
// @ai_optimization: 评估是否可向量化
// @ai_refactor: 考虑提取为独立函数
// @ai_doc: 更新API文档
// @ai_security: 检查边界条件

// ========== AI上下文标签 ==========

// @ai_context: 这是性能热点
// @ai_context: 涉及多线程同步
// @ai_context: 关键数据结构

// ========== AI边界标签 ==========

// @ai_boundary: 此处需处理空指针
// @ai_boundary: 此处需处理负数输入
// @ai_boundary: 此处需处理溢出

// ========== 使用示例 ==========

/**
 * @brief 执行一步推理
 * 
 * @ai_action: 生成集成测试
 * @ai_action: 生成性能基准测试
 * @ai_verification: 验证输出logits维度正确
 * @ai_verification: 验证采样结果在词汇表范围内
 */
std::optional<ForwardOutput> step(const ForwardInput& inputs);
```

### 3.2 结构化输出格式

```cpp
// ========== 结构化日志 ==========

// ✗ 差: 非结构化日志
LOG(INFO) << "Batch processed, size=" << batch.size();

// ✓ 好: 结构化JSON日志
XLLM_LOG(INFO, "BatchProcessed", {
    {"batch_id", batch.batch_id()},
    {"size", static_cast<int>(batch.size())},
    {"prefill_tokens", prefill_tokens},
    {"decode_tokens", decode_tokens},
    {"estimated_time_ms", estimated_time_ms},
    {"timestamp", absl::ToUnixNanos(absl::Now())}
});

// ========== 结构化错误 ==========

// ✗ 差: 简单异常
throw std::runtime_error("Allocation failed");

// ✓ 好: 结构化异常
class AllocationError : public std::runtime_error {
 public:
    AllocationError(size_t requested, size_t available, const std::string& reason)
        : std::runtime_error(fmt::format(
            "Allocation failed: requested={}, available={}, reason={}",
            requested, available, reason)),
          requested_(requested),
          available_(available),
          reason_(reason) {}
    
    size_t requested() const { return requested_; }
    size_t available() const { return available_; }
    const std::string& reason() const { return reason_; }
    
    // 结构化数据，便于AI解析
    nlohmann::json to_json() const {
        return {
            {"error_type", "AllocationError"},
            {"requested", requested_},
            {"available", available_},
            {"reason", reason_}
        };
    }

 private:
    size_t requested_;
    size_t available_;
    std::string reason_;
};

// ========== 结构化返回值 ==========

/**
 * @brief 估算KV Cache容量
 * @return {total_blocks, block_size_bytes}
 */
std::tuple<int64_t, int64_t> estimate_kv_cache_capacity();

// ========== 结构化配置 ==========

/**
 * @brief 调度器配置
 */
struct SchedulerConfig {
    // @ai_schema_type: int
    // @ai_schema_min: 1
    // @ai_schema_max: 1024
    // @ai_schema_default: 64
    int max_batch_size = 64;
    
    // @ai_schema_type: int
    // @ai_schema_unit: tokens
    // @ai_schema_default: 8192
    int max_prefill_tokens = 8192;
    
    // @ai_schema_type: double
    // @ai_schema_unit: seconds
    // @ai_schema_default: 0.1
    double step_timeout_seconds = 0.1;
};
```

### 3.3 正则可解析模式

```cpp
// ========== 一致的模式便于解析 ==========

// 类定义模式
// Pattern: class {Name} [: public|private {Base}] { "{" ...
// 用法: grep -E "^class \w+" xllm/core/**/*.h

// 方法定义模式
// Pattern: {ReturnType} {ClassName}::{MethodName} "(" ...
// 用法: grep -E "::step\(" xllm/core/**/*.cpp

// 虚函数标记
// Pattern: virtual {ReturnType} {MethodName} ...
// 用法: grep -E "^    virtual " xllm/core/**/*.h

// 常量定义模式
// Pattern: static constexpr {Type} k{Name} = {Value};
// 用法: grep -E "static constexpr" xllm/core/**/*.h

// 测试用例模式
// Pattern: TEST({Suite}, {Name}) { ... }
// 用法: grep -E "^TEST\(" tests/**/*.cpp

// ========== 使用辅助宏 ==========

// 定义可枚举的模式
#define XLLM_DEFINE_INTERFACE(InterfaceName) \
    class InterfaceName { \
     public: \
        virtual ~InterfaceName() = default;

// 接口方法
#define XLLM_DEFINE_METHOD(RetType, MethodName, ...) \
    virtual RetType MethodName(__VA_ARGS__) = 0;

// 使用示例
XLLM_DEFINE_INTERFACE(IWorker)
    XLLM_DEFINE_METHOD(bool, init_model, const std::string& path)
    XLLM_DEFINE_METHOD(ForwardOutput, step, const ForwardInput& input)
    XLLM_DEFINE_METHOD(bool, sleep, MasterStatus status)
XLLM_DEFINE_END_INTERFACE()

// AI可以轻松解析这些模式
// grep -E "XLLM_DEFINE_METHOD" xllm/core/**/*.h
```

---

## 4. 可验证设计

### 4.1 契约式设计

```cpp
// ========== 前置条件检查 ==========

/**
 * @brief 添加请求到调度器
 * @pre request != nullptr
 * @pre !request->finished()
 * @pre scheduler.pending_count() < max_pending_requests
 */
bool Scheduler::add_request(std::shared_ptr<Request>& request) {
    // 前置条件检查
    XLLM_DCHECK(request != nullptr);  // Debug断言
    XLLM_DCHECK(!request->finished());
    XLLM_DCHECK(pending_count_ < max_pending_requests_);
    
    // 业务逻辑
    pending_requests_.push(request);
    pending_count_++;
    
    return true;
}

// ========== 后置条件检查 ==========

/**
 * @brief 分配Block
 * @post result.size() == num_blocks
 * @post all_of(result, is_valid_block_id)
 */
std::vector<int64_t> allocate(int64_t num_blocks) {
    auto result = do_allocate(num_blocks);
    
    // 后置条件检查
    XLLM_DCHECK(result.size() == static_cast<size_t>(num_blocks));
    XLLM_DCHECK(std::all_of(result.begin(), result.end(), 
        [this](int64_t id) { return is_valid_block_id(id); }));
    
    return result;
}

// ========== 不变量检查 ==========

/**
 * @class Batch
 * @invariant size() > 0
 * @invariant all sequences are not finished
 */
class Batch {
 public:
    void add(Sequence* seq) {
        // 不变量检查
        XLLM_INVARIANT(seq != nullptr);
        XLLM_INVARIANT(!seq->is_finished());
        
        sequences_.push_back(seq);
        
        // 不变量仍然满足
        XLLM_INVARIANT(size() > 0);
    }
    
 private:
    std::vector<Sequence*> sequences_;
};
```

### 4.2 验证框架

```cpp
// ========== 验证宏定义 ==========

// 前置条件
#define XLLM_PRECOND(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Precondition failed: " message); \
        } \
    } while (false)

// 后置条件
#define XLLM_POSTCOND(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Postcondition failed: " message); \
        } \
    } while (false)

// 不变量
#define XLLM_INVARIANT(condition) \
    XLLM_POSTCOND(condition, "Invariant violated")

// Debug断言 (仅Debug模式生效)
#ifdef NDEBUG
#define XLLM_DCHECK(condition) ((void)0)
#define XLLM_DCHECK_EQ(a, b) ((void)0)
#else
#define XLLM_DCHECK(condition) \
    XLLM_PRECOND(condition, #condition)
#define XLLM_DCHECK_EQ(a, b) \
    XLLM_PRECOND((a) == (b), #a " == " #b)
#endif

// ========== 复杂验证 ==========

/**
 * @brief 验证KV Cache状态
 * @verification
 *   1. 所有分配的Block有valid的KV张量
 *   2. Block引用计数正确
 *   3. 序列到Block映射一致
 */
void KVCacheManager::verify_invariants() const {
    // 验证1: Block张量
    for (const auto& [block_id, block] : allocated_blocks_) {
        XLLM_INVARIANT(block->get_tensor(0).defined());
        XLLM_INVARIANT(block->get_tensor(1).defined());
    }
    
    // 验证2: 引用计数
    for (const auto& [seq_id, blocks] : sequence_to_blocks_) {
        int total_refs = 0;
        for (int64_t block_id : blocks) {
            total_refs += get_block(block_id)->ref_count();
        }
        XLLM_INVARIANT(total_refs >= 1);  // 至少被序列引用一次
    }
    
    // 验证3: 映射一致性
    for (const auto& [block_id, seq_id] : block_to_sequence_) {
        const auto& blocks = sequence_to_blocks_.at(seq_id);
        XLLM_INVARIANT(
            std::find(blocks.begin(), blocks.end(), block_id) != blocks.end()
        );
    }
}
```

### 4.3 测试用例标注

```cpp
// ========== AI可解析的测试用例 ==========

/**
 * @test BlockManager::allocate
 * @category unit
 * @priority high
 * @description 验证Block分配功能
 * 
 * @test_cases
 *   - name: "正常分配"
 *     input: {num_blocks: 10}
 *     expected: {size: 10, all_valid: true}
 *   
 *   - name: "分配全部"
 *     input: {num_blocks: 100}
 *     expected: {size: 100, free_count: 0}
 *   
 *   - name: "超额分配"
 *     input: {num_blocks: 150}
 *     expected: {throws: AllocationError}
 * 
 * @ai_generated: false
 * @ai_reviewed: true
 * @ai_coverage: branch=100%, path=critical
 */
TEST(BlockManager, Allocate) {
    BlockManager manager(/*total=*/100, shape, device);
    
    // Test case: 正常分配
    {
        auto blocks = manager.allocate(/*num=*/10);
        EXPECT_EQ(blocks.size(), 10);
        EXPECT_TRUE(std::all_of(blocks.begin(), blocks.end(),
            [&manager](int64_t id) { return manager.get_block(id) != nullptr; }));
    }
    
    // Test case: 分配全部
    {
        auto blocks = manager.allocate(/*num=*/100);
        EXPECT_EQ(blocks.size(), 100);
        EXPECT_EQ(manager.get_num_available_blocks(), 0);
    }
    
    // Test case: 超额分配
    {
        EXPECT_THROW(manager.allocate(/*num=*/150), AllocationError);
    }
}

/**
 * @test Scheduler::add_request
 * @category integration
 * @priority critical
 * @description 验证请求调度流程
 * 
 * @ai_test_scenarios
 *   - "多个并发请求"
 *   - "高优先级请求抢占"
 *   - "SLO超时场景"
 * 
 * @ai_performance_targets
 *   - latency_p50: < 1ms
 *   - throughput: > 10000 req/s
 */
TEST(SchedulerIntegration, AddRequest) {
    // ...
}
```

---

## 5. AI辅助代码审查

### 5.1 审查检查点

```yaml
ai_review_checklist:
  correctness:
    - name: "边界条件"
      description: "检查是否处理了所有边界情况"
      patterns:
        - "空指针"
        - "空容器"
        - "负数输入"
        - "溢出"
        
    - name: "状态一致性"
      description: "检查状态转换是否正确"
      patterns:
        - "锁的使用是否正确"
        - "原子变量的访问是否安全"
        
    - name: "资源管理"
      description: "检查是否有资源泄漏"
      patterns:
        - "内存分配/释放"
        - "文件句柄"
        - "GPU显存"
        
  performance:
    - name: "内存分配"
      description: "检查是否有不必要的内存分配"
      patterns:
        - "循环内分配"
        - "不必要的拷贝"
        
    - name: "计算复杂度"
      description: "检查算法复杂度是否最优"
      patterns:
        - "O(n^2) vs O(n)"
        - "重复计算"
        
    - name: "并行化"
      description: "检查是否可进一步并行化"
      patterns:
        - "串行循环"
        - "锁竞争"

  maintainability:
    - name: "代码重复"
      description: "检查是否有重复代码"
      patterns:
        - "copy-paste代码"
        - "魔法数字"
        
    - name: "依赖复杂度"
      description: "检查依赖关系是否清晰"
      patterns:
        - "循环依赖"
        - "过深调用栈"
```

### 5.2 AI审查提示

```cpp
// ========== 审查提示注释 ==========

/**
 * @brief 处理请求超时
 * 
 * @ai_review
 * @ai_review_concern: "可能的竞态条件"
 * @ai_review_suggestion: "考虑在锁内检查超时，而不是锁外"
 * 
 * @security_note: "需要验证超时值在合理范围内"
 */
void handle_timeout(Request* request) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // @ai_warning: "这里在锁外检查，可能导致竞态"
    // @ai_suggestion: "将超时检查移到锁内"
    if (is_expired(request)) {
        lock.unlock();  // @ai_nit: 不推荐手动unlock
        process_timeout(request);
    }
}

// @ai_refactor: "建议使用RAII自动管理锁"
void handle_timeout(Request* request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_expired(request)) {
        process_timeout(request);
    }
}
```

---

## 6. AI验收标准

### 6.1 代码质量指标

```yaml
ai_quality_metrics:
  # 圈复杂度
  cyclomatic_complexity:
    warning_threshold: 15
    error_threshold: 30
    measurement: "McCabe cyclomatic complexity"
    
  # 注释覆盖率
  comment_coverage:
    target: "> 30%"
    measurement: "lines_with_comments / total_lines"
    
  # 文档覆盖率
  documentation_coverage:
    public_api: "100%"
    complex_logic: "100%"
    measurement: "documented_items / total_items"
    
  # 测试覆盖率
  test_coverage:
    critical_path: "> 95%"
    normal_path: "> 80%"
    measurement: "covered_lines / total_lines"
    
  # 命名一致性
  naming_consistency:
    pattern: "PascalCase, camelCase, snake_case 混用"
    target: "100% 遵循项目规范"
```

### 6.2 AI验证清单

```yaml
ai_verification_checklist:
  before_submit:
    - "运行单元测试: `make test`"
    - "运行静态分析: `make lint`"
    - "检查性能基准: `make benchmark`"
    - "生成文档: `make doc`"
    - "AI代码审查: `ai-review --diff <commit>`"
    
  before_merge:
    - "所有CI测试通过"
    - "代码审查通过"
    - "AI验收标准满足"
    - "性能无退化"
    
  after_merge:
    - "更新相关文档"
    - "通知相关团队"
    - "记录变更日志"
```

### 6.3 自动化验证

```bash
#!/bin/bash
# xllm-ai-verify.sh - AI验收自动化脚本

set -e

echo "=== xLLM AI Verification ==="
echo ""

# 1. 语法检查
echo "[1/6] Running static analysis..."
cppcheck --enable=all --std=c++17 xllm/ 2>&1 | tee /tmp/cppcheck.log
if [ -s /tmp/cppcheck.log ]; then
    echo "Warning: Static analysis found issues"
fi

# 2. 格式化检查
echo "[2/6] Checking code format..."
clang-format --dry-run --Werror xllm/**/*.h xllm/**/*.cpp 2>&1 | tee /tmp/format.log
if [ -s /tmp/format.log ]; then
    echo "Error: Code not properly formatted"
    exit 1
fi

# 3. 单元测试
echo "[3/6] Running unit tests..."
ctest --output-on-failure -j 8
if [ $? -ne 0 ]; then
    echo "Error: Unit tests failed"
    exit 1
fi

# 4. 性能基准测试
echo "[4/6] Running performance benchmarks..."
./benchmark --suite=critical --threshold=0.95  # 性能退化不超过5%
if [ $? -ne 0 ]; then
    echo "Error: Performance regression detected"
    exit 1
fi

# 5. AI代码审查
echo "[5/6] Running AI code review..."
ai-review --model=gpt-4 --diff=HEAD --output=/tmp/ai-review.json
if [ -f /tmp/ai-review.json ]; then
    python3 parse_ai_review.py /tmp/ai-review.json
fi

# 6. 文档生成
echo "[6/6] Generating documentation..."
make doc
if [ $? -ne 0 ]; then
    echo "Warning: Documentation generation failed"
fi

echo ""
echo "=== Verification Complete ==="
```

---

## 7. 相关文档

| 文档 | 说明 |
|-----|------|
| [设计原则](./03_DESIGN_PRINCIPLES.md) | 设计原则与哲学 |
| [代码规范](./08_CODE_STANDARDS.md) | 详细编码规范 |
| [测试策略](./08_TEST_STRATEGY.md) | 测试策略 |
| [验收标准](./08_ACCEPTANCE_CRITERIA.md) | AI验收标准 |
