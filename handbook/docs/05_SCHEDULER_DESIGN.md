# xLLM Scheduler 设计详解

## 文档信息

```yaml
---
document_id: DESIGN-SCHEDULER-001
version: 1.0.0
category: component_design
owner: xllm-team
verification_level: BOTH
depends_on:
  - DESIGN-001  # Domain Model
---
```

---

## 1. Scheduler 架构概览

### 1.1 调度器家族

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Scheduler Family                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────────┐                                 │
│                         │  SchedulerBase  │                                 │
│                         │   (抽象基类)     │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                           │
│         ┌────────────────────────┼────────────────────────┐                   │
│         │                        │                        │                   │
│         ▼                        ▼                        ▼                   │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐            │
│  │  Scheduler  │        │ DitScheduler│        │  (预留扩展)  │            │
│  │ (通用调度器) │        │ (DIT专用)   │        │             │            │
│  └──────┬──────┘        └─────────────┘        └─────────────┘            │
│          │                                                                │
│          │                                                                │
│          ▼                                                                │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                        SchedulerImpl                               │     │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐│     │
│  │  │ Continuous     │ │ DisaggPD       │ │ ZeroEviction          ││     │
│  │  │ Scheduler      │ │ Scheduler      │ │ Scheduler             ││     │
│  │  │                │ │                │ │                        ││     │
│  │  │ - 在线服务      │ │ - P/D分离部署  │ │ - 资源受限场景        ││     │
│  │  │ - 高吞吐低延迟  │ │ - Prefill节点  │ │ - 防止重要请求被驱逐  ││     │
│  │  │ - 动态批处理    │ │ - Decode节点   │ │                        ││     │
│  │  └────────────────┘ └────────────────┘ └────────────────────────┘│     │
│  │  ┌────────────────┐ ┌────────────────┐                            │     │
│  │  │ FixedSteps     │ │ PDooc          │                            │     │
│  │  │ Scheduler      │ │ Scheduler      │                            │     │
│  │  │                │ │                │                            │     │
│  │  │ - 离线批处理   │ │ - P/D+OOC组合 │                            │     │
│  │  │ - 固定步数生成 │ │ - 超长序列处理 │                            │     │
│  │  └────────────────┘ └────────────────┘                            │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心类图

```cpp
// 代码位置: xllm/core/scheduler/scheduler.h

// 调度器基类
class SchedulerBase {
 public:
    virtual ~SchedulerBase() = default;
    
    // 执行调度
    virtual void step(const absl::Duration& timeout) = 0;
    
    // 离线生成
    virtual void generate() = 0;
    
    // 待处理请求数管理
    virtual void incr_pending_requests(size_t count) {}
    virtual void decr_pending_requests() {}
    virtual size_t num_pending_requests() { return 0; }
};

// 通用调度器接口
class Scheduler : public SchedulerBase {
 public:
    virtual ~Scheduler() = default;
    
    // 添加请求
    virtual bool add_request(std::shared_ptr<Request>& request) = 0;
    
    // 获取等待数
    virtual uint32_t get_waiting_requests_num() const = 0;
    
    // 获取延迟指标
    virtual void get_latency_metrics(std::vector<int64_t>& ttft,
                                    std::vector<int64_t>& tbt) = 0;
    
    // 获取实例信息
    virtual const InstanceInfo& get_instance_info() = 0;
    
    // 重置前缀缓存
    virtual void reset_prefix_cache() {}
};
```

---

## 2. Continuous Scheduler 详解

### 2.1 设计目标

| 目标 | 说明 | 优先级 |
|-----|------|-------|
| 高吞吐 | 最大化GPU利用率 | P0 |
| 低延迟 | 最小化TTFT和TPOT | P0 |
| SLO保障 | 满足TTFT/TPOT/TTLT SLO | P1 |
| 公平性 | 防止饥饿 | P1 |

**代码位置**: `xllm/core/scheduler/continuous_scheduler.h`

### 2.2 核心实现

```cpp
// continuous_scheduler.h
class ContinuousScheduler : public Scheduler {
 public:
    ContinuousScheduler(
        std::shared_ptr<SchedulerPolicy> policy,
        std::shared_ptr<BatchManager> batch_manager,
        const SchedulerOptions& options);
    
    // ========== Scheduler 接口实现 ==========
    
    bool add_request(std::shared_ptr<Request>& request) override;
    uint32_t get_waiting_requests_num() const override;
    void step(const absl::Duration& timeout) override;
    void get_latency_metrics(std::vector<int64_t>& ttft,
                            std::vector<int64_t>& tbt) override;
    const InstanceInfo& get_instance_info() override;
    
 private:
    // ========== 内部方法 ==========
    
    /**
     * @brief 调度主循环
     */
    void schedule();
    
    /**
     * @brief 填充预填充批次
     */
    void fill_prefill_batch(Batch& batch);
    
    /**
     * @brief 填充解码批次
     */
    void fill_decode_batch(Batch& batch);
    
    /**
     * @brief 计算调度优先级
     */
    Urgency calculate_urgency(const Request& request);
    
    /**
     * @brief 尝试前缀缓存匹配
     */
    bool try_prefix_cache(std::shared_ptr<Request>& request);
    
    // ========== 状态 ==========
    
    std::shared_ptr<SchedulerPolicy> policy_;
    std::shared_ptr<BatchManager> batch_manager_;
    SchedulerOptions options_;
    
    // 请求队列
    PriorityQueue waiting_requests_;    // 等待调度的请求
    PriorityQueue running_requests_;     // 正在运行的请求
    std::deque<std::shared_ptr<Request>> finished_requests_;  // 已完成的请求
    
    // 统计
    size_t total_prefill_tokens_ = 0;
    size_t total_decode_tokens_ = 0;
};
```

### 2.3 调度流程

```cpp
// continuous_scheduler.cpp

void ContinuousScheduler::step(const absl::Duration& timeout) {
    auto start = absl::Now();
    
    while (absl::Now() - start < timeout) {
        // 1. 处理新到达的请求
        process_arrived_requests();
        
        // 2. 检查完成的请求
        process_finished_requests();
        
        // 3. 执行调度
        schedule();
        
        // 4. 检查是否有批次需要执行
        if (has_ready_batch()) {
            execute_batch();
        }
    }
}

void ContinuousScheduler::schedule() {
    // 创建批次
    auto batch = batch_manager_->create_batch();
    
    // 优先级: 饥饿请求 > 紧急请求 > 正常请求
    auto starved_requests = waiting_requests_.get_starved();
    for (auto& req : starved_requests) {
        batch.add(req);
    }
    
    if (batch.empty()) {
        auto urgent_requests = waiting_requests_.get_urgent();
        for (auto& req : urgent_requests) {
            if (batch.can_add(req)) {
                batch.add(req);
            }
        }
    }
    
    if (batch.empty()) {
        // 填充策略: 优先填入预填充请求
        fill_prefill_batch(batch);
    }
    
    if (batch.empty()) {
        // 填充解码请求
        fill_decode_batch(batch);
    }
    
    // 更新批次的前向类型
    batch.refresh_forward_type();
}

void ContinuousScheduler::fill_prefill_batch(Batch& batch) {
    // 获取所有待预填充的请求
    auto prefill_candidates = waiting_requests_.get_pending_prefill();
    
    // 按长度排序（短在前，优先完成）
    sort_by_length(prefill_candidates, /* ascending */ true);
    
    size_t total_tokens = 0;
    for (auto& req : prefill_candidates) {
        size_t prompt_tokens = req->num_prompt_tokens();
        
        // 检查是否超过批处理限制
        if (total_tokens + prompt_tokens > options_.max_prefill_tokens) {
            break;
        }
        
        // 检查是否能分配KV Cache
        if (!can_allocate_kv_cache(req)) {
            continue;
        }
        
        batch.add(req.get());
        total_tokens += prompt_tokens;
        waiting_requests_.mark_running(req);
    }
}

void ContinuousScheduler::fill_decode_batch(Batch& batch) {
    // 获取所有正在解码的请求
    auto decode_candidates = running_requests_.get_decoding();
    
    // 按优先级和SLO紧迫度排序
    sort_by_slo_urgency(decode_candidates);
    
    size_t max_decode_tokens = options_.max_decode_tokens;
    size_t used_tokens = 0;
    
    for (auto& req : decode_candidates) {
        size_t decode_tokens = req->num_decode_tokens();
        
        if (used_tokens + decode_tokens > max_decode_tokens) {
            break;
        }
        
        batch.add(req.get());
        used_tokens += decode_tokens;
    }
}
```

### 2.4 优先级计算

```cpp
// 优先级计算逻辑
Urgency ContinuousScheduler::calculate_urgency(const Request& request) {
    int32_t remaining = request.get_remaining_time();
    
    // SLO违规风险检查
    if (remaining <= 0) {
        return Urgency::STARVED;  // 已超时，饥饿
    }
    
    // 获取各SLO的剩余时间
    int32_t ttft_remaining = request.ttft_slo_ms() - request.get_elapsed_time_ms();
    int32_t tpot_remaining = request.tpot_slo_ms();
    
    // TTFT 紧迫
    if (ttft_remaining <= 0 && request.num_tokens() == request.num_prompt_tokens()) {
        return Urgency::STARVED;
    }
    
    // TPOT 紧迫
    if (remaining <= tpot_remaining * 2) {
        return Urgency::URGENT;
    }
    
    return Urgency::NORMAL;
}

// 请求打分用于优先级队列排序
int64_t score_request(const Request& request) {
    int32_t remaining = request.get_remaining_time();
    int32_t priority_weight = request.ttft_priority_weight();
    
    // 分数 = 剩余时间 * 优先级权重
    // 分数越低，优先级越高
    return remaining / priority_weight;
}
```

---

## 3. DisaggPD Scheduler 详解

### 3.1 设计背景

P/D分离架构：将Prefill和Decode分离到不同节点，实现独立扩缩容。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Prefill-Decode Disaggregation                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Client                                                                   │
│     │                                                                       │
│     │  Request                                                              │
│     ▼                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      PD Scheduler                                      │  │
│   │  ┌────────────────┐         ┌────────────────┐                       │  │
│   │  │  Prefill Node  │ ←─────→ │  Decode Node   │                       │  │
│   │  │                │  KV     │                │                       │  │
│   │  │  • 高吞吐计算  │  Cache  │  • 低延迟服务   │                       │  │
│   │  │  • 批处理优化  │  Transfer│  • Streaming   │                       │  │
│   │  │  • 长序列处理  │         │  • 小Batch     │                       │  │
│   │  └────────────────┘         └────────────────┘                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**代码位置**: `xllm/core/scheduler/disagg_pd_scheduler.h`

### 3.2 核心实现

```cpp
// disagg_pd_scheduler.h
class DisaggPDScheduler : public Scheduler {
 public:
    DisaggPDScheduler(
        std::shared_ptr<PDSchedulerPolicy> policy,
        std::shared_ptr<PDClient> pd_client,
        const PDSchedulerOptions& options);
    
    // ========== 调度决策 ==========
    
    /**
     * @brief 决定请求应该在Prefill还是Decode节点执行
     */
    ExecutionTarget decide_execution_target(const Request& request);
    
    /**
     * @brief 发送请求到目标节点
     */
    bool forward_to_target(std::shared_ptr<Request>& request,
                          ExecutionTarget target);
    
    /**
     * @brief 触发KV Cache传输
     */
    void trigger_kv_transfer(const Request& request,
                           const PrefillOutput& prefill_result);

 private:
    // Prefill节点调度
    void schedule_prefill();
    
    // Decode节点调度
    void schedule_decode();
    
    // 监控KV传输进度
    void monitor_kv_transfer();
    
    // ========== 状态 ==========
    
    std::shared_ptr<PDClient> pd_client_;
    PDSchedulerOptions options_;
    
    // Prefill队列
    std::deque<std::shared_ptr<Request>> prefill_queue_;
    
    // Decode队列
    std::deque<std::shared_ptr<Request>> decode_queue_;
    
    // KV传输中
    std::unordered_map<std::string, KVTransferState> transfers_inflight_;
};
```

### 3.3 KV传输流程

```cpp
// KV Cache传输流程
void DisaggPDScheduler::trigger_kv_transfer(
    const Request& request,
    const PrefillOutput& prefill_result) {
    
    // 1. 从Prefill结果提取KV Cache信息
    auto kv_blocks = prefill_result.get_kv_blocks();
    auto kv_shape = prefill_result.get_kv_shape();
    
    // 2. 构建传输请求
    BlockTransferInfo transfer_info;
    transfer_info.src_blocks = kv_blocks;
    transfer_info.num_tokens = request.num_tokens();
    transfer_info.kv_shape = kv_shape;
    
    // 3. 发送到Decode节点
    std::string decode_addr = get_decode_node_addr(request);
    
    // 4. 异步传输
    auto future = pd_client_->transfer_kv_blocks(decode_addr, transfer_info);
    
    // 5. 记录传输状态
    transfers_inflight_[request.request_id()] = {
        .future = std::move(future),
        .start_time = absl::Now(),
        .num_blocks = kv_blocks.size()
    };
}
```

---

## 4. Scheduler Policy

### 4.1 策略基类

```cpp
// 代码位置: xllm/core/scheduler/scheduler_policy.h

/**
 * @class SchedulerPolicy
 * @brief 调度策略基类 - 封装调度决策逻辑
 * 
 * @design 策略模式，允许运行时切换调度策略
 */
class SchedulerPolicy {
 public:
    virtual ~SchedulerPolicy() = default;
    
    // ========== 批次组成策略 ==========
    
    /**
     * @brief 计算最佳批大小
     */
    virtual BatchConfig compute_batch_config(
        const std::vector<Request*>& candidates,
        const SystemResources& resources) = 0;
    
    /**
     * @brief 排序候选请求
     */
    virtual std::vector<Request*> prioritize_requests(
        const std::vector<Request*>& candidates) = 0;
    
    // ========== 前缀缓存策略 ==========
    
    /**
     * @brief 决定是否使用前缀缓存
     */
    virtual bool should_use_prefix_cache(const Request& request);
    
    // ========== 抢占策略 ==========
    
    /**
     * @brief 决定是否需要抢占
     */
    virtual bool should_preempt(const std::vector<Request*>& running,
                              const Request& incoming);
};

// 具体策略
class PrefillFirstPolicy : public SchedulerPolicy {
    // 优先调度预填充请求
};

class DecodeFirstPolicy : public SchedulerPolicy {
    // 优先调度解码请求
};

class UnifiedPolicy : public SchedulerPolicy {
    // 统一调度，动态平衡
};
```

### 4.2 统一策略实现

```cpp
// unified_policy.cpp
class UnifiedPolicy : public SchedulerPolicy {
 public:
    BatchConfig compute_batch_config(
        const std::vector<Request*>& candidates,
        const SystemResources& resources) override {
        
        BatchConfig config;
        
        // 1. 估算各请求类型的计算量
        size_t prefill_load = estimate_prefill_load(candidates);
        size_t decode_load = estimate_decode_load(candidates);
        
        // 2. 动态调整批组成
        if (prefill_load > decode_load * 2) {
            // Prefill密集型场景
            config.type = BatchType::PREFILL_HEAVY;
            config.max_prefill_tokens = resources.max_compute;
            config.max_decode_tokens = resources.max_compute / 4;
        } else if (decode_load > prefill_load * 2) {
            // Decode密集型场景
            config.type = BatchType::DECODE_HEAVY;
            config.max_decode_tokens = resources.max_compute;
            config.max_prefill_tokens = resources.max_compute / 8;
        } else {
            // 混合场景
            config.type = BatchType::MIXED;
            config.max_prefill_tokens = resources.max_compute / 2;
            config.max_decode_tokens = resources.max_compute / 2;
        }
        
        // 3. 考虑SLO
        adjust_for_slo(config, candidates);
        
        return config;
    }
    
    std::vector<Request*> prioritize_requests(
        const std::vector<Request*>& candidates) override {
        
        // 按SLO紧迫度分组
        std::vector<Request*> starved, urgent, normal;
        
        for (auto* req : candidates) {
            switch (calculate_urgency(*req)) {
                case Urgency::STARVED:
                    starved.push_back(req);
                    break;
                case Urgency::URGENT:
                    urgent.push_back(req);
                    break;
                default:
                    normal.push_back(req);
            }
        }
        
        // 合并结果
        std::vector<Request*> result;
        result.insert(result.end(), starved.begin(), starved.end());
        result.insert(result.end(), urgent.begin(), urgent.end());
        
        // normal按SLO紧迫度排序
        sort_by_deadline(normal);
        result.insert(result.end(), normal.begin(), normal.end());
        
        return result;
    }
};
```

---

## 5. 调度器工厂

### 5.1 工厂模式

```cpp
// 代码位置: xllm/core/scheduler/scheduler_factory.h

/**
 * @class SchedulerFactory
 * @brief 调度器工厂 - 根据配置创建调度器
 */
class SchedulerFactory {
 public:
    /**
     * @brief 创建调度器
     * @param type 调度器类型
     * @param options 调度选项
     * @param policy 调度策略
     * @return 创建的调度器
     */
    static std::unique_ptr<Scheduler> create(
        SchedulerType type,
        const SchedulerOptions& options,
        std::shared_ptr<SchedulerPolicy> policy = nullptr);
    
    /**
     * @brief 创建默认策略
     */
    static std::shared_ptr<SchedulerPolicy> create_default_policy(
        SchedulerType type);
};

// 工厂注册
namespace {
    RegisterSchedulerFactory registration("continuous", [](auto&&... args) {
        return std::make_unique<ContinuousScheduler>(std::forward<decltype(args)>(args)...);
    });
    
    RegisterSchedulerFactory registration("disagg_pd", [](auto&&... args) {
        return std::make_unique<DisaggPDScheduler>(std::forward<decltype(args)>(args)...);
    });
}
```

### 5.2 使用示例

```cpp
// 创建调度器
auto options = SchedulerOptions::from_json(config);
auto policy = SchedulerFactory::create_default_policy(options.type);
auto scheduler = SchedulerFactory::create(options.type, options, policy);

// 使用调度器
while (running) {
    scheduler->step(absl::Seconds(1));
}
```

---

## 6. 性能指标

### 6.1 关键指标

| 指标 | 说明 | 目标 |
|-----|------|-----|
| TTFT | Time To First Token | P50 < 100ms |
| TPOT | Time Per Output Token | P50 < 20ms |
| TTLT | Time To Last Token | P99 < SLO |
| GPU Utilization | GPU利用率 | > 85% |
| Batch Size | 批大小 | 动态最优 |
| Queue Length | 等待队列长度 | < 1000 |

### 6.2 指标收集

```cpp
// 指标收集
struct SchedulerMetrics {
    // 延迟指标
    std::vector<int64_t> ttft_samples;
    std::vector<int64_t> tpot_samples;
    std::vector<int64_t> ttlt_samples;
    
    // 吞吐量指标
    size_t total_requests = 0;
    size_t total_tokens = 0;
    size_t total_prefill_tokens = 0;
    size_t total_decode_tokens = 0;
    
    // 资源指标
    double avg_batch_size = 0;
    double gpu_utilization = 0;
    size_t queue_size = 0;
    
    // SLO合规率
    double ttft_slo_compliance = 0;
    double tpot_slo_compliance = 0;
    double ttlt_slo_compliance = 0;
};

void ContinuousScheduler::get_latency_metrics(
    std::vector<int64_t>& ttft,
    std::vector<int64_t>& tbt) {
    
    ttft.clear();
    tbt.clear();
    
    for (const auto& req : finished_requests_) {
        ttft.push_back(req->time_to_first_token_ms());
        
        const auto& tokens = req->get_token_ids();
        for (size_t i = 1; i < tokens.size(); ++i) {
            tbt.push_back(/* token interval */);
        }
    }
}
```

---

## 7. AI验收标准

### 7.1 功能验证

```yaml
ai_verification:
  scheduler:
    - name: "请求调度正确性"
      test: |
        1. 提交多个请求，验证都被调度执行
        2. 验证优先级排序正确
        3. 验证SLO时间正确追踪
        
    - name: "批处理正确性"
      test: |
        1. 验证批次大小符合限制
        2. 验证前向类型正确设置
        3. 验证Batch ID唯一性
        
    - name: "前缀缓存"
      test: |
        1. 提交相同系统提示的多个请求
        2. 验证前缀被正确复用
        3. 验证引用计数正确
```

### 7.2 性能验证

```cpp
// 性能测试用例
TEST(ContinuousScheduler, ThroughputBenchmark) {
    auto scheduler = create_scheduler();
    const int num_requests = 10000;
    const int tokens_per_request = 512;
    
    auto start = absl::Now();
    
    // 提交请求
    for (int i = 0; i < num_requests; ++i) {
        auto request = create_request(tokens_per_request);
        scheduler->add_request(request);
    }
    
    // 等待完成
    while (scheduler->num_pending_requests() > 0) {
        scheduler->step(absl::Seconds(1));
    }
    
    auto elapsed = absl::Now() - start;
    double throughput = num_requests / absl::ToDoubleSeconds(elapsed);
    
    // 验证吞吐量
    EXPECT_GT(throughput, 100);  // 至少100 req/s
}
```

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [领域模型](./04_DOMAIN_MODEL.md) | DDD领域划分 |
| [Batch设计](./06_BATCH_DESIGN.md) | Batch实现详解 |
| [Worker设计](./07_WORKER_DESIGN.md) | Worker实现详解 |
| [API设计](./09_API_DESIGN.md) | 调度器API |
