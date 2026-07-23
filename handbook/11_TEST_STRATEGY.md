# xLLM 测试策略

## 文档信息

```yaml
---
document_id: TEST-001
version: 1.0.0
category: testing
owner: xllm-team
verification_level: BOTH
---
```

---

## 1. 测试金字塔

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Testing Pyramid                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                                    ▲                                         │
│                                   /E2E\                                      │
│                                  /     \                                     │
│                                 / 集成  \                                    │
│                                /   测试   \                                  │
│                               /────────────\                                 │
│                              /    组件      \                                │
│                             /     测试       \                               │
│                            /──────────────────\                              │
│                           /      单元          \                             │
│                          /       测试           \                            │
│                         /────────────────────────\                           │
│                                                                              │
│   比例:  单元测试 70% | 组件测试 20% | 集成测试 8% | E2E测试 2%               │
│                                                                              │
│   速度:  单元测试 ms  | 组件测试 ms-s | 集成测试 s-m | E2E测试 m-h            │
│                                                                              │
│   成本:  单元测试 $   | 组件测试 $$ | 集成测试 $$$ | E2E测试 $$$$             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 单元测试

### 2.1 目录结构

```bash
tests/
├── unit/                          # 单元测试
│   ├── core/
│   │   ├── framework/
│   │   │   ├── batch/            # Batch单元测试
│   │   │   │   ├── batch_test.cpp
│   │   │   │   └── batch_forward_test.cpp
│   │   │   ├── block/            # Block管理器测试
│   │   │   │   ├── block_manager_test.cpp
│   │   │   │   ├── single_block_manager_test.cpp
│   │   │   │   └── concurrent_block_manager_test.cpp
│   │   │   ├── kv_cache/         # KV Cache测试
│   │   │   │   ├── kv_cache_test.cpp
│   │   │   │   └── kv_cache_shape_test.cpp
│   │   │   ├── request/          # Request测试
│   │   │   │   ├── request_test.cpp
│   │   │   │   ├── sequence_test.cpp
│   │   │   │   └── sequences_group_test.cpp
│   │   │   ├── scheduler/        # 调度器测试
│   │   │   │   ├── continuous_scheduler_test.cpp
│   │   │   │   └── scheduler_policy_test.cpp
│   │   │   └── model/            # 模型测试
│   │   │       └── model_args_test.cpp
│   │   ├── runtime/              # 运行时测试
│   │   │   ├── worker_test.cpp
│   │   │   └── executor_test.cpp
│   │   └── common/               # 公共组件测试
│   │       ├── options_test.cpp
│   │       └── rate_limiter_test.cpp
│   ├── api_service/              # API服务测试
│   │   ├── openai_service_test.cpp
│   │   └── anthropic_service_test.cpp
│   └── utils/                    # 工具测试
│       ├── threadpool_test.cpp
│       └── tensor_utils_test.cpp
│
├── component/                     # 组件测试
│   ├── scheduler_integration_test.cpp
│   ├── worker_integration_test.cpp
│   └── kv_cache_integration_test.cpp
│
├── integration/                   # 集成测试
│   ├── end_to_end_test.cpp
│   └── model_loading_test.cpp
│
└── performance/                   # 性能测试
    ├── benchmark_test.cpp
    └── throughput_test.cpp
```

### 2.2 单元测试框架

```cpp
// 代码位置: tests/core/framework/block/block_manager_test.cpp

#include <gtest/gtest.h>
#include "framework/block/block_manager.h"
#include "framework/kv_cache/kv_cache_shape.h"

/**
 * @test BlockManager - 核心功能测试
 * @category unit
 * @priority high
 */

// ========== Fixtures ==========

class BlockManagerTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // 创建测试配置
        shape_ = KVCacheShape{
            .num_heads = 32,
            .num_kv_heads = 8,
            .head_dim = 128,
            .block_seq_len = 16,
            .num_layers = 32
        };
        
        manager_ = std::make_unique<BlockManager>(
            /*num_blocks=*/100,
            shape_,
            torch::kCUDA
        );
    }
    
    std::unique_ptr<BlockManager> manager_;
    KVCacheShape shape_;
};

// ========== 功能测试 ==========

/**
 * @brief 测试正常分配
 * @test_cases
 *   - name: "分配单个Block"
 *     input: {num_blocks: 1}
 *     expected: {size: 1}
 *   
 *   - name: "分配多个Block"
 *     input: {num_blocks: 10}
 *     expected: {size: 10, no_overlap: true}
 *   
 *   - name: "分配全部"
 *     input: {num_blocks: 100}
 *     expected: {size: 100, available: 0}
 */
TEST_F(BlockManagerTest, Allocate) {
    // Test: 分配单个Block
    {
        auto blocks = manager_->allocate(/*num=*/1);
        EXPECT_EQ(blocks.size(), 1);
        EXPECT_GE(blocks[0], 0);
        EXPECT_LT(blocks[0], 100);
    }
    
    // Test: 分配多个Block
    {
        auto blocks = manager_->allocate(/*num=*/10);
        EXPECT_EQ(blocks.size(), 10);
        
        // 检查无重叠
        std::set<int64_t> unique_blocks(blocks.begin(), blocks.end());
        EXPECT_EQ(unique_blocks.size(), 10);
    }
    
    // Test: 分配全部
    {
        auto blocks = manager_->allocate(/*num=*/100);
        EXPECT_EQ(blocks.size(), 100);
        EXPECT_EQ(manager_->get_num_available_blocks(), 0);
    }
}

/**
 * @brief 测试超额分配
 * @test_cases
 *   - name: "超额分配"
 *     input: {num_blocks: 101}
 *     expected: {throws: std::runtime_error}
 */
TEST_F(BlockManagerTest, AllocateExceed) {
    EXPECT_THROW(manager_->allocate(/*num=*/101), std::runtime_error);
}

/**
 * @brief 测试释放
 * @test_cases
 *   - name: "释放后重新分配"
 *     input: {allocate: 10, free: [0-9], reallocate: 10}
 *     expected: {same_blocks: true}
 */
TEST_F(BlockManagerTest, FreeAndReallocate) {
    // 分配10个Block
    auto blocks1 = manager_->allocate(/*num=*/10);
    
    // 释放全部
    manager_->free(blocks1);
    EXPECT_EQ(manager_->get_num_available_blocks(), 100);
    
    // 重新分配应该得到相同的Block ID
    auto blocks2 = manager_->allocate(/*num=*/10);
    EXPECT_EQ(blocks2, blocks1);
}

/**
 * @brief 测试序列Block管理
 */
TEST_F(BlockManagerTest, SequenceBlocks) {
    const int64_t seq_id = 12345;
    
    // 分配给序列
    auto blocks = manager_->allocate(/*num=*/5);
    for (int64_t block_id : blocks) {
        manager_->get_block(block_id)->mark_allocated(seq_id);
    }
    
    // 查询序列的Blocks
    auto seq_blocks = manager_->get_sequence_blocks(seq_id);
    EXPECT_EQ(seq_blocks.size(), 5);
    
    // 释放序列
    manager_->free_sequence(seq_id);
    EXPECT_EQ(manager_->get_num_available_blocks(), 100);
}

/**
 * @brief 测试引用计数 (Prefix Cache)
 * @complexity O(1)
 */
TEST_F(BlockManagerTest, ReferenceCount) {
    auto block = manager_->get_block(0);
    
    // 初始引用计数为1
    EXPECT_EQ(block->ref_count(), 1);
    
    // 添加引用
    block->add_ref();
    EXPECT_EQ(block->ref_count(), 2);
    
    // 释放引用
    block->release_ref();
    EXPECT_EQ(block->ref_count(), 1);
    
    // 释放到0，应该被释放
    block->release_ref();
    EXPECT_FALSE(block->is_allocated());
}
```

### 2.3 Mock框架

```cpp
// 代码位置: tests/mocks/mock_scheduler.h

/**
 * @class MockScheduler
 * @brief Scheduler Mock - 用于测试
 */
class MockScheduler : public Scheduler {
 public:
    MOCK_METHOD(bool, add_request, (std::shared_ptr<Request>&), (override));
    MOCK_METHOD(uint32_t, get_waiting_requests_num, (), (const override));
    MOCK_METHOD(void, step, (const absl::Duration&), (override));
    MOCK_METHOD(void, get_latency_metrics, 
                (std::vector<int64_t>&, std::vector<int64_t>&), (override));
    
    // 可配置的默认行为
    void ConfigureDefaultBehavior() {
        ON_CALL(*this, add_request)
            .WillByDefault(Return(true));
        ON_CALL(*this, get_waiting_requests_num)
            .WillByDefault(Return(0));
    }
};

// ========== 使用Mock ==========

TEST(SchedulerTest, AddRequestCallsPolicy) {
    auto mock_policy = std::make_unique<MockPolicy>();
    auto scheduler = std::make_unique<ContinuousScheduler>(std::move(mock_policy));
    
    auto request = std::make_shared<Request>(...);
    
    EXPECT_CALL(*mock_policy, can_schedule)
        .Times(1);
    
    scheduler->add_request(request);
}
```

---

## 3. 组件测试

### 3.1 Scheduler组件测试

```cpp
// 代码位置: tests/component/scheduler_integration_test.cpp

/**
 * @test Scheduler组件集成测试
 * @category component
 * @priority high
 * @hardware_required cuda
 */

class SchedulerIntegrationTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // 创建测试组件
        config_ = SchedulerConfig{
            .max_batch_size = 64,
            .max_prefill_tokens = 8192,
            .max_decode_tokens = 4096
        };
        
        policy_ = std::make_unique<UnifiedPolicy>();
        batch_manager_ = std::make_unique<BatchManager>(/*capacity=*/10000);
        scheduler_ = std::make_unique<ContinuousScheduler>(
            std::move(policy_),
            std::move(batch_manager_),
            config_
        );
    }
    
    std::unique_ptr<ContinuousScheduler> scheduler_;
    SchedulerConfig config_;
};

/**
 * @brief 测试多优先级请求调度
 * @complexity O(n log n)
 */
TEST_F(SchedulerIntegrationTest, PriorityScheduling) {
    // 创建不同优先级的请求
    std::vector<std::shared_ptr<Request>> requests;
    
    for (int i = 0; i < 10; ++i) {
        auto req = std::make_shared<Request>(...);
        req->set_urgency(Urgency::NORMAL);
        requests.push_back(req);
    }
    
    // 添加高优先级请求
    auto urgent_req = std::make_shared<Request>(...);
    urgent_req->set_urgency(Urgency::STARVED);
    scheduler_->add_request(urgent_req);
    
    for (auto& req : requests) {
        scheduler_->add_request(req);
    }
    
    // 执行调度
    scheduler_->step(absl::Seconds(1));
    
    // 验证高优先级请求被优先调度
    auto metrics = scheduler_->get_latency_metrics(...);
    // ...
}

/**
 * @brief 测试前缀缓存命中
 */
TEST_F(SchedulerIntegrationTest, PrefixCacheHit) {
    // 创建共享前缀的请求
    const std::vector<int> shared_prefix = {1, 2, 3, 4, 5};
    
    auto req1 = create_request_with_prefix(shared_prefix, {6, 7, 8});
    auto req2 = create_request_with_prefix(shared_prefix, {9, 10, 11});
    auto req3 = create_request_with_prefix(shared_prefix, {12, 13, 14});
    
    scheduler_->add_request(req1);
    scheduler_->step(absl::Seconds(1));
    
    // req1完成后，req2和req3应该命中前缀缓存
    scheduler_->add_request(req2);
    scheduler_->add_request(req3);
    scheduler_->step(absl::Seconds(1));
    
    // 验证前缀缓存命中
    // 第二次调用的prefill_tokens应该更少
}

/**
 * @brief 测试SLO追踪
 */
TEST_F(SchedulerIntegrationTest, SLOPriority) {
    // 创建即将超时的请求
    auto urgent_req = std::make_shared<Request>(...);
    urgent_req->set_deadline_ms(/*remaining=*/50);  // 50ms剩余
    
    // 正常请求
    auto normal_req = std::make_shared<Request>(...);
    normal_req->set_deadline_ms(/*remaining=*/5000);  // 5s剩余
    
    scheduler_->add_request(normal_req);
    scheduler_->add_request(urgent_req);
    
    // 执行调度
    scheduler_->step(absl::Seconds(1));
    
    // 验证紧急请求被优先处理
    auto batches = scheduler_->get_last_batch();
    EXPECT_TRUE(contains_request(batches, urgent_req->request_id()));
}
```

---

## 4. 集成测试

### 4.1 端到端测试

```cpp
// 代码位置: tests/integration/end_to_end_test.cpp

/**
 * @test 端到端推理测试
 * @category integration
 * @priority critical
 * @model_required qwen2-7b
 */

class EndToEndTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // 启动测试服务器
        server_ = std::make_unique<TestServer>();
        server_->start();
        
        // 创建客户端
        client_ = std::make_unique<xLLMClient>(
            server_->url(),
            /*api_key=*/""
        );
    }
    
    void TearDown() override {
        server_->stop();
    }
    
    std::unique_ptr<TestServer> server_;
    std::unique_ptr<xLLMClient> client_;
};

/**
 * @brief 测试基本Chat Completion
 */
TEST_F(EndToEndTest, BasicChatCompletion) {
    auto response = client_->chat_completion(
        model="qwen2-7b",
        messages={
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"}
        },
        max_tokens=100
    );
    
    EXPECT_EQ(response.model, "qwen2-7b");
    EXPECT_FALSE(response.choices.empty());
    EXPECT_GT(response.choices[0].message["content"].length(), 0);
}

/**
 * @brief 测试流式输出
 */
TEST_F(EndToEndTest, StreamingChatCompletion) {
    std::string full_content;
    int chunk_count = 0;
    
    for (auto chunk : client_->chat_completion(
        model="qwen2-7b",
        messages={{"role": "user", "content": "讲个故事"}},
        max_tokens=200,
        stream=true
    )) {
        if (chunk.choices[0].delta.count("content")) {
            full_content += chunk.choices[0].delta["content"];
        }
        chunk_count++;
    }
    
    EXPECT_GT(chunk_count, 0);
    EXPECT_GT(full_content.length(), 0);
}

/**
 * @brief 测试并发请求
 */
TEST_F(EndToEndTest, ConcurrentRequests) {
    const int num_requests = 100;
    std::vector<std::future<ChatCompletion>> futures;
    
    for (int i = 0; i < num_requests; ++i) {
        futures.push_back(std::async(std::launch::async, [this, i] {
            return client_->chat_completion(
                model="qwen2-7b",
                messages={{"role": "user", "content": fmt::format("Query {}", i)}},
                max_tokens=50
            );
        }));
    }
    
    int success_count = 0;
    for (auto& f : futures) {
        try {
            auto response = f.get();
            if (!response.choices.empty()) {
                success_count++;
            }
        } catch (const std::exception& e) {
            // 记录错误但不失败
        }
    }
    
    EXPECT_GE(success_count, num_requests * 0.95);  // 95%成功率
}

/**
 * @brief 测试SLO合规
 */
TEST_F(EndToEndTest, SLOPerformance) {
    const int num_requests = 1000;
    
    std::vector<int64_t> ttft_samples;
    std::vector<int64_t> tpot_samples;
    
    for (int i = 0; i < num_requests; ++i) {
        auto start = absl::Now();
        
        auto response = client_->chat_completion(
            model="qwen2-7b",
            messages={{"role": "user", "content": "测试"}},
            max_tokens=100,
            extra_body={
                {"ttft_slo_ms", 1000},
                {"tpot_slo_ms", 100}
            }
        );
        
        auto elapsed = absl::ToInt64Milliseconds(absl::Now() - start);
        
        if (response.xllm_extra.count("ttft_ms")) {
            ttft_samples.push_back(response.xllm_extra["ttft_ms"]);
        }
    }
    
    // 计算P50/P95/P99
    std::sort(ttft_samples.begin(), ttft_samples.end());
    int64_t p50 = ttft_samples[num_requests * 0.5];
    int64_t p95 = ttft_samples[num_requests * 0.95];
    int64_t p99 = ttft_samples[num_requests * 0.99];
    
    EXPECT_LT(p50, 500);   // P50 < 500ms
    EXPECT_LT(p95, 1000);  // P95 < 1000ms (SLO)
    EXPECT_LT(p99, 2000);  // P99 < 2000ms
}
```

---

## 5. 性能测试

### 5.1 Benchmark框架

```cpp
// 代码位置: tests/performance/benchmark_test.cpp

/**
 * @test 性能基准测试
 * @category performance
 * @hardware_required cuda
 */

class BenchmarkTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // 加载模型
        model_loader_ = std::make_unique<ModelLoader>(model_path_);
        model_ = model_loader_->load();
        
        // 创建执行器
        executor_ = std::make_unique<CUDAGraphExecutor>(device_);
    }
    
    std::unique_ptr<Model> model_;
    std::unique_ptr<Executor> executor_;
};

/**
 * @benchmark Forward Throughput
 * @metric tokens/second
 * @target > 10000 tokens/s for batch_size=32
 */
TEST_F(BenchmarkTest, ForwardThroughput) {
    const int batch_size = 32;
    const int seq_len = 512;
    
    // 预热
    for (int i = 0; i < 10; ++i) {
        auto input = create_random_input(batch_size, seq_len);
        executor_->forward(input);
    }
    
    // 实际测试
    const int iterations = 100;
    auto start = absl::Now();
    
    int64_t total_tokens = 0;
    for (int i = 0; i < iterations; ++i) {
        auto input = create_random_input(batch_size, seq_len);
        auto output = executor_->forward(input);
        total_tokens += batch_size;
    }
    
    auto duration = absl::Now() - start;
    double throughput = total_tokens / absl::ToDoubleSeconds(duration);
    
    EXPECT_GT(throughput, 10000);  // > 10K tokens/s
    std::cout << "Throughput: " << throughput << " tokens/s\n";
}

/**
 * @benchmark KV Cache Allocation
 * @metric blocks/second
 */
TEST_F(BenchmarkTest, KVCacheAllocationThroughput) {
    const int num_blocks = 10000;
    auto manager = std::make_unique<BlockManager>(num_blocks, shape_, device_);
    
    const int iterations = 1000;
    auto start = absl::Now();
    
    for (int i = 0; i < iterations; ++i) {
        auto blocks = manager->allocate(/*num=*/8);
        manager->free(blocks);
    }
    
    auto duration = absl::Now() - start;
    double throughput = iterations / absl::ToDoubleSeconds(duration);
    
    EXPECT_GT(throughput, 50000);  // > 50K alloc+free/s
    std::cout << "KV Cache ops: " << throughput << " ops/s\n";
}

/**
 * @benchmark Memory Efficiency
 * @metric memory_utilization
 */
TEST_F(BenchmarkTest, KVCacheMemoryUtilization) {
    // 模拟真实负载
    std::vector<std::shared_ptr<Request>> requests;
    
    for (int i = 0; i < 1000; ++i) {
        auto req = std::make_shared<Request>(...);
        requests.push_back(req);
        
        // 模拟生成完成
        if (i % 10 == 0) {
            requests.erase(requests.begin(), requests.begin() + 10);
        }
    }
    
    // 计算内存利用率
    double utilization = calculate_memory_utilization(requests, total_blocks_);
    
    EXPECT_GT(utilization, 0.8);  // > 80% 利用率
    std::cout << "Memory utilization: " << (utilization * 100) << "%\n";
}
```

### 5.2 性能指标定义

```cpp
/**
 * @struct PerformanceMetrics
 * @brief 性能指标
 */
struct PerformanceMetrics {
    // 吞吐量
    double throughput_tokens_per_second = 0;
    double throughput_requests_per_second = 0;
    
    // 延迟
    double latency_p50_ms = 0;
    double latency_p95_ms = 0;
    double latency_p99_ms = 0;
    double latency_avg_ms = 0;
    
    // TTFT/TPOT
    double ttft_p50_ms = 0;
    double ttft_p99_ms = 0;
    double tpot_avg_ms = 0;
    
    // GPU利用率
    double gpu_utilization_percent = 0;
    double gpu_memory_utilization_percent = 0;
    
    // Batch统计
    double avg_batch_size = 0;
    double max_batch_size = 0;
    
    // 错误统计
    int error_count = 0;
    double error_rate_percent = 0;
    
    // SLO合规
    double ttft_slo_compliance_percent = 0;
    double tpot_slo_compliance_percent = 0;
    double ttlt_slo_compliance_percent = 0;
};
```

---

## 6. AI验收标准

### 6.1 测试覆盖率要求

```yaml
coverage_requirements:
  unit_tests:
    # 语句覆盖率
    statement_coverage: "> 80%"
    # 分支覆盖率
    branch_coverage: "> 70%"
    # 函数覆盖率
    function_coverage: "> 90%"
    
  critical_paths:
    # 调度关键路径
    scheduler_critical: "100%"
    # KV Cache分配
    kv_cache_critical: "100%"
    # 模型推理
    inference_critical: "> 95%"
    
  api_endpoints:
    # 所有公开API
    public_api: "100%"
    # 错误处理路径
    error_paths: "> 90%"
```

### 6.2 CI/CD集成

```yaml
# .github/workflows/test.yml

name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit_test:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      
      - name: Build
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build --target xllm_unit_tests -j$(nproc)
      
      - name: Run Unit Tests
        run: |
          ./build/tests/xllm_unit_tests \
            --gtest_output=xml:unit_test_results.xml \
            --gtest_filter=-*SlowTest:*Integration*
      
      - name: Coverage
        run: |
          # 生成覆盖率报告
          llvm-cov show build/tests/xllm_unit_tests \
            --format=html \
            --output-dir=coverage \
            --ignore-filename-regex="tests/.*"
      
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          files: coverage/coverage.json
          fail_ci_if_error: true

  component_test:
    runs-on: [self-hosted, gpu]
    needs: unit_test
    steps:
      - uses: actions/checkout@v4
      
      - name: Build with GPU support
        run: |
          cmake -B build -DXLLM_BUILD_TESTS=ON -DXLLM_ENABLE_CUDA=ON
          cmake --build build --target xllm_component_tests -j$(nproc)
      
      - name: Run Component Tests
        run: |
          ./build/tests/xllm_component_tests \
            --gtest_output=xml:component_test_results.xml

  integration_test:
    runs-on: [self-hosted, gpu]
    needs: component_test
    steps:
      - uses: actions/checkout@v4
      
      - name: Start Server
        run: |
          ./build/xllm_server --config=test_config.json &
          sleep 10  # 等待启动
      
      - name: Run Integration Tests
        run: |
          ./build/tests/xllm_integration_tests \
            --server_url=http://localhost:8080 \
            --gtest_output=xml:integration_test_results.xml
      
      - name: Performance Benchmark
        run: |
          ./build/tests/xllm_benchmark \
            --suite=critical \
            --output=benchmark_results.json
      
      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: |
            *_test_results.xml
            benchmark_results.json

  performance_gate:
    runs-on: [self-hosted, gpu]
    needs: integration_test
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Check Performance
        run: |
          python3 scripts/check_performance.py \
            --current=benchmark_results.json \
            --baseline=performance_baseline.json \
            --threshold=0.95
        # 性能退化不能超过5%
```

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [部署指南](./10_DEPLOYMENT.md) | 部署配置 |
| [性能优化](./11_PERFORMANCE.md) | 性能优化指南 |
| [Benchmark](./11_BENCHMARK.md) | 性能基准定义 |
