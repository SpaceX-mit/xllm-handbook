# xLLM 开发者手册 - 索引

## 文档导航

### 📚 按层级导航

| 层级 | 文档 | 说明 |
|-----|------|------|
| **概念层** | [总览](./00_OVERVIEW.md) | 文档体系结构 |
| | [术语表](./01_GLOSSARY.md) | 核心概念定义 |
| **架构层** | [架构设计](./02_ARCHITECTURE.md) | 系统整体架构 |
| | [设计原则](./03_DESIGN_PRINCIPLES.md) | 设计哲学与原则 |
| **模型层** | [领域模型](./04_DOMAIN_MODEL.md) | DDD领域设计 |
| **组件层** | [Scheduler设计](./05_SCHEDULER_DESIGN.md) | 调度器实现 |
| | [Worker设计](./06_WORKER_DESIGN.md) | Worker实现 |
| | [KV Cache设计](./07_KV_CACHE_DESIGN.md) | 缓存管理实现 |
| **规范层** | [AI Native开发](./08_AI_NATIVE_DEVELOPMENT.md) | AI辅助开发规范 |
| | [API设计](./09_API_DESIGN.md) | 接口定义 |
| | [配置Schema](./10_CONFIG_SCHEMA.md) | 配置规范 |
| | [测试策略](./11_TEST_STRATEGY.md) | 测试规范 |

---

## 🔍 快速查找

### 按主题查找

#### 核心概念
- **Attention机制** → [术语表](./01_GLOSSARY.md#attention)
- **Batch处理** → [术语表](./01_GLOSSARY.md#batch), [架构](./02_ARCHITECTURE.md#2-batch-处理流程)
- **KV Cache** → [术语表](./01_GLOSSARY.md#kv-cache), [KV Cache设计](./07_KV_CACHE_DESIGN.md)
- **Scheduler** → [术语表](./01_GLOSSARY.md#scheduler), [Scheduler设计](./05_SCHEDULER_DESIGN.md)
- **Worker** → [术语表](./01_GLOSSARY.md#worker), [Worker设计](./06_WORKER_DESIGN.md)

#### 并行策略
- **Tensor Parallel** → [术语表](./01_GLOSSARY.md#tp), [架构](./02_ARCHITECTURE.md#3-并行策略)
- **Pipeline Parallel** → [术语表](./01_GLOSSARY.md#pp)
- **Data Parallel** → [术语表](./01_GLOSSARY.md#dp)
- **Expert Parallel** → [术语表](./01_GLOSSARY.md#ep)

#### 开发规范
- **代码规范** → [AI Native开发](./08_AI_NATIVE_DEVELOPMENT.md)
- **测试规范** → [测试策略](./11_TEST_STRATEGY.md)
- **API规范** → [API设计](./09_API_DESIGN.md)
- **配置规范** → [配置Schema](./10_CONFIG_SCHEMA.md)

---

## 📊 代码位置索引

### 核心组件

| 组件 | 头文件 | 实现文件 |
|-----|-------|---------|
| **Scheduler** | `xllm/core/scheduler/scheduler.h` | `xllm/core/scheduler/*.cpp` |
| ContinuousScheduler | `continuous_scheduler.h` | `continuous_scheduler.cpp` |
| DisaggPDScheduler | `disagg_pd_scheduler.h` | `disagg_pd_scheduler.cpp` |
| **Worker** | `xllm/core/runtime/worker.h` | `xllm/core/runtime/worker.cpp` |
| WorkerImpl | `worker_impl.h` | `worker_impl.cpp` |
| LLMWorkerImpl | `llm_worker_impl.h` | `llm_worker_impl.cpp` |
| **Batch** | `xllm/core/framework/batch/batch.h` | `xllm/core/framework/batch/batch.cpp` |
| **Request** | `xllm/core/framework/request/request.h` | `xllm/core/framework/request/request.cpp` |
| **Sequence** | `xllm/core/framework/request/sequence.h` | `xllm/core/framework/request/sequence.cpp` |
| **KVCache** | `xllm/core/framework/kv_cache/kv_cache.h` | `xllm/core/framework/kv_cache/kv_cache.cpp` |
| **BlockManager** | `xllm/core/framework/block/block_manager.h` | `xllm/core/framework/block/block_manager.cpp` |
| **CausalLM** | `xllm/core/framework/model/causal_lm.h` | - |

### API服务

| 服务 | 文件 |
|-----|-----|
| OpenAI API | `xllm/api_service/openai_service.*` |
| Anthropic API | `xllm/api_service/anthropic_service.*` |
| Protocol Buffers | `xllm/proto/*.proto` |

### 测试

| 测试类型 | 目录 |
|---------|------|
| 单元测试 | `tests/unit/` |
| 组件测试 | `tests/component/` |
| 集成测试 | `tests/integration/` |
| 性能测试 | `tests/performance/` |

---

## 🗂️ 按职责查找

### 调度相关
1. [Scheduler设计](./05_SCHEDULER_DESIGN.md) - 调度器架构
2. [调度策略](./05_SCHEDULER_DESIGN.md#4-scheduler-policy) - 调度算法
3. [ContinuousScheduler](./05_SCHEDULER_DESIGN.md#2-continuous-scheduler-详解) - 连续批处理
4. [DisaggPDScheduler](./05_SCHEDULER_DESIGN.md#3-disaggpd-scheduler-详解) - P/D分离

### 推理相关
1. [Worker设计](./06_WORKER_DESIGN.md) - Worker架构
2. [LLMWorkerImpl](./06_WORKER_DESIGN.md#2-llmworkerimpl-详解) - LLM推理
3. [VLMWorkerImpl](./06_WORKER_DESIGN.md#3-vlmworkerimpl-详解) - 多模态推理
4. [Executor](./06_WORKER_DESIGN.md#5-executor-架构) - 执行器

### 缓存相关
1. [KV Cache设计](./07_KV_CACHE_DESIGN.md) - 缓存架构
2. [Block管理](./07_KV_CACHE_DESIGN.md#3-block-管理) - Block分配
3. [Prefix Cache](./07_KV_CACHE_DESIGN.md#4-prefix-cache) - 前缀缓存
4. [KV传输](./07_KV_CACHE_DESIGN.md#5-kv-cache-传输-pd-分离) - P/D传输

### 数据相关
1. [领域模型](./04_DOMAIN_MODEL.md) - 数据聚合
2. [Batch](./04_DOMAIN_MODEL.md#23-batch-批处理) - 批处理数据
3. [Request](./04_DOMAIN_MODEL.md#21-requestaggregate-请求聚合) - 请求数据

---

## 📈 AI辅助开发检查清单

### 新功能开发
- [ ] 阅读相关[领域模型](./04_DOMAIN_MODEL.md)
- [ ] 遵循[设计原则](./03_DESIGN_PRINCIPLES.md)
- [ ] 实现符合[AI Native规范](./08_AI_NATIVE_DEVELOPMENT.md)
- [ ] 编写单元测试，遵循[测试策略](./11_TEST_STRATEGY.md)
- [ ] 更新相关配置[Schema](./10_CONFIG_SCHEMA.md)

### 代码审查
- [ ] 检查[架构合规](./03_DESIGN_PRINCIPLES.md#2-架构设计原则)
- [ ] 验证[接口设计](./09_API_DESIGN.md)
- [ ] 检查[代码自解释性](./08_AI_NATIVE_DEVELOPMENT.md#2-代码自解释规范)
- [ ] 验证[可测试性](./11_TEST_STRATEGY.md)

### 性能优化
- [ ] 参考[性能设计原则](./03_DESIGN_PRINCIPLES.md#6-性能设计原则)
- [ ] 使用[Benchmark](./11_TEST_STRATEGY.md#5-性能测试)验证
- [ ] 检查[内存效率](./03_DESIGN_PRINCIPLES.md#62-内存效率原则)

---

## 🔗 外部链接

- [xLLM GitHub](https://github.com/jd-opensource/xllm)
- [xLLM文档](https://docs.xllm-ai.com/)
- [OpenAI API参考](https://platform.openai.com/docs/api-reference)
- [Anthropic API参考](https://docs.anthropic.com/)

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|-----|------|---------|
| 1.0.0 | 2026-07-23 | 初始版本，包含完整文档体系 |

---

## 贡献指南

欢迎贡献文档！请遵循以下步骤：

1. Fork 仓库
2. 创建分支 `docs/your-feature`
3. 提交更改
4. 创建 Pull Request

文档使用 Markdown 编写，遵循 [AI Native 规范](./08_AI_NATIVE_DEVELOPMENT.md)。

---

*最后更新: 2026-07-23*
