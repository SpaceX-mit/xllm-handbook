# xLLM 项目剖析指南 - LLM Wiki

> 本文档提供系统性剖析 xLLM 大模型推理框架的完整方法论和知识体系。

---

## 📋 目录

1. [项目概览](#1-项目概览)
2. [剖析步骤总览](#2-剖析步骤总览)
3. [第一阶段：宏观认知](#3-第一阶段宏观认知)
4. [第二阶段：核心模块深入](#4-第二阶段核心模块深入)
5. [第三阶段：硬件适配层](#5-第三阶段硬件适配层)
6. [第四阶段：服务与接口](#6-第四阶段服务与接口)
7. [剖析方法论](#7-剖析方法论)
8. [关键代码路径](#8-关键代码路径)
9. [常见问题与调试](#9-常见问题与调试)
10. [扩展与贡献指南](#10-扩展与贡献指南)

---

## 1. 项目概览

### 1.1 项目定位

| 属性 | 描述 |
|------|------|
| **项目名称** | xLLM |
| **类型** | 大模型推理框架 |
| **核心技术** | 高效推理、国产芯片优化、企业级部署 |
| **语言** | C++ (核心) + Python (接口) |
| **许可** | Apache 2.0 |

### 1.2 支持的硬件平台

```
┌─────────────────────────────────────────────────────────────┐
│                     硬件支持矩阵                              │
├─────────────┬──────┬──────────┬─────────────────────────────┤
│ 硬件类型     │ 简称 │ 型号      │ 备注                        │
├─────────────┼──────┼──────────┼─────────────────────────────┤
│ Ascend NPU  │ NPU  │ A2, A3   │ HDK Driver 25.2.0+          │
│ Cambricon   │ MLU  │ MLU系列   │ 寒武纪                      │
│ Moore Threads│ MUSA │ S5000    │ 摩尔线程                    │
│ Hygon DCU   │ DCU  │ BW1000   │ 海光                        │
│ MetaX MACA  │ MACA │ MXC500   │ 沐曦                        │
│ Iluvatar    │ ILU  │ BI150    │ 芯动科技                    │
└─────────────┴──────┴──────────┴─────────────────────────────┘
```

### 1.3 核心特性

- **P/D 分离架构**: Prefill-Decode 分布式推理
- **KV Cache 管理**: 基于 Mooncake 的多级 KV 缓存
- **Continuous Batching**: 连续批处理调度
- **推测解码**: Speculative Decoding 支持
- **多模态支持**: VLM (视觉语言模型)
- **Function Call**: 工具调用能力

---

## 2. 剖析步骤总览

```
┌────────────────────────────────────────────────────────────────┐
│                    项目剖析流程图                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐ │
│  │ 阶段一   │ -> │ 阶段二   │ -> │ 阶段三   │ -> │ 阶段四  │ │
│  │ 宏观认知 │    │ 核心模块 │    │ 硬件适配 │    │ 服务接口 │ │
│  └──────────┘    └──────────┘    └──────────┘    └─────────┘ │
│       │               │               │               │       │
│       v               v               v               v       │
│  • 目录结构      • Runtime        • Kernels       • API Service│
│  • 架构图        • Scheduler      • 平台抽象       • Server    │
│  • 入口点        • Framework      • 执行器实现     • Protocol  │
│  • 依赖关系      • 模型层         • 算子优化       • Python API│
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 剖析时间建议

| 角色 | 建议时长 | 目标 |
|------|----------|------|
| 快速概览 | 1-2 小时 | 了解整体架构和目录结构 |
| 核心模块 | 1-2 天 | 掌握 Runtime、Scheduler、Framework |
| 硬件适配 | 2-3 天 | 理解多平台抽象和 kernel 实现 |
| 服务部署 | 1 天 | 掌握部署和调优 |
| 深入贡献 | 持续 | 根据具体模块深入 |

---

## 3. 第一阶段：宏观认知

### 3.1 目录结构详解

```
xllm-handbook/
├── xllm/                          # 核心源码目录
│   ├── api_service/               # 🌟 API 服务层
│   │   ├── chat_service_impl.cpp  # Chat API 实现
│   │   ├── completion_service_impl.cpp
│   │   ├── embedding_service_impl.cpp
│   │   ├── anthropic_service_impl.cpp
│   │   └── ...
│   │
│   ├── c_api/                     # C 语言 API
│   ├── cc_api/                    # C++ 类 API
│   │
│   ├── core/                      # 🌟🌟 核心引擎
│   │   ├── common/                # 通用工具
│   │   ├── distributed_runtime/   # 🌟 分布式运行
│   │   │   ├── llm_engine.cpp     # LLM 引擎
│   │   │   ├── vlm_engine.cpp     # VLM 引擎
│   │   │   ├── rec_engine.cpp     # 推荐引擎
│   │   │   ├── master.cpp         # 主节点协调
│   │   │   ├── worker_service.cpp # Worker 服务
│   │   │   └── disagg_pd_*.cpp    # P/D 分离
│   │   │
│   │   ├── framework/             # 🌟 执行框架
│   │   │   ├── batch/             # 批处理管理
│   │   │   ├── block/             # 内存块管理
│   │   │   ├── kv_cache/         # KV 缓存
│   │   │   ├── model/             # 模型定义
│   │   │   ├── request/           # 请求管理
│   │   │   ├── sampling/          # 采样策略
│   │   │   └── parallel_state/    # 并行策略
│   │   │
│   │   ├── kernels/               # 🌟 算子层
│   │   │   ├── cuda/              # CUDA 算子
│   │   │   ├── npu/               # NPU 算子
│   │   │   ├── mlu/               # MLU 算子
│   │   │   ├── dcu/               # DCU 算子
│   │   │   ├── musa/              # MUSA 算子
│   │   │   ├── ilu/               # ILU 算子
│   │   │   └── ops_api.cpp        # 统一算子接口
│   │   │
│   │   ├── layers/                # 模型层实现
│   │   ├── platform/              # 平台抽象
│   │   ├── runtime/               # 🌟 执行器
│   │   │   ├── executor.cpp       # 执行器基类
│   │   │   ├── worker.cpp         # Worker 基类
│   │   │   ├── worker_impl.cpp    # Worker 实现
│   │   │   ├── cuda_graph_executor_impl.cpp
│   │   │   ├── acl_graph_executor_impl.cpp
│   │   │   ├── mlu_graph_executor_impl.cpp
│   │   │   └── dcu_graph_executor_impl.cpp
│   │   │
│   │   ├── scheduler/             # 🌟 调度器
│   │   │   ├── continuous_scheduler.cpp  # 连续批处理
│   │   │   ├── disagg_pd_scheduler.cpp   # P/D 分离调度
│   │   │   ├── pd_ooc_scheduler.cpp      # P/D OOC 调度
│   │   │   └── scheduler_policy.cpp      # 调度策略
│   │   │
│   │   ├── triton_jit/            # Triton JIT
│   │   └── util/                   # 工具类
│   │
│   ├── function_call/             # Function Call 解析
│   ├── models/                    # 模型实现
│   │   ├── llm/                   # LLM 模型
│   │   ├── vlm/                   # VLM 模型
│   │   ├── dit/                   # DiT 模型
│   │   ├── rec/                   # 推荐模型
│   │   └── model_registry.cpp     # 模型注册表
│   │
│   ├── parser/                    # 推理解析
│   ├── processors/                # 🌟 多模态预处理
│   │   ├── image_processor.h      # 图像处理基类
│   │   ├── video_processor.h      # 视频处理基类
│   │   ├── qwen2_vl_*.cpp         # Qwen2-VL 处理器
│   │   ├── glm4v_*.cpp            # GLM-4V 处理器
│   │   └── clip_*.cpp             # CLIP 处理器
│   │
│   ├── proto/                     # 🌟 通信协议
│   │   ├── chat.proto             # Chat 协议
│   │   ├── worker.proto           # Worker 通信
│   │   ├── disagg_pd.proto        # P/D 分离协议
│   │   └── ...
│   │
│   ├── pybind/                    # 🌟 Python 绑定
│   │   ├── bind.cpp               # 绑定入口
│   │   ├── llm.py                 # LLM Python 接口
│   │   └── vlm.py                 # VLM Python 接口
│   │
│   ├── python/                    # Python 层
│   ├── server/                    # 服务入口
│   └── xllm.cpp                   # 主入口
│
├── examples/                      # 使用示例
├── tests/                         # 测试代码
├── docs/                          # 文档
├── tools/                         # 工具脚本
└── xllm-handbook/AGENTS.md        # Agent 指令
```

### 3.2 核心架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          xLLM 整体架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Service Layer (服务层)                       │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │   │
│  │  │   Chat    │ │Completion │ │Embedding  │ │  Anthropic│       │   │
│  │  │  Service  │ │  Service  │ │  Service  │ │  Service  │       │   │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘       │   │
│  └────────┼─────────────┼─────────────┼─────────────┼──────────────┘   │
│           └─────────────┴──────┬──────┴─────────────┘                  │
│                                │                                       │
│  ┌─────────────────────────────┼───────────────────────────────────┐   │
│  │                     Engine Layer (引擎层)                        │   │
│  │    ┌────────────┐    ┌────────────┐    ┌────────────┐          │   │
│  │    │  LLM Engine │    │  VLM Engine │    │  REC Engine │          │   │
│  │    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │   │
│  │           └──────────┬───────┘                  │                │   │
│  │              ┌──────┴──────┐                    │                │   │
│  │              │   Scheduler │◄───────────────────┘                │   │
│  │              │   调度器     │                                      │   │
│  │              └──────┬──────┘                                      │   │
│  └─────────────────────┼─────────────────────────────────────────────┘   │
│                        │                                                │
│  ┌─────────────────────┼─────────────────────────────────────────────┐   │
│  │                 Worker Layer (Worker 层)                          │   │
│  │    ┌────────────┐  │  ┌────────────┐  │  ┌────────────┐         │   │
│  │    │  Prefill   │◄─┼─►│   Batch    │◄─┼─►│   Decode    │         │   │
│  │    │   Worker   │  │  │  Manager   │  │  │   Worker    │         │   │
│  │    └────────────┘  │  └────────────┘  │  └────────────┘         │   │
│  │                    │                   │                          │   │
│  │         ┌──────────┴───────────────────┴──────────┐              │   │
│  │         │              KV Cache Manager            │              │   │
│  │         │           (基于 Mooncake 多级缓存)        │              │   │
│  │         └──────────────────────────────────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 Platform Layer (平台层)                          │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │   │
│  │  │  CUDA  │ │  NPU   │ │  MLU   │ │  DCU   │ │  MUSA  │ ...   │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │   │
│  │                                                                 │   │
│  │  ┌────────────────────────────────────────────────────────┐    │   │
│  │  │                    Kernels (算子)                       │    │   │
│  │  │  Attention │ Linear │ RMSNorm │ RoPE │ MoE │ ...      │    │   │
│  │  └────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 入口点分析

#### C++ 主入口

```cpp
// xllm/xllm.cpp - 主入口点
int main(int argc, char* argv[]) {
    // 1. 解析命令行参数
    // 2. 初始化配置
    // 3. 创建 xLLM Server
    // 4. 启动服务
}
```

#### Python 入口

```python
# xllm/launch_server.py - Python 启动脚本
# 支持多种启动模式:
# - 单机模式
# - P/D 分离模式
# - 分布式模式
```

#### API 服务入口

```cpp
// xllm/server/xllm_server.cpp
class XLLMServer {
    // 初始化 brpc 服务
    // 注册各个 Service 实现
    // 启动 HTTP/gRPC 服务
}
```

---

## 4. 第二阶段：核心模块深入

### 4.1 Runtime 模块 (执行层)

#### 核心类关系

```
┌──────────────────────────────────────────────────────────────┐
│                    Runtime 类层次结构                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐                                          │
│  │ Executor (基类) │                                          │
│  └────────┬────────┘                                          │
│           │                                                   │
│  ┌────────┴────────┐                                          │
│  │ BaseExecutorImpl│                                         │
│  └────────┬────────┘                                          │
│           │                                                   │
│  ┌────────┴──────────────────────────────────┐              │
│  │         GraphExecutorImpl (图形化执行)      │              │
│  ├── CUDAGraphExecutorImpl                   │              │
│  ├── ACLGraphExecutorImpl (NPU)              │              │
│  ├── MLUGraphExecutorImpl                    │              │
│  └── DCUGraphExecutorImpl                    │              │
│                                                              │
│  ┌─────────────────┐                                          │
│  │   Worker (基类) │                                          │
│  └────────┬────────┘                                          │
│           │                                                   │
│  ┌────────┴────────┐                                          │
│  │  WorkerImpl     │                                          │
│  ├─────────────────┤                                          │
│  │ LLMWorkerImpl   │  - 标准 LLM 推理                        │
│  │ VLMWorkerImpl   │  - 多模态推理                           │
│  │ RecWorkerImpl   │  - 推荐模型推理                         │
│  │ SpeculativeWorkerImpl │ - 推测解码                        │
│  │ MTPWorkerImpl   │  - Multi-Token Prediction               │
│  └─────────────────┘                                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### 关键文件

| 文件 | 功能 |
|------|------|
| `executor.h/cpp` | 执行器基类，定义执行接口 |
| `worker_impl.cpp` | 核心推理循环实现 |
| `forward_params.h` | 前向传播参数定义 |
| `cuda_graph_executor_impl.cpp` | CUDA 图执行器 |
| `acl_graph_executor_impl.cpp` | NPU 图执行器 |

#### 执行流程

```cpp
// 简化版执行流程
class WorkerImpl {
public:
    void Execute(Request& req) {
        // 1. 输入预处理
        Preprocess(req);
        
        // 2. KV Cache 查找
        auto kv_cache = FindKVCache(req);
        
        // 3. 执行前向传播
        for (auto& layer : layers_) {
            layer->Forward(kv_cache);
        }
        
        // 4. 采样
        auto token = Sampler::Sample(output);
        
        // 5. 后处理
        Postprocess(token);
        
        // 6. 返回结果
        return token;
    }
};
```

### 4.2 Scheduler 模块 (调度层)

#### 调度器类型

```
┌────────────────────────────────────────────────────────────────┐
│                      调度器类型                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Scheduler (基类)                                               │
│  │                                                              │
│  ├── ContinuousScheduler        # 连续批处理调度               │
│  │   └── 特点: 新请求动态加入，正在处理的请求动态完成             │
│  │                                                              │
│  ├── DisaggPDScheduler           # P/D 分离调度                │
│  │   ├── Prefill 请求发送到 Prefill 节点                       │
│  │   └── Decode 请求发送到 Decode 节点                         │
│  │                                                              │
│  ├── DisaggPDChunkedPrefillScheduler  # 分块预填充             │
│  │   └── 将长序列分块处理                                       │
│  │                                                              │
│  ├── PDOOCScheduler              # P/D Out-of-Order            │
│  │   └── 支持乱序执行                                          │
│  │                                                              │
│  ├── DitScheduler                # DiT 调度                   │
│  │                                                              │
│  └── FixedStepsScheduler          # 固定步数调度               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### Continuous Batching 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                  Continuous Batching 流程                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  时间 ─────────────────────────────────────────────────────────► │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Batch 1: [Req1, Req2, Req3]          生成 token         │    │
│  │    Req1: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░  8/20       │    │
│  │    Req2: ████████████████░░░░░░░░░░░░░░░░░░  16/25      │    │
│  │    Req3: ██████████████████████░░░░░░░░░░░░  22/30      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Batch 2: [Req1, Req2, Req3, Req4]    新请求加入         │    │
│  │    Req1: ████████████░░░░░░░░░░░░░░░░░░░░░░░░  12/20    │    │
│  │    Req2: ████████████████████████░░░░░░░░░░░  25/25 ✓   │    │
│  │    Req3: ████████████████████████████░░░░░░░  28/30     │    │
│  │    Req4: █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5/15       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Batch 3: [Req1, Req3, Req4, Req5]    Req2 完成，新请求  │    │
│  │    Req1: ████████████████░░░░░░░░░░░░░░░░░░░░░░  16/20   │    │
│  │    Req3: ██████████████████████████████░░░░░░░  30/30 ✓  │    │
│  │    Req4: █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  9/15     │    │
│  │    Req5: ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2/18     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 调度策略

| 策略类 | 位置 | 功能 |
|--------|------|------|
| `DecodeFirstPolicy` | scheduler_policy.cpp | Decode 优先 |
| `PrefillFirstPolicy` | prefill_first_policy.cpp | Prefill 优先 |
| `UnifiedPolicy` | unified_policy.cpp | 统一策略 |

### 4.3 Framework 模块 (框架层)

#### Framework 子模块

```
xllm/core/framework/
├── batch/              # 批处理管理
│   ├── batch_manager.h
│   └── batch_slot.h
│
├── block/             # 内存块管理
│   ├── block_manager.h
│   └── memory_pool.h
│
├── kv_cache/          # KV 缓存
│   ├── kv_cache_manager.h
│   ├── block_attention.h
│   └── prefix_cache.h
│
├── model/             # 模型结构
│   ├── model.h
│   └── layer.h
│
├── request/           # 请求管理
│   ├── request.h
│   └── request_queue.h
│
├── sampling/          # 采样策略
│   ├── sampler.h
│   ├── greedy_sampler.h
│   └── random_sampler.h
│
├── parallel_state/    # 并行策略
│   ├── tensor_parallel.h
│   └── pipeline_parallel.h
│
├── config/            # 配置解析
│   └── model_config.h
│
├── dit_cache/         # DiT 缓存
├── encoder_cache/     # Encoder 缓存
├── kv_cache_transfer/ # KV 传输
├── multimodal/        # 多模态支持
├── state_dict/        # 状态字典
├── tokenizer/         # 分词器
└── chat_template/     # Chat 模板
```

#### KV Cache 管理

```cpp
// Framework 中的 KV Cache 结构
class KVCacheManager {
    // 多级 KV Cache (基于 Mooncake)
    // - L1: GPU 显存
    // - L2: 主机内存
    // - L3: SSD (可选)
    
    // Block 分配策略
    // Paged Attention 支持
};
```

### 4.4 Models 模块 (模型层)

#### 模型注册机制

```cpp
// models/model_registry.cpp
class ModelRegistry {
    // 注册各类模型
    RegisterModel("Llama", LlamaModel::Create);
    RegisterModel("Qwen", QwenModel::Create);
    RegisterModel("GLM", GLMModel::Create);
    RegisterModel("DeepSeek", DeepSeekModel::Create);
    // ...
};
```

#### 模型类型

| 目录 | 模型类型 | 示例 |
|------|----------|------|
| `llm/` | 语言模型 | Llama, Qwen, GLM, DeepSeek |
| `vlm/` | 视觉语言模型 | Qwen2-VL, GLM-4V, MiniCPM-V |
| `dit/` | 扩散Transformer | DiT, LLMWalker |
| `rec/` | 推荐模型 | 推荐系统模型 |

---

## 5. 第三阶段：硬件适配层

### 5.1 平台抽象层

```
┌────────────────────────────────────────────────────────────────┐
│                    平台抽象层                                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌──────────────┐                             │
│                    │ PlatformBase │  (基类)                     │
│                    └───────┬──────┘                             │
│                            │                                    │
│        ┌───────────────────┼───────────────────┐               │
│        │                   │                   │               │
│        ▼                   ▼                   ▼               │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐           │
│  │   CUDA   │       │   NPU    │       │   MLU    │           │
│  │ Platform │       │ Platform │       │ Platform │           │
│  └──────────┘       └──────────┘       └──────────┘           │
│                                                                 │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐           │
│  │   DCU    │       │   MUSA   │       │   ILU    │           │
│  │ Platform │       │ Platform │       │ Platform │           │
│  └──────────┘       └──────────┘       └──────────┘           │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Kernel 实现

#### 核心算子

| 算子 | 功能 | 位置 |
|------|------|------|
| `Attention` | 自注意力 | kernels/*/attention*.cpp |
| `Linear` | 线性层 | kernels/*/linear*.cpp |
| `RMSNorm` | RMS 归一化 | kernels/*/rmsnorm*.cpp |
| `RoPE` | 旋转位置编码 | kernels/*/rope*.cpp |
| `MoE` | 混合专家 | kernels/*/moe*.cpp |
| `SiluAndMul` | SwiGLU 激活 | kernels/*/silu*.cpp |

#### 多平台 Kernel 目录

```
xllm/core/kernels/
├── cuda/           # CUDA (NVIDIA GPU)
│   ├── attention/
│   ├── linear/
│   └── ...
│
├── npu/            # NPU (华为 Ascend)
│   ├── attention/
│   ├── tilelang/   # TileLang DSL
│   └── ...
│
├── mlu/            # MLU (寒武纪)
│   ├── attention/
│   └── ...
│
├── dcu/            # DCU (海光)
│   └── ...
│
├── musa/           # MUSA (摩尔线程)
│   └── ...
│
├── ilu/            # ILU (芯动科技)
│   └── ...
│
├── ops_api.cpp     # 统一算子 API
├── ops_api.h       # 算子接口定义
└── param.h         # 算子参数定义
```

### 5.3 Graph Executor 实现

#### CudaGraphExecutorImpl

```cpp
// 使用 CUDA Graph 优化小算子融合
class CudaGraphExecutorImpl {
    // 1. 捕获计算图
    cudaGraph_t graph_;
    cudaGraphExec_t instance_;
    
    // 2. 优化小算子
    // 3. 批量执行
};
```

#### ACLGraphExecutorImpl (NPU)

```cpp
// 适配华为 Ascend
class ACLGraphExecutorImpl {
    // 使用 ACL (Ascend CL) 接口
    // 支持 A2, A3 系列
};
```

---

## 6. 第四阶段：服务与接口

### 6.1 API Service 层

#### 服务类型

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Service 类型                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                             │
│  │ ChatService     │  - 对话补全 (chat.completion)               │
│  │                 │  - 支持流式输出                              │
│  └─────────────────┘                                             │
│                                                                  │
│  ┌─────────────────┐                                             │
│  │ CompletionService│ - 通用补全 (completion)                   │
│  └─────────────────┘                                             │
│                                                                  │
│  ┌─────────────────┐                                             │
│  │ EmbeddingService│  - 向量嵌入                                  │
│  │                 │  - 支持批处理                                │
│  └─────────────────┘                                             │
│                                                                  │
│  ┌─────────────────┐                                             │
│  │ AnthropicService│ - Anthropic API 兼容                        │
│  │                 │  - Claude 接口                               │
│  └─────────────────┘                                             │
│                                                                  │
│  ┌─────────────────┐                                             │
│  │ RerankService   │  - 重排序服务                                │
│  └─────────────────┘                                             │
│                                                                  │
│  ┌─────────────────┐                                             │
│  │ VLMService      │  - 多模态服务                                │
│  └─────────────────┘                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 服务实现示例

```cpp
// api_service/chat_service_impl.cpp
class ChatServiceImpl : public brpc::Service {
public:
    void Chat(::google::protobuf::RpcController* controller,
             const ChatRequest* request,
             ChatResponse* response,
             ::google::protobuf::Closure* done) {
        // 1. 解析请求
        // 2. 预处理 (Chat Template)
        // 3. 提交到 Engine
        // 4. 后处理响应
        // 5. 返回结果
    }
};
```

### 6.2 Protocol (Protobuf)

```
proto/
├── chat.proto              # 对话协议
├── completion.proto        # 补全协议
├── embedding.proto         # 嵌入协议
├── worker.proto            # Worker 通信
├── disagg_pd.proto         # P/D 分离协议
├── multimodal.proto       # 多模态协议
├── common.proto           # 通用消息
└── xllm_service.proto     # xLLM 服务协议
```

### 6.3 Python 接口

```python
# pybind/llm.py
import xllm

# 创建 LLM 实例
llm = xllm.LLM(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    device="npu",  # 或 "cuda", "mlu", "dcu"
    tensor_parallel=1,
)

# 对话
response = llm.chat([
    {"role": "user", "content": "Hello!"}
])

# 流式对话
for chunk in llm.stream_chat([...]):
    print(chunk)
```

### 6.4 Server 架构

```cpp
// server/xllm_server.cpp
class XLLMServer {
    // 基于 brpc 构建高性能 HTTP/gRPC 服务
    
    void Start() {
        // 1. 初始化 brpc
        brpc::Server server;
        
        // 2. 注册服务
        server.AddService(new ChatServiceImpl());
        server.AddService(new CompletionServiceImpl());
        // ...
        
        // 3. 启动服务
        server.Start(port);
    }
};
```

---

## 7. 剖析方法论

### 7.1 自顶向下法

```
1. 架构概览
   ├── 阅读 README 和文档
   ├── 查看架构图
   └── 理解核心特性

2. 目录结构
   ├── 理解模块划分
   ├── 识别关键入口点
   └── 建立目录-功能映射

3. 核心流程
   ├── 请求处理流程
   ├── 推理执行流程
   └── 调度流程

4. 深入实现
   ├── 关键算法
   ├── 数据结构
   └── 优化策略
```

### 7.2 自底向上法

```
1. 基础组件
   ├── 工具类和数据结构
   ├── 平台抽象层
   └── 算子接口

2. 构建层
   ├── Kernel 实现
   ├── Layer 实现
   └── Model 实现

3. 执行层
   ├── Worker 实现
   ├── Executor 实现
   └── Scheduler 实现

4. 服务层
   ├── API Service
   ├── Protocol
   └── Server
```

### 7.3 追踪法

#### 追踪一次完整推理

```
1. 请求入口
   api_service/chat_service_impl.cpp::Chat()
   
2. 请求预处理
   processors/ - Chat Template 解析
   
3. 引擎调度
   distributed_runtime/llm_engine.cpp
   
4. 调度器
   scheduler/continuous_scheduler.cpp
   
5. Worker 执行
   runtime/worker_impl.cpp
   
6. 算子执行
   kernels/*/ops_api.cpp
   
7. 结果返回
   返回链路反向追溯
```

### 7.4 代码阅读技巧

| 技巧 | 应用场景 |
|------|----------|
| **搜索入口** | 使用 IDE 搜索 `class XxxServiceImpl` |
| **追踪继承** | 查看类的 `virtual` 方法 |
| **理解工厂** | Factory 类揭示对象创建模式 |
| **寻找注册** | `RegisterXxx` 揭示扩展点 |
| **关注配置** | `Options`, `Config` 类揭示参数 |

---

## 8. 关键代码路径

### 8.1 Chat 请求完整路径

```
┌─────────────────────────────────────────────────────────────────┐
│              Chat 请求处理完整路径                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HTTP Request                                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ api_service/chat_service_impl.cpp                         │  │
│  │ ChatServiceImpl::Chat()                                   │  │
│  │   - 解析 JSON 请求                                         │  │
│  │   - 提取 messages, parameters                             │  │
│  └────────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ core/framework/chat_template/                             │  │
│  │   - 应用 Chat Template                                     │  │
│  │   - Tokenize 输入                                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ core/distributed_runtime/llm_engine.cpp                    │  │
│  │ LLMEngine::AddRequest()                                    │  │
│  │   - 创建 Request 对象                                      │  │
│  │   - 加入调度队列                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ core/scheduler/continuous_scheduler.cpp                    │  │
│  │ ContinuousScheduler::Schedule()                           │  │
│  │   - Continuous Batching                                    │  │
│  │   - 合并多个请求到同一批次                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ core/runtime/worker_impl.cpp                              │  │
│  │ WorkerImpl::Forward()                                      │  │
│  │   - KV Cache 查找/分配                                     │  │
│  │   - 逐层执行 Transformer                                    │  │
│  │   - Attention, Linear, RMSNorm, RoPE...                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ core/kernels/*/ops_api.cpp                                 │  │
│  │   - 实际算子执行                                            │  │
│  │   - CUDA/NPU/MLU Kernel 调用                               │  │
│  └────────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ core/framework/sampling/                                   │  │
│  │   - 采样策略 (Greedy, Random, Beam Search)                  │  │
│  │   - 生成下一个 token                                       │  │
│  └────────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ core/distributed_runtime/                                  │  │
│  │   - 流式输出处理                                           │  │
│  │   - Async Response                                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  HTTP Response (Stream)                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 模型加载路径

```
┌─────────────────────────────────────────────────────────────────┐
│              模型加载完整路径                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  llm.chat(model="Qwen/Qwen2.5-7B")                              │
│       │                                                          │
│       ▼                                                          │
│  pybind/llm.py::LLM.__init__()                                  │
│       │                                                          │
│       ▼                                                          │
│  pybind/bind.cpp                                                │
│       │                                                          │
│       ▼                                                          │
│  core/framework/hf_model_loader.cpp                             │
│  core/framework/dit_model_loader.cpp                           │
│       │                                                          │
│       ▼                                                          │
│  core/models/llm/*.cpp (模型实现)                                │
│       │                                                          │
│       ▼                                                          │
│  加载权重到 device                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 P/D 分离请求路径

```
┌─────────────────────────────────────────────────────────────────┐
│              Prefill-Decode 分离请求路径                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐         ┌─────────────┐                        │
│  │ Prefill     │         │ Decode      │                        │
│  │ Node        │         │ Node        │                        │
│  │             │         │             │                        │
│  │ +---------+ │  KV     │ +---------+ │                        │
│  │ │Prefill  │ │ Cache   │ │Decode   │ │                        │
│  │ │Scheduler│ │────────►│ │Scheduler│ │                        │
│  │ +---------+ │ Transfer│ +---------+ │                        │
│  │      │      │         │      │      │                        │
│  │      ▼      │         │      ▼      │                        │
│  │ +---------+ │         │ +---------+ │                        │
│  │ │Prefill  │ │         │ │Decode   │ │                        │
│  │ │Worker   │ │         │ │Worker   │ │                        │
│  │ +---------+ │         │ +---------+ │                        │
│  └─────────────┘         └─────────────┘                        │
│                                                                  │
│  关键文件:                                                       │
│  - disagg_pd_scheduler.cpp     # P/D 调度                       │
│  - disagg_pd_service.cpp       # P/D 服务                        │
│  - disagg_pd_service_impl.cpp  # P/D 实现                        │
│  - disagg_pd.proto             # P/D 协议                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 常见问题与调试

### 9.1 调试入口

| 场景 | 调试方法 |
|------|----------|
| 请求处理 | 日志 + brpc debug 接口 |
| 推理执行 | GDB/LLDB 断点 |
| 内存问题 | Valgrind, AddressSanitizer |
| 性能问题 | Profiler (gprof, nvidia profiler) |
| GPU 问题 | CUDA API 错误检查 |

### 9.2 关键日志位置

```cpp
// 启用调试日志
FLAGS_v = 1;  // 级别

// 关键日志点
XLLM_LOG(INFO) << "Request received";
XLLM_LOG(DEBUG) << "Batch size: " << batch_size;
XLLM_LOG(ERROR) << "Kernel execution failed";
```

### 9.3 常见问题排查

| 问题 | 可能原因 | 排查方向 |
|------|----------|----------|
| OOM | KV Cache 过大 | 调整 block 数 |
| 慢推理 | Batch 太小 | 调整调度参数 |
| 内存泄漏 | 资源未释放 | 检查 Worker 生命周期 |
| 精度问题 | Kernel 实现 | 对比 PyTorch 实现 |

---

## 10. 扩展与贡献指南

### 10.1 添加新模型

```
步骤:
1. 在 models/llm/ 创建新模型目录
2. 实现模型类 (继承 Model 基类)
3. 在 model_registry.cpp 注册
4. 添加模型加载器 (如需要)
5. 添加 Chat Template
6. 编写测试

关键文件:
- models/llm/llama.cpp (参考实现)
- models/model_registry.cpp
- core/framework/chat_template/
```

### 10.2 添加新硬件支持

```
步骤:
1. 在 core/platform/ 创建平台目录
2. 实现 Platform 基类
3. 在 core/kernels/ 创建对应 kernel
4. 实现 GraphExecutorImpl
5. 实现 WorkerImpl
6. 添加到 executor_impl_factory

关键文件:
- core/platform/ (平台基类)
- core/kernels/*/ (各平台 Kernel)
- core/runtime/*_graph_executor_impl.cpp
```

### 10.3 添加新调度器

```
步骤:
1. 继承 Scheduler 基类
2. 实现调度算法
3. 在 scheduler_factory.cpp 注册
4. 添加配置选项

关键文件:
- core/scheduler/scheduler.h
- core/scheduler/continuous_scheduler.cpp (参考)
- core/scheduler/scheduler_factory.cpp
```

### 10.4 代码规范

> ⚠️ **重要**: 贡献代码前请阅读 [.agents/skills/code-review/references/custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md)

---

## 📚 附录

### A. 核心依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| brpc | latest | RPC 框架 |
| protobuf | 3.x | 序列化 |
| glog | latest | 日志 |
| gflags | latest | 命令行参数 |
| CUDA | 11.8+ | GPU 支持 |
| ACL | 25.2.0+ | NPU 支持 |

### B. 相关文档

- [官方文档](https://docs.xllm-ai.com/)
- [技术报告](https://arxiv.org/abs/2510.14686)
- [快速开始](https://docs.xllm-ai.com/zh/getting_started/quick_start/)

### C. 参考项目

- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM) - 架构参考
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - KV Cache 参考
- [vLLM](https://github.com/vllm-project/vllm) - 调度器参考

---

*文档版本: v1.0 | 更新日期: 2026-07-23*
