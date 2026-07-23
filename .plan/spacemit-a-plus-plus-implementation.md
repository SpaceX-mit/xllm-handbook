# 方案 A++ 实施计划

> **目标**：在 xLLM 中实现方案 A++（LibTorch + ggml-spacemit 零拷贝），在 K3 worker 上运行 Qwen3.5 2B Q4_0

---

## 1. 项目概述

### 1.1 目标

- 实现 xLLM 方案 A++（零拷贝集成 ggml-spacemit）
- 在 SpacemiT K3 worker 机（10.0.90.243）上运行
- 使用 Qwen3.5 2B Q4_0 模型
- 性能目标：~16.5 t/s（与 llama.cpp 持平）
- 完整测试保护功能

### 1.2 架构设计

```
xLLM 方案 A++
├─ Python 接口 (xllm.LLM)
├─ C++ 核心 (LibTorch)
│   ├─ Executor/Worker/Scheduler (保持不变)
│   └─ Platform::SpacemiT (新增)
│
├─ SpacemiT 平台层
│   ├─ GGMLBridge (零拷贝转换) ⭐核心
│   │   ├─ torch::Tensor → ggml_tensor
│   │   └─ ggml_tensor → torch::Tensor
│   │
│   ├─ GGMLBackend (ggml 计算后端)
│   │   ├─ 管理 ggml_context
│   │   ├─ 构建计算图
│   │   └─ 执行 ggml_spacemit kernels
│   │
│   └─ SpacemiT 算子实现
│       ├─ matmul (使用 ggml)
│       ├─ rms_norm (使用 ggml)
│       ├─ apply_rotary (使用 ggml)
│       └─ act_and_mul (使用 ggml)
│
└─ ggml-spacemit (第三方)
    ├─ IME2 kernels (A100)
    ├─ IME1 kernels (X100)
    └─ RVV fallback
```

---

## 2. 代码库分析

### 2.1 现有 xLLM 架构模式

**Platform 层：**
- `xllm/core/platform/platform.h` - 平台统一接口
- `xllm/core/platform/{cuda,npu,mlu}/` - 各平台实现
- 编译选项：`USE_CUDA`, `USE_NPU`, `USE_MLU` 等

**Kernel 层：**
- `xllm/core/kernels/ops_api.h` - 统一算子接口
- `xllm/core/kernels/ops_api.cpp` - 算子分发逻辑
- `xllm/core/kernels/{cuda,npu}/` - 各平台算子实现

**算子分发模式：**
```cpp
torch::Tensor matmul(MatmulParams& params) {
#if defined(USE_MLU)
  return mlu::matmul(...);
#elif defined(USE_NPU)
  return npu::matmul(...);
#elif defined(USE_CUDA)
  return cuda::matmul(...);
#elif defined(USE_SPACEMIT)  // ← 新增
  return spacemit::matmul(...);
#endif
}
```

### 2.2 llama.cpp ggml-spacemit 资源

**可用 kernels：**
- `/data/workspace2026-new/work0618/tech-analyais0713/llama.cpp-handbook/ggml/src/ggml-cpu/spacemit/`
  - `ime2_kernels.cpp` - A100 优化
  - `ime1_kernels.cpp` - X100 优化
  - `rvv_kernels.cpp` - 向量回退
  - `ime_kernels.h` - 接口定义

---

## 3. 实施步骤

### 阶段 1：基础设施（2-3 天）

#### Task 1.1: 创建目录结构

```bash
xllm/core/
├─ platform/spacemit/
│   ├─ ggml_bridge.h         # 零拷贝转换
│   ├─ ggml_bridge.cpp
│   ├─ ggml_backend.h        # ggml 后端封装
│   ├─ ggml_backend.cpp
│   └─ platform_spacemit.cpp # 平台检测
│
├─ kernels/spacemit/
│   ├─ matmul_ggml.cpp       # 矩阵乘法
│   ├─ rms_norm_ggml.cpp     # RMSNorm
│   ├─ rotary_ggml.cpp       # RoPE
│   └─ activation_ggml.cpp   # 激活函数
│
└─ third_party/
    └─ ggml-spacemit/        # 复制 llama.cpp kernels
```

#### Task 1.2: 引入 ggml-spacemit

```bash
# 复制 llama.cpp ggml-spacemit 到 third_party
mkdir -p third_party/ggml-spacemit
cp -r /path/to/llama.cpp-handbook/ggml/src/ggml-cpu/spacemit/* \
      third_party/ggml-spacemit/

# 同时需要 ggml 核心头文件
cp /path/to/llama.cpp-handbook/ggml/include/ggml*.h \
   third_party/ggml-spacemit/
```

#### Task 1.3: 扩展 Platform 类

```cpp
// xllm/core/platform/platform.h
class Platform final {
 public:
  // 新增 SpacemiT 检测
  static constexpr bool is_spacemit() {
#if defined(USE_SPACEMIT)
    return true;
#else
    return false;
#endif
  }
  
  // 运行时检测 IME 版本
  static bool is_spacemit_ime2();  // A100
  static bool is_spacemit_ime1();  // X100
};
```

---

### 阶段 2：零拷贝桥接层（3-4 天）

#### Task 2.1: 实现 GGMLBridge

**核心功能：torch::Tensor ↔ ggml_tensor 零拷贝转换**

```cpp
// xllm/core/platform/spacemit/ggml_bridge.h
#pragma once
#include <torch/torch.h>
#include "ggml.h"

namespace xllm::spacemit {

class GGMLBridge {
 public:
    // torch::Tensor → ggml_tensor（零拷贝）
    static ggml_tensor* to_ggml(
        ggml_context* ctx,
        const torch::Tensor& t
    );
    
    // ggml_tensor → torch::Tensor（零拷贝）
    static torch::Tensor from_ggml(ggml_tensor* t);
    
 private:
    static ggml_type to_ggml_type(torch::Dtype dtype);
    static torch::Dtype to_torch_dtype(ggml_type type);
};

} // namespace xllm::spacemit
```

**实现要点：**
```cpp
ggml_tensor* GGMLBridge::to_ggml(
    ggml_context* ctx,
    const torch::Tensor& t
) {
    // 确保连续性
    torch::Tensor contig = t.contiguous();
    
    // ⭐ 关键：零拷贝，直接共享数据指针
    return ggml_new_tensor_from_data(
        ctx,
        to_ggml_type(contig.dtype()),
        contig.dim(),
        contig.sizes().data(),
        contig.data_ptr()  // ← 零拷贝
    );
}
```

#### Task 2.2: 单元测试

```cpp
// test/platform/spacemit/test_ggml_bridge.cpp
TEST(GGMLBridgeTest, ZeroCopyTorchToGGML) {
    torch::Tensor x = torch::randn({2, 3});
    void* torch_ptr = x.data_ptr();
    
    ggml_context* ctx = ggml_init(...);
    ggml_tensor* y = GGMLBridge::to_ggml(ctx, x);
    
    // 验证零拷贝：指针地址相同
    ASSERT_EQ(torch_ptr, y->data);
    
    ggml_free(ctx);
}

TEST(GGMLBridgeTest, ZeroCopyGGMLToTorch) {
    ggml_context* ctx = ggml_init(...);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    void* ggml_ptr = x->data;
    
    torch::Tensor y = GGMLBridge::from_ggml(x);
    
    // 验证零拷贝：指针地址相同
    ASSERT_EQ(ggml_ptr, y.data_ptr());
    
    ggml_free(ctx);
}
```

---

### 阶段 3：ggml 后端封装（4-5 天）

#### Task 3.1: 实现 GGMLBackend

```cpp
// xllm/core/platform/spacemit/ggml_backend.h
namespace xllm::spacemit {

class GGMLBackend {
 public:
    GGMLBackend();
    ~GGMLBackend();
    
    // 执行 ggml 计算图
    torch::Tensor compute(
        std::function<ggml_tensor*(ggml_context*)> build_graph,
        const std::vector<torch::Tensor>& inputs
    );
    
    // 获取 ggml context
    ggml_context* context() { return ctx_; }
    
 private:
    ggml_context* ctx_;
    size_t context_size_;
    bool use_ime2_;  // A100 vs X100
};

} // namespace xllm::spacemit
```

**实现：**
```cpp
GGMLBackend::GGMLBackend() {
    // 检测 IME 版本
    use_ime2_ = detect_ime_version() == IMEVersion::IME2;
    
    // 初始化 ggml context
    context_size_ = 128 * 1024 * 1024;  // 128MB
    ggml_init_params params = {
        .mem_size = context_size_,
        .mem_buffer = nullptr,
        .no_alloc = false
    };
    ctx_ = ggml_init(params);
}

torch::Tensor GGMLBackend::compute(
    std::function<ggml_tensor*(ggml_context*)> build_graph,
    const std::vector<torch::Tensor>& inputs
) {
    // 1. 构建计算图
    ggml_tensor* output = build_graph(ctx_);
    
    // 2. 创建计算图对象
    ggml_cgraph* gf = ggml_new_graph(ctx_);
    ggml_build_forward_expand(gf, output);
    
    // 3. 执行计算（调用 ggml-spacemit）
    ggml_graph_compute_with_ctx(ctx_, gf, /*n_threads=*/8);
    
    // 4. 零拷贝转换结果
    return GGMLBridge::from_ggml(output);
}
```

---

### 阶段 4：算子实现（5-6 天）

#### Task 4.1: 实现 matmul

```cpp
// xllm/core/kernels/spacemit/matmul_ggml.cpp
namespace xllm::kernel::spacemit {

torch::Tensor matmul(
    const torch::Tensor& a,  // [M, K]
    const torch::Tensor& b,  // [K, N]
    const std::optional<torch::Tensor>& bias
) {
    GGMLBackend backend;
    
    auto build_graph = [&](ggml_context* ctx) -> ggml_tensor* {
        // 转换输入（零拷贝）
        ggml_tensor* ga = GGMLBridge::to_ggml(ctx, a);
        ggml_tensor* gb = GGMLBridge::to_ggml(ctx, b);
        
        // 矩阵乘法
        ggml_tensor* gc = ggml_mul_mat(ctx, ga, gb);
        
        // 可选的 bias
        if (bias.has_value()) {
            ggml_tensor* gb_bias = GGMLBridge::to_ggml(ctx, *bias);
            gc = ggml_add(ctx, gc, gb_bias);
        }
        
        return gc;
    };
    
    return backend.compute(build_graph, {a, b});
}

} // namespace xllm::kernel::spacemit
```

#### Task 4.2: 实现 rms_norm

```cpp
torch::Tensor rms_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float eps
) {
    GGMLBackend backend;
    
    auto build_graph = [&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = GGMLBridge::to_ggml(ctx, input);
        ggml_tensor* w = GGMLBridge::to_ggml(ctx, weight);
        
        // RMS Norm
        ggml_tensor* normed = ggml_rms_norm(ctx, x, eps);
        ggml_tensor* output = ggml_mul(ctx, normed, w);
        
        return output;
    };
    
    return backend.compute(build_graph, {input, weight});
}
```

#### Task 4.3: 其他算子

- `apply_rotary` - RoPE 位置编码
- `act_and_mul` - SwiGLU 激活
- `reshape_paged_cache` - KV Cache 操作

---

### 阶段 5：集成到 xLLM（3-4 天）

#### Task 5.1: 修改算子分发

```cpp
// xllm/core/kernels/ops_api.cpp
torch::Tensor matmul(MatmulParams& params) {
#if defined(USE_SPACEMIT)
  return spacemit::matmul(params.a, params.b, params.bias);
#elif defined(USE_MLU)
  return mlu::matmul(...);
#elif defined(USE_NPU)
  return npu::matmul(...);
#elif defined(USE_CUDA)
  return cuda::matmul(...);
#endif
}

// 同样修改其他算子...
```

#### Task 5.2: CMake 配置

```cmake
# cmake/spacemit.cmake

if(USE_SPACEMIT)
  # 添加编译选项
  add_compile_definitions(USE_SPACEMIT)
  add_compile_definitions(RISCV64_SPACEMIT_IME2)
  
  # 编译 ggml-spacemit
  add_library(ggml_spacemit STATIC
    third_party/ggml-spacemit/ime2_kernels.cpp
    third_party/ggml-spacemit/rvv_kernels.cpp
    third_party/ggml-spacemit/ime_env.cpp
  )
  
  # SpacemiT 平台源文件
  set(SPACEMIT_SOURCES
    xllm/core/platform/spacemit/ggml_bridge.cpp
    xllm/core/platform/spacemit/ggml_backend.cpp
    xllm/core/kernels/spacemit/matmul_ggml.cpp
    xllm/core/kernels/spacemit/rms_norm_ggml.cpp
  )
  
  target_sources(xllm_core PRIVATE ${SPACEMIT_SOURCES})
  target_link_libraries(xllm_core PRIVATE ggml_spacemit)
endif()
```

---

### 阶段 6：测试与验证（5-7 天）

#### Task 6.1: 单元测试套件

```cpp
// test/platform/spacemit/test_spacemit_ops.cpp

TEST(SpacemiTOpsTest, MatmulCorrectness) {
    torch::Tensor a = torch::randn({16, 512});
    torch::Tensor b = torch::randn({512, 1024});
    
    // SpacemiT 实现
    torch::Tensor c_spacemit = spacemit::matmul(a, b, std::nullopt);
    
    // PyTorch 参考实现
    torch::Tensor c_ref = torch::matmul(a, b);
    
    // 验证精度（误差 < 1e-3）
    ASSERT_TRUE(torch::allclose(c_spacemit, c_ref, 1e-3, 1e-3));
}

TEST(SpacemiTOpsTest, RMSNormCorrectness) {
    torch::Tensor input = torch::randn({16, 512});
    torch::Tensor weight = torch::ones({512});
    
    torch::Tensor output = spacemit::rms_norm(input, weight, 1e-5);
    
    // 验证输出不为 NaN/Inf
    ASSERT_FALSE(torch::isnan(output).any().item<bool>());
    ASSERT_FALSE(torch::isinf(output).any().item<bool>());
}
```

#### Task 6.2: 端到端测试

```python
# test/integration/test_qwen_spacemit.py

def test_qwen_inference_spacemit():
    # 加载模型
    model = xllm.LLM(
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        device="spacemit"
    )
    
    # 推理
    output = model.generate("Hello, how are you?", max_tokens=50)
    
    # 验证输出
    assert len(output) > 0
    assert not any(c in output for c in ['\x00', '�'])  # 无乱码
```

#### Task 6.3: 性能基准测试

```bash
# 在 K3 worker 上运行
sshpass -p 'bianbu' ssh bianbu@10.0.90.243

# 性能测试
./build/bin/xllm-bench \
  --model Qwen2.5-0.5B-Q4_0.gguf \
  --device spacemit \
  --batch-size 1 \
  --seq-len 128

# 预期结果：
# Decode: ~16.5 t/s (llama.cpp baseline: 16.49 t/s)
```

#### Task 6.4: 对比验证

```bash
# llama.cpp 基准
./llama-bench -m Qwen2.5-0.5B-Q4_0.gguf -p 128 -n 128

# xLLM 方案 A++
./xllm-bench --model Qwen2.5-0.5B-Q4_0.gguf --batch 1 --seq 128

# 对比输出一致性
./compare_outputs.py --llama output_llama.txt --xllm output_xllm.txt
```

---

## 4. 测试策略

### 4.1 测试金字塔

```
           /\
          /  \  E2E Tests (5%)
         /────\
        /      \  Integration Tests (15%)
       /────────\
      /          \  Unit Tests (80%)
     /────────────\
```

### 4.2 测试覆盖

| 测试类型 | 数量 | 覆盖范围 |
|---------|------|---------|
| **单元测试** | ~30 | 每个函数/类 |
| **集成测试** | ~10 | 算子组合、模型层 |
| **端到端测试** | ~3 | 完整推理流程 |
| **性能测试** | ~5 | 吞吐量、延迟 |

### 4.3 关键测试用例

1. **零拷贝验证**
   - 验证指针地址相同
   - 验证数据不被拷贝

2. **精度验证**
   - 对比 PyTorch 参考实现
   - 误差 < 1e-3

3. **性能验证**
   - 对比 llama.cpp 基准
   - 性能差距 < 5%

4. **功能完整性**
   - 所有算子正确性
   - Continuous Batching 工作
   - KV Cache 正常

5. **稳定性测试**
   - 连续运行 1000 次推理
   - 无内存泄漏
   - 无崩溃

---

## 5. 交付清单

### 5.1 代码交付

- [ ] `xllm/core/platform/spacemit/` - 平台层实现
- [ ] `xllm/core/kernels/spacemit/` - 算子实现
- [ ] `third_party/ggml-spacemit/` - ggml kernels
- [ ] `test/platform/spacemit/` - 单元测试
- [ ] `test/integration/` - 集成测试
- [ ] `cmake/spacemit.cmake` - 构建配置

### 5.2 文档交付

- [ ] `docs/spacemit-a-plus-plus-user-guide.md` - 用户指南
- [ ] `docs/spacemit-a-plus-plus-developer-guide.md` - 开发者指南
- [ ] `README-spacemit.md` - 快速开始

### 5.3 测试报告

- [ ] 单元测试覆盖率报告
- [ ] 性能测试报告（vs llama.cpp）
- [ ] 精度验证报告
- [ ] 稳定性测试报告

---

## 6. 时间估算

| 阶段 | 任务 | 时间 |
|------|------|------|
| 1 | 基础设施 | 2-3 天 |
| 2 | 零拷贝桥接 | 3-4 天 |
| 3 | ggml 后端 | 4-5 天 |
| 4 | 算子实现 | 5-6 天 |
| 5 | 集成 xLLM | 3-4 天 |
| 6 | 测试验证 | 5-7 天 |
| **总计** | | **22-29 天** |

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| ggml-spacemit 集成困难 | 中 | 高 | 提前验证单独编译 |
| 零拷贝生命周期问题 | 中 | 中 | 详细单元测试 |
| 性能不达标 | 低 | 高 | Profile 分析优化 |
| K3 环境问题 | 低 | 中 | 准备备用机器 |

---

## 8. 成功标准

✅ **功能标准**
- 所有算子单元测试通过
- 端到端推理成功
- 输出与 llama.cpp 一致

✅ **性能标准**
- Qwen3.5 2B Q4_0 Decode: ~16.5 t/s
- 与 llama.cpp 性能差距 < 5%
- 内存占用与 llama.cpp 相同

✅ **质量标准**
- 单元测试覆盖率 > 80%
- 无内存泄漏
- 连续运行 1000 次无崩溃

---

**Plan 准备完成，等待审批后开始实施！**
