# xLLM 接入 SpacemiT K3 平台技术分析（第二部分）

---

## 4. 核心差异：运行时流程对比

### 4.1 llama.cpp (GGUF Q4_0) 运行流程

```
═══════════════════════════════════════════════════════════
                    llama.cpp 推理流程
═══════════════════════════════════════════════════════════

[1] 模型加载阶段
────────────────────────────────────────────────────────
磁盘: Qwen2.5-7B-Q4_0.gguf (3.8 GB)
  ↓ mmap() 直接内存映射（零拷贝）
内存: INT4 量化权重 (已经是 INT4，无需转换)
  • layers.0.attention.wq: INT4 [4096, 4096]
  • layers.0.attention.wk: INT4 [512, 4096]
  • scale/zero_point: FP32 per block (256 elements)

加载时间: ~100ms (mmap 映射)
内存占用: 3.8 GB (仅权重)
────────────────────────────────────────────────────────

[2] 推理阶段 (单个 token 生成)
────────────────────────────────────────────────────────
输入: hidden_states [1, 4096] FP32
  ↓
【激活量化】spacemit_kernels::quantize_a_row_i8()
  hidden_states_fp32 → hidden_states_int8 + scale
  时间: ~5μs
  ↓
【矩阵乘法】ime2::gemm_kernel_i8i4() 
  • 输入: INT8 [1, 4096]
  • 权重: INT4 [4096, 4096] (从 mmap 直接读取)
  • 使用 IME2 加速 + TCM 缓存
  • 输出: FP32 [1, 4096]
  时间: ~150μs (A100 IME2)
  ↓
输出: hidden_states [1, 4096] FP32

总时间: ~155μs per layer
────────────────────────────────────────────────────────

优势分析:
✅ 权重已量化 → 无运行时量化开销
✅ mmap 零拷贝 → 启动快
✅ 内存占用小 → 可运行更大模型
✅ 直接调用 IME → 无额外抽象层开销
```

### 4.2 xLLM (方案 A) 运行流程

```
═══════════════════════════════════════════════════════════
                xLLM 方案 A 推理流程
═══════════════════════════════════════════════════════════

[1] 模型加载阶段
────────────────────────────────────────────────────────
磁盘: Qwen/Qwen2.5-7B-Instruct/*.safetensors (14 GB)
  ↓ safetensors::load() 
内存: torch::Tensor (FP16)
  • layers.0.attention.wq: FP16 [4096, 4096]
  • layers.0.attention.wk: FP16 [512, 4096]
  ↓
【首次推理时】权重量化 + 缓存
  FP16 → INT4 量化 (per-channel 或 per-block)
  存储到量化缓存 (额外内存)
  时间: ~500ms (一次性)
  ↓
内存布局:
  • 原始权重: torch::Tensor FP16 (14 GB)
  • 量化缓存: raw buffer INT4 (3.8 GB)

加载时间: ~2000ms (safetensors 加载) + 500ms (量化)
内存占用: 14 GB (FP16) + 3.8 GB (INT4 缓存) = 17.8 GB
────────────────────────────────────────────────────────

[2] 推理阶段 (单个 token 生成)
────────────────────────────────────────────────────────
输入: torch::Tensor hidden_states [1, 4096] FP32
  ↓
【PyTorch Tensor 操作】
  检查设备、数据类型、连续性
  时间: ~2μs
  ↓
【激活量化】spacemit::quantize_activation_i8()
  • torch::Tensor → raw pointer 转换
  • FP32 → INT8 量化
  • 重新包装为 torch::Tensor (optional，可优化)
  时间: ~8μs (PyTorch 开销 +3μs)
  ↓
【矩阵乘法】spacemit::matmul_impl()
  ┌────────────────────────────────────────┐
  │ 1. torch::Tensor → raw pointer         │
  │    hidden_int8.data_ptr<int8_t>()      │
  │    weight_int4.data_ptr<uint8_t>()     │
  │    时间: ~1μs                           │
  ├────────────────────────────────────────┤
  │ 2. 调用 IME kernel (与 llama.cpp 相同)│
  │    ime2::gemm_kernel_i8i4(...)         │
  │    时间: ~150μs                         │
  ├────────────────────────────────────────┤
  │ 3. raw pointer → torch::Tensor         │
  │    torch::from_blob(output_ptr, ...)   │
  │    时间: ~1μs                           │
  └────────────────────────────────────────┘
  时间: 152μs (IME) + 2μs (转换) = 154μs
  ↓
【PyTorch Tensor 返回】
  引用计数更新、设备信息记录
  时间: ~1μs
  ↓
输出: torch::Tensor [1, 4096] FP32

总时间: ~165μs per layer (比 llama.cpp 慢 6%)
────────────────────────────────────────────────────────

劣势分析:
❌ 内存占用 4.7x (17.8GB vs 3.8GB)
❌ 首次推理慢 (需要权重量化)
❌ PyTorch 开销 ~10μs per layer (6% 开销)

优势分析:
✅ 统一生态 (与 CUDA/NPU 共用代码)
✅ 灵活性高 (可动态切换量化策略)
✅ 兼容 HuggingFace 模型
```

### 4.3 关键性能瓶颈对比

| 操作 | llama.cpp | xLLM 方案 A | 差距 |
|------|-----------|-------------|------|
| **模型加载** | 100ms (mmap) | 2500ms (load + quant) | **25x** |
| **内存占用 (7B)** | 3.8 GB | 17.8 GB | **4.7x** |
| **单层推理** | 155μs | 165μs | **+6%** |
| **全模型推理** | 155μs × 32层 = 4.96ms | 165μs × 32层 = 5.28ms | **+6%** |
| **吞吐量 (tokens/s)** | 55.77 | ~52 | **-7%** |

**结论：推理性能差距可接受，内存占用是主要问题**

---

## 5. 接入方案设计

### 5.1 方案 A：标准 PyTorch Backend（推荐）

#### 目录结构

```
xllm/
├── core/
│   ├── platform/
│   │   ├── platform.h            # 扩展 is_spacemit()
│   │   ├── platform.cpp
│   │   └── spacemit/             # 新增 SpacemiT 平台
│   │       ├── platform_spacemit.h
│   │       ├── platform_spacemit.cpp
│   │       ├── device_spacemit.h
│   │       ├── device_spacemit.cpp
│   │       ├── allocator_spacemit.h      # 内存管理（含 TCM）
│   │       ├── allocator_spacemit.cpp
│   │       ├── stream_spacemit.h         # 流式执行（可选）
│   │       └── stream_spacemit.cpp
│   │
│   ├── kernels/
│   │   ├── ops_api.h             # 扩展算子接口
│   │   ├── ops_api.cpp           # 添加 spacemit 分发
│   │   └── spacemit/             # 新增 SpacemiT 算子
│   │       ├── ime_wrapper.h              # 封装 llama.cpp IME
│   │       ├── ime_wrapper.cpp
│   │       ├── linear_spacemit.cpp        # 矩阵乘法
│   │       ├── attention_spacemit.cpp     # 注意力
│   │       ├── activation_spacemit.cpp    # 激活函数
│   │       ├── rms_norm_spacemit.cpp      # RMSNorm
│   │       ├── rotary_spacemit.cpp        # RoPE
│   │       ├── quantization_spacemit.cpp  # 量化算子
│   │       └── paged_cache_spacemit.cpp   # KV Cache 操作
│   │
│   └── runtime/
│       └── spacemit/
│           ├── executor_impl_spacemit.h
│           ├── executor_impl_spacemit.cpp
│           └── worker_spacemit.cpp  # 可选：特化 Worker
│
├── third_party/
│   └── llama.cpp/
│       └── ggml/src/ggml-cpu/spacemit/  # 引入 llama.cpp IME kernels
│           ├── ime1_kernels.cpp
│           ├── ime2_kernels.cpp
│           ├── ime_kernels.h
│           ├── spine_tcm.h
│           └── ...
│
└── cmake/
    └── spacemit.cmake            # SpacemiT 编译配置
```

#### 核心实现步骤

##### Step 1: 扩展 Platform 类

```cpp
// xllm/core/platform/platform.h
class Platform final {
 public:
  // 新增 SpacemiT 平台检测
  static constexpr bool is_spacemit() {
#if defined(USE_SPACEMIT)
    return true;
#else
    return false;
#endif
  }
  
  // 运行时检测 IME 类型
  static bool is_spacemit_ime2();  // A100 (IME2 + TCM)
  static bool is_spacemit_ime1();  // X100 (IME1)
};
```

```cpp
// xllm/core/platform/platform.cpp
#if defined(USE_SPACEMIT)
#include "spacemit/platform_spacemit.h"
#endif

std::string Platform::type_str() {
#if defined(USE_SPACEMIT)
  return "spacemit";
#elif defined(USE_CUDA)
  return "cuda";
// ...
#endif
}

torch::DeviceType Platform::type_torch() {
#if defined(USE_SPACEMIT)
  // 选项 1: 复用 torch::kCPU (简单)
  return torch::kCPU;
  
  // 选项 2: 注册自定义设备 (复杂，更标准)
  // return torch::kPrivateUse1;  // 需要注册
#endif
}

bool Platform::is_spacemit_ime2() {
#if defined(USE_SPACEMIT)
  return spacemit::detect_ime_version() == 2;
#else
  return false;
#endif
}
```

##### Step 2: 实现 SpacemiT Platform

```cpp
// xllm/core/platform/spacemit/platform_spacemit.h
#pragma once

namespace xllm::spacemit {

enum class IMEVersion {
  NONE = 0,
  IME1 = 1,  // X100
  IME2 = 2,  // A100
};

// 检测 IME 版本
IMEVersion detect_ime_version();

// 检测 TCM 是否可用
bool is_tcm_available();

// 获取 CPU 核心数 (RISC-V)
int get_cpu_count();

} // namespace xllm::spacemit
```

```cpp
// xllm/core/platform/spacemit/platform_spacemit.cpp
#include "platform_spacemit.h"
#include <fstream>
#include <string>

namespace xllm::spacemit {

IMEVersion detect_ime_version() {
  // 方法 1: 读取 /proc/cpuinfo
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    // 查找 "uarch" 或 "model name"
    if (line.find("Spacemit(R) A100") != std::string::npos) {
      return IMEVersion::IME2;
    } else if (line.find("Spacemit(R) X60") != std::string::npos ||
               line.find("Spacemit(R) X100") != std::string::npos) {
      return IMEVersion::IME1;
    }
  }
  
  // 方法 2: 尝试调用 IME2 指令（需要运行时检测）
  // 如果 SIGILL 则降级到 IME1
  
  return IMEVersion::NONE;
}

bool is_tcm_available() {
  // TCM 只在 A100 (IME2) 上可用
  return detect_ime_version() == IMEVersion::IME2;
}

int get_cpu_count() {
  return std::thread::hardware_concurrency();
}

} // namespace xllm::spacemit
```

##### Step 3: 封装 llama.cpp IME Kernels

```cpp
// xllm/core/kernels/spacemit/ime_wrapper.h
#pragma once
#include <torch/torch.h>
#include <cstdint>

// 引入 llama.cpp 的 IME kernels（C++ 命名空间）
#include "ggml-cpu/spacemit/ime_kernels.h"

namespace xllm::kernel::spacemit {

// ═══════════════════════════════════════════════════════
// 1. 激活量化（输入 FP32/FP16 → INT8）
// ═══════════════════════════════════════════════════════
struct QuantizedActivation {
  torch::Tensor data;   // INT8 [M, K]
  torch::Tensor scale;  // FP32 [M] per-row scale
};

QuantizedActivation quantize_activation_i8(
    const torch::Tensor& input,  // FP32/FP16 [M, K]
    size_t block_len = 256       // 量化块大小
);

// ═══════════════════════════════════════════════════════
// 2. 权重量化（权重 FP16 → INT4，首次推理时调用）
// ═══════════════════════════════════════════════════════
struct QuantizedWeight {
  torch::Tensor data;       // UINT8 [N, K/2] packed INT4
  torch::Tensor scale;      // FP32 [N, K/block_len]
  torch::Tensor zero_point; // UINT8 [N, K/block_len] (可选)
  size_t block_len;
};

QuantizedWeight quantize_weight_i4(
    const torch::Tensor& weight,  // FP16 [N, K]
    size_t block_len = 256
);

// ═══════════════════════════════════════════════════════
// 3. IME 矩阵乘法（核心算子）
// ═══════════════════════════════════════════════════════
torch::Tensor ime_matmul(
    const torch::Tensor& input_int8,    // INT8 [M, K]
    const torch::Tensor& input_scale,   // FP32 [M]
    const torch::Tensor& weight_int4,   // UINT8 [N, K/2] packed
    const torch::Tensor& weight_scale,  // FP32 [N, K/block_len]
    const torch::Tensor& weight_zp,     // UINT8 [N, K/block_len] (optional)
    bool use_ime2,                      // true=A100, false=X100
    size_t block_len = 256
);

// ═══════════════════════════════════════════════════════
// 4. 端到端矩阵乘法（封装量化 + GEMM）
// ═══════════════════════════════════════════════════════
torch::Tensor matmul(
    const torch::Tensor& input,   // FP32/FP16 [M, K]
    const QuantizedWeight& weight // 预量化权重
);

} // namespace xllm::kernel::spacemit
```

```cpp
// xllm/core/kernels/spacemit/ime_wrapper.cpp
#include "ime_wrapper.h"
#include "xllm/core/platform/spacemit/platform_spacemit.h"
#include <cstring>

namespace xllm::kernel::spacemit {

// ────────────────────────────────────────────────────────
// 实现 1: 激活量化
// ────────────────────────────────────────────────────────
QuantizedActivation quantize_activation_i8(
    const torch::Tensor& input,
    size_t block_len
) {
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "Input must be 2D [M, K]");
  
  const int64_t M = input.size(0);
  const int64_t K = input.size(1);
  
  // 转换为 FP32（如果是 FP16）
  torch::Tensor input_fp32 = input.to(torch::kFloat32);
  
  // 分配输出
  torch::Tensor output_int8 = torch::empty(
      {M, K}, torch::dtype(torch::kInt8).device(torch::kCPU));
  torch::Tensor scale = torch::empty(
      {M}, torch::dtype(torch::kFloat32).device(torch::kCPU));
  
  // 调用 llama.cpp IME kernel
  const float* input_ptr = input_fp32.data_ptr<float>();
  int8_t* output_ptr = output_int8.data_ptr<int8_t>();
  float* scale_ptr = scale.data_ptr<float>();
  
  bool is_ime2 = xllm::spacemit::detect_ime_version() == 
                 xllm::spacemit::IMEVersion::IME2;
  
  if (is_ime2) {
    // 使用 IME2 kernel（支持 4 行并行）
    for (int64_t i = 0; i < M; i += 4) {
      int64_t rows = std::min(4L, M - i);
      if (rows == 4) {
        spacemit_kernels::ime2::quantize_a_4row_i8(
            block_len,
            input_ptr + i * K,
            K,
            reinterpret_cast<uint8_t*>(output_ptr + i * K)
        );
        // 提取 scale（llama.cpp 将 scale 写入前 4 个 float）
        std::memcpy(scale_ptr + i, output_ptr + i * K, 4 * sizeof(float));
      } else {
        // 处理剩余行（逐行）
        for (int64_t j = 0; j < rows; ++j) {
          spacemit_kernels::ime1::quantize_a_row_i8(
              block_len,
              input_ptr + (i + j) * K,
              K,
              reinterpret_cast<uint8_t*>(output_ptr + (i + j) * K)
          );
        }
      }
    }
  } else {
    // 使用 IME1 kernel（逐行）
    for (int64_t i = 0; i < M; ++i) {
      spacemit_kernels::ime1::quantize_a_row_i8(
          block_len,
          input_ptr + i * K,
          K,
          reinterpret_cast<uint8_t*>(output_ptr + i * K)
      );
    }
  }
  
  return {output_int8, scale};
}

// ────────────────────────────────────────────────────────
// 实现 2: IME 矩阵乘法
// ────────────────────────────────────────────────────────
torch::Tensor ime_matmul(
    const torch::Tensor& input_int8,
    const torch::Tensor& input_scale,
    const torch::Tensor& weight_int4,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zp,
    bool use_ime2,
    size_t block_len
) {
  const int64_t M = input_int8.size(0);
  const int64_t K = input_int8.size(1);
  const int64_t N = weight_int4.size(0);
  
  // 分配输出
  torch::Tensor output = torch::zeros(
      {M, N}, torch::dtype(torch::kFloat32).device(torch::kCPU));
  
  // 调用 llama.cpp IME GEMM kernel
  const uint8_t* input_ptr = 
      reinterpret_cast<const uint8_t*>(input_int8.data_ptr<int8_t>());
  const uint8_t* weight_ptr = weight_int4.data_ptr<uint8_t>();
  const uint8_t* zp_ptr = weight_zp.defined() ? 
      weight_zp.data_ptr<uint8_t>() : nullptr;
  float* output_ptr = output.data_ptr<float>();
  
  size_t k_blks = (K + block_len - 1) / block_len;
  
  if (use_ime2) {
    spacemit_kernels::ime2::gemm_kernel_i8i4(
        block_len,
        input_ptr,
        weight_ptr,
        zp_ptr,
        output_ptr,
        M,  // count_m
        N,  // count_n
        k_blks,
        N   // ldc (leading dimension of C)
    );
  } else {
    spacemit_kernels::ime1::gemm_kernel_i8i4(
        block_len,
        input_ptr,
        weight_ptr,
        zp_ptr,
        output_ptr,
        M, N, k_blks, N
    );
  }
  
  // 应用 scale（output * input_scale * weight_scale）
  // 这里简化处理，实际需要 per-channel 缩放
  output.mul_(input_scale.unsqueeze(1));
  
  return output;
}

} // namespace xllm::kernel::spacemit
```

**（第二部分完成，包含运行流程对比、方案 A 详细设计）**
