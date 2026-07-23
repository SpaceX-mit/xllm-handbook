# xLLM 接入 SpacemiT K3 平台技术分析（第三部分）

---

## 6. 方案 A 完整实施（续）

### Step 4: 实现算子适配器

```cpp
// xllm/core/kernels/spacemit/linear_spacemit.cpp
#include "ime_wrapper.h"
#include "xllm/core/kernels/ops_api.h"
#include <unordered_map>
#include <mutex>

namespace xllm::kernel::spacemit {

// ═══════════════════════════════════════════════════════
// 权重量化缓存（全局，首次推理时填充）
// ═══════════════════════════════════════════════════════
class WeightQuantCache {
 public:
  static WeightQuantCache& instance() {
    static WeightQuantCache cache;
    return cache;
  }
  
  // 获取或创建量化权重
  const QuantizedWeight& get_or_quantize(
      const std::string& key,
      const torch::Tensor& weight_fp16
  ) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;  // 缓存命中
    }
    
    // 首次访问，执行量化
    LOG(INFO) << "Quantizing weight: " << key 
              << " [" << weight_fp16.size(0) << ", " 
              << weight_fp16.size(1) << "]";
    
    auto quant_weight = quantize_weight_i4(weight_fp16, /*block_len=*/256);
    cache_[key] = std::move(quant_weight);
    return cache_[key];
  }
  
 private:
  std::unordered_map<std::string, QuantizedWeight> cache_;
  std::mutex mutex_;
};

} // namespace xllm::kernel::spacemit

namespace xllm::kernel {

// ═══════════════════════════════════════════════════════
// 矩阵乘法算子实现（被 ops_api.cpp 调用）
// ═══════════════════════════════════════════════════════
torch::Tensor matmul(MatmulParams& params) {
#if defined(USE_SPACEMIT)
  // 检测平台类型
  bool use_ime2 = Platform::is_spacemit_ime2();
  
  // 获取量化权重（首次会执行量化，后续从缓存读取）
  std::string weight_key = params.weight_name;  // 从参数传入
  const auto& quant_weight = 
      spacemit::WeightQuantCache::instance().get_or_quantize(
          weight_key, params.weight
      );
  
  // 执行矩阵乘法
  return spacemit::matmul(params.input, quant_weight);
  
#elif defined(USE_CUDA)
  return cuda::matmul(params);
#elif defined(USE_NPU)
  return npu::matmul(params);
  // ...
#endif
}

// ═══════════════════════════════════════════════════════
// RMSNorm 算子实现
// ═══════════════════════════════════════════════════════
std::tuple<torch::Tensor, torch::Tensor> rms_norm_dynamic_quant(
    RmsNormDynamicQuantParams& params
) {
#if defined(USE_SPACEMIT)
  // SpacemiT 没有专用 RMSNorm 硬件，使用 RVV 优化
  
  const auto& input = params.input;
  const auto& weight = params.weight;
  const float eps = params.eps;
  
  // 使用 PyTorch 实现（后续可用 RVV intrinsics 优化）
  auto variance = input.pow(2).mean(-1, /*keepdim=*/true);
  auto normalized = input * torch::rsqrt(variance + eps);
  auto output = normalized * weight;
  
  // 量化输出（如果需要）
  if (params.need_quantize) {
    auto quant = spacemit::quantize_activation_i8(output);
    return {quant.data, quant.scale};
  } else {
    return {output, torch::Tensor()};
  }
  
#elif defined(USE_CUDA)
  return cuda::rms_norm_dynamic_quant(params);
  // ...
#endif
}

// ═══════════════════════════════════════════════════════
// RoPE (Rotary Position Embedding) 实现
// ═══════════════════════════════════════════════════════
void apply_rotary(RotaryParams& params) {
#if defined(USE_SPACEMIT)
  // 使用 RVV 优化的 RoPE 实现
  // 或直接使用 PyTorch 实现（首版）
  
  auto& query = params.query;   // [batch, seq_len, num_heads, head_dim]
  auto& key = params.key;
  const auto& cos = params.cos_cache;
  const auto& sin = params.sin_cache;
  
  // 标准 RoPE 计算
  // q_rotated = q * cos + rotate_half(Tensor q) * sin
  auto rotate_half = [](torch::Tensor x) {
    auto x1 = x.slice(-1, 0, x.size(-1) / 2);
    auto x2 = x.slice(-1, x.size(-1) / 2);
    return torch::cat({-x2, x1}, /*dim=*/-1);
  };
  
  query = query * cos + rotate_half(query) * sin;
  key = key * cos + rotate_half(key) * sin;
  
#elif defined(USE_CUDA)
  cuda::apply_rotary(params);
  // ...
#endif
}

} // namespace xllm::kernel
```

### Step 5: 实现 Executor

```cpp
// xllm/core/runtime/spacemit/executor_impl_spacemit.h
#pragma once
#include "xllm/core/runtime/executor_impl.h"
#include "xllm/core/platform/spacemit/platform_spacemit.h"

namespace xllm::runtime {

class ExecutorImplSpacemiT : public ExecutorImpl {
 public:
  ExecutorImplSpacemiT(
      CausalLM* model,
      const ModelArgs& args,
      const torch::Device& device,
      const Options& options
  );
  
  ~ExecutorImplSpacemiT() override;
  
  // 准备前向输入
  ForwardInput prepare_inputs(Batch& batch) override;
  
  // 执行前向传播
  ModelOutput forward(
      const torch::Tensor& tokens,
      const torch::Tensor& positions,
      std::vector<KVCache>& kv_caches,
      const ModelInputParams& params
  ) override;
  
  // 图优化（SpacemiT 暂不支持，预留接口）
  void prepare_graph_input(...) override {}
  
 private:
  // IME 类型
  spacemit::IMEVersion ime_version_;
  
  // TCM 内存管理（仅 A100）
  std::unique_ptr<spacemit::TCMAllocator> tcm_allocator_;
  
  // 性能统计
  struct {
    int64_t total_forward_calls = 0;
    int64_t total_forward_time_us = 0;
  } stats_;
};

} // namespace xllm::runtime
```

```cpp
// xllm/core/runtime/spacemit/executor_impl_spacemit.cpp
#include "executor_impl_spacemit.h"
#include "xllm/core/kernels/ops_api.h"

namespace xllm::runtime {

ExecutorImplSpacemiT::ExecutorImplSpacemiT(
    CausalLM* model,
    const ModelArgs& args,
    const torch::Device& device,
    const Options& options
) : ExecutorImpl(model, args, device, options) {
  
  // 检测 IME 版本
  ime_version_ = spacemit::detect_ime_version();
  LOG(INFO) << "Detected IME version: " 
            << static_cast<int>(ime_version_);
  
  // 初始化 TCM allocator（如果是 A100）
  if (ime_version_ == spacemit::IMEVersion::IME2) {
    if (spacemit::is_tcm_available()) {
      tcm_allocator_ = std::make_unique<spacemit::TCMAllocator>();
      LOG(INFO) << "TCM allocator initialized";
    }
  }
}

ExecutorImplSpacemiT::~ExecutorImplSpacemiT() {
  LOG(INFO) << "SpacemiT Executor stats:";
  LOG(INFO) << "  Total forward calls: " << stats_.total_forward_calls;
  LOG(INFO) << "  Average forward time: " 
            << (stats_.total_forward_time_us / 
                std::max(1L, stats_.total_forward_calls)) 
            << " us";
}

ForwardInput ExecutorImplSpacemiT::prepare_inputs(Batch& batch) {
  // 准备输入张量
  // SpacemiT 使用 CPU 设备，数据已经在主机内存
  ForwardInput input;
  
  // 收集 token IDs
  std::vector<int32_t> token_ids;
  std::vector<int32_t> positions;
  
  for (auto* seq : batch.sequences_) {
    const auto& tokens = seq->get_token_ids();
    token_ids.insert(token_ids.end(), tokens.begin(), tokens.end());
    
    // 生成 position IDs
    int32_t start_pos = seq->num_computed_tokens();
    for (size_t i = 0; i < tokens.size(); ++i) {
      positions.push_back(start_pos + i);
    }
  }
  
  // 转换为 torch::Tensor
  input.tokens = torch::from_blob(
      token_ids.data(),
      {static_cast<int64_t>(token_ids.size())},
      torch::kInt32
  ).clone();  // clone 避免悬空指针
  
  input.positions = torch::from_blob(
      positions.data(),
      {static_cast<int64_t>(positions.size())},
      torch::kInt32
  ).clone();
  
  return input;
}

ModelOutput ExecutorImplSpacemiT::forward(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& params
) {
  auto start = std::chrono::high_resolution_clock::now();
  
  // 调用模型的前向传播
  // 模型内部会调用 xllm::kernel::* 算子
  // 这些算子会自动路由到 spacemit 实现
  ModelOutput output = model_->forward(
      tokens, positions, kv_caches, params
  );
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_us = 
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  
  stats_.total_forward_calls++;
  stats_.total_forward_time_us += duration_us;
  
  return output;
}

} // namespace xllm::runtime
```

### Step 6: CMake 配置

```cmake
# cmake/spacemit.cmake

if(NOT USE_SPACEMIT)
  return()
endif()

message(STATUS "Building with SpacemiT support")

# ═══════════════════════════════════════════════════════
# 1. 检测工具链
# ═══════════════════════════════════════════════════════
if(NOT DEFINED ENV{SPACEMIT_TOOLCHAIN_PATH})
  message(FATAL_ERROR 
    "SpacemiT toolchain not found. Please set SPACEMIT_TOOLCHAIN_PATH:\n"
    "  export SPACEMIT_TOOLCHAIN_PATH=/path/to/spacemit-toolchain-linux-glibc-x86_64-v1.2.7\n"
    "  Or download from: https://nexus.bianbu.xyz/repository/toolchain/llvm-gcc/..."
  )
endif()

set(SPACEMIT_TOOLCHAIN $ENV{SPACEMIT_TOOLCHAIN_PATH})
message(STATUS "Using SpacemiT toolchain: ${SPACEMIT_TOOLCHAIN}")

# ═══════════════════════════════════════════════════════
# 2. 设置交叉编译工具链（如果是 x86 开发机）
# ═══════════════════════════════════════════════════════
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(STATUS "Cross-compiling for riscv64")
  
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR riscv64)
  
  set(CMAKE_C_COMPILER ${SPACEMIT_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER ${SPACEMIT_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)
  
  set(CMAKE_FIND_ROOT_PATH ${SPACEMIT_TOOLCHAIN}/sysroot)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
else()
  message(STATUS "Native compilation on riscv64")
endif()

# ═══════════════════════════════════════════════════════
# 3. 编译选项
# ═══════════════════════════════════════════════════════
add_compile_definitions(USE_SPACEMIT)

# RISC-V ISA 扩展
set(RISCV_ARCH "rv64gcv")  # 基础指令集 + 向量扩展
set(RISCV_ARCH "${RISCV_ARCH}_zfh")      # FP16 支持
set(RISCV_ARCH "${RISCV_ARCH}_zba_zbb_zbc_zbs")  # 位操作扩展

add_compile_options(
  -march=${RISCV_ARCH}
  -mabi=lp64d
  -mtune=spacemit-x60  # 或 spacemit-k1
)

# IME 版本选择（根据目标平台）
option(SPACEMIT_USE_IME2 "Build for A100 (IME2 + TCM)" ON)
option(SPACEMIT_USE_IME1 "Build for X100 (IME1)" OFF)

if(SPACEMIT_USE_IME2)
  add_compile_definitions(RISCV64_SPACEMIT_IME2)
  message(STATUS "Building for SpacemiT A100 (IME2 + TCM)")
elseif(SPACEMIT_USE_IME1)
  add_compile_definitions(RISCV64_SPACEMIT_IME1)
  message(STATUS "Building for SpacemiT X100 (IME1)")
else()
  message(WARNING "No IME version specified, falling back to RVV")
endif()

# ═══════════════════════════════════════════════════════
# 4. 引入 llama.cpp IME kernels
# ═══════════════════════════════════════════════════════
set(LLAMA_CPP_SPACEMIT_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/src/ggml-cpu/spacemit)

if(NOT EXISTS ${LLAMA_CPP_SPACEMIT_DIR})
  message(FATAL_ERROR 
    "llama.cpp spacemit kernels not found at: ${LLAMA_CPP_SPACEMIT_DIR}\n"
    "Please copy from llama.cpp project:\n"
    "  cp -r /path/to/llama.cpp/ggml/src/ggml-cpu/spacemit third_party/llama.cpp/ggml/src/ggml-cpu/"
  )
endif()

# 编译 IME kernels 为静态库
add_library(ggml_spacemit STATIC
  ${LLAMA_CPP_SPACEMIT_DIR}/ime1_kernels.cpp
  ${LLAMA_CPP_SPACEMIT_DIR}/ime2_kernels.cpp
  ${LLAMA_CPP_SPACEMIT_DIR}/rvv_kernels.cpp
  ${LLAMA_CPP_SPACEMIT_DIR}/ime_env.cpp
  ${LLAMA_CPP_SPACEMIT_DIR}/spine_mem_pool.cpp
  ${LLAMA_CPP_SPACEMIT_DIR}/repack.cpp
)

target_include_directories(ggml_spacemit PUBLIC
  ${LLAMA_CPP_SPACEMIT_DIR}
)

# ═══════════════════════════════════════════════════════
# 5. 添加 SpacemiT 源文件
# ═══════════════════════════════════════════════════════
set(SPACEMIT_SOURCES
  xllm/core/platform/spacemit/platform_spacemit.cpp
  xllm/core/platform/spacemit/device_spacemit.cpp
  xllm/core/platform/spacemit/allocator_spacemit.cpp
  xllm/core/kernels/spacemit/ime_wrapper.cpp
  xllm/core/kernels/spacemit/linear_spacemit.cpp
  xllm/core/kernels/spacemit/activation_spacemit.cpp
  xllm/core/kernels/spacemit/rms_norm_spacemit.cpp
  xllm/core/runtime/spacemit/executor_impl_spacemit.cpp
)

target_sources(xllm_core PRIVATE ${SPACEMIT_SOURCES})

# 链接 IME kernels
target_link_libraries(xllm_core PRIVATE ggml_spacemit)

# ═══════════════════════════════════════════════════════
# 6. PyTorch 配置（使用 CPU backend）
# ═══════════════════════════════════════════════════════
# SpacemiT 使用 torch::kCPU 设备类型
# 无需额外配置

message(STATUS "SpacemiT build configuration complete")
```

### Step 7: 编译脚本

```bash
#!/bin/bash
# build_spacemit.sh

set -e

# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════
XLLM_ROOT=$(pwd)
BUILD_DIR=${XLLM_ROOT}/build_spacemit
TOOLCHAIN_URL="https://nexus.bianbu.xyz/repository/toolchain/llvm-gcc/spacemit-toolchain-linux-glibc-x86_64-v1.2.7.tar.xz"
TOOLCHAIN_DIR=${XLLM_ROOT}/toolchain/spacemit-v1.2.7

# 目标平台
TARGET_PLATFORM="ime2"  # 或 "ime1"

# ═══════════════════════════════════════════════════════
# 1. 下载工具链（如果不存在）
# ═══════════════════════════════════════════════════════
if [ ! -d "${TOOLCHAIN_DIR}" ]; then
  echo "Downloading SpacemiT toolchain..."
  mkdir -p $(dirname ${TOOLCHAIN_DIR})
  wget -O /tmp/spacemit-toolchain.tar.xz ${TOOLCHAIN_URL}
  tar -xf /tmp/spacemit-toolchain.tar.xz -C $(dirname ${TOOLCHAIN_DIR})
  rm /tmp/spacemit-toolchain.tar.xz
fi

export SPACEMIT_TOOLCHAIN_PATH=${TOOLCHAIN_DIR}

# ═══════════════════════════════════════════════════════
# 2. 复制 llama.cpp IME kernels
# ═══════════════════════════════════════════════════════
LLAMA_CPP_SRC="/data/workspace2026-new/work0618/tech-analyais0713/llama.cpp-handbook"
SPACEMIT_KERNELS_SRC="${LLAMA_CPP_SRC}/ggml/src/ggml-cpu/spacemit"
SPACEMIT_KERNELS_DST="${XLLM_ROOT}/third_party/llama.cpp/ggml/src/ggml-cpu/spacemit"

if [ ! -d "${SPACEMIT_KERNELS_DST}" ]; then
  echo "Copying llama.cpp spacemit kernels..."
  mkdir -p $(dirname ${SPACEMIT_KERNELS_DST})
  cp -r ${SPACEMIT_KERNELS_SRC} ${SPACEMIT_KERNELS_DST}
fi

# ═══════════════════════════════════════════════════════
# 3. 配置 CMake
# ═══════════════════════════════════════════════════════
cmake -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SPACEMIT=ON \
  -DSPACEMIT_USE_IME2=$([ "$TARGET_PLATFORM" = "ime2" ] && echo "ON" || echo "OFF") \
  -DSPACEMIT_USE_IME1=$([ "$TARGET_PLATFORM" = "ime1" ] && echo "ON" || echo "OFF") \
  -DCMAKE_INSTALL_PREFIX=${BUILD_DIR}/install \
  -DGGML_RVV=ON \
  -DGGML_RV_ZVFH=ON \
  -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZICBOP=ON \
  -DGGML_RV_ZIHINTPAUSE=ON \
  -DGGML_RV_ZBA=ON

# ═══════════════════════════════════════════════════════
# 4. 编译
# ═══════════════════════════════════════════════════════
cmake --build ${BUILD_DIR} --parallel $(nproc) --config Release

# ═══════════════════════════════════════════════════════
# 5. 安装
# ═══════════════════════════════════════════════════════
cmake --install ${BUILD_DIR}

echo "Build complete! Output: ${BUILD_DIR}/install"
echo ""
echo "To run on SpacemiT K3:"
echo "  sshpass -p 'bianbu' scp -r ${BUILD_DIR}/install bianbu@10.0.90.243:/home/bianbu/"
echo "  sshpass -p 'bianbu' ssh bianbu@10.0.90.243"
echo "  cd /home/bianbu/install && ./bin/xllm --model /path/to/model"
```

---

## 7. 性能优化策略

### 7.1 内存优化：支持 GGUF 格式（方案 A+）

```cpp
// xllm/core/framework/gguf_model_loader.h
#pragma once
#include <torch/torch.h>
#include <string>
#include <unordered_map>

extern "C" {
#include "gguf.h"  // 来自 llama.cpp
}

namespace xllm {

class GGUFModelLoader {
 public:
  // 加载 GGUF 模型
  std::unordered_map<std::string, torch::Tensor> load(
      const std::string& gguf_path
  );
  
 private:
  // GGUF 类型 → torch dtype 映射
  torch::Dtype gguf_type_to_torch(enum ggml_type type);
  
  // 解析 GGUF 元数据
  ModelConfig parse_metadata(struct gguf_context* ctx);
};

} // namespace xllm
```

**优势：**
- 内存占用降低到与 llama.cpp 相同 (3.8GB vs 17.8GB)
- 启动速度提升 (无需运行时量化)
- 兼容 llama.cpp 生态模型

### 7.2 计算优化：零拷贝 Tensor 封装

```cpp
// xllm/core/kernels/spacemit/zero_copy_tensor.h
namespace xllm::kernel::spacemit {

// 零拷贝包装：raw buffer → torch::Tensor (不拷贝数据)
torch::Tensor wrap_as_tensor(
    void* data,
    std::vector<int64_t> sizes,
    torch::Dtype dtype,
    std::function<void(void*)> deleter = nullptr
) {
  return torch::from_blob(
      data, sizes, deleter,
      torch::TensorOptions().dtype(dtype).device(torch::kCPU)
  );
}

} // namespace xllm::kernel::spacemit
```

**优势：**
- 消除 torch::Tensor ↔ raw pointer 转换开销
- 减少内存拷贝

### 7.3 并行优化：多核调度

```cpp
// xllm/core/runtime/spacemit/executor_impl_spacemit.cpp

// 使用 OpenMP 并行处理 batch
ModelOutput ExecutorImplSpacemiT::forward(...) {
  const int num_threads = spacemit::get_cpu_count();
  
  #pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < batch_size; ++i) {
    // 并行处理每个序列
    process_sequence(i);
  }
  
  return output;
}
```

### 7.4 TCM 优化（A100 专用）

```cpp
// xllm/core/platform/spacemit/tcm_allocator.cpp
namespace xllm::spacemit {

class TCMAllocator {
 public:
  // 分配 TCM 内存（低延迟）
  void* allocate_tcm(size_t size);
  
  // 预加载权重到 TCM
  void prefetch_weight_to_tcm(const torch::Tensor& weight);
  
 private:
  void* tcm_base_;
  size_t tcm_size_;
  size_t tcm_used_;
};

} // namespace xllm::spacemit
```

**（第三部分完成，包含完整实施步骤、CMake 配置、性能优化）**
