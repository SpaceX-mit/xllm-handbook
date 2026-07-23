# SpacemiT K3 平台 CMake 配置详解

> **用于 xLLM 项目的 SpacemiT 平台编译配置**

---

## 1. 顶层 CMakeLists.txt 修改

### 1.1 添加 SpacemiT 选项

```cmake
# CMakeLists.txt (项目根目录)

cmake_minimum_required(VERSION 3.18)
project(xllm VERSION 1.0.0 LANGUAGES CXX C)

# ═══════════════════════════════════════════════════════
# 平台选项
# ═══════════════════════════════════════════════════════
option(USE_CUDA "Build with CUDA support" OFF)
option(USE_NPU "Build with Ascend NPU support" OFF)
option(USE_MLU "Build with Cambricon MLU support" OFF)
option(USE_DCU "Build with Hygon DCU support" OFF)
option(USE_MUSA "Build with Moore Threads MUSA support" OFF)
option(USE_ILU "Build with Iluvatar ILU support" OFF)
option(USE_SPACEMIT "Build with SpacemiT support" OFF)  # 新增

# SpacemiT 子选项
if(USE_SPACEMIT)
  option(SPACEMIT_USE_IME2 "Build for SpacemiT A100 (IME2 + TCM)" ON)
  option(SPACEMIT_USE_IME1 "Build for SpacemiT X100 (IME1)" OFF)
  option(SPACEMIT_ENABLE_GGUF "Enable GGUF format support" ON)
endif()

# ═══════════════════════════════════════════════════════
# 引入平台配置
# ═══════════════════════════════════════════════════════
if(USE_SPACEMIT)
  include(cmake/spacemit.cmake)
endif()

# ═══════════════════════════════════════════════════════
# 添加子目录
# ═══════════════════════════════════════════════════════
add_subdirectory(xllm)
```

---

## 2. SpacemiT 平台配置文件

### 2.1 cmake/spacemit.cmake

```cmake
# cmake/spacemit.cmake

if(NOT USE_SPACEMIT)
  return()
endif()

message(STATUS "======================================")
message(STATUS "  SpacemiT Platform Configuration")
message(STATUS "======================================")

# ═══════════════════════════════════════════════════════
# 1. 检测工具链
# ═══════════════════════════════════════════════════════
if(NOT DEFINED ENV{SPACEMIT_TOOLCHAIN_PATH})
  message(FATAL_ERROR 
    "SpacemiT toolchain not found. Please set SPACEMIT_TOOLCHAIN_PATH:\n"
    "  export SPACEMIT_TOOLCHAIN_PATH=/path/to/spacemit-toolchain-linux-glibc-x86_64-v1.2.7\n"
    "\n"
    "Download from:\n"
    "  https://nexus.bianbu.xyz/repository/toolchain/llvm-gcc/spacemit-toolchain-linux-glibc-x86_64-v1.2.7.tar.xz\n"
    "\n"
    "Or use backup version:\n"
    "  https://nexus.bianbu.xyz/repository/toolchain/llvm-gcc/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz"
  )
endif()

set(SPACEMIT_TOOLCHAIN $ENV{SPACEMIT_TOOLCHAIN_PATH})
message(STATUS "Using SpacemiT toolchain: ${SPACEMIT_TOOLCHAIN}")

# 验证工具链存在
if(NOT EXISTS "${SPACEMIT_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc")
  message(FATAL_ERROR 
    "SpacemiT toolchain not found at: ${SPACEMIT_TOOLCHAIN}\n"
    "Please check SPACEMIT_TOOLCHAIN_PATH environment variable"
  )
endif()

# ═══════════════════════════════════════════════════════
# 2. 设置交叉编译工具链（如果需要）
# ═══════════════════════════════════════════════════════
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(STATUS "Cross-compiling from x86_64 to riscv64")
  
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR riscv64)
  
  # 设置编译器
  set(CMAKE_C_COMPILER ${SPACEMIT_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER ${SPACEMIT_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)
  set(CMAKE_AR ${SPACEMIT_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ar)
  set(CMAKE_RANLIB ${SPACEMIT_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ranlib)
  
  # 设置 sysroot
  set(CMAKE_SYSROOT ${SPACEMIT_TOOLCHAIN}/sysroot)
  set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
  
  # 查找策略
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
  
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "riscv64")
  message(STATUS "Native compilation on riscv64")
  
  # 本地编译，使用系统编译器
  # 但仍需确保使用 SpacemiT 工具链（支持 IME 指令）
  if(NOT CMAKE_C_COMPILER MATCHES "spacemit")
    message(WARNING 
      "Using system compiler on riscv64. "
      "For best performance, use SpacemiT toolchain."
    )
  endif()
  
else()
  message(FATAL_ERROR 
    "Unsupported host architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}\n"
    "SpacemiT build requires x86_64 (cross-compile) or riscv64 (native)"
  )
endif()

# ═══════════════════════════════════════════════════════
# 3. 编译选项
# ═══════════════════════════════════════════════════════
add_compile_definitions(USE_SPACEMIT)

# RISC-V ISA 扩展
set(RISCV_ARCH "rv64gcv")  # 基础指令集 + 向量扩展

# 添加扩展
set(RISCV_ARCH "${RISCV_ARCH}_zfh")      # FP16 支持
set(RISCV_ARCH "${RISCV_ARCH}_zba")      # 地址生成
set(RISCV_ARCH "${RISCV_ARCH}_zbb")      # 基础位操作
set(RISCV_ARCH "${RISCV_ARCH}_zbc")      # 进位位操作
set(RISCV_ARCH "${RISCV_ARCH}_zbs")      # 单比特操作

message(STATUS "RISC-V ISA: ${RISCV_ARCH}")

# 编译器标志
add_compile_options(
  -march=${RISCV_ARCH}
  -mabi=lp64d
  -mtune=spacemit-x60
)

# 优化标志
add_compile_options(
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-DNDEBUG>
  $<$<CONFIG:Release>:-ffast-math>
  $<$<CONFIG:Release>:-funroll-loops>
)

# Vector 扩展启用
add_compile_definitions(
  GGML_USE_RVV
  __riscv_v
  __riscv_v_intrinsic
  __riscv_zfh
)

# IME 版本选择
if(SPACEMIT_USE_IME2)
  add_compile_definitions(RISCV64_SPACEMIT_IME2)
  message(STATUS "Building for SpacemiT A100 (IME2 + TCM)")
  set(SPACEMIT_TARGET_NAME "A100")
elseif(SPACEMIT_USE_IME1)
  add_compile_definitions(RISCV64_SPACEMIT_IME1)
  message(STATUS "Building for SpacemiT X100 (IME1)")
  set(SPACEMIT_TARGET_NAME "X100")
else()
  message(WARNING "No IME version specified, will fallback to RVV")
  set(SPACEMIT_TARGET_NAME "RVV")
endif()

# ═══════════════════════════════════════════════════════
# 4. 依赖项检查
# ═══════════════════════════════════════════════════════

# 检查 PyTorch
find_package(Torch REQUIRED)
message(STATUS "PyTorch version: ${Torch_VERSION}")
message(STATUS "PyTorch include: ${TORCH_INCLUDE_DIRS}")
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")

# ═══════════════════════════════════════════════════════
# 5. 引入 llama.cpp IME kernels
# ═══════════════════════════════════════════════════════
set(LLAMA_CPP_ROOT ${CMAKE_SOURCE_DIR}/third_party/llama.cpp)
set(LLAMA_CPP_SPACEMIT_DIR ${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu/spacemit)

if(NOT EXISTS ${LLAMA_CPP_SPACEMIT_DIR})
  message(WARNING 
    "llama.cpp spacemit kernels not found at:\n"
    "  ${LLAMA_CPP_SPACEMIT_DIR}\n"
    "\n"
    "Please copy from llama.cpp project:\n"
    "  mkdir -p ${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu\n"
    "  cp -r /path/to/llama.cpp/ggml/src/ggml-cpu/spacemit ${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu/\n"
    "\n"
    "SpacemiT backend will use fallback implementations"
  )
  set(SPACEMIT_HAS_IME_KERNELS FALSE)
else()
  message(STATUS "Found llama.cpp IME kernels: ${LLAMA_CPP_SPACEMIT_DIR}")
  set(SPACEMIT_HAS_IME_KERNELS TRUE)
endif()

# 编译 IME kernels 为静态库
if(SPACEMIT_HAS_IME_KERNELS)
  set(GGML_SPACEMIT_SOURCES
    ${LLAMA_CPP_SPACEMIT_DIR}/ime_env.cpp
    ${LLAMA_CPP_SPACEMIT_DIR}/repack.cpp
    ${LLAMA_CPP_SPACEMIT_DIR}/rvv_kernels.cpp
    ${LLAMA_CPP_SPACEMIT_DIR}/spine_mem_pool.cpp
  )
  
  # 根据 IME 版本添加对应的 kernels
  if(SPACEMIT_USE_IME1)
    list(APPEND GGML_SPACEMIT_SOURCES
      ${LLAMA_CPP_SPACEMIT_DIR}/ime1_kernels.cpp
    )
  endif()
  
  if(SPACEMIT_USE_IME2)
    list(APPEND GGML_SPACEMIT_SOURCES
      ${LLAMA_CPP_SPACEMIT_DIR}/ime2_kernels.cpp
    )
  endif()
  
  add_library(ggml_spacemit STATIC ${GGML_SPACEMIT_SOURCES})
  
  target_include_directories(ggml_spacemit PUBLIC
    ${LLAMA_CPP_SPACEMIT_DIR}
    ${LLAMA_CPP_ROOT}/ggml/include
    ${LLAMA_CPP_ROOT}/include
  )
  
  target_compile_options(ggml_spacemit PRIVATE
    -Wno-unused-parameter
    -Wno-cast-qual
  )
  
  # 安装
  install(TARGETS ggml_spacemit
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()

# ═══════════════════════════════════════════════════════
# 6. GGUF 支持（可选）
# ═══════════════════════════════════════════════════════
if(SPACEMIT_ENABLE_GGUF)
  message(STATUS "GGUF format support: ENABLED")
  add_compile_definitions(SPACEMIT_ENABLE_GGUF)
  
  # 需要 llama.cpp 的 GGUF 库
  if(NOT EXISTS ${LLAMA_CPP_ROOT}/src/gguf.cpp)
    message(WARNING 
      "llama.cpp GGUF implementation not found.\n"
      "GGUF support will be disabled."
    )
    set(SPACEMIT_ENABLE_GGUF OFF)
  else()
    # 编译 GGUF 库
    add_library(gguf STATIC
      ${LLAMA_CPP_ROOT}/src/gguf.cpp
      ${LLAMA_CPP_ROOT}/ggml/src/ggml.c
      ${LLAMA_CPP_ROOT}/ggml/src/ggml-alloc.c
    )
    
    target_include_directories(gguf PUBLIC
      ${LLAMA_CPP_ROOT}/include
      ${LLAMA_CPP_ROOT}/ggml/include
    )
    
    install(TARGETS gguf
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
    )
  endif()
else()
  message(STATUS "GGUF format support: DISABLED")
endif()

# ═══════════════════════════════════════════════════════
# 7. 导出配置变量
# ═══════════════════════════════════════════════════════
set(SPACEMIT_INCLUDE_DIRS
  ${LLAMA_CPP_SPACEMIT_DIR}
  ${LLAMA_CPP_ROOT}/ggml/include
  ${LLAMA_CPP_ROOT}/include
  CACHE INTERNAL "SpacemiT include directories"
)

set(SPACEMIT_LIBRARIES
  CACHE INTERNAL "SpacemiT libraries"
)

if(SPACEMIT_HAS_IME_KERNELS)
  list(APPEND SPACEMIT_LIBRARIES ggml_spacemit)
endif()

if(SPACEMIT_ENABLE_GGUF)
  list(APPEND SPACEMIT_LIBRARIES gguf)
endif()

# ═══════════════════════════════════════════════════════
# 8. 打印配置摘要
# ═══════════════════════════════════════════════════════
message(STATUS "======================================")
message(STATUS "SpacemiT Configuration Summary:")
message(STATUS "  Target: ${SPACEMIT_TARGET_NAME}")
message(STATUS "  Toolchain: ${SPACEMIT_TOOLCHAIN}")
message(STATUS "  IME Kernels: ${SPACEMIT_HAS_IME_KERNELS}")
message(STATUS "  GGUF Support: ${SPACEMIT_ENABLE_GGUF}")
message(STATUS "  ISA: ${RISCV_ARCH}")
message(STATUS "======================================")
```

---

## 3. xLLM 核心库 CMakeLists.txt 修改

### 3.1 xllm/core/CMakeLists.txt

```cmake
# xllm/core/CMakeLists.txt

# ═══════════════════════════════════════════════════════
# xLLM 核心库
# ═══════════════════════════════════════════════════════
set(XLLM_CORE_SOURCES
  # ... 现有源文件 ...
)

# ───────────────────────────────────────────────────────
# SpacemiT 平台源文件
# ───────────────────────────────────────────────────────
if(USE_SPACEMIT)
  list(APPEND XLLM_CORE_SOURCES
    # Platform 层
    platform/spacemit/platform_spacemit.cpp
    platform/spacemit/device_spacemit.cpp
    platform/spacemit/allocator_spacemit.cpp
    
    # Kernels 层
    kernels/spacemit/ime_wrapper.cpp
    kernels/spacemit/linear_spacemit.cpp
    kernels/spacemit/activation_spacemit.cpp
    kernels/spacemit/rms_norm_spacemit.cpp
    kernels/spacemit/rotary_spacemit.cpp
    kernels/spacemit/paged_cache_spacemit.cpp
    
    # Runtime 层
    runtime/spacemit/executor_impl_spacemit.cpp
  )
  
  # GGUF 支持
  if(SPACEMIT_ENABLE_GGUF)
    list(APPEND XLLM_CORE_SOURCES
      framework/gguf_model_loader.cpp
    )
  endif()
endif()

# ═══════════════════════════════════════════════════════
# 构建核心库
# ═══════════════════════════════════════════════════════
add_library(xllm_core SHARED ${XLLM_CORE_SOURCES})

# ───────────────────────────────────────────────────────
# 头文件路径
# ───────────────────────────────────────────────────────
target_include_directories(xllm_core PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/xllm
  ${TORCH_INCLUDE_DIRS}
)

if(USE_SPACEMIT)
  target_include_directories(xllm_core PUBLIC
    ${SPACEMIT_INCLUDE_DIRS}
  )
endif()

# ───────────────────────────────────────────────────────
# 链接库
# ───────────────────────────────────────────────────────
target_link_libraries(xllm_core PUBLIC
  ${TORCH_LIBRARIES}
  # ... 其他依赖 ...
)

if(USE_SPACEMIT)
  target_link_libraries(xllm_core PRIVATE
    ${SPACEMIT_LIBRARIES}
  )
endif()

# ───────────────────────────────────────────────────────
# 安装
# ───────────────────────────────────────────────────────
install(TARGETS xllm_core
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
```

---

## 4. 使用示例

### 4.1 配置与编译（交叉编译，x86 → riscv64）

```bash
#!/bin/bash
# build_spacemit_cross.sh

set -e

# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════
XLLM_ROOT=$(pwd)
BUILD_DIR=${XLLM_ROOT}/build_spacemit
INSTALL_DIR=${BUILD_DIR}/install

# SpacemiT 工具链路径
export SPACEMIT_TOOLCHAIN_PATH=/opt/spacemit-toolchain-v1.2.7

# 目标平台
TARGET_PLATFORM="ime2"  # 或 "ime1"

# ═══════════════════════════════════════════════════════
# 清理旧构建
# ═══════════════════════════════════════════════════════
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

# ═══════════════════════════════════════════════════════
# CMake 配置
# ═══════════════════════════════════════════════════════
cmake -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  \
  -DUSE_SPACEMIT=ON \
  -DSPACEMIT_USE_IME2=$([ "$TARGET_PLATFORM" = "ime2" ] && echo "ON" || echo "OFF") \
  -DSPACEMIT_USE_IME1=$([ "$TARGET_PLATFORM" = "ime1" ] && echo "ON" || echo "OFF") \
  -DSPACEMIT_ENABLE_GGUF=ON \
  \
  -DUSE_CUDA=OFF \
  -DUSE_NPU=OFF \
  -DUSE_MLU=OFF

# ═══════════════════════════════════════════════════════
# 编译
# ═══════════════════════════════════════════════════════
cmake --build ${BUILD_DIR} \
  --parallel $(nproc) \
  --config Release \
  --target all

# ═══════════════════════════════════════════════════════
# 安装
# ═══════════════════════════════════════════════════════
cmake --install ${BUILD_DIR}

# ═══════════════════════════════════════════════════════
# 打包
# ═══════════════════════════════════════════════════════
cd ${BUILD_DIR}
tar -czf xllm-spacemit-${TARGET_PLATFORM}.tar.gz install/

echo "Build complete!"
echo "  Output: ${BUILD_DIR}/xllm-spacemit-${TARGET_PLATFORM}.tar.gz"
echo ""
echo "To deploy to SpacemiT K3:"
echo "  sshpass -p 'bianbu' scp ${BUILD_DIR}/xllm-spacemit-${TARGET_PLATFORM}.tar.gz bianbu@10.0.90.243:/home/bianbu/"
```

### 4.2 本地编译（在 K3 上）

```bash
#!/bin/bash
# build_spacemit_native.sh

set -e

# 在 K3 上本地编译
BUILD_DIR=build_spacemit_native

cmake -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SPACEMIT=ON \
  -DSPACEMIT_USE_IME2=ON \
  -DSPACEMIT_ENABLE_GGUF=ON

cmake --build ${BUILD_DIR} --parallel $(nproc)
cmake --install ${BUILD_DIR}
```

---

## 5. 验证编译结果

```bash
# 检查生成的库
ls -lh build_spacemit/install/lib/

# 应该包含：
# - libxllm_core.so
# - libggml_spacemit.a
# - libgguf.a (如果启用 GGUF)

# 检查符号
nm -D build_spacemit/install/lib/libxllm_core.so | grep spacemit

# 应该看到：
# - xllm::spacemit::* 符号
# - spacemit_kernels::ime1::* 或 ime2::* 符号
```

---

**（CMake 配置完成）**
