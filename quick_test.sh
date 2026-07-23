#!/bin/bash
# 简化测试：直接在 K3 上编译和测试单元测试

set -e

echo "=========================================="
echo "  简化方案：K3 上直接测试"
echo "=========================================="

# 创建测试目录
mkdir -p spacemit_test
cd spacemit_test

# 复制核心文件
cp -r ../xllm/core/platform/spacemit platform_spacemit
cp -r ../xllm/core/kernels/spacemit kernels_spacemit
cp -r ../test/platform/spacemit test_platform
cp -r ../test/kernels/spacemit test_kernels
cp -r ../third_party/ggml-spacemit ggml-spacemit

# 创建简单的 CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(spacemit_test CXX)

set(CMAKE_CXX_STANDARD 17)

# 添加 RISC-V 标志
add_compile_options(-march=rv64gcv_zfh_zba_zbb_zbc_zbs)
add_compile_definitions(USE_SPACEMIT RISCV64_SPACEMIT_IME2)

# 查找 PyTorch
find_package(Torch REQUIRED)

# ggml-spacemit 库
add_library(ggml_spacemit STATIC
  ggml-spacemit/ggml.c
  ggml-spacemit/ggml-alloc.c
  ggml-spacemit/ime_env.cpp
  ggml-spacemit/ime2_kernels.cpp
  ggml-spacemit/rvv_kernels.cpp
)
target_include_directories(ggml_spacemit PUBLIC ggml-spacemit)

# 测试程序
add_executable(test_bridge
  test_platform/test_ggml_bridge.cpp
  platform_spacemit/ggml_bridge.cpp
  platform_spacemit/ggml_backend.cpp
)
target_link_libraries(test_bridge ${TORCH_LIBRARIES} ggml_spacemit glog)
EOF

echo "✅ 测试环境准备完成"
echo "传输到 K3: sshpass -p 'bianbu' scp -r spacemit_test bianbu@10.0.90.243:~/"
