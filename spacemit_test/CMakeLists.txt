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
