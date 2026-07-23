include(cc_library)

# SpacemiT accelerated kernels (matmul, rms_norm) built on top of the
# zero-copy ggml bridge in xllm/core/platform/spacemit.
cc_library(
  NAME
    spacemit_kernels
  HDRS
    spacemit_ops.h
  SRCS
    matmul_ggml.cpp
    rms_norm_ggml.cpp
  DEPS
    torch
    glog::glog
    :spacemit_bridge
    $<$<BOOL:${USE_SPACEMIT}>:ggml-spacemit>
)

target_include_directories(spacemit_kernels PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/ggml-spacemit
)
