include(cc_library)

# SpacemiT zero-copy bridge between torch::Tensor and ggml_tensor.
# Backed by the ggml-spacemit static library (IME2/IME1/RVV kernels).
cc_library(
  NAME
    spacemit_bridge
  HDRS
    ggml_bridge.h
    ggml_backend.h
  SRCS
    ggml_bridge.cpp
    ggml_backend.cpp
  DEPS
    torch
    glog::glog
    $<$<BOOL:${USE_SPACEMIT}>:ggml-spacemit>
)

target_include_directories(spacemit_bridge PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/ggml-spacemit
)
