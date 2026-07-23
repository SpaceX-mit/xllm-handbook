/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>
#include <vector>
#include <cstdint>

// Forward declare ggml types
struct ggml_context;
struct ggml_tensor;
enum ggml_type : int;

namespace xllm::spacemit {

/**
 * GGMLBridge: Zero-copy conversion between torch::Tensor and ggml_tensor
 *
 * Key feature: Share memory pointers instead of copying data
 */
class GGMLBridge {
 public:
    /**
     * Convert torch::Tensor to ggml_tensor (zero-copy)
     *
     * @param ctx ggml context
     * @param t torch tensor (must be contiguous)
     * @return ggml_tensor pointer sharing the same data
     */
    static ggml_tensor* to_ggml(
        ggml_context* ctx,
        const torch::Tensor& t
    );

    /**
     * Convert ggml_tensor to torch::Tensor (zero-copy)
     *
     * @param t ggml tensor
     * @return torch::Tensor sharing the same data
     */
    static torch::Tensor from_ggml(ggml_tensor* t);

 private:
    // Type conversion helpers
    static ggml_type to_ggml_type(torch::Dtype dtype);
    static torch::Dtype to_torch_dtype(ggml_type type);

    // Shape helpers
    static std::vector<int64_t> compute_shape(ggml_tensor* t);
};

} // namespace xllm::spacemit
