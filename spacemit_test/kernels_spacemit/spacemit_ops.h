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
#include <optional>

namespace xllm::kernel::spacemit {

// Matrix multiplication with optional bias
torch::Tensor matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const std::optional<torch::Tensor>& bias
);

// RMS Normalization
torch::Tensor rms_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float eps
);

// TODO: Add more operators as needed
// - apply_rotary (RoPE)
// - act_and_mul (SwiGLU)
// - reshape_paged_cache (KV Cache)

} // namespace xllm::kernel::spacemit
