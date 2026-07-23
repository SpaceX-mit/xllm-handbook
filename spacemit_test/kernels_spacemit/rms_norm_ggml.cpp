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

#include "xllm/core/platform/spacemit/ggml_backend.h"
#include "xllm/core/platform/spacemit/ggml_bridge.h"

extern "C" {
#include "third_party/ggml-spacemit/ggml.h"
}

#include <glog/logging.h>

namespace xllm::kernel::spacemit {

/**
 * RMS Normalization using ggml
 *
 * output = input * rsqrt(mean(input^2) + eps) * weight
 */
torch::Tensor rms_norm(
    const torch::Tensor& input,   // [batch, seq_len, hidden_size]
    const torch::Tensor& weight,  // [hidden_size]
    float eps
) {
    CHECK(input.defined()) << "rms_norm: input is not defined";
    CHECK(weight.defined()) << "rms_norm: weight is not defined";
    CHECK_EQ(input.size(-1), weight.size(0))
        << "rms_norm: weight size mismatch";

    // Create ggml backend
    xllm::spacemit::GGMLBackend backend;

    // Build computation graph
    auto build_graph = [&](ggml_context* ctx) -> ggml_tensor* {
        // Convert inputs (zero-copy)
        ggml_tensor* x = xllm::spacemit::GGMLBridge::to_ggml(ctx, input);
        ggml_tensor* w = xllm::spacemit::GGMLBridge::to_ggml(ctx, weight);

        // RMS Norm
        ggml_tensor* normed = ggml_rms_norm(ctx, x, eps);

        // Scale by weight
        ggml_tensor* output = ggml_mul(ctx, normed, w);

        return output;
    };

    // Execute computation
    return backend.compute(build_graph, {input, weight});
}

} // namespace xllm::kernel::spacemit
