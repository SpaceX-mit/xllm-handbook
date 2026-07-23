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
 * Matrix multiplication using ggml-spacemit
 *
 * C = A @ B + bias (optional)
 *
 * Uses IME2 (A100) or IME1 (X100) acceleration via ggml-spacemit
 */
torch::Tensor matmul(
    const torch::Tensor& a,  // [M, K]
    const torch::Tensor& b,  // [K, N]
    const std::optional<torch::Tensor>& bias = std::nullopt  // [N]
) {
    CHECK(a.dim() == 2) << "matmul: a must be 2D, got " << a.dim();
    CHECK(b.dim() == 2) << "matmul: b must be 2D, got " << b.dim();
    CHECK_EQ(a.size(1), b.size(0)) << "matmul: incompatible shapes";

    if (bias.has_value()) {
        CHECK(bias->dim() == 1) << "matmul: bias must be 1D";
        CHECK_EQ(bias->size(0), b.size(1)) << "matmul: bias size mismatch";
    }

    // Create ggml backend
    xllm::spacemit::GGMLBackend backend;

    // Build computation graph
    auto build_graph = [&](ggml_context* ctx) -> ggml_tensor* {
        // Convert inputs (zero-copy)
        ggml_tensor* ga = xllm::spacemit::GGMLBridge::to_ggml(ctx, a);
        ggml_tensor* gb = xllm::spacemit::GGMLBridge::to_ggml(ctx, b);

        // Matrix multiplication
        // Note: ggml_mul_mat expects transposed layout
        ggml_tensor* gc = ggml_mul_mat(ctx, gb, ga);

        // Add bias if provided
        if (bias.has_value()) {
            ggml_tensor* gb_bias = xllm::spacemit::GGMLBridge::to_ggml(
                ctx, *bias
            );
            // Broadcast bias across rows
            gb_bias = ggml_repeat(ctx, gb_bias, gc);
            gc = ggml_add(ctx, gc, gb_bias);
        }

        return gc;
    };

    // Execute computation
    std::vector<torch::Tensor> inputs = {a, b};
    if (bias.has_value()) {
        inputs.push_back(*bias);
    }

    return backend.compute(build_graph, inputs);
}

} // namespace xllm::kernel::spacemit
