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
#include <functional>
#include <vector>

// Forward declare ggml types
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace xllm::spacemit {

/**
 * GGMLBackend: Manages ggml context and executes computation graphs
 *
 * Uses ggml-spacemit kernels for IME/TCM acceleration
 */
class GGMLBackend {
 public:
    GGMLBackend();
    ~GGMLBackend();

    // Disable copy
    GGMLBackend(const GGMLBackend&) = delete;
    GGMLBackend& operator=(const GGMLBackend&) = delete;

    /**
     * Execute a ggml computation graph
     *
     * @param build_graph Function that builds the computation graph
     * @param inputs Input tensors (for lifetime management)
     * @return Output tensor (zero-copy from ggml result)
     */
    torch::Tensor compute(
        std::function<ggml_tensor*(ggml_context*)> build_graph,
        const std::vector<torch::Tensor>& inputs
    );

    /**
     * Get ggml context (for advanced usage)
     */
    ggml_context* context() { return ctx_; }

 private:
    ggml_context* ctx_;
    size_t context_size_;
    bool use_ime2_;  // A100 vs X100

    // Thread count for parallel execution
    int n_threads_;

    // Detect IME version at runtime
    void detect_platform();
};

} // namespace xllm::spacemit
