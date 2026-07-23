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

#include "ggml_backend.h"
#include "ggml_bridge.h"

extern "C" {
#include "third_party/ggml-spacemit/ggml.h"
}

#include <glog/logging.h>
#include <thread>

namespace xllm::spacemit {

GGMLBackend::GGMLBackend() {
    // Detect platform capabilities
    detect_platform();

    // Initialize ggml context (128 MB work buffer)
    context_size_ = 128 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = context_size_,
        .mem_buffer = nullptr,      // Let ggml allocate
        .no_alloc   = false,        // Allocate memory
    };

    ctx_ = ggml_init(params);
    CHECK(ctx_ != nullptr) << "Failed to initialize ggml context";

    // Use all available CPU cores
    n_threads_ = std::thread::hardware_concurrency();
    if (n_threads_ == 0) {
        n_threads_ = 8;  // Fallback
    }

    LOG(INFO) << "GGMLBackend initialized:";
    LOG(INFO) << "  Context size: " << context_size_ / (1024 * 1024) << " MB";
    LOG(INFO) << "  Threads: " << n_threads_;
    LOG(INFO) << "  IME version: " << (use_ime2_ ? "IME2 (A100)" : "IME1/RVV");
}

GGMLBackend::~GGMLBackend() {
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

torch::Tensor GGMLBackend::compute(
    std::function<ggml_tensor*(ggml_context*)> build_graph,
    const std::vector<torch::Tensor>& inputs
) {
    CHECK(ctx_ != nullptr) << "ggml context is null";

    // Build computation graph
    ggml_tensor* output = build_graph(ctx_);
    CHECK(output != nullptr) << "build_graph returned null";

    // Create graph object
    struct ggml_cgraph* gf = ggml_new_graph(ctx_);
    ggml_build_forward_expand(gf, output);

    // Execute graph (calls ggml-spacemit kernels)
    ggml_graph_compute_with_ctx(ctx_, gf, n_threads_);

    // Convert result back to torch (zero-copy)
    torch::Tensor result = GGMLBridge::from_ggml(output);

    // Clone to make it safe (ggml context will be reused)
    return result.clone();
}

void GGMLBackend::detect_platform() {
    // Detect IME version by checking CPU info
    // For now, assume IME2 (A100) if available
    // TODO: Implement proper detection via /proc/cpuinfo

    use_ime2_ = true;  // Default to A100 (IME2 + TCM)

    // Could check environment variable for override
    const char* env = std::getenv("SPACEMIT_IME_VERSION");
    if (env != nullptr) {
        if (std::string(env) == "ime1") {
            use_ime2_ = false;
        }
    }
}

} // namespace xllm::spacemit
