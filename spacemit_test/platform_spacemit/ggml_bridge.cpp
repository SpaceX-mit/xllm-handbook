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

#include "ggml_bridge.h"

// Include ggml after our header to avoid conflicts
extern "C" {
#include "third_party/ggml-spacemit/ggml.h"
}

#include <glog/logging.h>

namespace xllm::spacemit {

ggml_tensor* GGMLBridge::to_ggml(
    ggml_context* ctx,
    const torch::Tensor& t
) {
    CHECK(ctx != nullptr) << "ggml_context is null";
    CHECK(t.defined()) << "torch::Tensor is not defined";

    // Ensure contiguous for zero-copy
    torch::Tensor contig = t.contiguous();

    // Convert dtype
    ggml_type type = to_ggml_type(contig.dtype());

    // Get shape (ggml uses int64_t for dimensions)
    int n_dims = contig.dim();
    CHECK_LE(n_dims, GGML_MAX_DIMS) << "Too many dimensions: " << n_dims;

    int64_t ne[GGML_MAX_DIMS];
    for (int i = 0; i < n_dims; ++i) {
        ne[i] = contig.size(i);
    }

    // ⭐ Zero-copy: create ggml_tensor from torch data pointer
    ggml_tensor* result = ggml_new_tensor(ctx, type, n_dims, ne);
    result->data = contig.data_ptr();

    return result;
}

torch::Tensor GGMLBridge::from_ggml(ggml_tensor* t) {
    CHECK(t != nullptr) << "ggml_tensor is null";
    CHECK(t->data != nullptr) << "ggml_tensor data is null";

    // Convert dtype
    torch::Dtype dtype = to_torch_dtype(t->type);

    // Get shape
    std::vector<int64_t> shape = compute_shape(t);

    // ⭐ Zero-copy: wrap ggml data pointer as torch::Tensor
    // Use empty deleter to avoid freeing ggml-managed memory
    torch::Tensor result = torch::from_blob(
        t->data,
        shape,
        [](void*) {},  // Empty deleter
        torch::TensorOptions().dtype(dtype).device(torch::kCPU)
    );

    return result;
}

ggml_type GGMLBridge::to_ggml_type(torch::Dtype dtype) {
    if (dtype == torch::kFloat32) {
        return GGML_TYPE_F32;
    } else if (dtype == torch::kFloat16) {
        return GGML_TYPE_F16;
    } else if (dtype == torch::kInt32) {
        return GGML_TYPE_I32;
    } else if (dtype == torch::kInt16) {
        return GGML_TYPE_I16;
    } else if (dtype == torch::kInt8) {
        return GGML_TYPE_I8;
    } else if (dtype == torch::kUInt8) {
        return GGML_TYPE_I8;  // ggml doesn't have unsigned int8
    } else {
        LOG(FATAL) << "Unsupported torch dtype: " << dtype;
        return GGML_TYPE_F32;  // Unreachable
    }
}

torch::Dtype GGMLBridge::to_torch_dtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return torch::kFloat32;
        case GGML_TYPE_F16:
            return torch::kFloat16;
        case GGML_TYPE_I32:
            return torch::kInt32;
        case GGML_TYPE_I16:
            return torch::kInt16;
        case GGML_TYPE_I8:
            return torch::kInt8;
        default:
            // For quantized types, return uint8 (raw bytes)
            return torch::kUInt8;
    }
}

std::vector<int64_t> GGMLBridge::compute_shape(ggml_tensor* t) {
    std::vector<int64_t> shape;
    for (int i = 0; i < GGML_MAX_DIMS && t->ne[i] > 0; ++i) {
        shape.push_back(t->ne[i]);
    }
    return shape;
}

} // namespace xllm::spacemit
