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

#include "xllm/core/platform/spacemit/ggml_bridge.h"
#include "xllm/core/platform/spacemit/ggml_backend.h"

extern "C" {
#include "third_party/ggml-spacemit/ggml.h"
}

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::spacemit {

class GGMLBridgeTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // Initialize ggml context
        struct ggml_init_params params = {
            .mem_size = 16 * 1024 * 1024,  // 16 MB
            .mem_buffer = nullptr,
            .no_alloc = false,
        };
        ctx_ = ggml_init(params);
        ASSERT_NE(ctx_, nullptr);
    }

    void TearDown() override {
        if (ctx_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ggml_context* ctx_;
};

TEST_F(GGMLBridgeTest, ZeroCopyTorchToGGML) {
    // Create torch tensor
    torch::Tensor x = torch::randn({2, 3}, torch::kFloat32);
    void* torch_ptr = x.data_ptr();

    // Convert to ggml (zero-copy)
    ggml_tensor* y = GGMLBridge::to_ggml(ctx_, x);

    // Verify zero-copy: pointer addresses should be the same
    EXPECT_EQ(torch_ptr, y->data) << "Zero-copy failed: different pointers";

    // Verify shape
    EXPECT_EQ(y->ne[0], 2);
    EXPECT_EQ(y->ne[1], 3);

    // Verify type
    EXPECT_EQ(y->type, GGML_TYPE_F32);
}

TEST_F(GGMLBridgeTest, ZeroCopyGGMLToTorch) {
    // Create ggml tensor
    ggml_tensor* x = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, 2, 3);
    void* ggml_ptr = x->data;

    // Convert to torch (zero-copy)
    torch::Tensor y = GGMLBridge::from_ggml(x);

    // Verify zero-copy: pointer addresses should be the same
    EXPECT_EQ(ggml_ptr, y.data_ptr()) << "Zero-copy failed: different pointers";

    // Verify shape
    EXPECT_EQ(y.size(0), 2);
    EXPECT_EQ(y.size(1), 3);

    // Verify dtype
    EXPECT_EQ(y.dtype(), torch::kFloat32);
}

TEST_F(GGMLBridgeTest, DataConsistency) {
    // Create torch tensor with known values
    torch::Tensor x = torch::tensor({{1.0f, 2.0f, 3.0f},
                                      {4.0f, 5.0f, 6.0f}});

    // Convert to ggml
    ggml_tensor* y = GGMLBridge::to_ggml(ctx_, x);

    // Verify data values
    float* data = static_cast<float*>(y->data);
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
    EXPECT_FLOAT_EQ(data[4], 5.0f);
    EXPECT_FLOAT_EQ(data[5], 6.0f);
}

TEST_F(GGMLBridgeTest, RoundTrip) {
    // Create torch tensor
    torch::Tensor x = torch::randn({4, 5}, torch::kFloat32);

    // torch → ggml → torch
    ggml_tensor* y = GGMLBridge::to_ggml(ctx_, x);
    torch::Tensor z = GGMLBridge::from_ggml(y);

    // Should be identical (same pointer)
    EXPECT_TRUE(torch::equal(x, z));
    EXPECT_EQ(x.data_ptr(), z.data_ptr());
}

TEST_F(GGMLBridgeTest, TypeConversions) {
    // Test FP32
    {
        torch::Tensor x = torch::randn({2, 3}, torch::kFloat32);
        ggml_tensor* y = GGMLBridge::to_ggml(ctx_, x);
        EXPECT_EQ(y->type, GGML_TYPE_F32);
    }

    // Test FP16
    {
        torch::Tensor x = torch::randn({2, 3}, torch::kFloat16);
        ggml_tensor* y = GGMLBridge::to_ggml(ctx_, x);
        EXPECT_EQ(y->type, GGML_TYPE_F16);
    }

    // Test INT32
    {
        torch::Tensor x = torch::randint(0, 100, {2, 3}, torch::kInt32);
        ggml_tensor* y = GGMLBridge::to_ggml(ctx_, x);
        EXPECT_EQ(y->type, GGML_TYPE_I32);
    }
}

} // namespace xllm::spacemit
