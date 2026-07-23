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

#include "xllm/core/kernels/spacemit/spacemit_ops.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::kernel::spacemit {

class SpacemiTOpsTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // Set random seed for reproducibility
        torch::manual_seed(42);
    }
};

TEST_F(SpacemiTOpsTest, MatmulCorrectness) {
    // Create test matrices
    torch::Tensor a = torch::randn({16, 512}, torch::kFloat32);
    torch::Tensor b = torch::randn({512, 1024}, torch::kFloat32);

    // SpacemiT implementation
    torch::Tensor c_spacemit = matmul(a, b, std::nullopt);

    // PyTorch reference implementation
    torch::Tensor c_ref = torch::matmul(a, b);

    // Verify correctness (allow small numerical error)
    EXPECT_TRUE(torch::allclose(c_spacemit, c_ref, /*rtol=*/1e-3, /*atol=*/1e-3))
        << "Matmul output differs from reference";

    // Verify shape
    EXPECT_EQ(c_spacemit.size(0), 16);
    EXPECT_EQ(c_spacemit.size(1), 1024);
}

TEST_F(SpacemiTOpsTest, MatmulWithBias) {
    torch::Tensor a = torch::randn({4, 64}, torch::kFloat32);
    torch::Tensor b = torch::randn({64, 128}, torch::kFloat32);
    torch::Tensor bias = torch::randn({128}, torch::kFloat32);

    // SpacemiT implementation
    torch::Tensor c_spacemit = matmul(a, b, bias);

    // PyTorch reference
    torch::Tensor c_ref = torch::matmul(a, b) + bias;

    // Verify
    EXPECT_TRUE(torch::allclose(c_spacemit, c_ref, 1e-3, 1e-3))
        << "Matmul with bias output differs from reference";
}

TEST_F(SpacemiTOpsTest, MatmulNoNaN) {
    torch::Tensor a = torch::randn({8, 256}, torch::kFloat32);
    torch::Tensor b = torch::randn({256, 512}, torch::kFloat32);

    torch::Tensor c = matmul(a, b, std::nullopt);

    // Verify no NaN or Inf
    EXPECT_FALSE(torch::isnan(c).any().item<bool>())
        << "Matmul output contains NaN";
    EXPECT_FALSE(torch::isinf(c).any().item<bool>())
        << "Matmul output contains Inf";
}

TEST_F(SpacemiTOpsTest, RMSNormCorrectness) {
    torch::Tensor input = torch::randn({4, 512}, torch::kFloat32);
    torch::Tensor weight = torch::ones({512}, torch::kFloat32);
    float eps = 1e-5;

    // SpacemiT implementation
    torch::Tensor output = rms_norm(input, weight, eps);

    // PyTorch reference implementation
    torch::Tensor variance = input.pow(2).mean(-1, /*keepdim=*/true);
    torch::Tensor normed = input * torch::rsqrt(variance + eps);
    torch::Tensor output_ref = normed * weight;

    // Verify correctness
    EXPECT_TRUE(torch::allclose(output, output_ref, 1e-3, 1e-3))
        << "RMSNorm output differs from reference";
}

TEST_F(SpacemiTOpsTest, RMSNormNoNaN) {
    torch::Tensor input = torch::randn({16, 1024}, torch::kFloat32);
    torch::Tensor weight = torch::ones({1024}, torch::kFloat32);

    torch::Tensor output = rms_norm(input, weight, 1e-5);

    // Verify no NaN or Inf
    EXPECT_FALSE(torch::isnan(output).any().item<bool>())
        << "RMSNorm output contains NaN";
    EXPECT_FALSE(torch::isinf(output).any().item<bool>())
        << "RMSNorm output contains Inf";
}

TEST_F(SpacemiTOpsTest, RMSNormShape) {
    torch::Tensor input = torch::randn({2, 3, 768}, torch::kFloat32);
    torch::Tensor weight = torch::ones({768}, torch::kFloat32);

    torch::Tensor output = rms_norm(input, weight, 1e-5);

    // Verify shape unchanged
    EXPECT_EQ(output.sizes(), input.sizes());
}

// Stress test with realistic model dimensions
TEST_F(SpacemiTOpsTest, MatmulStressTest) {
    // Qwen 2B dimensions
    torch::Tensor a = torch::randn({1, 2048}, torch::kFloat32);  // Hidden state
    torch::Tensor b = torch::randn({2048, 2048}, torch::kFloat32);  // Weight

    torch::Tensor c = matmul(a, b, std::nullopt);

    EXPECT_FALSE(torch::isnan(c).any().item<bool>());
    EXPECT_FALSE(torch::isinf(c).any().item<bool>());
    EXPECT_EQ(c.size(0), 1);
    EXPECT_EQ(c.size(1), 2048);
}

} // namespace xllm::kernel::spacemit
