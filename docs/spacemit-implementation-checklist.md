# xLLM SpacemiT K3 平台接入 - 实施清单

> **完整实施指南** - 从设计到部署的一站式参考

---

## 📚 文档体系

### 核心文档（必读）

| 文档 | 用途 | 阅读时间 |
|------|------|---------|
| [执行摘要](./spacemit-k3-adaptation-summary.md) | 快速了解可行性 | 5 分钟 |
| [完整索引](./spacemit-k3-adaptation-index.md) | 导航所有文档 | 5 分钟 |
| [README-spacemit](./README-spacemit.md) | 文档目录与快速参考 | 10 分钟 |

### 技术分析（深入理解）

| 文档 | 内容 | 阅读时间 |
|------|------|---------|
| [第一部分](./spacemit-k3-adaptation-part1.md) | 架构对比与基础分析 | 20 分钟 |
| [第二部分](./spacemit-k3-adaptation-part2.md) | 运行流程对比与方案设计 | 30 分钟 |
| [第三部分](./spacemit-k3-adaptation-part3.md) | 完整实施与优化策略 | 40 分钟 |
| [第四部分](./spacemit-k3-adaptation-part4.md) | 性能评估与总结 | 30 分钟 |

### 实施指南（动手实践）

| 文档 | 内容 | 用途 |
|------|------|------|
| [GGUF 加载器](./spacemit-plan-a-plus-gguf-loader.md) | 方案 A+ GGUF 格式支持 | 代码参考 |
| [CMake 配置](./spacemit-cmake-config.md) | 编译系统配置 | 编译参考 |
| **本文档** | 实施清单与检查点 | 项目管理 |

---

## 🎯 实施路径选择

### 决策树

```
开始
  │
  ├─ 需要快速验证（2-3周）？
  │   └─ YES → 方案 B（外部调用）
  │   └─ NO → 继续
  │
  ├─ 内存充足（32GB+ RAM）？
  │   └─ YES → 方案 A（标准 Backend）
  │   └─ NO → 方案 A+（GGUF 支持）
  │
  ├─ 已有 x86 机器 + K3 混合部署？
  │   └─ YES → 可考虑方案 C（混合部署）
  │   └─ NO → 继续
  │
  └─ 推荐：方案 A（短期） → 方案 A+（长期）
```

### 方案对比

| 方案 | 时间 | 内存 (7B) | 性能 | 适用场景 |
|------|------|-----------|------|---------|
| **B 外部调用** | 2-3 周 | 3.8 GB | -20%~-30% | 快速验证 |
| **A 标准 Backend** | 3-6 月 | 17.8 GB | -5%~-10% | 生产部署 |
| **A+ GGUF 支持** | 4-6 月 | 3.8 GB | -3%~-5% | 性能优化 |
| **C 混合部署** | 2-3 月 | 3.8 GB | -10%~-15% | 特殊场景 |

---

## 📋 Phase 1: 环境准备（第 1-2 周）

### ✅ Checkpoint 1.1: 下载工具链

```bash
# 1. 创建工作目录
mkdir -p ~/spacemit-dev
cd ~/spacemit-dev

# 2. 下载 SpacemiT 工具链
wget https://nexus.bianbu.xyz/repository/toolchain/llvm-gcc/spacemit-toolchain-linux-glibc-x86_64-v1.2.7.tar.xz

# 3. 解压
tar -xf spacemit-toolchain-linux-glibc-x86_64-v1.2.7.tar.xz

# 4. 设置环境变量
export SPACEMIT_TOOLCHAIN_PATH=$(pwd)/spacemit-toolchain-linux-glibc-x86_64-v1.2.7

# 5. 验证
${SPACEMIT_TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-gnu-gcc --version
```

**预期输出：**
```
riscv64-unknown-linux-gnu-gcc (GCC) 13.2.0
...
```

### ✅ Checkpoint 1.2: 验证 K3 环境

```bash
# 连接到 K3 worker 机
sshpass -p 'bianbu' ssh bianbu@10.0.90.243

# 检查 CPU 信息
cat /proc/cpuinfo | grep -E "(model name|isa|mvendorid|marchid)"

# 检查内存
free -h

# 检查磁盘空间
df -h /home/bianbu
```

**预期输出（A100）：**
```
model name      : Spacemit(R) A100
isa             : rv64imafdcvh_zicbom_zicbop_...
mvendorid       : 0x710
marchid         : 0x8000000041000002
...
```

### ✅ Checkpoint 1.3: 验证 llama.cpp 性能

```bash
# 在 K3 上运行 llama.cpp
cd /data/workspace2026-new/work0618/tech-analyais0713/llama.cpp-handbook

# 如果没有编译，先编译
cmake -B build -DGGML_CPU_RISCV64_SPACEMIT=ON -DGGML_RVV=ON
cmake --build build --parallel $(nproc)

# 下载测试模型（如果没有）
# wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf

# 运行推理测试
./build/bin/llama-cli \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  -p "Hello, how are you?" \
  -n 50 \
  -t 8

# 运行性能基准
./build/bin/llama-bench \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  -p 128 \
  -n 128
```

**预期输出（A100）：**
```
pp 128: 500+ tokens/s
tg 128: 50+ tokens/s
```

### ✅ Checkpoint 1.4: 准备 xLLM 代码

```bash
# 返回开发机
cd ~/spacemit-dev

# 克隆或同步 xLLM 代码
git clone <xllm-repo> xllm
cd xllm

# 创建 SpacemiT 分支
git checkout -b feature/spacemit-k3-integration

# 复制 llama.cpp IME kernels
mkdir -p third_party/llama.cpp/ggml/src/ggml-cpu
cp -r /data/workspace2026-new/work0618/tech-analyais0713/llama.cpp-handbook/ggml/src/ggml-cpu/spacemit \
     third_party/llama.cpp/ggml/src/ggml-cpu/

# 复制必要的头文件
cp /data/workspace2026-new/work0618/tech-analyais0713/llama.cpp-handbook/ggml/include/ggml*.h \
   third_party/llama.cpp/ggml/include/
```

---

## 📋 Phase 2: 最小可行原型（第 3-6 周）

### ✅ Checkpoint 2.1: 创建目录结构

```bash
cd ~/spacemit-dev/xllm

# 创建 SpacemiT 平台目录
mkdir -p xllm/core/platform/spacemit
mkdir -p xllm/core/kernels/spacemit
mkdir -p xllm/core/runtime/spacemit

# 创建 CMake 配置
mkdir -p cmake
touch cmake/spacemit.cmake

# 验证
tree xllm/core/platform/spacemit
tree xllm/core/kernels/spacemit
```

### ✅ Checkpoint 2.2: 实现 Platform 层

**文件清单：**
- [x] `xllm/core/platform/platform.h` - 添加 `is_spacemit()`
- [x] `xllm/core/platform/platform.cpp` - 实现 SpacemiT 检测
- [x] `xllm/core/platform/spacemit/platform_spacemit.h`
- [x] `xllm/core/platform/spacemit/platform_spacemit.cpp`
- [x] `xllm/core/platform/spacemit/device_spacemit.h`
- [x] `xllm/core/platform/spacemit/device_spacemit.cpp`

**参考实现：** 见技术分析第二、三部分

**验证：**
```bash
# 编译测试
cd ~/spacemit-dev/xllm
mkdir -p build_test
cd build_test

cmake .. \
  -DUSE_SPACEMIT=ON \
  -DSPACEMIT_USE_IME2=ON \
  -DCMAKE_BUILD_TYPE=Debug

make -j$(nproc) 2>&1 | tee build.log

# 检查是否有编译错误
grep -i error build.log
```

### ✅ Checkpoint 2.3: 封装 IME Kernels

**文件清单：**
- [x] `xllm/core/kernels/spacemit/ime_wrapper.h`
- [x] `xllm/core/kernels/spacemit/ime_wrapper.cpp`

**参考实现：** 见技术分析第二部分 Step 3

**单元测试：**
```cpp
// test/kernels/spacemit/test_ime_wrapper.cpp
#include <gtest/gtest.h>
#include "xllm/core/kernels/spacemit/ime_wrapper.h"

TEST(IMEWrapperTest, QuantizeActivationI8) {
    // 创建测试输入
    torch::Tensor input = torch::randn({4, 1024}, torch::kFloat32);
    
    // 执行量化
    auto result = xllm::kernel::spacemit::quantize_activation_i8(input);
    
    // 验证形状
    EXPECT_EQ(result.data.sizes(), input.sizes());
    EXPECT_EQ(result.data.dtype(), torch::kInt8);
    EXPECT_EQ(result.scale.size(0), 4);
    
    // 验证精度（反量化后误差 < 1%）
    torch::Tensor dequant = result.data.to(torch::kFloat32) * 
                            result.scale.unsqueeze(1);
    float mse = torch::mse_loss(dequant, input).item<float>();
    EXPECT_LT(mse, 0.01);
}

TEST(IMEWrapperTest, IMEMatmul) {
    // 创建测试输入
    torch::Tensor input = torch::randn({2, 512}, torch::kFloat32);
    
    // 创建量化权重（模拟）
    auto weight = create_mock_quantized_weight(1024, 512);
    
    // 执行矩阵乘法
    torch::Tensor output = xllm::kernel::spacemit::matmul(input, weight);
    
    // 验证形状
    EXPECT_EQ(output.sizes(), std::vector<int64_t>({2, 1024}));
    
    // 验证数值范围
    EXPECT_FALSE(torch::isnan(output).any().item<bool>());
    EXPECT_FALSE(torch::isinf(output).any().item<bool>());
}
```

**运行测试：**
```bash
cd ~/spacemit-dev/xllm/build_test
ctest --output-on-failure -R IMEWrapper
```

### ✅ Checkpoint 2.4: 单层推理验证

**测试代码：**
```cpp
// test/integration/test_single_layer.cpp
#include <torch/torch.h>
#include "xllm/models/llm/qwen2/qwen2_attention.h"

int main() {
    // 加载单层 Transformer
    Qwen2Attention attention(/*config*/);
    
    // 准备输入
    torch::Tensor hidden = torch::randn({1, 128, 4096});
    torch::Tensor position = torch::arange(128).unsqueeze(0);
    
    // 前向传播
    auto output = attention.forward(hidden, position);
    
    // 验证输出
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Output mean: " << output.mean().item<float>() << std::endl;
    std::cout << "Output std: " << output.std().item<float>() << std::endl;
    
    return 0;
}
```

**在 K3 上运行：**
```bash
# 将编译好的程序传到 K3
sshpass -p 'bianbu' scp build_test/bin/test_single_layer \
  bianbu@10.0.90.243:/home/bianbu/

# 在 K3 上运行
sshpass -p 'bianbu' ssh bianbu@10.0.90.243 \
  "cd /home/bianbu && ./test_single_layer"
```

**预期输出：**
```
Output shape: [1, 128, 4096]
Output mean: 0.0012
Output std: 0.8765
```

---

## 📋 Phase 3: 完整实现（第 7-14 周）

### ✅ Checkpoint 3.1: 实现所有算子

**算子清单：**
- [ ] `matmul` - 矩阵乘法（核心）
- [ ] `rms_norm` - RMS Normalization
- [ ] `apply_rotary` - RoPE 位置编码
- [ ] `activation` - SwiGLU / GELU
- [ ] `softmax` - Softmax
- [ ] `reshape_paged_cache` - KV Cache 操作
- [ ] `reshape_from_cache` - KV Cache 读取

**进度追踪：**
```bash
# 创建进度表
cat > progress.txt << EOF
[x] matmul - 已完成
[ ] rms_norm - 进行中
[ ] apply_rotary - 待开始
[ ] activation - 待开始
[ ] softmax - 待开始
[ ] reshape_paged_cache - 待开始
[ ] reshape_from_cache - 待开始
EOF
```

### ✅ Checkpoint 3.2: 实现 Executor

**文件清单：**
- [ ] `xllm/core/runtime/spacemit/executor_impl_spacemit.h`
- [ ] `xllm/core/runtime/spacemit/executor_impl_spacemit.cpp`

**参考实现：** 见技术分析第三部分 Step 5

### ✅ Checkpoint 3.3: 完整模型推理

**测试：**
```bash
# 加载完整 Qwen3 0.6B 模型
cd ~/spacemit-dev/xllm

# 准备模型
mkdir -p models
cd models
# 下载或软链接 HuggingFace 模型
ln -s /path/to/Qwen/Qwen2.5-0.5B-Instruct .

# 运行推理
cd ..
./build_test/bin/xllm-cli \
  --model models/Qwen2.5-0.5B-Instruct \
  --prompt "Hello, how are you?" \
  --max-tokens 50

# 运行性能测试
./build_test/bin/xllm-bench \
  --model models/Qwen2.5-0.5B-Instruct \
  --batch-size 1 \
  --seq-len 128
```

**性能目标（A100）：**
```
Prefill: 450+ tokens/s
Decode: 45+ tokens/s
```

---

## 📋 Phase 4: 方案 A+ 实施（第 15-20 周）

### ✅ Checkpoint 4.1: 实现 GGUF 加载器

**文件清单：**
- [ ] `xllm/core/framework/gguf_model_loader.h`
- [ ] `xllm/core/framework/gguf_model_loader.cpp`

**参考实现：** 见 [GGUF 加载器文档](./spacemit-plan-a-plus-gguf-loader.md)

**验证：**
```cpp
// 测试 GGUF 加载
#include "xllm/core/framework/gguf_model_loader.h"

int main() {
    GGUFModelLoader loader("models/qwen2.5-0.5b-q4_0.gguf");
    
    // 加载权重
    auto tensors = loader.load_all_tensors(/*dequantize=*/false);
    
    std::cout << "Loaded tensors: " << tensors.size() << std::endl;
    
    // 测试单个张量
    auto wq = loader.load_tensor("layers.0.attention.wq");
    std::cout << "wq shape: " << wq.sizes() << std::endl;
    
    return 0;
}
```

### ✅ Checkpoint 4.2: 集成 GGUF 到模型加载流程

**修改：**
```cpp
// xllm/core/framework/model_loader.cpp
std::unique_ptr<Model> load_model(const std::string& path) {
    // 检测文件类型
    if (path.ends_with(".gguf")) {
        // 使用 GGUF 加载器
        return load_model_from_gguf(path);
    } else if (path.ends_with(".safetensors") || 
               std::filesystem::is_directory(path)) {
        // 使用 safetensors 加载器
        return load_model_from_safetensors(path);
    } else {
        throw std::runtime_error("Unsupported model format");
    }
}
```

### ✅ Checkpoint 4.3: 性能对比测试

**测试脚本：**
```bash
#!/bin/bash
# benchmark_gguf_vs_safetensors.sh

MODEL_NAME="Qwen2.5-0.5B"

echo "Testing safetensors (Plan A)..."
./xllm-bench \
  --model models/${MODEL_NAME}-Instruct \
  --batch-size 1 \
  --seq-len 128 \
  > results_safetensors.txt

echo "Testing GGUF Q4_0 (Plan A+)..."
./xllm-bench \
  --model models/${MODEL_NAME}-q4_0.gguf \
  --batch-size 1 \
  --seq-len 128 \
  > results_gguf.txt

echo "Comparison:"
echo "======================="
echo "Safetensors:"
grep "tokens/s" results_safetensors.txt
echo ""
echo "GGUF:"
grep "tokens/s" results_gguf.txt
```

**目标：**
- GGUF 内存占用与 llama.cpp 相同
- GGUF 性能达到 safetensors 的 95%+

---

## 📋 Phase 5: 生产就绪（第 21-24 周）

### ✅ Checkpoint 5.1: 监控与 Metrics

**实现：**
- [ ] 性能指标采集（TTFT/TPOT/TTLT）
- [ ] 内存使用监控
- [ ] IME/TCM 利用率监控
- [ ] Prometheus exporter

### ✅ Checkpoint 5.2: Doctor 工具

```bash
# xllm/tools/spacemit_doctor.py
#!/usr/bin/env python3

import subprocess
import sys

def check_ime_version():
    """检测 IME 版本"""
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
            if "Spacemit(R) A100" in content:
                return "IME2"
            elif "Spacemit(R) X60" in content:
                return "IME1"
    except:
        pass
    return "Unknown"

def check_tcm_available():
    """检测 TCM 是否可用"""
    ime = check_ime_version()
    return ime == "IME2"

def check_toolchain():
    """检测工具链"""
    # 实现检测逻辑
    pass

if __name__ == "__main__":
    print("SpacemiT Platform Doctor")
    print("=" * 40)
    print(f"IME Version: {check_ime_version()}")
    print(f"TCM Available: {check_tcm_available()}")
    # ...
```

### ✅ Checkpoint 5.3: 文档与示例

**清单：**
- [ ] 部署文档
- [ ] 性能调优指南
- [ ] 故障排查手册
- [ ] API 使用示例
- [ ] 端到端 Demo

---

## 🔍 质量检查点

### 代码质量

- [ ] 通过所有单元测试
- [ ] 通过集成测试
- [ ] 代码覆盖率 > 80%
- [ ] 无内存泄漏（valgrind 检查）
- [ ] 符合代码规范（参考 CLAUDE.md）

### 性能质量

- [ ] Prefill 吞吐量 >= llama.cpp 的 90%
- [ ] Decode 吞吐量 >= llama.cpp 的 90%
- [ ] 内存占用 <= llama.cpp 的 1.1x（方案 A+）
- [ ] 启动时间 <= llama.cpp 的 1.2x

### 稳定性质量

- [ ] 连续运行 24 小时无崩溃
- [ ] 处理 10,000+ 请求无内存泄漏
- [ ] 支持并发请求
- [ ] 优雅处理异常情况

---

## 📊 进度追踪

### 甘特图（示例）

```
Week 1-2   : [################] 环境准备
Week 3-6   : [################] MVP 开发
Week 7-14  : [################] 完整实现
Week 15-20 : [################] 方案 A+ 实施
Week 21-24 : [################] 生产就绪
```

### 里程碑

| 日期 | 里程碑 | 交付物 |
|------|--------|--------|
| Week 2 | 环境就绪 | 工具链 + K3 验证 |
| Week 6 | MVP 完成 | 单层推理验证 |
| Week 14 | 完整实现 | 完整模型推理 |
| Week 20 | 方案 A+ | GGUF 支持 |
| Week 24 | 生产就绪 | 文档 + 监控 |

---

## 🚨 风险管理

### 高风险项

| 风险 | 缓解措施 | 负责人 |
|------|---------|--------|
| PyTorch 开销过大 | 提前实施零拷贝优化 | 核心工程师 |
| 内存不足 | 优先实施 GGUF 支持 | 架构师 |
| 精度损失 | 每个算子逐层验证 | QA |
| K3 硬件不稳定 | 使用 QEMU 模拟开发 | DevOps |

---

## 📞 支持与协作

### 团队分工

- **架构师**：整体设计、技术决策
- **核心工程师 1**：Platform 层 + Kernels
- **核心工程师 2**：Runtime 层 + Executor
- **QA 工程师**：测试、验证、性能基准
- **DevOps**：编译系统、CI/CD

### 定期会议

- **每日站会**：15 分钟同步进度
- **每周回顾**：1 小时回顾问题与解决方案
- **里程碑评审**：2 小时评审交付物

---

**实施清单文档完成！** 🎉
