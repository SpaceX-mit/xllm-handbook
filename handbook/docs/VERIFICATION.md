# xLLM 验收标准文档 (VERIFICATION)

> **文档版本**: v1.0  
> **项目**: xLLM 大模型推理框架  
> **日期**: 2026-07-23

---

## 1. 验收策略

### 1.1 验收层次

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         验收层次结构                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Level 1: 单元测试 (Unit Tests)                                        │
│  ├── 目标: 验证每个函数/类的正确性                                       │
│  ├── 覆盖: 核心算法、数据结构、接口                                       │
│  └── 标准: 分支覆盖 > 80%                                               │
│                                                                         │
│  Level 2: 集成测试 (Integration Tests)                                 │
│  ├── 目标: 验证模块间交互正确性                                          │
│  ├── 覆盖: API、Engine、Scheduler、Worker                                 │
│  └── 标准: 关键路径 100% 覆盖                                            │
│                                                                         │
│  Level 3: 系统测试 (System Tests)                                      │
│  ├── 目标: 验证端到端功能正确性                                          │
│  ├── 覆盖: 推理流程、SLO 达成、并发处理                                  │
│  └── 标准: 性能指标达标                                                 │
│                                                                         │
│  Level 4: 性能测试 (Performance Tests)                                  │
│  ├── 目标: 验证性能指标达标                                             │
│  ├── 覆盖: 延迟、吞吐、内存                                             │
│  └── 标准: 达到性能基线                                                 │
│                                                                         │
│  Level 5: 回归测试 (Regression Tests)                                  │
│  ├── 目标: 确保变更不引入缺陷                                            │
│  ├── 覆盖: 历史 Bug、已知问题                                            │
│  └── 标准: 零回归                                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 验收流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         验收流程                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────┐ │
│  │ 代码    │ -> │ 单元    │ -> │ 集成    │ -> │ 系统    │ -> │ 发布  │ │
│  │ 提交    │    │ 测试    │    │ 测试    │    │ 测试    │    │ 评估  │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └───────┘ │
│      │              │              │              │                │     │
│      ▼              ▼              ▼              ▼                ▼     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │ CI/CD   │    │ L1 通过 │    │ L2 通过 │    │ L3+L4   │    │ 评审   │ │
│  │ 自动触发│    │ L5 通过 │    │         │    │ 通过    │    │ 通过   │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 功能验收标准

### 2.1 核心功能

#### F1: 模型加载

| 测试编号 | F1.1 |
|----------|------|
| **测试名称** | HuggingFace 格式模型加载 |
| **前置条件** | 模型文件存在于指定路径 |
| **测试步骤** | 1. 调用 `LLM(model_path="path/to/model")`<br>2. 检查模型配置正确加载<br>3. 检查权重正确加载 |
| **验收标准** | - 模型加载时间 < 60s (7B 模型)<br>- 模型配置解析正确<br>- 权重数据验证通过 |
| **通过标准** | ✓ 通过 |

| 测试编号 | F1.2 |
|----------|------|
| **测试名称** | 多硬件模型加载 |
| **前置条件** | 各硬件平台环境正常 |
| **测试步骤** | 在 CUDA、NPU、MLU、DCU 平台分别加载模型 |
| **验收标准** | 所有平台模型加载成功 |
| **通过标准** | ✓ 所有平台通过 |

#### F2: 基础推理

| 测试编号 | F2.1 |
|----------|------|
| **测试名称** | 单请求推理 |
| **前置条件** | 模型加载成功 |
| **测试步骤** | 1. 发送文本 "Hello, world!"<br>2. 获取生成结果<br>3. 验证输出非空 |
| **验收标准** | - 成功返回结果<br>- 输出符合语言模型格式<br>- 无异常 |
| **通过标准** | ✓ 通过 |

| 测试编号 | F2.2 |
|----------|------|
| **测试名称** | 采样参数控制 |
| **前置条件** | 模型加载成功 |
| **测试步骤** | 使用不同参数测试：<br>1. `temperature=0` (贪婪)<br>2. `temperature=1.0`<br>3. `top_p=0.9`<br>4. `top_k=50` |
| **验收标准** | 参数对输出有预期影响 |
| **通过标准** | ✓ 通过 |

#### F3: 对话功能

| 测试编号 | F3.1 |
|----------|------|
| **测试名称** | Chat API 对话 |
| **前置条件** | 模型加载成功 |
| **测试步骤** | 1. 发送多轮对话消息<br>2. 检查历史上下文保持<br>3. 验证回复相关性 |
| **验收标准** | - 多轮对话正确处理<br>- 上下文理解准确 |
| **通过标准** | ✓ 通过 |

| 测试编号 | F3.2 |
|----------|------|
| **测试名称** | 流式输出 |
| **前置条件** | 模型加载成功 |
| **测试步骤** | 1. 发送请求并设置 `stream=True`<br>2. 收集流式响应块<br>3. 验证完整性和顺序 |
| **验收标准** | - 流式响应正确<br>- 块顺序正确<br>- 最终结果完整 |
| **通过标准** | ✓ 通过 |

#### F4: 多模态功能

| 测试编号 | F4.1 |
|----------|------|
| **测试名称** | 图像理解 |
| **前置条件** | VLM 模型加载成功 |
| **测试步骤** | 1. 发送包含图像的请求<br>2. 询问图像内容<br>3. 验证回复正确 |
| **验收标准** | - 图像正确处理<br>- 理解准确 |
| **通过标准** | ✓ 通过 |

---

## 3. 性能验收标准

### 3.1 延迟指标

#### P1: TTFT (Time To First Token)

| 测试场景 | 模型规模 | 并发数 | TTFT 目标 | TTFT 基线 | 验收阈值 |
|----------|----------|--------|-----------|-----------|----------|
| Short prompt | 7B | 1 | < 50ms | 80ms | ≤ 基线 * 1.2 |
| Medium prompt | 7B | 1 | < 100ms | 150ms | ≤ 基线 * 1.2 |
| Long prompt | 7B | 1 | < 200ms | 300ms | ≤ 基线 * 1.2 |
| Short prompt | 70B | 1 | < 150ms | 250ms | ≤ 基线 * 1.2 |
| Short prompt | 7B | 16 | < 200ms | 350ms | ≤ 基线 * 1.2 |

**测试方法**:

```python
def test_ttft():
    llm = xllm.LLM(model_path="Qwen2.5-7B")
    
    ttft_list = []
    for i in range(100):
        start = time.time()
        response = llm.chat([{"role": "user", "content": "Hello!"}])
        ttft = time.time() - start  # 测量首 Token 延迟
        ttft_list.append(ttft)
    
    p50 = np.percentile(ttft_list, 50)
    p99 = np.percentile(ttft_list, 99)
    
    assert p50 <= TARGET_TTFT, f"TTFT P50 {p50*1000:.2f}ms exceeds target"
    assert p99 <= TARGET_TTFT * 2, f"TTFT P99 {p99*1000:.2f}ms exceeds limit"
```

#### P2: TPOT (Time Per Output Token)

| 测试场景 | 模型规模 | 并发数 | TPOT 目标 | 验收阈值 |
|----------|----------|--------|-----------|----------|
| Decode | 7B | 1 | < 10ms | ≤ 基线 * 1.2 |
| Decode | 7B | 16 | < 15ms | ≤ 基线 * 1.2 |
| Decode | 70B | 1 | < 30ms | ≤ 基线 * 1.2 |

**测试方法**:

```python
def test_tpot():
    llm = xllm.LLM(model_path="Qwen2.5-7B")
    
    total_time = 0
    total_tokens = 0
    
    for i in range(100):
        start = time.time()
        response = llm.chat([{"role": "user", "content": "Count to 100"}],
                          max_tokens=100)
        elapsed = time.time() - start
        
        total_time += elapsed
        total_tokens += response.usage.completion_tokens
    
    tpot = total_time / total_tokens
    assert tpot <= TARGET_TPOT, f"TPOT {tpot*1000:.2f}ms exceeds target"
```

### 3.2 吞吐指标

#### P3: Throughput

| 测试场景 | 模型规模 | Batch Size | 吞吐目标 | 验收阈值 |
|----------|----------|------------|----------|----------|
| 小批量 | 7B | 8 | > 500 tokens/s | ≥ 基线 * 0.9 |
| 中批量 | 7B | 32 | > 1500 tokens/s | ≥ 基线 * 0.9 |
|大批量 | 7B | 64 | > 3000 tokens/s | ≥ 基线 * 0.9 |

**测试方法**:

```python
def test_throughput():
    llm = xllm.LLM(model_path="Qwen2.5-7B")
    
    prompts = ["Hello!"] * 64
    
    start = time.time()
    results = llm.batch_generate(prompts, max_tokens=100)
    elapsed = time.time() - start
    
    total_tokens = sum(len(r) for r in results)
    throughput = total_tokens / elapsed
    
    assert throughput >= TARGET_THROUGHPUT, \
        f"Throughput {throughput:.2f} tokens/s below target"
```

### 3.3 资源指标

#### P4: 内存占用

| 指标 | 7B 模型 (FP16) | 7B 模型 (INT8) | 验收阈值 |
|------|-----------------|----------------|----------|
| GPU 显存 | ~16GB | ~10GB | ≤ 基线 * 1.1 |
| KV Cache | 动态 | 动态 | ≤ 基线 * 1.2 |
| 模型权重 | ~14GB | ~7GB | = 预期值 |

**测试方法**:

```python
def test_memory_usage():
    llm = xllm.LLM(model_path="Qwen2.5-7B", dtype="float16")
    
    # 获取 GPU 内存使用
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated()
    
    # 执行推理
    llm.chat([{"role": "user", "content": "Hello!"}])
    
    peak_mem = torch.cuda.max_memory_allocated()
    model_mem = peak_mem - initial_mem
    
    expected_mem = 14 * 1024**3  # 14GB for FP16 7B
    
    assert model_mem <= expected_mem * 1.2, \
        f"Memory {model_mem/1024**3:.2f}GB exceeds expected"
```

---

## 4. SLO 验收标准

### 4.1 SLO 定义

| SLO 指标 | 定义 | 目标值 | 告警阈值 |
|----------|------|--------|----------|
| **可用性** | 服务可用时间 / 总时间 | 99.9% | < 99.5% |
| **TTFT P50** | 首 Token 延迟中位数 | < 100ms | > 150ms |
| **TTFT P99** | 首 Token 延迟 P99 | < 500ms | > 800ms |
| **TPOT P50** | 每 Token 延迟中位数 | < 15ms | > 25ms |
| **错误率** | 失败请求 / 总请求 | < 0.1% | > 0.5% |

### 4.2 SLO 测试用例

#### SLO-1: TTFT SLO 达成

```python
def test_slo_ttft():
    """
    SLO: TTFT P50 < 100ms, TTFT P99 < 500ms
    
    测试方法:
    1. 发送 1000 个并发请求
    2. 收集每个请求的 TTFT
    3. 计算 P50, P99
    4. 与 SLO 对比
    """
    llm = xllm.LLM(model_path="Qwen2.5-7B")
    
    ttft_samples = []
    
    # 并发测试
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i in range(1000):
            future = executor.submit(measure_ttft, llm, f"Test {i}")
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            ttft = future.result()
            ttft_samples.append(ttft)
    
    p50 = np.percentile(ttft_samples, 50)
    p99 = np.percentile(ttft_samples, 99)
    
    # 验收
    assert p50 < 0.1, f"TTFT P50 {p50*1000:.2f}ms exceeds SLO 100ms"
    assert p99 < 0.5, f"TTFT P99 {p99*1000:.2f}ms exceeds SLO 500ms"
```

#### SLO-2: 高负载下 SLO 达成

```python
def test_slo_under_load():
    """
    SLO: 在 16 并发下，TTFT P99 < 500ms
    
    测试方法:
    1. 持续发送 16 并发请求
    2. 持续时间: 10 分钟
    3. 收集所有 TTFT 样本
    4. 计算 P50, P99, 错误率
    """
    llm = xllm.LLM(model_path="Qwen2.5-7B")
    
    ttft_samples = []
    errors = 0
    total = 0
    
    start_time = time.time()
    duration = 600  # 10 分钟
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        while time.time() - start_time < duration:
            futures = []
            for _ in range(16):
                future = executor.submit(measure_ttft, llm)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                total += 1
                try:
                    ttft = future.result()
                    ttft_samples.append(ttft)
                except Exception as e:
                    errors += 1
    
    # 计算指标
    p50 = np.percentile(ttft_samples, 50)
    p99 = np.percentile(ttft_samples, 99)
    error_rate = errors / total
    
    # 验收
    assert p50 < 0.15, f"P50 {p50*1000:.2f}ms exceeds threshold"
    assert p99 < 0.5, f"P99 {p99*1000:.2f}ms exceeds SLO"
    assert error_rate < 0.001, f"Error rate {error_rate*100:.2f}% exceeds 0.1%"
```

---

## 5. 兼容性验收

### 5.1 模型兼容性

| 模型系列 | 模型名称 | 状态 | 测试覆盖 |
|----------|----------|------|----------|
| **LLaMA** | LLaMA-2-7B | ✓ 支持 | ✓ 测试 |
| | LLaMA-2-13B | ✓ 支持 | ✓ 测试 |
| | LLaMA-3-8B | ✓ 支持 | ✓ 测试 |
| **Qwen** | Qwen2-1.5B | ✓ 支持 | ✓ 测试 |
| | Qwen2-7B | ✓ 支持 | ✓ 测试 |
| | Qwen2-VL-7B | ✓ 支持 | ✓ 测试 |
| **GLM** | GLM-4-9B | ✓ 支持 | ✓ 测试 |
| | GLM-4V-9B | ✓ 支持 | ✓ 测试 |
| **DeepSeek** | DeepSeek-V2.5 | ✓ 支持 | ✓ 测试 |
| | DeepSeek-V3 | ✓ 支持 | ✓ 测试 |
| **MiniMax** | MiniMax-M3 | ✓ 支持 | ✓ 测试 |

### 5.2 硬件兼容性

| 硬件 | 型号 | 状态 | 测试覆盖 |
|------|------|------|----------|
| **NVIDIA GPU** | A100 | ✓ 支持 | ✓ 测试 |
| | H100 | ✓ 支持 | ✓ 测试 |
| | RTX 4090 | ✓ 支持 | ✓ 测试 |
| **华为 Ascend** | NPU A2 | ✓ 支持 | ✓ 测试 |
| | NPU A3 | ✓ 支持 | ✓ 测试 |
| **寒武纪** | MLU370 | ✓ 支持 | ✓ 测试 |
| **海光** | DCU Z100 | ✓ 支持 | ✓ 测试 |
| **摩尔线程** | MUSA S5000 | ✓ 支持 | ✓ 测试 |

---

## 6. AI 复刻验收标准

### 6.1 AI 可复刻性定义

**AI 复刻**: AI 助手（如 Claude、ChatGPT）能够根据文档完整实现或复现项目功能。

### 6.2 复刻验收检查点

#### A1: 需求可理解性

| 检查点 | 描述 | 验收标准 |
|--------|------|----------|
| **功能描述完整性** | 每个功能有明确的输入、输出、约束 | ✓ 所有功能有完整描述 |
| **状态机清晰** | 对象状态转换明确 | ✓ 状态图清晰 |
| **异常处理明确** | 错误情况有定义 | ✓ 错误码完整 |

#### A2: 设计可实现性

| 检查点 | 描述 | 验收标准 |
|--------|------|----------|
| **类结构完整** | 类定义包含所有必要成员 | ✓ 成员完整 |
| **接口签名明确** | 函数签名清晰 | ✓ 无歧义 |
| **算法可推导** | 复杂算法有伪代码或公式 | ✓ 可直接实现 |

#### A3: 代码可对齐性

| 检查点 | 描述 | 验收标准 |
|--------|------|----------|
| **关键代码示例** | 提供核心代码示例 | ✓ 示例完整 |
| **类型定义准确** | 类型定义与实现一致 | ✓ 无偏差 |
| **常量定义明确** | Magic number 有定义 | ✓ 无隐含常量 |

### 6.3 AI 复刻测试

**测试方法**: 使用 AI 助手根据文档生成代码，然后与原始代码对比。

**验收标准**:

| 指标 | 目标 | 说明 |
|------|------|------|
| **功能完整性** | ≥ 95% | AI 生成代码能实现的功能比例 |
| **接口一致性** | 100% | 公共 API 完全一致 |
| **核心算法准确率** | ≥ 90% | 关键算法实现正确率 |
| **可编译率** | 100% | AI 生成代码可直接编译 |

---

## 7. 验收检查清单

### 7.1 功能验收清单

```
□ F1.1: HuggingFace 模型加载
□ F1.2: 多硬件模型加载
□ F2.1: 单请求推理
□ F2.2: 采样参数控制
□ F3.1: Chat API 对话
□ F3.2: 流式输出
□ F4.1: 图像理解
□ F4.2: 视频理解 (如支持)
□ F5.1: Function Call
□ F5.2: Embedding API
```

### 7.2 性能验收清单

```
□ P1.1: TTFT P50 达标
□ P1.2: TTFT P99 达标
□ P2.1: TPOT P50 达标
□ P2.2: TPOT P99 达标
□ P3.1: 小批量吞吐达标
□ P3.2: 中批量吞吐达标
□ P3.3: 大批量吞吐达标
□ P4.1: GPU 显存占用达标
□ P4.2: 内存峰值达标
```

### 7.3 SLO 验收清单

```
□ SLO-1: TTFT SLO 达成
□ SLO-2: TPOT SLO 达成
□ SLO-3: 高负载 SLO 达成
□ SLO-4: 错误率达标
□ SLO-5: 可用性达标
```

### 7.4 兼容性验收清单

```
□ M1.1: LLaMA 系列模型支持
□ M1.2: Qwen 系列模型支持
□ M1.3: GLM 系列模型支持
□ M1.4: DeepSeek 系列模型支持
□ H1.1: NVIDIA GPU 支持
□ H1.2: 华为 NPU 支持
□ H1.3: 寒武纪 MLU 支持
□ H1.4: 海光 DCU 支持
□ H1.5: 摩尔线程 MUSA 支持
```

### 7.5 AI 复刻验收清单

```
□ A1.1: 功能描述完整性
□ A1.2: 状态机清晰
□ A1.3: 异常处理明确
□ A2.1: 类结构完整
□ A2.2: 接口签名明确
□ A2.3: 算法可推导
□ A3.1: 关键代码示例
□ A3.2: 类型定义准确
□ A3.3: 常量定义明确
□ A4.1: 功能完整性 ≥ 95%
□ A4.2: 接口一致性 100%
□ A4.3: 核心算法准确率 ≥ 90%
```

---

## 8. 验收报告模板

```markdown
# xLLM 验收报告

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 版本 | v1.0.0 |
| 验收日期 | 2026-07-23 |
| 验收人员 | [姓名] |
| 环境 | [硬件配置] |

## 2. 验收结果汇总

| 类别 | 通过数 | 总数 | 通过率 |
|------|--------|------|--------|
| 功能验收 | 10 | 10 | 100% |
| 性能验收 | 9 | 9 | 100% |
| SLO 验收 | 5 | 5 | 100% |
| 兼容性验收 | 14 | 14 | 100% |
| AI 复刻验收 | 12 | 12 | 100% |

## 3. 详细结果

### 3.1 功能验收

| 测试项 | 状态 | 备注 |
|--------|------|------|
| F1.1 | ✅ 通过 | |
| F1.2 | ✅ 通过 | |
| ... | ... | ... |

### 3.2 性能验收

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| TTFT P50 | < 100ms | 85ms | ✅ 通过 |
| TTFT P99 | < 500ms | 320ms | ✅ 通过 |
| ... | ... | ... | ... |

## 4. 遗留问题

| 问题 ID | 描述 | 严重度 | 状态 | 计划 |
|---------|------|--------|------|------|
| - | 无 | - | - | - |

## 5. 最终结论

**验收结论**: ✅ 通过

**签字**:
- 验收负责人: __________ 日期: __________
- 产品负责人: __________ 日期: __________
- 技术负责人: __________ 日期: __________
```

---

**文档结束**
