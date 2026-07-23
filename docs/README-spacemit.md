# SpacemiT K3 平台适配文档

本目录包含 xLLM 接入 SpacemiT K3 (riscv64) 平台的完整技术分析文档。

---

## 📑 文档列表

### 🎯 [执行摘要](./spacemit-k3-adaptation-summary.md)
**快速阅读（5 分钟）**
- 核心结论
- 三个关键问题的答案
- 推荐方案
- 性能预期
- 实施计划概览

**适合：** 决策者、项目经理、需要快速了解可行性的人员

---

### 📚 [完整文档索引](./spacemit-k3-adaptation-index.md)
**导航中心**
- 四个部分的导航
- 核心问题快速查找
- 性能数据汇总
- 技术栈概览
- 参考资料链接

**适合：** 所有读者（作为导航起点）

---

### 📖 详细技术分析（四部分）

#### [第一部分：架构对比与基础分析](./spacemit-k3-adaptation-part1.md)
**阅读时间：20 分钟**

内容：
- xLLM 与 llama.cpp 架构对比
- xLLM 是否是 C++ 实现？
- 模型格式差异（GGUF vs safetensors）
- llama.cpp 的 ggml-spacemit 实现详解
- xLLM 的平台抽象设计

**适合：** 架构师、技术负责人

---

#### [第二部分：运行流程对比与方案设计](./spacemit-k3-adaptation-part2.md)
**阅读时间：30 分钟**

内容：
- llama.cpp 运行时流程详解
- xLLM 方案 A 运行时流程详解
- 性能瓶颈对比分析
- 方案 A 目录结构设计
- Platform 层实现（Step 1-3）
- IME Kernels 封装详解

**适合：** 系统工程师、性能优化工程师

---

#### [第三部分：完整实施与优化策略](./spacemit-k3-adaptation-part3.md)
**阅读时间：40 分钟**

内容：
- 算子适配器实现（Step 4）
- Executor 实现（Step 5）
- CMake 配置详解（Step 6）
- 编译脚本（Step 7）
- 性能优化策略
  - 方案 A+: GGUF 格式支持
  - 零拷贝 Tensor 封装
  - 多核并行调度
  - TCM 内存优化

**适合：** 开发工程师、需要实际实施的人员

---

#### [第四部分：性能评估与总结](./spacemit-k3-adaptation-part4.md)
**阅读时间：30 分钟**

内容：
- 性能评估与预测
- 三个方案详细对比
- 四阶段实施路线图
- 风险评估与应对
- 总结与建议
- 附录：关键问题解答

**适合：** 项目经理、质量保证、技术决策者

---

## 🎯 推荐阅读路径

### 路径 1：快速了解（15 分钟）
1. [执行摘要](./spacemit-k3-adaptation-summary.md) - 5 分钟
2. [第四部分](./spacemit-k3-adaptation-part4.md) 第 12-13 节 - 10 分钟

### 路径 2：技术评估（1 小时）
1. [执行摘要](./spacemit-k3-adaptation-summary.md) - 5 分钟
2. [第一部分](./spacemit-k3-adaptation-part1.md) - 20 分钟
3. [第二部分](./spacemit-k3-adaptation-part2.md) 第 4 节 - 15 分钟
4. [第四部分](./spacemit-k3-adaptation-part4.md) 第 8-10 节 - 20 分钟

### 路径 3：完整学习（2-3 小时）
1. [完整文档索引](./spacemit-k3-adaptation-index.md) - 导航
2. [第一部分](./spacemit-k3-adaptation-part1.md) - 20 分钟
3. [第二部分](./spacemit-k3-adaptation-part2.md) - 30 分钟
4. [第三部分](./spacemit-k3-adaptation-part3.md) - 40 分钟
5. [第四部分](./spacemit-k3-adaptation-part4.md) - 30 分钟

### 路径 4：实施开发（按需查阅）
1. [第三部分](./spacemit-k3-adaptation-part3.md) - 实施步骤
2. [第四部分](./spacemit-k3-adaptation-part4.md) 第 10 节 - 路线图
3. 随时参考前两部分的技术细节

---

## 📊 核心数据速查

### 性能对比（SpacemiT A100）

| 模型 | llama.cpp | xLLM 方案 A | xLLM 方案 A+ |
|------|-----------|-------------|--------------|
| Qwen3 0.6B | 55.77 t/s | 50-53 t/s | 53-55 t/s |
| Qwen3 4B | 11.29 t/s | 10.2-10.7 t/s | 10.7-11.1 t/s |

### 内存占用（7B 模型）

| 方案 | 内存占用 |
|------|---------|
| llama.cpp GGUF | 3.8 GB |
| xLLM 方案 A (FP16) | 17.8 GB ⚠️ |
| xLLM 方案 A+ (GGUF) | 3.8 GB ✅ |

### 实施周期

| 阶段 | 时间 |
|------|------|
| Phase 1: MVP | 3-4 周 |
| Phase 2: 完整实现 | 6-8 周 |
| Phase 3: 性能优化 | 6-8 周 |
| Phase 4: 生产就绪 | 4-6 周 |
| **总计** | **4-6 个月** |

---

## 🔑 关键问题快速解答

### Q: xLLM 是 C++ 实现还是 Python？
**A:** C++ 实现，基于 PyTorch C++ API (libtorch)。

### Q: xLLM 能用 GGUF 模型吗？
**A:** 默认不支持，但方案 A+ 会实现 GGUF 加载器。

### Q: 性能会比 llama.cpp 差多少？
**A:** 方案 A 慢 5-10%，方案 A+ 慢 3-5%。

### Q: 内存占用会大多少？
**A:** 方案 A 大 4.7 倍，方案 A+ 相同。

### Q: 多久能完成？
**A:** 4-6 个月，2-3 名工程师。

---

## 🛠️ 技术栈

- **核心框架**: PyTorch C++ API (libtorch)
- **硬件加速**: IME1/IME2 + TCM
- **参考实现**: llama.cpp ggml-spacemit
- **工具链**: SpacemiT toolchain v1.2.7
- **目标平台**: SpacemiT K3 (riscv64)

---

## 📞 相关资源

### 代码仓库
- llama.cpp: `/data/workspace2026-new/work0618/tech-analyais0713/llama.cpp-handbook`
- xLLM: `/data/workspace2026-new/work0618/tech-analyais0713/xllm-handbook`

### 关键文档
- xLLM 架构: `../handbook/ARCHITECTURE.md`
- 项目指令: `../CLAUDE.md`
- llama.cpp SpacemiT: `llama.cpp-handbook/docs/build-riscv64-spacemit.md`

### K3 Worker 机
- 地址: `10.0.90.243`
- 用户: `bianbu / bianbu`
- 路径: `/home/bianbu/bianbu-agentos`

---

## 📝 文档维护

- **版本**: v1.0
- **创建日期**: 2026-07-23
- **作者**: Analysis Agent
- **更新频率**: 根据实施进度更新

如需更新或补充内容，请：
1. 修改对应的 part1-4 文档
2. 同步更新 index 和 summary
3. 更新本 README

---

**祝接入顺利！🚀**
