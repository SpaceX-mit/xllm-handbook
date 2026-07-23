# xLLM Coding Agent Instructions

## Directory Structure

```
├── xllm/
|   : main source folder
│   ├── api_service/               # code for api services
│   ├── c_api/                     # code for c api
│   ├── cc_api/                    # code for cc api 
│   ├── core/  
│   │   : xllm core features folder
│   │   ├── common/                
│   │   ├── distributed_runtime/   # code for distributed and pd serving
│   │   ├── framework/             # code for execution orchestration
│   │   ├── kernels/               # adaption for npu kernels adaption
│   │   ├── layers/                # model layers impl
│   │   ├── platform/              # adaption for various platform
│   │   ├── runtime/               # code for worker and executor
│   │   ├── scheduler/             # code for batch and pd scheduler
│   │   └── util/
│   ├── function_call              # code for tool call parser
│   ├── models/                    # models impl
│   ├── parser/                    # parser reasoning
│   ├── processors/                # code for vlm pre-processing
│   ├── proto/                     # communication protocol
│   ├── pybind/                    # code for python bind
|   └── server/                    # xLLM server
├── examples/                      # examples of calling xLLM
├── tools/                         # code for npu time generations
└── xllm.cpp                       # entrypoint of xLLM
```

## Code Style Guide

* Before editing, creating, refactoring, or reviewing any file under `xllm/`, you **MUST** read [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md).
* The file above is a **required instruction file**, not an optional reference. Do not skip reading it.
* Apply the rules in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) to **both code generation and code review**.
* Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* If [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) specifies a rule, that rule takes precedence over the Google C++/Python Style Guide.
* Use the Google C++/Python Style Guide only for cases not specified in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md).

## Review Instructions

* For code review tasks, you **MUST** first read [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Then read [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) and apply it during the review.
* Review code changes for quality, security, performance, correctness, and maintainability following the project-specific standards.
* Review code changes for DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Use the review workflow, checklist, severity rules, and output format defined in [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Apply the Google C++/Python Style Guide only when the project-specific style guide does not define the rule.
* Focus the review on the requested diff or changed files. Do not comment on unrelated code.

## Worker 机器
| 项 | 值 |
|----|----|
| 主机 | `10.0.90.243` |
| 用户名 | `bianbu` |
| 密码 | `bianbu` |
| 工作路径 | `/home/bianbu/bianbu-agentos` |

**连接规则：SSH 连接时不要交互式提示输入密码，直接使用上面的密码。** 用 `sshpass` 携带密码非交互连接：

```bash
# 直接执行远程命令
sshpass -p 'bianbu' ssh -o StrictHostKeyChecking=no bianbu@10.0.90.243 'cd /home/bianbu/bianbu-agentos && <command>'

# 交互式登录
sshpass -p 'bianbu' ssh -o StrictHostKeyChecking=no bianbu@10.0.90.243

# 传文件
sshpass -p 'bianbu' scp -o StrictHostKeyChecking=no <local> bianbu@10.0.90.243:/home/bianbu/bianbu-agentos/
```

若无 `sshpass`，先安装（Debian/Ubuntu：`apt-get install -y sshpass`）；无安装权限时可用 Python `paramiko` 携带密码连接。


## 运行平台与本地 AI 栈（重要，后续开发基线）

目标运行平台为 worker 机（SpacemiT K3，`10.0.90.243`）。**本地 AI 推理使用 SpacemiT 定制栈，不是通用 llama.cpp / 通用 onnxruntime / 纯 RVV**。设计与实现必须按此栈落地：

### 硬件加速分层
- **IME**（Integrated Matrix Engine，矩阵加速引擎）+ **TCM**（Tightly-Coupled Memory，紧耦合内存）是本地推理的主加速通路。
- **A100 簇** → 使用 **IME2 + TCM**。
- **X100 簇** → 使用 **IME1**。
- 算子不被 IME 覆盖时 **回退 RVV**（RISC-V Vector），再回退 **CPU 标量**。

### 软件栈分工
| 任务类别 | 运行栈 | 说明 |
|----------|--------|------|
| LLM 本地推理 | **定制版 `llama.cpp` + `ggml-spacemit` 后端** | A100=IME2+TCM，X100=IME1，回退 RVV/CPU |
| 图形 / 视觉类（OCR/Vision 等） | **`onnxruntime` + `spacemit-ep`**（SpacemiT Execution Provider，riscv64 IME/TCM 适配版） | 非通用 CPU EP |

### 编译前提（关键约束）
- 编译 `llama.cpp` 与 `onnxruntime` 均需 **SpacemiT 定制版 toolchain** 才能启用 IME/TCM 支持，**标准 riscv64-linux-gnu 工具链无法编出加速版本**。
- 因此 HLD/LLD 的 AI HAL 后端、AI Runtime EP Dispatcher、性能 baseline（[TBD-15]）均须以此栈为准：本地 LLM 后端 = ggml-spacemit；视觉/OCR 后端 = onnxruntime + spacemit-ep；EP 调度顺序 = IME(2/1)+TCM → RVV → CPU。
- doctor 工具须检测：定制 toolchain、ggml-spacemit、spacemit-ep、IME/TCM 驱动是否就绪，缺失时给出提示。

### 交叉编译工具链（面向 x86 开发机）
运行目标为 worker 机（SpacemiT K3，riscv64）时，在开发主机（x86_64 Linux）上**交叉编译** `llama.cpp`、`onnxruntime` 及一般 C/C++ 项目，统一使用 SpacemiT 官方交叉编译工具链（即上文"SpacemiT 定制 toolchain"，含 LLVM + GCC，能编出 IME/TCM 加速版本）：在K3 riscv64 上不需要使用交叉编译工具链。直接使用本机上安装的编译工具链即可。

| 项 | 值 |
|----|----|
| 工具链 | `spacemit-toolchain-linux-glibc-x86_64-v1.2.7`（LLVM+GCC） |
| 下载 | `https://nexus.bianbu.xyz/repository/toolchain/llvm-gcc/spacemit-toolchain-linux-glibc-x86_64-v1.2.7.tar.xz` |
| host → target | x86_64 Linux（开发机） → riscv64 glibc（K3 / worker） |
| 用途 | 交叉编译 ggml-spacemit / spacemit-ep 及所有下发到 worker 的 C/C++ 产物 |

- **备用版本**：若 `v1.2.7` 编译异常，回退使用 `v1.1.2`：`https://nexus.bianbu.xyz/repository/toolchain/llvm-gcc/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz`
- 标准 `riscv64-linux-gnu` 工具链**不可用于**本项目 AI 栈编译（编不出 IME/TCM 加速）。
- 版本随官方发布更新，当前基线 `v1.2.7`、备用 `v1.1.2`；升级时同步修订本条。
