# xLLM 开发者手册

## 文档目录

```
handbook/
├── README.md                    # 本文档
└── docs/                       # 完整文档体系
    ├── 00_OVERVIEW.md         # 文档总览与导航
    ├── 01_GLOSSARY.md         # 核心概念术语表
    ├── 02_ARCHITECTURE.md     # 系统架构设计
    ├── 03_DESIGN_PRINCIPLES.md # 设计原则与哲学
    ├── 04_DOMAIN_MODEL.md     # DDD领域模型设计
    ├── 05_SCHEDULER_DESIGN.md # Scheduler调度器详解
    ├── 06_WORKER_DESIGN.md    # Worker推理引擎详解
    ├── 07_KV_CACHE_DESIGN.md # KV Cache缓存管理详解
    ├── 08_AI_NATIVE_DEVELOPMENT.md # AI Native开发规范
    ├── 09_API_DESIGN.md      # API设计规范
    ├── 10_CONFIG_SCHEMA.md    # 配置Schema定义
    ├── 11_TEST_STRATEGY.md    # 测试策略与规范
    └── 12_DEPLOYMENT.md      # 部署指南
```

## 快速导航

| 主题 | 文档 | 说明 |
|-----|------|------|
| 核心概念 | [docs/01_GLOSSARY.md](./docs/01_GLOSSARY.md) | 术语表 (A-Z) |
| 系统架构 | [docs/02_ARCHITECTURE.md](./docs/02_ARCHITECTURE.md) | 架构设计 |
| 设计原则 | [docs/03_DESIGN_PRINCIPLES.md](./docs/03_DESIGN_PRINCIPLES.md) | 设计哲学 |
| 领域模型 | [docs/04_DOMAIN_MODEL.md](./docs/04_DOMAIN_MODEL.md) | DDD设计 |
| Scheduler | [docs/05_SCHEDULER_DESIGN.md](./docs/05_SCHEDULER_DESIGN.md) | 调度器 |
| Worker | [docs/06_WORKER_DESIGN.md](./docs/06_WORKER_DESIGN.md) | 推理引擎 |
| KV Cache | [docs/07_KV_CACHE_DESIGN.md](./docs/07_KV_CACHE_DESIGN.md) | 缓存管理 |
| AI开发 | [docs/08_AI_NATIVE_DEVELOPMENT.md](./docs/08_AI_NATIVE_DEVELOPMENT.md) | 开发规范 |
| API | [docs/09_API_DESIGN.md](./docs/09_API_DESIGN.md) | 接口定义 |
| 配置 | [docs/10_CONFIG_SCHEMA.md](./docs/10_CONFIG_SCHEMA.md) | 配置规范 |
| 测试 | [docs/11_TEST_STRATEGY.md](./docs/11_TEST_STRATEGY.md) | 测试策略 |
| 部署 | [docs/12_DEPLOYMENT.md](./docs/12_DEPLOYMENT.md) | 部署指南 |

---

## 文档特色

- **AI可复刻验收**: 每个文档包含结构化验收标准
- **直击代码本质**: 完整代码引用与文件位置标注
- **深度架构图**: Mermaid格式架构图
- **IPD开发流程**: 对齐集成产品开发流程

---

*最后更新: 2026-07-23*
