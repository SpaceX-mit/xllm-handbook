# xLLM 部署指南

## 文档信息

```yaml
---
document_id: DEPLOY-001
version: 1.0.0
category: deployment
owner: xllm-team
verification_level: BOTH
---
```

---

## 1. 部署架构

### 1.1 单机部署

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       Single Node Deployment                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        ┌─────────────────┐                                   │
│                        │   xLLM Server   │                                   │
│                        │                 │                                   │
│                        │  • API Service  │                                   │
│                        │  • Scheduler    │                                   │
│                        │  • Worker Pool  │                                   │
│                        └────────┬────────┘                                   │
│                                 │                                            │
│         ┌───────────────────────┼───────────────────────┐                   │
│         │                       │                       │                    │
│         ▼                       ▼                       ▼                    │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐             │
│  │    GPU 0    │         │    GPU 1    │         │    GPU 2    │             │
│  │             │         │             │         │             │             │
│  │  • Model    │         │  • Model    │         │  • Model    │             │
│  │  • KVCache  │         │  • KVCache  │         │  • KVCache  │             │
│  │  • Executor │         │  • Executor │         │  • Executor │             │
│  └─────────────┘         └─────────────┘         └─────────────┘             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 P/D分离部署

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   Prefill-Decode Separation Deployment                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Load Balancer                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                    │                          │                              │
│                    ▼                          ▼                              │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                │
│  │    Prefill Cluster      │    │     Decode Cluster       │                │
│  │                         │    │                          │                │
│  │  ┌─────────────────┐   │    │  ┌─────────────────┐    │                │
│  │  │  Prefill Node 1 │   │    │  │  Decode Node 1  │    │                │
│  │  │  • High Memory  │   │    │  │  • Low Latency  │    │                │
│  │  │  • Batch Opt    │   │    │  │  • Streaming    │    │                │
│  │  └─────────────────┘   │    │  └─────────────────┘    │                │
│  │  ┌─────────────────┐   │    │  ┌─────────────────┐    │                │
│  │  │  Prefill Node 2 │   │    │  │  Decode Node 2  │    │                │
│  │  └─────────────────┘   │    │  └─────────────────┘    │                │
│  │                         │    │                          │                │
│  └─────────────────────────┘    └─────────────────────────┘                │
│                 │                                │                          │
│                 │         KV Cache Transfer       │                          │
│                 └────────────────────────┬────────┘                          │
│                                          │                                   │
│                                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    RDMA Network / Mooncake                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 多节点部署

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Node Deployment                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────────┐                                  │
│                         │   Coordinator   │                                  │
│                         │                 │                                  │
│                         │  • Etcd/Redis   │                                  │
│                         │  • Service Disc │                                  │
│                         └────────┬────────┘                                  │
│                                  │                                           │
│         ┌───────────────────────┼───────────────────────┐                   │
│         │                       │                       │                    │
│         ▼                       ▼                       ▼                    │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐           │
│  │   Node 0    │         │   Node 1    │         │   Node 2    │           │
│  │             │         │             │         │             │           │
│  │  TP Group 0 │◀──────▶│  TP Group 0 │◀──────▶│  TP Group 0 │           │
│  │  ┌───────┐  │       │  ┌───────┐  │       │  ┌───────┐  │           │
│  │  │ GPU 0 │  │       │  │ GPU 0 │  │       │  │ GPU 0 │  │           │
│  │  │ GPU 1 │  │       │  │ GPU 1 │  │       │  │ GPU 1 │  │           │
│  │  │ GPU 2 │  │       │  │ GPU 2 │  │       │  │ GPU 2 │  │           │
│  │  │ GPU 3 │  │       │  │ GPU 3 │  │       │  │ GPU 3 │  │           │
│  │  └───────┘  │       │  └───────┘  │       │  └───────┘  │           │
│  └─────────────┘         └─────────────┘         └─────────────┘           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    NCCL/GLOO Communication                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 快速开始

### 2.1 Docker部署

```bash
# 拉取镜像
docker pull quay.io/jd_xllm/xllm-ai:latest

# 运行容器
docker run -d \
  --gpus all \
  --network host \
  --name xllm \
  -v /path/to/models:/models \
  -v /path/to/config.json:/config/config.json \
  xllm-ai:latest \
  xllm_server --config /config/config.json
```

### 2.2 源码部署

```bash
# 1. 安装依赖
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libtorch-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libbrpc-dev \
    libssl-dev

# 2. 编译
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. 配置
cp config/xllm_config.example.json config/xllm_config.json
vim config/xllm_config.json

# 4. 运行
./build/xllm_server --config config/xllm_config.json
```

### 2.3 Kubernetes部署

```yaml
# xllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xllm-server
  labels:
    app: xllm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: xllm
  template:
    metadata:
      labels:
        app: xllm
    spec:
      containers:
      - name: xllm
        image: quay.io/jd_xllm/xllm-ai:latest
        ports:
        - containerPort: 8080
        - containerPort: 9090  # metrics
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        volumeMounts:
        - name: model
          mountPath: /models
        - name: config
          mountPath: /config
        env:
        - name: XLLM_CONFIG_PATH
          value: /config/config.json
        - name: XLLM_MODEL_PATH
          value: /models/qwen2-7b
      volumes:
      - name: model
        persistentVolumeClaim:
          claimName: xllm-models
      - name: config
        configMap:
          name: xllm-config
---
apiVersion: v1
kind: Service
metadata:
  name: xllm-service
spec:
  type: LoadBalancer
  ports:
  - port: 8080
    targetPort: 8080
  selector:
    app: xllm
```

---

## 3. 配置指南

### 3.1 基础配置

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "num_workers": 4
  },
  "model": {
    "path": "/models/qwen2-7b",
    "type": "causal_lm"
  },
  "hardware": {
    "type": "nvidia_gpu"
  }
}
```

### 3.2 高吞吐配置

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "max_connections": 2000
  },
  "scheduler": {
    "type": "continuous",
    "max_batch_size": 64,
    "max_prefill_tokens": 16384
  },
  "kv_cache": {
    "reserved_memory_mb": 4096
  }
}
```

---

## 4. 运维监控

### 4.1 监控指标

| 指标 | 说明 | Prometheus名称 |
|-----|------|---------------|
| 请求延迟 | TTFT/TPOT/TTLT | `xllm_request_latency_*` |
| 吞吐量 | QPS/TPS | `xllm_throughput_*` |
| GPU利用率 | 利用率/显存 | `xllm_gpu_*` |
| 批处理统计 | 批次大小 | `xllm_batch_*` |
| 缓存命中率 | Prefix Cache | `xllm_prefix_cache_*` |

### 4.2 日志配置

```json
{
  "logging": {
    "level": "info",
    "format": "json",
    "output": "file",
    "file_path": "/var/log/xllm/server.log",
    "max_file_size_mb": 100,
    "max_backup_files": 10
  }
}
```

---

## 5. 故障排查

### 5.1 常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 启动失败 | 模型路径错误 | 检查 `--model_path` 配置 |
| OOM | KV Cache过大 | 减小 `max_batch_size` |
| 高延迟 | GPU利用率低 | 增加 `max_batch_size` |
| 内存泄漏 | Block未释放 | 检查 `free_*` 调用 |

### 5.2 调试命令

```bash
# 查看日志
tail -f /var/log/xllm/server.log | jq

# 检查GPU状态
nvidia-smi

# 查看指标
curl http://localhost:9090/metrics

# 健康检查
curl http://localhost:8080/health
```

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [配置Schema](./10_CONFIG_SCHEMA.md) | 完整配置定义 |
| [测试策略](./11_TEST_STRATEGY.md) | 测试规范 |
| [监控指南](https://docs.xllm-ai.com/monitoring) | 监控配置 |
