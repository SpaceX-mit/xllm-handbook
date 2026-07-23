# xLLM 配置 Schema

## 文档信息

```yaml
---
document_id: CONFIG-001
version: 1.0.0
category: configuration
owner: xllm-team
verification_level: BOTH
---
```

---

## 1. 配置结构概览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         xLLM Configuration Structure                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  xllm_config.json                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  {                                                                  │  │
│  │    "server": { ... },       // 服务配置                              │  │
│  │    "model": { ... },        // 模型配置                              │  │
│  │    "scheduler": { ... },    // 调度器配置                           │  │
│  │    "runtime": { ... },      // 运行时配置                           │  │
│  │    "kv_cache": { ... },     // KV Cache配置                         │  │
│  │    "parallel": { ... },     // 并行策略配置                         │  │
│  │    "hardware": { ... },     // 硬件配置                             │  │
│  │    "logging": { ... },     // 日志配置                             │  │
│  │    "metrics": { ... }       // 指标配置                             │  │
│  │  }                                                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 完整配置Schema

```json
// 代码位置: xllm/core/framework/config/schema.json

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "xLLM Configuration",
  "description": "xLLM Inference Engine Configuration Schema",
  "type": "object",
  "required": ["model", "hardware"],
  
  "properties": {
    "server": {
      "type": "object",
      "description": "Server configuration",
      "properties": {
        "host": {
          "type": "string",
          "default": "0.0.0.0",
          "description": "Server bind address"
        },
        "port": {
          "type": "integer",
          "minimum": 1,
          "maximum": 65535,
          "default": 8080,
          "description": "Server port"
        },
        "num_workers": {
          "type": "integer",
          "minimum": 1,
          "default": 1,
          "description": "Number of worker processes"
        },
        "max_connections": {
          "type": "integer",
          "minimum": 1,
          "default": 1000,
          "description": "Maximum concurrent connections"
        },
        "request_timeout_seconds": {
          "type": "number",
          "minimum": 0,
          "default": 300,
          "description": "Request timeout in seconds"
        },
        "enable_cors": {
          "type": "boolean",
          "default": false,
          "description": "Enable CORS"
        }
      }
    },
    
    "model": {
      "type": "object",
      "description": "Model configuration",
      "required": ["path"],
      "properties": {
        "path": {
          "type": "string",
          "description": "Model weight path (local or HuggingFace)"
        },
        "type": {
          "type": "string",
          "enum": ["causal_lm", "causal_vlm", "embedding", "dit"],
          "default": "causal_lm",
          "description": "Model type"
        },
        "config_file": {
          "type": "string",
          "description": "Path to model config.json"
        },
        "tokenizer": {
          "type": "string",
          "description": "Tokenizer path (default: auto-detect)"
        },
        "trust_remote_code": {
          "type": "boolean",
          "default": false,
          "description": "Trust remote code in model config"
        },
        "revision": {
          "type": "string",
          "default": "main",
          "description": "Model revision for HuggingFace"
        }
      }
    },
    
    "scheduler": {
      "type": "object",
      "description": "Scheduler configuration",
      "properties": {
        "type": {
          "type": "string",
          "enum": [
            "continuous",
            "disagg_pd",
            "fixed_steps",
            "zero_eviction",
            "pd_ooc"
          ],
          "default": "continuous",
          "description": "Scheduler type"
        },
        "max_batch_size": {
          "type": "integer",
          "minimum": 1,
          "default": 64,
          "description": "Maximum batch size"
        },
        "max_prefill_tokens": {
          "type": "integer",
          "minimum": 1,
          "default": 8192,
          "description": "Maximum prefill tokens per step"
        },
        "max_decode_tokens": {
          "type": "integer",
          "minimum": 1,
          "default": 4096,
          "description": "Maximum decode tokens per step"
        },
        "chunked_prefill_threshold": {
          "type": "integer",
          "minimum": 0,
          "default": 8192,
          "description": "Enable chunked prefill for sequences longer than this"
        },
        "prefix_cache_enabled": {
          "type": "boolean",
          "default": true,
          "description": "Enable prefix caching"
        },
        "prefix_cache_max_size": {
          "type": "integer",
          "minimum": 0,
          "default": 10000,
          "description": "Maximum cached prefix entries"
        },
        "policy": {
          "type": "object",
          "description": "Scheduling policy",
          "properties": {
            "type": {
              "type": "string",
              "enum": ["prefill_first", "decode_first", "unified"],
              "default": "unified"
            },
            "slo_weight_ttft": {
              "type": "number",
              "minimum": 0,
              "default": 1.0,
              "description": "SLO weight for TTFT"
            },
            "slo_weight_tpot": {
              "type": "number",
              "minimum": 0,
              "default": 1.0,
              "description": "SLO weight for TPOT"
            },
            "slo_weight_ttlt": {
              "type": "number",
              "minimum": 0,
              "default": 1.0,
              "description": "SLO weight for TTLT"
            }
          }
        }
      }
    },
    
    "runtime": {
      "type": "object",
      "description": "Runtime configuration",
      "properties": {
        "device": {
          "type": "string",
          "enum": ["cuda", "npu", "mlu", "dcu", "cpu"],
          "default": "cuda",
          "description": "Compute device"
        },
        "device_id": {
          "type": "integer",
          "minimum": 0,
          "default": 0,
          "description": "Primary device ID"
        },
        "dtype": {
          "type": "string",
          "enum": ["float32", "float16", "bfloat16", "int8", "fp8"],
          "default": "float16",
          "description": "Model data type"
        },
        "max_active_models": {
          "type": "integer",
          "minimum": 1,
          "default": 1,
          "description": "Maximum active models in memory"
        },
        "enable_profile": {
          "type": "boolean",
          "default": false,
          "description": "Enable profiling"
        },
        "enable_memory_profiling": {
          "type": "boolean",
          "default": false,
          "description": "Enable memory profiling"
        }
      }
    },
    
    "kv_cache": {
      "type": "object",
      "description": "KV Cache configuration",
      "properties": {
        "block_size": {
          "type": "integer",
          "minimum": 1,
          "default": 16,
          "description": "KV cache block size in tokens"
        },
        "num_blocks": {
          "type": "integer",
          "minimum": 1,
          "description": "Total number of blocks (auto-calculated if not set)"
        },
        "reserved_memory_mb": {
          "type": "integer",
          "minimum": 0,
          "default": 2048,
          "description": "Reserved memory in MB"
        },
        "enable_quantization": {
          "type": "boolean",
          "default": false,
          "description": "Enable KV cache quantization"
        },
        "quantization_type": {
          "type": "string",
          "enum": ["int8", "fp8", "nf4"],
          "default": "int8",
          "description": "Quantization type"
        },
        "enable_offload": {
          "type": "boolean",
          "default": false,
          "description": "Enable KV cache offload"
        },
        "offload_device": {
          "type": "string",
          "enum": ["cpu", "disk"],
          "default": "cpu",
          "description": "Offload target device"
        },
        "prefetch_enabled": {
          "type": "boolean",
          "default": false,
          "description": "Enable KV prefetching"
        }
      }
    },
    
    "parallel": {
      "type": "object",
      "description": "Parallelism configuration",
      "properties": {
        "tensor_parallel_size": {
          "type": "integer",
          "minimum": 1,
          "default": 1,
          "description": "Tensor parallelism size"
        },
        "pipeline_parallel_size": {
          "type": "integer",
          "minimum": 1,
          "default": 1,
          "description": "Pipeline parallelism size"
        },
        "context_parallel_size": {
          "type": "integer",
          "minimum": 1,
          "default": 1,
          "description": "Context parallelism size"
        },
        "expert_parallel_size": {
          "type": "integer",
          "minimum": 1,
          "default": 1,
          "description": "Expert parallelism size (MoE only)"
        },
        "data_parallel_size": {
          "type": "integer",
          "minimum": 1,
          "default": 1,
          "description": "Data parallelism size"
        }
      }
    },
    
    "quantization": {
      "type": "object",
      "description": "Model quantization configuration",
      "properties": {
        "enabled": {
          "type": "boolean",
          "default": false,
          "description": "Enable quantization"
        },
        "method": {
          "type": "string",
          "enum": ["awq", "gptq", "smoothquant", "bitsandbytes"],
          "description": "Quantization method"
        },
        "bits": {
          "type": "integer",
          "enum": [4, 8, 16],
          "default": 8,
          "description": "Quantization bits"
        },
        "group_size": {
          "type": "integer",
          "minimum": -1,
          "default": 128,
          "description": "Quantization group size (-1 for per-channel)"
        },
        "zero_point": {
          "type": "boolean",
          "default": false,
          "description": "Use zero point quantization"
        }
      }
    },
    
    "hardware": {
      "type": "object",
      "description": "Hardware-specific configuration",
      "required": ["type"],
      "properties": {
        "type": {
          "type": "string",
          "enum": ["nvidia_gpu", "ascend_npu", "cambricon_mlu", "hygon_dcu"],
          "description": "Hardware type"
        },
        "cuda": {
          "type": "object",
          "description": "NVIDIA GPU specific config",
          "properties": {
            "enable_cudnn": {
              "type": "boolean",
              "default": true
            },
            "enable_tf32": {
              "type": "boolean",
              "default": true
            },
            "cudnn_benchmark": {
              "type": "boolean",
              "default": true
            },
            "enable_flash_attention": {
              "type": "boolean",
              "default": true
            },
            "enable_flashinfer": {
              "type": "boolean",
              "default": false
            }
          }
        },
        "npu": {
          "type": "object",
          "description": "Ascend NPU specific config",
          "properties": {
            "device_id": {
              "type": "integer",
              "default": 0
            },
            "enable_stream_parallel": {
              "type": "boolean",
              "default": true
            },
            "matmul_split_dim": {
              "type": "integer",
              "default": 0
            }
          }
        }
      }
    },
    
    "logging": {
      "type": "object",
      "description": "Logging configuration",
      "properties": {
        "level": {
          "type": "string",
          "enum": ["trace", "debug", "info", "warning", "error", "fatal"],
          "default": "info"
        },
        "format": {
          "type": "string",
          "enum": ["text", "json"],
          "default": "json"
        },
        "output": {
          "type": "string",
          "enum": ["stdout", "stderr", "file"],
          "default": "stdout"
        },
        "file_path": {
          "type": "string",
          "description": "Log file path when output=file"
        },
        "max_file_size_mb": {
          "type": "integer",
          "default": 100
        },
        "max_backup_files": {
          "type": "integer",
          "default": 10
        },
        "enable_request_logging": {
          "type": "boolean",
          "default": false,
          "description": "Log all requests"
        }
      }
    },
    
    "metrics": {
      "type": "object",
      "description": "Metrics and monitoring configuration",
      "properties": {
        "enabled": {
          "type": "boolean",
          "default": true
        },
        "port": {
          "type": "integer",
          "default": 9090,
          "description": "Metrics server port"
        },
        "path": {
          "type": "string",
          "default": "/metrics",
          "description": "Metrics endpoint path"
        },
        "export_interval_seconds": {
          "type": "integer",
          "minimum": 1,
          "default": 10
        },
        "enable_prometheus": {
          "type": "boolean",
          "default": true
        },
        "enable_opentelemetry": {
          "type": "boolean",
          "default": false
        }
      }
    },
    
    "slo": {
      "type": "object",
      "description": "SLO configuration",
      "properties": {
        "default_ttft_slo_ms": {
          "type": "integer",
          "minimum": 0,
          "default": 1000,
          "description": "Default TTFT SLO in milliseconds"
        },
        "default_tpot_slo_ms": {
          "type": "integer",
          "minimum": 0,
          "default": 100,
          "description": "Default TPOT SLO in milliseconds"
        },
        "default_ttlt_slo_ms": {
          "type": "integer",
          "minimum": 0,
          "default": 30000,
          "description": "Default TTLT SLO in milliseconds"
        }
      }
    }
  }
}
```

---

## 3. 配置示例

### 3.1 基础配置

```json
// 单卡推理配置
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080
  },
  "model": {
    "path": "/models/qwen2-7b",
    "type": "causal_lm"
  },
  "hardware": {
    "type": "nvidia_gpu",
    "cuda": {
      "enable_flash_attention": true
    }
  }
}
```

### 3.2 生产配置

```json
// 生产环境高吞吐配置
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "max_connections": 2000
  },
  "model": {
    "path": "/models/qwen2-72b-instruct",
    "type": "causal_lm"
  },
  "scheduler": {
    "type": "continuous",
    "max_batch_size": 64,
    "max_prefill_tokens": 16384,
    "prefix_cache_enabled": true
  },
  "runtime": {
    "device": "cuda",
    "dtype": "float16"
  },
  "kv_cache": {
    "block_size": 16,
    "reserved_memory_mb": 4096
  },
  "parallel": {
    "tensor_parallel_size": 8,
    "context_parallel_size": 1
  },
  "quantization": {
    "enabled": true,
    "method": "awq",
    "bits": 4
  },
  "metrics": {
    "enabled": true,
    "port": 9090
  },
  "logging": {
    "level": "info",
    "format": "json"
  }
}
```

### 3.3 P/D分离配置

```json
// Prefill-Decode分离部署
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "role": "scheduler"
  },
  "model": {
    "path": "/models/qwen2-7b",
    "type": "causal_lm"
  },
  "scheduler": {
    "type": "disagg_pd",
    "prefill_node_addr": "10.0.0.1:8001",
    "decode_node_addr": "10.0.0.2:8002"
  },
  "runtime": {
    "device": "cuda"
  }
}
```

---

## 4. 配置加载

```cpp
// 代码位置: xllm/core/framework/config/config_loader.h

/**
 * @class ConfigLoader
 * @brief 配置加载器
 */
class ConfigLoader {
 public:
    /**
     * @brief 从JSON文件加载配置
     */
    static xllm::Config from_json_file(const std::string& path);
    
    /**
     * @brief 从JSON字符串加载配置
     */
    static xllm::Config from_json_string(const std::string& json);
    
    /**
     * @brief 合并多个配置源
     */
    static xllm::Config merge(
        const std::vector<std::string>& config_files,
        const std::string& env_prefix = "XLLM_"
    );
    
    /**
     * @brief 验证配置
     */
    static void validate(const xllm::Config& config);
    
    /**
     * @brief 应用环境变量覆盖
     */
    static void apply_env_overrides(xllm::Config& config, 
                                   const std::string& prefix = "XLLM_");
    
    /**
     * @brief 转换为运行时选项
     */
    static runtime::Options to_runtime_options(const xllm::Config& config);
    
    /**
     * @brief 导出配置为文档
     */
    static std::string to_markdown(const xllm::Config& config);
};
```

---

## 5. AI验收标准

### 5.1 配置验证

```yaml
ai_verification:
  config:
    - name: "必需字段"
      check: |
        验证 model.path 和 hardware.type 存在
        
    - name: "值范围"
      check: |
        验证所有带min/max限制的值在范围内
        
    - name: "一致性"
      check: |
        验证 parallel 配置一致性 (TP * PP * CP <= 总设备数)
        验证 dtype 和 quantization 兼容性
```

### 5.2 配置测试

```cpp
// 配置验证测试
TEST(ConfigValidation, RequiredFields) {
    auto config = ConfigLoader::from_json_string(R"({
        "model": {"path": "/test"},
        "hardware": {"type": "nvidia_gpu"}
    })");
    
    EXPECT_NO_THROW(ConfigLoader::validate(config));
}

TEST(ConfigValidation, InvalidValue) {
    auto config = ConfigLoader::from_json_string(R"({
        "model": {"path": "/test"},
        "hardware": {"type": "nvidia_gpu"},
        "server": {"port": 70000}  // 无效端口
    })");
    
    EXPECT_THROW(ConfigLoader::validate(config), ConfigValidationError);
}

TEST(ConfigValidation, ParallelConsistency) {
    auto config = ConfigLoader::from_json_string(R"({
        "model": {"path": "/test"},
        "hardware": {"type": "nvidia_gpu"},
        "parallel": {
            "tensor_parallel_size": 16,  // 超过可用GPU
            "pipeline_parallel_size": 1
        }
    })");
    
    // 应该警告或调整
    ConfigLoader::validate(config);
}
```

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [API设计](./09_API_DESIGN.md) | API定义 |
| [架构设计](./02_ARCHITECTURE.md) | 系统架构 |
| [部署指南](./10_DEPLOYMENT.md) | 部署配置 |
