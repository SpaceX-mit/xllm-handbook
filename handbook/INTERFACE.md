# xLLM 接口设计文档 (INTERFACE)

> **文档版本**: v1.0  
> **项目**: xLLM 大模型推理框架  
> **日期**: 2026-07-23

---

## 1. 接口概览

### 1.1 接口层次

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           接口层次图                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Public API (公开 API)                          │   │
│  │                                                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │  Python    │  │     C      │  │   gRPC     │             │   │
│  │  │   API      │  │    API     │  │    API     │             │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Internal API (内部 API)                        │   │
│  │                                                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │   C++      │  │  Protobuf   │  │   Kernel   │             │   │
│  │  │  Interface  │  │   Interface │  │  Interface │             │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Python API

### 2.1 LLM 类

**文件**: `pybind/llm.py`

```python
class LLM:
    """
    xLLM LLM 推理引擎
    
    Example:
        >>> llm = xllm.LLM(model_path="Qwen/Qwen2.5-7B-Instruct")
        >>> response = llm.chat([{"role": "user", "content": "Hello!"}])
        >>> print(response.content)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        tensor_parallel: int = 1,
        dtype: str = "float16",
        max_batch_size: int = 64,
        max_sequence_length: int = 8192,
        **kwargs
    ):
        """
        初始化 LLM 推理引擎
        
        Args:
            model_path: 模型路径 (HuggingFace 格式)
            device: 设备类型 ("cuda", "npu", "mlu", "dcu")
            tensor_parallel: 张量并行度
            dtype: 数据类型 ("float16", "bfloat16", "int8")
            max_batch_size: 最大批次大小
            max_sequence_length: 最大序列长度
        
        Raises:
            RuntimeError: 模型加载失败
        """
        ...
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None,
        stream: bool = False
    ) -> ChatResponse:
        """
        对话补全 (同步)
        
        Args:
            messages: 对话历史，格式为:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ]
            sampling_params: 采样参数
            stream: 是否返回生成器 (仅用于流式)
        
        Returns:
            ChatResponse: 对话响应，包含 content, finish_reason, usage 等
        
        Example:
            >>> response = llm.chat([
            ...     {"role": "user", "content": "What is 2+2?"}
            ... ])
            >>> print(response.content)
            '4'
        """
        ...
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None
    ) -> Generator[str, None, None]:
        """
        流式对话补全
        
        Args:
            messages: 对话历史
            sampling_params: 采样参数
        
        Yields:
            str: 增量生成的文本片段
        
        Example:
            >>> for chunk in llm.stream_chat([
            ...     {"role": "user", "content": "Count to 5"}
            ... ]):
            ...     print(chunk, end="", flush=True)
            1 2 3 4 5
        """
        ...
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """
        通用文本补全
        
        Args:
            prompts: 提示文本 (单条或批量)
            sampling_params: 采样参数
        
        Returns:
            List[str]: 生成结果列表
        
        Example:
            >>> results = llm.generate([
            ...     "The capital of France is",
            ...     "The largest planet is"
            ... ])
            >>> print(results)
            ['Paris', 'Jupiter']
        """
        ...
    
    def batch_generate(
        self,
        requests: List[GenerationRequest]
    ) -> List[GenerationResult]:
        """
        批量生成 (高性能)
        
        Args:
            requests: 批量生成请求
        
        Returns:
            List[GenerationResult]: 批量生成结果
        """
        ...
    
    @property
    def model_config(self) -> ModelConfig:
        """获取模型配置"""
        ...
    
    @property
    def tokenizer(self) -> Tokenizer:
        """获取分词器"""
        ...
```

### 2.2 SamplingParams 类

**文件**: `pybind/llm.py`

```python
@dataclass
class SamplingParams:
    """
    采样参数配置
    
    Attributes:
        temperature: 温度参数，控制随机性
            - 0.0: 确定性采样 (greedy)
            - > 0.0: 概率分布采样
            - 推荐范围: 0.7 - 1.0
        
        top_p: Nucleus 采样阈值
            - 1.0: 保留所有 token
            - 0.9: 保留累积概率 >= 0.9 的最小 token 集合
            - 推荐范围: 0.9 - 0.95
        
        top_k: Top-K 采样
            - -1: 禁用
            - N: 只保留概率 top N 的 token
        
        max_tokens: 最大生成 token 数
            - 默认为 4096
            - 用于控制生成长度
        
        stop: 停止词列表
            - 生成到此词时停止
        
        repetition_penalty: 重复惩罚
            - 1.0: 无惩罚
            - > 1.0: 惩罚重复
            - 推荐范围: 1.0 - 1.2
        
        presence_penalty: 存在惩罚 (影响词是否再次出现)
        frequency_penalty: 频率惩罚 (根据词出现次数惩罚)
        
        seed: 随机种子 (用于确定性采样)
        
        beam_size: Beam 宽度 (用于 beam search)
        
        length_penalty: 长度惩罚 (beam search)
    """
    
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 4096
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    beam_size: int = 1
    length_penalty: float = 1.0
    
    def validate(self) -> None:
        """验证参数合法性"""
        if not 0.0 <= self.temperature:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError(f"top_k must be -1 or >= 1, got {self.top_k}")
```

### 2.3 VLM 类 (视觉语言模型)

**文件**: `pybind/vlm.py`

```python
class VLM:
    """
    视觉语言模型推理引擎
    
    Example:
        >>> vlm = xllm.VLM(model_path="Qwen/Qwen2-VL-7B-Instruct")
        >>> response = vlm.chat([
        ...     {"role": "user", "content": "What is in this image?",
        ...      "image": "image.jpg"}
        ... ])
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        **kwargs
    ):
        """初始化 VLM 引擎"""
        ...
    
    def chat(
        self,
        messages: List[Dict],
        sampling_params: Optional[SamplingParams] = None
    ) -> ChatResponse:
        """
        多模态对话
        
        Args:
            messages: 支持图片的消息列表:
                [
                    {"role": "user", "content": "What is this?",
                     "image": "path/to/image.jpg"},
                    {"role": "assistant", "content": "It's a cat."},
                    {"role": "user", "content": "What color is it?",
                     "images": ["image1.jpg", "image2.jpg"]},
                ]
            sampling_params: 采样参数
        
        Returns:
            ChatResponse: 对话响应
        """
        ...
```

---

## 3. C API

### 3.1 核心 API

**文件**: `c_api/xllm.h`

```c
/**
 * @file xllm.h
 * @brief xLLM C API - 跨语言绑定接口
 */

#ifndef XLLM_C_API_H
#define XLLM_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

// ===== 类型定义 =====

/**
 * xLLM 句柄 - 模型实例
 */
typedef struct XLLMHandle_ *XLLMHandle;

/**
 * 请求句柄
 */
typedef struct XLLMRequest_ *XLLMRequest;

/**
 * 模型配置
 */
typedef struct XLLMModelConfig {
    const char* model_path;       // 模型路径
    const char* device;           // 设备: "cuda", "npu", "mlu"
    int tensor_parallel;          // 张量并行度
    const char* dtype;            // 数据类型
    int max_batch_size;          // 最大批次
    int max_sequence_length;     // 最大序列长度
} XLLMModelConfig;

/**
 * 采样参数
 */
typedef struct XLLMSamplingParams {
    float temperature;            // 温度
    float top_p;                 // Nucleus 采样
    int top_k;                   // Top-K
    int max_tokens;              // 最大 token 数
    float repetition_penalty;    // 重复惩罚
    const char* stop;            // 停止词
} XLLMSamplingParams;

/**
 * 请求结果
 */
typedef struct XLLMResult {
    char* content;               // 生成内容
    int finish_reason;           // 结束原因: 0=length, 1=stop
    int generated_tokens;         // 生成的 token 数
    double latency_ms;           // 延迟 (毫秒)
} XLLMResult;

// ===== 生命周期 API =====

/**
 * 创建 xLLM 句柄
 * @param config 模型配置
 * @return 句柄，失败返回 NULL
 * @note 使用完毕后必须调用 xllm_destroy_handle 释放
 */
XLLMHandle xllm_create_handle(const XLLMModelConfig* config);

/**
 * 销毁 xLLM 句柄
 * @param handle 句柄
 */
void xllm_destroy_handle(XLLMHandle handle);

/**
 * 检查句柄是否有效
 * @param handle 句柄
 * @return 1=有效, 0=无效
 */
int xllm_is_valid(XLLMHandle handle);

// ===== 请求 API =====

/**
 * 创建请求
 * @param handle 句柄
 * @param prompt 输入提示
 * @param params 采样参数
 * @return 请求句柄
 */
XLLMRequest xllm_create_request(
    XLLMHandle handle,
    const char* prompt,
    const XLLMSamplingParams* params
);

/**
 * 添加多模态数据
 * @param request 请求
 * @param image_path 图片路径
 */
void xllm_request_add_image(
    XLLMRequest request,
    const char* image_path
);

/**
 * 销毁请求
 * @param request 请求
 */
void xllm_destroy_request(XLLMRequest request);

// ===== 执行 API =====

/**
 * 添加请求到执行队列
 * @param handle 句柄
 * @param request 请求
 * @return 0=成功, -1=失败
 */
int xllm_add_request(XLLMHandle handle, XLLMRequest request);

/**
 * 执行一步推理
 * @param handle 句柄
 * @return 0=成功, -1=失败
 * @note 需要循环调用直到请求完成
 */
int xllm_step(XLLMHandle handle);

/**
 * 获取请求结果
 * @param request 请求
 * @return 结果结构体
 * @note 结果需要调用 xllm_free_result 释放
 */
XLLMResult xllm_get_result(XLLMRequest request);

/**
 * 释放结果内存
 * @param result 结果
 */
void xllm_free_result(XLLMResult result);

/**
 * 检查请求是否完成
 * @param request 请求
 * @return 1=完成, 0=未完成
 */
int xllm_request_is_finished(XLLMRequest request);

// ===== 批量 API =====

/**
 * 批量添加请求
 * @param handle 句柄
 * @param requests 请求数组
 * @param count 请求数量
 * @return 0=成功, -1=失败
 */
int xllm_add_requests_batch(
    XLLMHandle handle,
    XLLMRequest* requests,
    int count
);

/**
 * 获取批量结果
 * @param requests 请求数组
 * @param count 请求数量
 * @param results 结果数组 (输出)
 */
void xllm_get_results_batch(
    XLLMRequest* requests,
    int count,
    XLLMResult* results
);

// ===== 配置 API =====

/**
 * 获取模型配置信息
 * @param handle 句柄
 * @param config 输出配置
 */
void xllm_get_config(XLLMHandle handle, XLLMModelConfig* config);

/**
 * 获取设备信息
 * @param handle 句柄
 * @return 设备描述字符串
 */
const char* xllm_get_device_info(XLLMHandle handle);

#ifdef __cplusplus
}
#endif

#endif // XLLM_C_API_H
```

### 3.2 使用示例

```c
#include "xllm.h"
#include <stdio.h>
#include <string.h>

int main() {
    // 1. 创建配置
    XLLMModelConfig config = {
        .model_path = "/path/to/model",
        .device = "cuda",
        .tensor_parallel = 1,
        .dtype = "float16",
        .max_batch_size = 64,
        .max_sequence_length = 8192
    };
    
    // 2. 创建句柄
    XLLMHandle handle = xllm_create_handle(&config);
    if (!handle) {
        fprintf(stderr, "Failed to create handle\n");
        return 1;
    }
    
    // 3. 创建请求
    XLLMSamplingParams params = {
        .temperature = 0.7f,
        .top_p = 0.95f,
        .top_k = -1,
        .max_tokens = 100,
        .repetition_penalty = 1.1f
    };
    
    XLLMRequest request = xllm_create_request(
        handle,
        "The capital of France is",
        &params
    );
    
    // 4. 添加到队列
    xllm_add_request(handle, request);
    
    // 5. 循环推理直到完成
    while (!xllm_request_is_finished(request)) {
        xllm_step(handle);
    }
    
    // 6. 获取结果
    XLLMResult result = xllm_get_result(request);
    printf("Generated: %s\n", result.content);
    printf("Tokens: %d, Latency: %.2fms\n", 
           result.generated_tokens, result.latency_ms);
    
    // 7. 清理
    xllm_free_result(result);
    xllm_destroy_request(request);
    xllm_destroy_handle(handle);
    
    return 0;
}
```

---

## 4. gRPC API

### 4.1 Protocol Buffers 定义

**文件**: `proto/chat.proto`

```protobuf
syntax = "proto3";

package xllm;

// ===== 消息定义 =====

message Message {
    string role = 1;         // "system", "user", "assistant"
    string content = 2;      // 文本内容
    repeated ContentPart content_parts = 3;  // 多模态内容
}

message ContentPart {
    oneof type {
        TextPart text = 1;
        ImagePart image = 2;
        VideoPart video = 3;
    }
}

message TextPart {
    string text = 1;
}

message ImagePart {
    string url = 1;              // URL 或 base64
    string detail = 2;           // "low", "high", "auto"
}

message VideoPart {
    string url = 1;
}

message ChatRequest {
    string model = 1;                    // 模型名称
    repeated Message messages = 2;        // 对话历史
    SamplingParams sampling_params = 3; // 采样参数
    bool stream = 4;                     // 是否流式
    string user_id = 5;                  // 用户 ID (可选)
    map<string, string> metadata = 6;    // 元数据
}

message ChatResponse {
    string id = 1;                        // 响应 ID
    string model = 2;                     // 模型名称
    Choice choice = 3;                   // 选择结果
    Usage usage = 4;                     // 使用统计
    int64 created = 5;                   // 时间戳
}

message ChatCompleteRequest {
    string model = 1;
    repeated Message messages = 2;
    SamplingParams sampling_params = 3;
    string user_id = 4;
}

message ChatCompleteResponse {
    string id = 1;
    string model = 2;
    string content = 3;
    string finish_reason = 4;
    Usage usage = 5;
    int64 created = 6;
}

message Choice {
    int32 index = 1;
    Message message = 2;
    string finish_reason = 3;
}

message Usage {
    int32 prompt_tokens = 1;     // 输入 token 数
    int32 completion_tokens = 2;  // 输出 token 数
    int32 total_tokens = 3;      // 总 token 数
}

message SamplingParams {
    float temperature = 1;
    float top_p = 2;
    int32 top_k = 3;
    int32 max_tokens = 4;
    repeated string stop = 5;
    float repetition_penalty = 6;
    int32 seed = 7;
    int32 beam_size = 8;
}

// ===== 流式消息 =====

message ChatStreamResponse {
    string id = 1;
    int32 index = 2;
    ChoiceDelta delta = 3;
    int64 created = 4;
    bool finish = 5;
    Usage usage = 6;
}

message ChoiceDelta {
    string content = 1;
    string role = 2;
}

// ===== 服务定义 =====

service ChatService {
    // 流式对话
    rpc Chat(stream ChatRequest) returns (stream ChatStreamResponse);
    
    // 非流式对话
    rpc ChatComplete(ChatCompleteRequest) returns (ChatCompleteResponse);
    
    // 健康检查
    rpc Health(HealthRequest) returns (HealthResponse);
}

message HealthRequest {}

message HealthResponse {
    bool healthy = 1;
    string version = 2;
    int32 gpu_count = 3;
    string gpu_info = 4;
}
```

### 4.2 HTTP API

基于 RESTful 的 HTTP 接口，与 gRPC 功能等价。

**端点**:

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/v1/chat/completions` | 对话补全 |
| POST | `/v1/completions` | 文本补全 |
| POST | `/v1/embeddings` | 向量嵌入 |
| GET | `/v1/models` | 模型列表 |
| GET | `/health` | 健康检查 |

**Chat Completions 示例**:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": false
  }'
```

**响应**:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1699999999,
  "model": "Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 15,
    "total_tokens": 35
  }
}
```

---

## 5. C++ 内部接口

### 5.1 核心接口类

```cpp
// core/runtime/worker.h

namespace xllm {

/**
 * Worker - 模型执行工作器
 * 
 * 生命周期:
 * 1. Construction: 创建 Worker 实例
 * 2. Init: init_model() 加载模型
 * 3. Allocate: allocate_kv_cache() 分配 KV Cache
 * 4. Execute: 循环 step() 执行推理
 * 5. Sleep/Wakeup: 深度睡眠和恢复
 * 6. Destruction: 自动释放资源
 */
class Worker {
 public:
    /**
     * 构造函数
     * @param parallel_args 并行配置
     * @param device 设备
     * @param options 运行时选项
     * @param worker_type Worker 类型
     */
    Worker(const ParallelArgs& parallel_args,
           const torch::Device& device,
           const runtime::Options& options,
           WorkerType worker_type);
    
    // ===== 生命周期 =====
    
    /**
     * 初始化模型 (同步)
     * @param weights_path 权重路径
     * @param random_seed 随机种子
     * @param status 主节点状态
     * @return 成功返回 true
     */
    bool init_model(const std::string& weights_path,
                   int32_t random_seed,
                   MasterStatus status);
    
    /**
     * 初始化模型 (异步)
     */
    folly::SemiFuture<bool> init_model_async(...);
    
    /**
     * 深度睡眠 (释放显存)
     */
    bool sleep(MasterStatus status);
    
    /**
     * 唤醒
     */
    bool wakeup(const WakeupOptions& options);
    
    /**
     * 更新权重
     */
    bool update_weights(const std::string& weights_path);
    
    // ===== KV Cache =====
    
    /**
     * 分配 KV Cache
     */
    bool allocate_kv_cache(const KVCacheShape& shape);
    
    /**
     * 估算 KV Cache 容量
     */
    std::tuple<int64_t, int64_t> estimate_kv_cache_capacity();
    
    // ===== 执行 =====
    
    /**
     * 准备输入
     */
    ForwardInput prepare_inputs(Batch& batch);
    
    /**
     * 执行一步
     */
    std::optional<ForwardOutput> step(const ForwardInput& inputs);
    
    /**
     * 执行一步 (异步)
     */
    folly::SemiFuture<std::optional<ForwardOutput>> 
    step_async(const ForwardInput& inputs);
    
    // ===== Profile =====
    
    bool start_profile();
    bool stop_profile();
    
    // ===== P2P 通信 =====
    
    bool link_p2p(const std::string& remote_addr);
    bool unlink_p2p(const std::string& remote_addr);
};

}  // namespace xllm
```

### 5.2 关键数据结构

```cpp
// core/runtime/forward_params.h

namespace xllm {

/**
 * ForwardInput - 前向传播输入
 * 
 * 封装一次前向传播所需的所有输入数据
 */
struct ForwardInput {
    // ===== Token 输入 =====
    torch::Tensor input_tokens;      // [batch_size, seq_len]
    torch::Tensor positions;         // 位置 ID
    
    // ===== 注意力参数 =====
    torch::Tensor cos_cached;        // RoPE cos
    torch::Tensor sin_cached;        // RoPE sin
    
    // ===== 采样参数 =====
    SamplingParameters sampling_params;
    
    // ===== KV Cache =====
    std::vector<KVCache> kv_caches;
    
    // ===== 多模态 =====
    std::vector<MMData> mm_inputs;
};

/**
 * ForwardOutput - 前向传播输出
 * 
 * 封装一次前向传播的输出结果
 */
struct ForwardOutput {
    torch::Tensor logits;            // [batch_size, vocab_size]
    torch::Tensor hidden_states;     // 隐藏状态 (可选)
    
    // 采样结果
    std::vector<int32_t> output_tokens;
    std::vector<float> log_probs;
};

/**
 * KVCacheShape - KV Cache 形状
 */
struct KVCacheShape {
    int32_t num_layers;            // 层数
    int32_t num_heads;              // 头数
    int32_t head_dim;               // 头维度
    int32_t max_blocks;             // 最大块数
    int32_t block_size;             // 块大小
};

}  // namespace xllm
```

---

## 6. Kernel 接口

### 6.1 算子注册

```cpp
// core/kernels/ops_api.h

namespace xllm {

/**
 * 算子注册表 - 统一管理所有 Kernel 实现
 */
class OpsRegistry {
 public:
    // ===== Attention 算子 =====
    static void attention(
        const Tensor& q,           // Query [batch, heads, seq, head_dim]
        const Tensor& k,           // Key   [batch, heads, seq, head_dim]
        const Tensor& v,           // Value [batch, heads, seq, head_dim]
        Tensor& output,            // Output [batch, heads, seq, head_dim]
        const AttentionParams& params  // 注意力参数
    );
    
    // ===== Linear 算子 =====
    static void linear(
        const Tensor& input,        // [*, in_features]
        const Tensor& weight,      // [out_features, in_features]
        const Tensor& bias,        // [out_features] (可选)
        Tensor& output,             // [*, out_features]
        const LinearParams& params
    );
    
    // ===== RMSNorm 算子 =====
    static void rms_norm(
        const Tensor& input,        // [*, hidden_size]
        const Tensor& weight,      // [hidden_size]
        float eps,                 // 数值稳定项
        Tensor& output             // [*, hidden_size]
    );
    
    // ===== RoPE 算子 =====
    static void rotary_embedding(
        const Tensor& q,           // Query
        const Tensor& k,           // Key
        const Tensor& cos,         // RoPE cos
        const Tensor& sin,         // RoPE sin
        Tensor& q_out,
        Tensor& k_out
    );
    
    // ===== MoE 算子 =====
    static void moe_forward(
        const Tensor& hidden_states,
        const Tensor& gate_weight,
        const Tensor& experts_weight,
        Tensor& output,
        const MoEParams& params
    );
};

/**
 * Attention 参数
 */
struct AttentionParams {
    bool is_causal;               // 是否因果注意力
    float attn_scalar;            // 注意力缩放
    int32_t seq_len;              // 序列长度
    int32_t num_heads;            // Q 头数
    int32_t num_kv_heads;         // KV 头数
    int32_t head_dim;             // 头维度
    torch::Tensor alibi_slopes;   // ALiBi 斜率
};

/**
 * Linear 参数
 */
struct LinearParams {
    bool transposed;              // 是否转置
    c10::ScalarType out_dtype;   // 输出类型
};

}  // namespace xllm
```

---

## 7. 接口兼容性

### 7.1 版本策略

| 版本 | 策略 | 说明 |
|------|------|------|
| Major | 不兼容 | API 重大变更 |
| Minor | 向后兼容 | 新增 API |
| Patch | 向前兼容 | Bug 修复 |

### 7.2 废弃策略

1. **废弃警告期**: 2 个 Minor 版本
2. **废弃标记**: `@deprecated` 注解
3. **迁移指南**: 提供迁移文档

### 7.3 API 稳定性保证

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         API 稳定性保证                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Public API:  ✓ 完全稳定                                               │
│  ├── Python API                                                         │
│  ├── C API                                                              │
│  └── gRPC/HTTP API                                                      │
│                                                                         │
│  Internal API: ✗ 可能变更                                              │
│  ├── C++ Internal API                                                   │
│  ├── Kernel API                                                         │
│  └── Protocol Buffers (Internal)                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**文档结束**
