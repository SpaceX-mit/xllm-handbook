# xLLM API 设计规范

## 文档信息

```yaml
---
document_id: API-001
version: 1.0.0
category: api_design
owner: xllm-team
verification_level: BOTH
---
```

---

## 1. API 层级结构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           xLLM API Layers                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Public API (公开API)                               │ │
│  │  • OpenAI兼容API: /v1/chat/completions                                 │ │
│  │  • Anthropic兼容API: /v1/messages                                      │ │
│  │  • REST/JSON格式                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Internal API (内部API)                               │ │
│  │  • Protocol Buffers定义                                               │ │
│  │  • brpc服务接口                                                       │ │
│  │  • 请求/响应消息                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    SDK API (SDK接口)                                     │ │
│  │  • Python SDK: xllm.ChatCompletion()                                   │ │
│  │  • C++ SDK: xllm::Client                                               │ │
│  │  • Java SDK: XLLMClient                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. OpenAI兼容API

### 2.1 Chat Completions

```yaml
endpoint: POST /v1/chat/completions
version: "2024-01"
compatibility: OpenAI v1.x
```

**Request:**

```yaml
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer {api_key}

{
  # ========== 必需参数 ==========
  
  "model": "string",                    # 模型名称
  
  "messages": [                         # 消息列表
    {
      "role": "system|user|assistant", # 角色
      "content": "string",             # 内容
      "name": "string (可选)"          # 命名
    }
  ],
  
  # ========== 生成参数 ==========
  
  "max_tokens": "integer (可选)",       # 最大生成token数
  "temperature": "number (可选)",       # 温度 (0-2)
  "top_p": "number (可选)",             # Top-p采样
  "n": "integer (可选)",               # 生成数量
  "stream": "boolean (可选)",          # 流式响应
  "stop": "string|string[] (可选)",     # 停止词
  "presence_penalty": "number (可选)",  # 存在惩罚
  "frequency_penalty": "number (可选)", # 频率惩罚
  "logit_bias": "object (可选)",       # logit偏差
  "user": "string (可选)",             # 用户标识
  
  # ========== xLLM扩展 ==========
  
  "extra_body": {
    "request_id": "string (可选)",      # 请求ID
    "priority": "integer (可选)",      # 优先级 (0-9)
    "ttft_slo_ms": "integer (可选)",   # TTFT SLO (ms)
    "tpot_slo_ms": "integer (可选)",   # TPOT SLO (ms)
    "seed": "integer (可选)",          # 随机种子
    "response_format": "object (可选)" # 响应格式
  }
}
```

**Response (non-streaming):**

```yaml
{
  "id": "chatcmpl-123",                # 完成ID
  "object": "chat.completion",         # 对象类型
  "created": 1700000000,               # 创建时间戳
  "model": "qwen2-7b",                 # 模型名称
  "choices": [
    {
      "index": 0,                      # 选择索引
      "message": {                     # 消息
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop",         # 结束原因
      "logprobs": {                   # log概率 (可选)
        "content": [...]
      }
    }
  ],
  "usage": {                           # 使用统计
    "prompt_tokens": 50,
    "completion_tokens": 30,
    "total_tokens": 80
  },
  "xllm_extra": {                      # xLLM扩展
    "ttft_ms": 120,                   # 实际TTFT
    "total_time_ms": 500,              # 总时间
    "cached_tokens": 32                # 缓存命中的token数
  }
}
```

**Response (streaming):**

```yaml
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"qwen2-7b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"qwen2-7b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: [DONE]
```

### 2.2 Completions

```yaml
endpoint: POST /v1/completions
version: "2024-01"
```

**Request:**

```yaml
{
  "model": "string",
  "prompt": "string|string[]|token[][]",  # 提示词
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "n": 1,
  "stream": false,
  "logprobs": null,
  "echo": false,
  "stop": null,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "best_of": 1,
  "seed": null,
  ...
}
```

---

## 3. Protocol Buffers 定义

### 3.1 核心消息

```protobuf
// 代码位置: xllm/proto/xllm.proto

syntax = "proto3";

package xllm;

option cc_enable_arenas = true;

// ========== 公共类型 ==========

message TensorProto {
    string name = 1;
    DataType dtype = 2;
    repeated int64 shape = 3;
    bytes raw_data = 4;  // 序列化数据
    repeated float float_data = 5;
    repeated int32 int_data = 6;
    string tensor_name = 7;
}

enum DataType {
    DT_INVALID = 0;
    DT_FLOAT = 1;
    DT_INT32 = 2;
    DT_INT64 = 3;
    DT_INT16 = 4;
    DT_INT8 = 5;
    DT_UINT8 = 6;
    DT_BOOL = 7;
    DT_HALF = 8;  // float16
    DT_BFLOAT16 = 9;
}

// ========== 请求消息 ==========

message ForwardRequest {
    uint64 batch_id = 1;
    repeated int64 input_ids = 2;
    repeated int64 position_ids = 3;
    int64 attention_mask = 4;  // TensorProto
    
    // 序列信息
    repeated SequenceInfo sequences = 5;
    
    // 采样参数
    SamplingParams sampling_params = 6;
    
    // 额外参数
    map<string, string> extra_params = 7;
}

message SequenceInfo {
    int64 request_id = 1;
    int64 sequence_index = 2;
    repeated int64 token_ids = 3;
    bool is_prefill = 4;
    int64 num_prompt_tokens = 5;
}

message SamplingParams {
    int32 beam_width = 1;
    float temperature = 2;
    float top_p = 3;
    int32 top_k = 4;
    int32 max_tokens = 5;
    repeated string stop_strings = 6;
}

// ========== 响应消息 ==========

message ForwardResponse {
    uint64 batch_id = 1;
    repeated SamplingResult results = 2;
    int32 num_decoded_tokens = 3;
    map<string, TensorProto> extra_outputs = 4;
}

message SamplingResult {
    int64 request_id = 1;
    int64 sequence_index = 2;
    repeated int32 output_token_ids = 3;
    repeated float logprobs = 4;
    bool finished = 5;
    FinishReason finish_reason = 6;
    int32 num_tokens = 7;
}

enum FinishReason {
    FINISH_UNSPECIFIED = 0;
    FINISH_STOP = 1;
    FINISH_LENGTH = 2;
    FINISH_STOP_STRING = 3;
}

// ========== KV Cache传输 ==========

message BlockTransferRequest {
    uint64 batch_id = 1;
    string src_addr = 2;
    string dst_addr = 3;
    repeated BlockInfo blocks = 4;
    int32 priority = 5;
}

message BlockInfo {
    int64 block_id = 1;
    int32 layer_id = 2;
    int32 num_tokens = 3;
    bytes data = 4;
}

message BlockTransferResponse {
    bool success = 1;
    string error_message = 2;
    int64 bytes_transferred = 3;
}

// ========== 服务定义 ==========

service XLLMService {
    // 推理
    rpc Forward(ForwardRequest) returns (ForwardResponse);
    rpc ForwardStream(ForwardRequest) returns (stream ForwardResponse);
    
    // 异步推理
    rpc ForwardAsync(ForwardRequest) returns (stream ForwardResponse);
    
    // KV Cache传输
    rpc TransferKVBlocks(stream BlockTransferRequest) 
        returns (stream BlockTransferResponse);
    
    // 集群管理
    rpc LinkCluster(LinkClusterRequest) returns (LinkClusterResponse);
    rpc UnlinkCluster(UnlinkClusterRequest) returns (UnlinkClusterResponse);
    
    // 模型管理
    rpc UpdateWeights(UpdateWeightsRequest) returns (UpdateWeightsResponse);
}

message LinkClusterRequest {
    repeated uint64 cluster_ids = 1;
    repeated string addresses = 2;
    repeated uint32 ports = 3;
}

message LinkClusterResponse {
    bool success = 1;
    string error_message = 2;
}
```

---

## 4. Python SDK

### 4.1 基础用法

```python
# 代码位置: xllm/pybind/xllm_pybind.cc
# 文档位置: docs/sdk/python.md

"""
xLLM Python SDK

提供简洁的Python接口调用xLLM推理服务。
"""

from __future__ import annotations
from typing import Optional, List, Union, Dict, Any, Iterator, AsyncIterator
from dataclasses import dataclass
import json

class xLLMClient:
    """
    xLLM客户端
    
    Args:
        base_url: xLLM服务地址
        api_key: API密钥 (可选)
        timeout: 请求超时时间 (秒)
        max_retries: 最大重试次数
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
    
    # ========== Chat Completions ==========
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        发送Chat Completion请求
        
        Args:
            model: 模型名称
            messages: 消息列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            n: 生成数量
            stream: 是否流式响应
            stop: 停止字符串
            
        Returns:
            ChatCompletion: 非流式响应
            Iterator[ChatCompletionChunk]: 流式响应
            
        Raises:
            APIError: API调用失败
            RateLimitError: 限流
            AuthenticationError: 认证失败
        """
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            **kwargs
        )
        
        if stream:
            return self._stream_chat_completion(request)
        else:
            return self._sync_chat_completion(request)
    
    # ========== Embedding ==========
    
    def create_embedding(
        self,
        model: str,
        input: Union[str, List[str]],
        encoding_format: str = "float"
    ) -> EmbeddingResponse:
        """
        创建Embedding向量
        
        Args:
            model: 模型名称
            input: 输入文本
            encoding_format: 编码格式
            
        Returns:
            EmbeddingResponse: Embedding响应
        """
        ...
    
    # ========== 辅助方法 ==========
    
    def _sync_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletion:
        """同步发送Chat Completion请求"""
        response = self._post("/v1/chat/completions", request.to_dict())
        return ChatCompletion.from_dict(response)
    
    def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> Iterator[ChatCompletionChunk]:
        """流式发送Chat Completion请求"""
        for chunk_data in self._stream_post("/v1/chat/completions", request.to_dict()):
            yield ChatCompletionChunk.from_dict(chunk_data)


@dataclass
class ChatCompletionRequest:
    """Chat Completion请求"""
    model: str
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.stream
        }
        if self.max_tokens:
            result["max_tokens"] = self.max_tokens
        if self.stop:
            result["stop"] = self.stop
        return result


@dataclass 
class ChatCompletion:
    """Chat Completion响应"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatCompletion":
        return cls(
            id=data["id"],
            object=data["object"],
            created=data["created"],
            model=data["model"],
            choices=[Choice.from_dict(c) for c in data["choices"]],
            usage=Usage.from_dict(data["usage"])
        )


@dataclass
class Choice:
    """响应选择"""
    index: int
    message: Dict[str, str]
    finish_reason: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Choice":
        return cls(
            index=data["index"],
            message=data["message"],
            finish_reason=data.get("finish_reason", "")
        )


@dataclass
class Usage:
    """使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Usage":
        return cls(
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            total_tokens=data["total_tokens"]
        )
```

### 4.2 使用示例

```python
# ========== 基础用法 ==========

from xllm import xLLMClient

client = xLLMClient("http://localhost:8080")

# 非流式调用
response = client.chat_completion(
    model="qwen2-7b",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "什么是大语言模型？"}
    ],
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].message["content"])
print(f"Usage: {response.usage}")

# ========== 流式调用 ==========

for chunk in client.chat_completion(
    model="qwen2-7b",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
):
    if chunk.choices[0].delta.get("content"):
        print(chunk.choices[0].delta["content"], end="", flush=True)

# ========== 批量调用 ==========

responses = client.batch_chat_completion([
    {"model": "qwen2-7b", "messages": [{"role": "user", "content": "你好"}]},
    {"model": "qwen2-7b", "messages": [{"role": "user", "content": "再见"}]},
])
```

---

## 5. C++ SDK

### 5.1 头文件

```cpp
// 代码位置: xllm/c_api/xllm_c_api.h

/**
 * @file xllm_c_api.h
 * @brief xLLM C API - 跨语言绑定接口
 */

#ifdef __cplusplus
extern "C" {
#endif

// ========== 类型定义 ==========

typedef struct XLLMClient XLLMClient;
typedef struct XLLMCompletion XLLMCompletion;
typedef struct XLLMStreamCompletion XLLMStreamCompletion;

// ========== 错误处理 ==========

typedef enum XLLMErrorCode {
    XLLM_OK = 0,
    XLLM_ERROR_INVALID_REQUEST = 1,
    XLLM_ERROR_NETWORK = 2,
    XLLM_ERROR_TIMEOUT = 3,
    XLLM_ERROR_AUTHENTICATION = 4,
    XLLM_ERROR_RATE_LIMIT = 5,
    XLLM_ERROR_SERVER = 6,
    XLLM_ERROR_UNKNOWN = 99
} XLLMErrorCode;

typedef struct XLLMError {
    XLLMErrorCode code;
    char message[512];
    char* details;  // 额外信息，调用者需释放
} XLLMError;

// ========== 客户端创建/销毁 ==========

/**
 * @brief 创建客户端
 * @param base_url 服务地址
 * @param api_key API密钥 (可为NULL)
 * @param error 错误信息 (可为NULL)
 * @return 客户端指针，失败返回NULL
 */
XLLMClient* xllm_client_create(
    const char* base_url,
    const char* api_key,
    XLLMError* error
);

/**
 * @brief 销毁客户端
 */
void xllm_client_destroy(XLLMClient* client);

// ========== Chat Completion ==========

/**
 * @brief Chat Completion请求参数
 */
typedef struct {
    const char* model;
    const char* messages_json;  // JSON格式的消息数组
    int max_tokens;
    float temperature;
    float top_p;
    int n;
    int stream;  // 0=非流式, 1=流式
    const char* stop;  // JSON数组或单个字符串
    
    // 额外参数 (JSON格式)
    const char* extra_params_json;
} XLLMChatCompletionParams;

/**
 * @brief 非流式Chat Completion
 * @param client 客户端
 * @param params 参数
 * @param completion 输出 (调用者需释放)
 * @param error 错误信息 (可为NULL)
 * @return 成功返回0
 */
int xllm_chat_completion(
    XLLMClient* client,
    const XLLMChatCompletionParams* params,
    XLLMCompletion** completion,
    XLLMError* error
);

/**
 * @brief 创建流式Chat Completion
 */
XLLMStreamCompletion* xllm_chat_completion_stream(
    XLLMClient* client,
    const XLLMChatCompletionParams* params,
    XLLMError* error
);

/**
 * @brief 获取下一个流式块
 * @return 0=成功, 1=流结束, -1=错误
 */
int xllm_stream_next(XLLMStreamCompletion* stream, XLLMError* error);

/**
 * @brief 获取当前块内容
 */
const char* xllm_stream_get_content(XLLMStreamCompletion* stream);
const char* xllm_stream_get_full_response_json(XLLMStreamCompletion* stream);

/**
 * @brief 销毁流
 */
void xllm_stream_destroy(XLLMStreamCompletion* stream);

// ========== Completion对象 ==========

/**
 * @brief 获取响应JSON
 */
const char* xllm_completion_to_json(const XLLMCompletion* completion);

/**
 * @brief 解析响应
 */
const char* xllm_completion_get_content(const XLLMCompletion* completion);
int xllm_completion_get_usage(const XLLMCompletion* completion,
                               int* prompt_tokens,
                               int* completion_tokens,
                               int* total_tokens);

/**
 * @brief 销毁Completion
 */
void xllm_completion_destroy(XLLMCompletion* completion);

// ========== 错误处理辅助 ==========

const char* xllm_error_get_message(const XLLMError* error);
void xllm_error_free(XLLMError* error);

#ifdef __cplusplus
}
#endif
```

### 5.2 使用示例

```cpp
// 代码位置: xllm/c_api/examples/completion_example.cc

#include "xllm_c_api.h"
#include <stdio.h>
#include <string.h>

int main() {
    // 创建客户端
    XLLMError error;
    XLLMClient* client = xllm_client_create(
        "http://localhost:8080",
        NULL,  // 无API密钥
        &error
    );
    
    if (!client) {
        fprintf(stderr, "Failed to create client: %s\n", error.message);
        xllm_error_free(&error);
        return 1;
    }
    
    // 准备消息
    const char* messages_json = 
        "[{\"role\":\"system\",\"content\":\"你是一个有帮助的助手\"},"
         "{\"role\":\"user\",\"content\":\"什么是AI？\"}]";
    
    // 设置参数
    XLLMChatCompletionParams params = {
        .model = "qwen2-7b",
        .messages_json = messages_json,
        .max_tokens = 512,
        .temperature = 0.7f,
        .top_p = 1.0f,
        .n = 1,
        .stream = 0,
        .stop = NULL,
        .extra_params_json = NULL
    };
    
    // 发送请求
    XLLMCompletion* completion = NULL;
    int ret = xllm_chat_completion(client, &params, &completion, &error);
    
    if (ret != 0) {
        fprintf(stderr, "API call failed: %s\n", error.message);
        xllm_error_free(&error);
        xllm_client_destroy(client);
        return 1;
    }
    
    // 处理响应
    const char* content = xllm_completion_get_content(completion);
    printf("Response: %s\n", content);
    
    int prompt_tokens, completion_tokens, total_tokens;
    xllm_completion_get_usage(completion, &prompt_tokens, 
                               &completion_tokens, &total_tokens);
    printf("Usage: prompt=%d, completion=%d, total=%d\n",
           prompt_tokens, completion_tokens, total_tokens);
    
    // 清理
    xllm_completion_destroy(completion);
    xllm_client_destroy(client);
    
    return 0;
}
```

---

## 6. 错误处理

### 6.1 错误码

```cpp
// 代码位置: xllm/api_service/error.h

namespace xllm {
namespace api {

/**
 * @enum APIErrorCode
 * @brief API错误码定义
 */
enum class APIErrorCode {
    // 成功
    OK = 0,
    
    // 客户端错误 (4xx)
    BAD_REQUEST = 400,
    UNAUTHORIZED = 401,
    FORBIDDEN = 403,
    NOT_FOUND = 404,
    METHOD_NOT_ALLOWED = 405,
    TIMEOUT = 408,
    CONFLICT = 409,
    UNPROCESSABLE_ENTITY = 422,
    TOO_MANY_REQUESTS = 429,
    
    // 服务器错误 (5xx)
    INTERNAL_SERVER_ERROR = 500,
    NOT_IMPLEMENTED = 501,
    BAD_GATEWAY = 502,
    SERVICE_UNAVAILABLE = 503,
    GATEWAY_TIMEOUT = 504,
    
    // xLLM特定错误
    MODEL_NOT_FOUND = 1001,
    CONTEXT_LENGTH_EXCEEDED = 1002,
    INVALID_API_KEY = 1003,
    RATE_LIMIT_EXCEEDED = 1004,
    QUOTA_EXCEEDED = 1005,
    ENGINE_OVERLOADED = 1006,
    
    // 推理错误
    INFERENCE_ERROR = 2001,
    MODEL_NOT_LOADED = 2002,
    KV_CACHE_EXHAUSTED = 2003,
    WORKER_UNAVAILABLE = 2004
};

/**
 * @class APIError
 * @brief API异常
 */
class APIError : public std::runtime_error {
 public:
    APIError(APIErrorCode code, const std::string& message)
        : std::runtime_error(message),
          code_(code),
          message_(message) {}
    
    APIErrorCode code() const { return code_; }
    
    /**
     * @brief 转换为HTTP状态码
     */
    int http_status() const {
        return static_cast<int>(code_);
    }
    
    /**
     * @brief 转换为JSON
     */
    nlohmann::json to_json() const {
        return {
            {"error", {
                {"code", static_cast<int>(code_)},
                {"message", message_},
                {"type", error_type()}
            }}
        };
    }
    
    /**
     * @brief 获取错误类型字符串
     */
    const char* error_type() const {
        switch (code_) {
            case APIErrorCode::UNAUTHORIZED:
            case APIErrorCode::INVALID_API_KEY:
                return "authentication_error";
            case APIErrorCode::RATE_LIMIT_EXCEEDED:
                return "rate_limit_error";
            case APIErrorCode::CONTEXT_LENGTH_EXCEEDED:
                return "invalid_request_error";
            case APIErrorCode::MODEL_NOT_FOUND:
            case APIErrorCode::MODEL_NOT_LOADED:
                return "not_found_error";
            case APIErrorCode::INFERENCE_ERROR:
            case APIErrorCode::WORKER_UNAVAILABLE:
                return "server_error";
            default:
                return "api_error";
        }
    }

 private:
    APIErrorCode code_;
    std::string message_;
};

/**
 * @brief 错误处理宏
 */
#define XLLM_API_TRY try

#define XLLM_API_CATCH(code) \
    catch (const APIError& e) { \
        throw; \
    } \
    catch (const std::exception& e) { \
        throw APIError(code, e.what()); \
    }

}  // namespace api
}  // namespace xllm
```

---

## 7. AI验收标准

### 7.1 API验证检查点

```yaml
ai_verification:
  api:
    - name: "OpenAI兼容性"
      check: |
        1. /v1/chat/completions 端点存在
        2. 请求/响应格式符合OpenAI规范
        3. 错误格式符合OpenAI规范
        
    - name: "参数验证"
      check: |
        1. 必需参数缺失返回400错误
        2. 无效参数值返回422错误
        3. 未知参数被忽略或警告
        
    - name: "错误处理"
      check: |
        1. 所有错误返回JSON格式
        2. HTTP状态码正确
        3. 错误信息包含足够诊断信息
```

### 7.2 SDK验证

```cpp
// Python SDK测试
TEST(PythonSDK, ChatCompletion) {
    auto client = xLLMClient("http://localhost:8080");
    
    auto response = client.chat_completion(
        model="qwen2-7b",
        messages={{"role": "user", "content": "Hello"}}
    );
    
    EXPECT_EQ(response.model, "qwen2-7b");
    EXPECT_FALSE(response.choices.empty());
}

// C SDK测试
TEST(CSdk, Completion) {
    auto* client = xllm_client_create("http://localhost:8080", NULL, &error);
    ASSERT_NE(client, nullptr);
    
    auto* completion = xllm_chat_completion(client, &params, &error);
    ASSERT_NE(completion, nullptr);
    
    const char* content = xllm_completion_get_content(completion);
    EXPECT_STRNE(content, "");
}
```

---

## 相关文档

| 文档 | 说明 |
|-----|------|
| [协议定义](./09_PROTO_DEFINITIONS.md) | Protocol Buffers完整定义 |
| [配置Schema](./09_CONFIG_SCHEMA.md) | 配置参数定义 |
| [Python SDK文档](../07_SDK/python_sdk.md) | Python SDK详细文档 |
| [C SDK文档](../07_SDK/c_sdk.md) | C SDK详细文档 |
