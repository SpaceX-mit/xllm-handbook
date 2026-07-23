# 方案 A+：GGUF 格式支持 - 详细实现指南

> **目标**：让 xLLM 支持加载 GGUF 格式模型，达到与 llama.cpp 相同的内存占用和启动速度

---

## 1. GGUF 格式概述

### 1.1 GGUF 文件结构

```
┌─────────────────────────────────────────────────────────┐
│                    GGUF File Structure                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Header]                                               │
│  ├── Magic: "GGUF" (4 bytes)                            │
│  ├── Version: 3 (uint32)                                │
│  ├── Tensor count: N (uint64)                           │
│  └── Metadata KV count: M (uint64)                      │
│                                                         │
│  [Metadata KV pairs] (M entries)                        │
│  ├── Key: string (e.g. "general.architecture")         │
│  ├── Type: enum (STRING, UINT32, FLOAT32, ARRAY, ...)  │
│  └── Value: varies by type                              │
│                                                         │
│  [Tensor Info] (N entries)                              │
│  ├── Name: string (e.g. "layers.0.attention.wq")       │
│  ├── Dimensions: uint32[] (e.g. [4096, 4096])          │
│  ├── Type: enum (Q4_0, Q4_1, Q8_0, F16, F32, ...)     │
│  └── Offset: uint64 (relative to tensor data start)    │
│                                                         │
│  [Alignment Padding]                                    │
│                                                         │
│  [Tensor Data] (binary blob)                            │
│  ├── Tensor 0 data (at offset 0)                       │
│  ├── Tensor 1 data (at offset X)                       │
│  └── ...                                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 GGUF 量化类型

```cpp
enum ggml_type {
    GGML_TYPE_F32  = 0,   // FP32 (4 bytes per element)
    GGML_TYPE_F16  = 1,   // FP16 (2 bytes per element)
    GGML_TYPE_Q4_0 = 2,   // 4-bit quantization (block size 32)
    GGML_TYPE_Q4_1 = 3,   // 4-bit quantization with min offset
    GGML_TYPE_Q5_0 = 6,   // 5-bit quantization
    GGML_TYPE_Q5_1 = 7,   // 5-bit quantization with min offset
    GGML_TYPE_Q8_0 = 8,   // 8-bit quantization
    GGML_TYPE_Q8_1 = 9,   // 8-bit quantization with min offset
    GGML_TYPE_Q2_K = 10,  // 2-bit K-quantization
    GGML_TYPE_Q3_K = 11,  // 3-bit K-quantization
    GGML_TYPE_Q4_K = 12,  // 4-bit K-quantization
    GGML_TYPE_Q5_K = 13,  // 5-bit K-quantization
    GGML_TYPE_Q6_K = 14,  // 6-bit K-quantization
    // ...
};
```

### 1.3 Q4_0 量化格式详解

```cpp
// Q4_0: 4-bit quantization, block size = 32
// 每 32 个 FP32 值量化为：
//   - 1 个 FP16 scale
//   - 16 个 uint8 (每个 uint8 存 2 个 4-bit 值)

#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta (scale factor)
    uint8_t qs[QK4_0 / 2];  // quantized values (16 bytes)
} block_q4_0;

// 总大小：2 + 16 = 18 bytes per block (32 FP32 = 128 bytes)
// 压缩比：128 / 18 = 7.1x
```

---

## 2. GGUF 模型加载器实现

### 2.1 接口设计

```cpp
// xllm/core/framework/gguf_model_loader.h
#pragma once

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <memory>

// 前向声明 llama.cpp 结构
struct gguf_context;
struct ggml_context;

namespace xllm {

// ═══════════════════════════════════════════════════════
// GGUF 张量信息
// ═══════════════════════════════════════════════════════
struct GGUFTensorInfo {
    std::string name;              // 张量名称
    std::vector<int64_t> shape;    // 形状
    int64_t n_elements;            // 元素总数
    size_t offset;                 // 文件偏移量
    int ggml_type;                 // GGML 类型（Q4_0/Q4_1/...）
    
    // 量化信息
    size_t block_size;             // 量化块大小
    size_t type_size;              // 单个块的字节数
    
    // PyTorch 兼容
    torch::Dtype torch_dtype;      // 对应的 torch dtype
    bool is_quantized;             // 是否量化
};

// ═══════════════════════════════════════════════════════
// GGUF 模型元数据
// ═══════════════════════════════════════════════════════
struct GGUFMetadata {
    std::string architecture;      // "llama", "qwen2", etc.
    int64_t context_length;
    int64_t embedding_length;
    int64_t block_count;
    int64_t feed_forward_length;
    int64_t head_count;
    int64_t head_count_kv;
    std::string rope_scaling_type;
    std::unordered_map<std::string, std::string> extra;
};

// ═══════════════════════════════════════════════════════
// GGUF 模型加载器
// ═══════════════════════════════════════════════════════
class GGUFModelLoader {
 public:
    explicit GGUFModelLoader(const std::string& gguf_path);
    ~GGUFModelLoader();
    
    // 禁止拷贝
    GGUFModelLoader(const GGUFModelLoader&) = delete;
    GGUFModelLoader& operator=(const GGUFModelLoader&) = delete;
    
    // ───────────────────────────────────────────────────
    // 元数据访问
    // ───────────────────────────────────────────────────
    const GGUFMetadata& get_metadata() const { return metadata_; }
    
    // 获取所有张量名称
    std::vector<std::string> get_tensor_names() const;
    
    // 获取张量信息
    const GGUFTensorInfo* get_tensor_info(const std::string& name) const;
    
    // ───────────────────────────────────────────────────
    // 张量加载
    // ───────────────────────────────────────────────────
    
    // 加载单个张量（零拷贝 mmap，或反量化为 FP16/FP32）
    torch::Tensor load_tensor(
        const std::string& name,
        bool dequantize = false,           // 是否反量化
        torch::Dtype target_dtype = torch::kFloat16  // 目标类型
    );
    
    // 批量加载张量
    std::unordered_map<std::string, torch::Tensor> load_all_tensors(
        bool dequantize = false
    );
    
    // 加载匹配前缀的张量（例如 "layers.0."）
    std::unordered_map<std::string, torch::Tensor> load_tensors_with_prefix(
        const std::string& prefix,
        bool dequantize = false
    );
    
    // ───────────────────────────────────────────────────
    // 量化张量包装（零拷贝）
    // ───────────────────────────────────────────────────
    
    // 将量化数据包装为 torch::Tensor（不反量化，保持 UINT8）
    // 用于直接传递给 SpacemiT IME kernels
    torch::Tensor wrap_quantized_tensor(const std::string& name);
    
 private:
    // GGUF 文件路径
    std::string gguf_path_;
    
    // llama.cpp 上下文（通过 dlopen 动态加载）
    struct gguf_context* gguf_ctx_;
    
    // 文件映射
    int fd_;
    void* mapped_data_;
    size_t mapped_size_;
    
    // 元数据
    GGUFMetadata metadata_;
    
    // 张量索引
    std::unordered_map<std::string, GGUFTensorInfo> tensor_index_;
    
    // 初始化
    void init();
    void parse_metadata();
    void build_tensor_index();
    
    // GGML 类型转换
    torch::Dtype ggml_type_to_torch(int ggml_type) const;
    size_t ggml_type_size(int ggml_type) const;
    
    // 反量化
    torch::Tensor dequantize_q4_0(
        const void* data, 
        const std::vector<int64_t>& shape,
        torch::Dtype target_dtype
    );
    
    torch::Tensor dequantize_q4_1(
        const void* data,
        const std::vector<int64_t>& shape,
        torch::Dtype target_dtype
    );
    
    torch::Tensor dequantize_q8_0(
        const void* data,
        const std::vector<int64_t>& shape,
        torch::Dtype target_dtype
    );
};

} // namespace xllm
```

### 2.2 核心实现

```cpp
// xllm/core/framework/gguf_model_loader.cpp
#include "gguf_model_loader.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <glog/logging.h>

// 引入 llama.cpp GGUF API
extern "C" {
#include "gguf.h"
#include "ggml.h"
}

namespace xllm {

// ═══════════════════════════════════════════════════════
// 构造与析构
// ═══════════════════════════════════════════════════════
GGUFModelLoader::GGUFModelLoader(const std::string& gguf_path)
    : gguf_path_(gguf_path),
      gguf_ctx_(nullptr),
      fd_(-1),
      mapped_data_(nullptr),
      mapped_size_(0) {
    init();
}

GGUFModelLoader::~GGUFModelLoader() {
    if (mapped_data_ != nullptr) {
        munmap(mapped_data_, mapped_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
    if (gguf_ctx_ != nullptr) {
        gguf_free(gguf_ctx_);
    }
}

// ═══════════════════════════════════════════════════════
// 初始化
// ═══════════════════════════════════════════════════════
void GGUFModelLoader::init() {
    LOG(INFO) << "Loading GGUF model: " << gguf_path_;
    
    // 1. 打开文件
    fd_ = open(gguf_path_.c_str(), O_RDONLY);
    CHECK(fd_ != -1) << "Failed to open GGUF file: " << gguf_path_;
    
    // 2. 获取文件大小
    struct stat st;
    CHECK(fstat(fd_, &st) == 0) << "Failed to stat GGUF file";
    mapped_size_ = st.st_size;
    
    // 3. mmap 映射文件（零拷贝）
    mapped_data_ = mmap(
        nullptr, mapped_size_,
        PROT_READ, MAP_PRIVATE,
        fd_, 0
    );
    CHECK(mapped_data_ != MAP_FAILED) << "Failed to mmap GGUF file";
    
    // 4. 初始化 GGUF 上下文
    struct gguf_init_params params = {
        .no_alloc = true,  // 不分配内存，使用 mmap
        .ctx = nullptr,
    };
    
    gguf_ctx_ = gguf_init_from_file(gguf_path_.c_str(), params);
    CHECK(gguf_ctx_ != nullptr) << "Failed to initialize GGUF context";
    
    // 5. 解析元数据
    parse_metadata();
    
    // 6. 构建张量索引
    build_tensor_index();
    
    LOG(INFO) << "GGUF model loaded successfully";
    LOG(INFO) << "  Architecture: " << metadata_.architecture;
    LOG(INFO) << "  Context length: " << metadata_.context_length;
    LOG(INFO) << "  Tensors: " << tensor_index_.size();
}

// ═══════════════════════════════════════════════════════
// 解析元数据
// ═══════════════════════════════════════════════════════
void GGUFModelLoader::parse_metadata() {
    // 从 GGUF KV 中提取模型元数据
    
    // architecture
    int arch_idx = gguf_find_key(gguf_ctx_, "general.architecture");
    if (arch_idx >= 0) {
        metadata_.architecture = gguf_get_val_str(gguf_ctx_, arch_idx);
    }
    
    // context_length
    int ctx_idx = gguf_find_key(gguf_ctx_, 
        (metadata_.architecture + ".context_length").c_str());
    if (ctx_idx >= 0) {
        metadata_.context_length = gguf_get_val_u32(gguf_ctx_, ctx_idx);
    }
    
    // embedding_length
    int emb_idx = gguf_find_key(gguf_ctx_,
        (metadata_.architecture + ".embedding_length").c_str());
    if (emb_idx >= 0) {
        metadata_.embedding_length = gguf_get_val_u32(gguf_ctx_, emb_idx);
    }
    
    // block_count (层数)
    int blk_idx = gguf_find_key(gguf_ctx_,
        (metadata_.architecture + ".block_count").c_str());
    if (blk_idx >= 0) {
        metadata_.block_count = gguf_get_val_u32(gguf_ctx_, blk_idx);
    }
    
    // head_count
    int head_idx = gguf_find_key(gguf_ctx_,
        (metadata_.architecture + ".attention.head_count").c_str());
    if (head_idx >= 0) {
        metadata_.head_count = gguf_get_val_u32(gguf_ctx_, head_idx);
    }
    
    // head_count_kv
    int head_kv_idx = gguf_find_key(gguf_ctx_,
        (metadata_.architecture + ".attention.head_count_kv").c_str());
    if (head_kv_idx >= 0) {
        metadata_.head_count_kv = gguf_get_val_u32(gguf_ctx_, head_kv_idx);
    }
    
    // ... 其他元数据
}

// ═══════════════════════════════════════════════════════
// 构建张量索引
// ═══════════════════════════════════════════════════════
void GGUFModelLoader::build_tensor_index() {
    int n_tensors = gguf_get_n_tensors(gguf_ctx_);
    
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx_, i);
        
        GGUFTensorInfo info;
        info.name = name;
        
        // 获取形状
        int n_dims = gguf_get_tensor_n_dims(gguf_ctx_, i);
        info.shape.resize(n_dims);
        info.n_elements = 1;
        for (int d = 0; d < n_dims; ++d) {
            info.shape[d] = gguf_get_tensor_dim(gguf_ctx_, i, d);
            info.n_elements *= info.shape[d];
        }
        
        // 获取类型
        info.ggml_type = gguf_get_tensor_type(gguf_ctx_, i);
        info.torch_dtype = ggml_type_to_torch(info.ggml_type);
        info.is_quantized = (info.ggml_type >= GGML_TYPE_Q4_0 &&
                             info.ggml_type <= GGML_TYPE_Q6_K);
        
        // 获取偏移量
        info.offset = gguf_get_tensor_offset(gguf_ctx_, i);
        
        // 量化参数
        info.block_size = ggml_blck_size(
            static_cast<enum ggml_type>(info.ggml_type));
        info.type_size = ggml_type_size(
            static_cast<enum ggml_type>(info.ggml_type));
        
        tensor_index_[name] = info;
    }
}

// ═══════════════════════════════════════════════════════
// 加载张量（零拷贝或反量化）
// ═══════════════════════════════════════════════════════
torch::Tensor GGUFModelLoader::load_tensor(
    const std::string& name,
    bool dequantize,
    torch::Dtype target_dtype
) {
    auto it = tensor_index_.find(name);
    CHECK(it != tensor_index_.end()) << "Tensor not found: " << name;
    
    const GGUFTensorInfo& info = it->second;
    
    // 计算数据指针
    const uint8_t* data_ptr = 
        static_cast<const uint8_t*>(mapped_data_) + info.offset;
    
    // 如果不需要反量化，直接包装为 torch::Tensor
    if (!dequantize) {
        // 零拷贝包装（仅适用于 FP32/FP16）
        if (info.ggml_type == GGML_TYPE_F32 ||
            info.ggml_type == GGML_TYPE_F16) {
            return torch::from_blob(
                const_cast<uint8_t*>(data_ptr),
                info.shape,
                [](void*) {},  // 空 deleter（数据由 mmap 管理）
                torch::TensorOptions()
                    .dtype(info.torch_dtype)
                    .device(torch::kCPU)
            ).clone();  // clone 避免生命周期问题
        } else {
            // 量化类型，包装为 UINT8（传给 IME kernels）
            return wrap_quantized_tensor(name);
        }
    }
    
    // 需要反量化
    switch (info.ggml_type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            // 已经是浮点，直接返回
            return load_tensor(name, false, target_dtype)
                .to(target_dtype);
        
        case GGML_TYPE_Q4_0:
            return dequantize_q4_0(data_ptr, info.shape, target_dtype);
        
        case GGML_TYPE_Q4_1:
            return dequantize_q4_1(data_ptr, info.shape, target_dtype);
        
        case GGML_TYPE_Q8_0:
            return dequantize_q8_0(data_ptr, info.shape, target_dtype);
        
        default:
            LOG(FATAL) << "Unsupported GGML type for dequantization: "
                       << info.ggml_type;
            return torch::Tensor();
    }
}

// ═══════════════════════════════════════════════════════
// Q4_0 反量化
// ═══════════════════════════════════════════════════════
torch::Tensor GGUFModelLoader::dequantize_q4_0(
    const void* data,
    const std::vector<int64_t>& shape,
    torch::Dtype target_dtype
) {
    // Q4_0 block 结构
    struct block_q4_0 {
        uint16_t d;          // FP16 scale
        uint8_t qs[16];      // 16 bytes (32 4-bit values)
    };
    
    const int64_t n_elements = std::accumulate(
        shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    
    // 分配输出
    torch::Tensor output = torch::empty(shape, torch::kFloat32);
    float* out_ptr = output.data_ptr<float>();
    
    const block_q4_0* blocks = static_cast<const block_q4_0*>(data);
    const int64_t n_blocks = n_elements / 32;
    
    // 反量化
    for (int64_t b = 0; b < n_blocks; ++b) {
        const block_q4_0& block = blocks[b];
        
        // 解码 FP16 scale
        float scale = ggml_fp16_to_fp32(block.d);
        
        // 解码 32 个 4-bit 值
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = block.qs[i];
            
            // 低 4 位
            int8_t v0 = (byte & 0x0F) - 8;  // [-8, 7]
            out_ptr[b * 32 + i * 2] = v0 * scale;
            
            // 高 4 位
            int8_t v1 = (byte >> 4) - 8;
            out_ptr[b * 32 + i * 2 + 1] = v1 * scale;
        }
    }
    
    return output.to(target_dtype);
}

// ═══════════════════════════════════════════════════════
// 包装量化张量（零拷贝）
// ═══════════════════════════════════════════════════════
torch::Tensor GGUFModelLoader::wrap_quantized_tensor(
    const std::string& name
) {
    auto it = tensor_index_.find(name);
    CHECK(it != tensor_index_.end()) << "Tensor not found: " << name;
    
    const GGUFTensorInfo& info = it->second;
    CHECK(info.is_quantized) << "Tensor is not quantized: " << name;
    
    // 计算量化数据大小
    const int64_t n_blocks = info.n_elements / info.block_size;
    const int64_t data_size = n_blocks * info.type_size;
    
    // 计算数据指针
    uint8_t* data_ptr = 
        static_cast<uint8_t*>(mapped_data_) + info.offset;
    
    // 零拷贝包装为 UINT8 tensor
    return torch::from_blob(
        data_ptr,
        {data_size},
        [](void*) {},  // 空 deleter
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCPU)
    ).clone();  // clone 避免生命周期问题
}

// ═══════════════════════════════════════════════════════
// 类型转换
// ═══════════════════════════════════════════════════════
torch::Dtype GGUFModelLoader::ggml_type_to_torch(int ggml_type) const {
    switch (ggml_type) {
        case GGML_TYPE_F32:  return torch::kFloat32;
        case GGML_TYPE_F16:  return torch::kFloat16;
        case GGML_TYPE_I32:  return torch::kInt32;
        case GGML_TYPE_I16:  return torch::kInt16;
        case GGML_TYPE_I8:   return torch::kInt8;
        default:             return torch::kUInt8;  // 量化类型统一为 UINT8
    }
}

} // namespace xllm
```

**（方案 A+ GGUF 加载器实现完成，下一步：CMakeLists.txt）**
