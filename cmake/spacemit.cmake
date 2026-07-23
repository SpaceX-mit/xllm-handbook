# SpacemiT Platform Configuration for xLLM
# Enables SpacemiT K3 (RISC-V) support with ggml-spacemit acceleration

if(NOT USE_SPACEMIT)
  return()
endif()

message(STATUS "========================================")
message(STATUS "  Configuring SpacemiT Platform")
message(STATUS "========================================")

# Add compile definition
add_compile_definitions(USE_SPACEMIT)

# ============================================================================
# SpacemiT Toolchain Detection
# ============================================================================

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "riscv64")
  message(STATUS "Native compilation on RISC-V")
else()
  message(STATUS "Cross-compilation to RISC-V")
  # Note: Toolchain file should be set via CMAKE_TOOLCHAIN_FILE
endif()

# ============================================================================
# RISC-V ISA Configuration
# ============================================================================

set(RISCV_ARCH "rv64gcv")  # Base + Vector extension

# Add sub-extensions
set(RISCV_ARCH "${RISCV_ARCH}_zfh")   # FP16
set(RISCV_ARCH "${RISCV_ARCH}_zba")   # Address generation
set(RISCV_ARCH "${RISCV_ARCH}_zbb")   # Basic bit manipulation
set(RISCV_ARCH "${RISCV_ARCH}_zbc")   # Carry-less multiplication
set(RISCV_ARCH "${RISCV_ARCH}_zbs")   # Single-bit operations

message(STATUS "RISC-V ISA: ${RISCV_ARCH}")

# Add RISC-V specific compile flags
add_compile_options(
  -march=${RISCV_ARCH}
  -mabi=lp64d
)

# Vector extension flags
add_compile_definitions(
  GGML_USE_RVV
  __riscv_v
  __riscv_v_intrinsic
  __riscv_zfh
)

# ============================================================================
# IME Version Selection
# ============================================================================

option(SPACEMIT_USE_IME2 "Build for SpacemiT A100 (IME2 + TCM)" ON)
option(SPACEMIT_USE_IME1 "Build for SpacemiT X100 (IME1)" OFF)

if(SPACEMIT_USE_IME2)
  add_compile_definitions(RISCV64_SPACEMIT_IME2)
  message(STATUS "Target: SpacemiT A100 (IME2 + TCM)")
  set(SPACEMIT_TARGET "A100")
elseif(SPACEMIT_USE_IME1)
  add_compile_definitions(RISCV64_SPACEMIT_IME1)
  message(STATUS "Target: SpacemiT X100 (IME1)")
  set(SPACEMIT_TARGET "X100")
else()
  message(STATUS "Target: RISC-V with RVV fallback")
  set(SPACEMIT_TARGET "RVV")
endif()

# ============================================================================
# Build ggml-spacemit library
# ============================================================================

set(GGML_SPACEMIT_DIR ${CMAKE_SOURCE_DIR}/third_party/ggml-spacemit)

if(NOT EXISTS ${GGML_SPACEMIT_DIR})
  message(FATAL_ERROR
    "ggml-spacemit not found at: ${GGML_SPACEMIT_DIR}\n"
    "Please run: cp -r /path/to/llama.cpp/ggml/src/ggml-cpu/spacemit third_party/ggml-spacemit/"
  )
endif()

message(STATUS "ggml-spacemit directory: ${GGML_SPACEMIT_DIR}")

# Collect source files
set(GGML_SPACEMIT_SOURCES
  ${GGML_SPACEMIT_DIR}/ime_env.cpp
  ${GGML_SPACEMIT_DIR}/spine_mem_pool.cpp
  ${GGML_SPACEMIT_DIR}/repack.cpp
  ${GGML_SPACEMIT_DIR}/rvv_kernels.cpp
)

# Add IME-specific kernels
if(SPACEMIT_USE_IME1)
  list(APPEND GGML_SPACEMIT_SOURCES ${GGML_SPACEMIT_DIR}/ime1_kernels.cpp)
endif()

if(SPACEMIT_USE_IME2)
  list(APPEND GGML_SPACEMIT_SOURCES ${GGML_SPACEMIT_DIR}/ime2_kernels.cpp)
endif()

# Create ggml-spacemit library
add_library(ggml_spacemit STATIC ${GGML_SPACEMIT_SOURCES})

target_include_directories(ggml_spacemit PUBLIC
  ${GGML_SPACEMIT_DIR}
)

target_compile_options(ggml_spacemit PRIVATE
  -Wno-unused-parameter
  -Wno-cast-qual
  -Wno-sign-compare
)

# ============================================================================
# SpacemiT Platform Sources
# ============================================================================

set(SPACEMIT_PLATFORM_SOURCES
  ${CMAKE_SOURCE_DIR}/xllm/core/platform/spacemit/ggml_bridge.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/platform/spacemit/ggml_backend.cpp
)

set(SPACEMIT_KERNEL_SOURCES
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/spacemit/matmul_ggml.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/spacemit/rms_norm_ggml.cpp
)

set(SPACEMIT_SOURCES
  ${SPACEMIT_PLATFORM_SOURCES}
  ${SPACEMIT_KERNEL_SOURCES}
)

# ============================================================================
# Export Variables
# ============================================================================

set(SPACEMIT_INCLUDE_DIRS
  ${GGML_SPACEMIT_DIR}
  CACHE INTERNAL "SpacemiT include directories"
)

set(SPACEMIT_LIBRARIES
  ggml_spacemit
  CACHE INTERNAL "SpacemiT libraries"
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "========================================")
message(STATUS "SpacemiT Configuration Summary:")
message(STATUS "  Target: ${SPACEMIT_TARGET}")
message(STATUS "  ISA: ${RISCV_ARCH}")
message(STATUS "  Sources: ${SPACEMIT_SOURCES}")
message(STATUS "========================================")
