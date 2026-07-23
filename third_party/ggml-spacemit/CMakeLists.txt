# ggml-spacemit: ggml core + SpacemiT IME2/IME1/RVV CPU kernels.
#
# Build flags mirror the configuration verified on the SpacemiT K3 worker
# (see project docs): IME2 is selected on A100 clusters, IME1 on X100, with
# RVV and scalar CPU fallbacks. Requires the SpacemiT custom toolchain to emit
# the IME/TCM accelerated instructions; a stock riscv64 toolchain builds the
# scalar/RVV path only.

set(GGML_SPACEMIT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Core ggml sources present in this vendored copy.
set(GGML_SPACEMIT_SRCS
  ${GGML_SPACEMIT_DIR}/ggml.c
  ${GGML_SPACEMIT_DIR}/ggml-alloc.c
  ${GGML_SPACEMIT_DIR}/ggml-backend.cpp
)

# Optional core sources (present depending on vendoring completeness).
foreach(_opt ggml-quants.c ime.cpp ime_env.cpp spine_mem_pool.cpp repack.cpp
              ime2_kernels.cpp ime1_kernels.cpp rvv_kernels.cpp)
  if(EXISTS ${GGML_SPACEMIT_DIR}/${_opt})
    list(APPEND GGML_SPACEMIT_SRCS ${GGML_SPACEMIT_DIR}/${_opt})
  endif()
endforeach()

add_library(ggml-spacemit STATIC ${GGML_SPACEMIT_SRCS})

target_include_directories(ggml-spacemit PUBLIC ${GGML_SPACEMIT_DIR})

target_compile_definitions(ggml-spacemit PUBLIC
  GGML_USE_SPACEMIT
  GGML_USE_CPU_RISCV64_SPACEMIT
  RISCV64_SPACEMIT_IME1
  RISCV64_SPACEMIT_IME2
)

# RISC-V vector + SpacemiT bit-manipulation / dot-product extensions.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
  set(_smt_march "-march=rv64gcv_zfh_zvfh_zba_zicbop")
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
    string(APPEND _smt_march "_xsmtvdotii")
  endif()
  target_compile_options(ggml-spacemit PRIVATE ${_smt_march})
endif()

target_link_libraries(ggml-spacemit PUBLIC m pthread)

set_target_properties(ggml-spacemit PROPERTIES POSITION_INDEPENDENT_CODE ON)

message(STATUS "ggml-spacemit: SpacemiT IME2/IME1/RVV target configured")
