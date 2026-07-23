#!/usr/bin/env python3
"""Deploy and build on K3 with system header isolation"""

import paramiko
import os
import sys

def connect_k3():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname="10.0.90.243",
        username="bianbu",
        password="bianbu",
        allow_agent=False,
        look_for_keys=False
    )
    return client

def exec_cmd(client, cmd, workdir=None):
    if workdir:
        cmd = f"cd {workdir} && {cmd}"
    stdin, stdout, stderr = client.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    return exit_code, output, error

def upload_file(client, local, remote):
    sftp = client.open_sftp()
    remote_dir = os.path.dirname(remote)
    exec_cmd(client, f"mkdir -p {remote_dir}")
    try:
        sftp.put(local, remote)
        sftp.close()
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        sftp.close()
        return False

def main():
    print("Building SpacemiT with system header isolation...")

    client = connect_k3()
    print("✓ Connected to K3")

    base = "/home/bianbu/xllm-spacemit-final"
    local_base = "/data/workspace2026-new/work0618/tech-analyais0713/xllm-handbook"

    # Prepare
    print("\nPreparing...")
    exec_cmd(client, f"rm -rf {base} && mkdir -p {base}")

    # Upload implementation files
    print("\nUploading implementation...")
    impl_files = [
        "xllm/core/platform/spacemit/ggml_bridge.h",
        "xllm/core/platform/spacemit/ggml_bridge.cpp",
        "xllm/core/platform/spacemit/ggml_backend.h",
        "xllm/core/platform/spacemit/ggml_backend.cpp",
        "xllm/core/kernels/spacemit/matmul_ggml.cpp",
        "xllm/core/kernels/spacemit/rms_norm_ggml.cpp",
        "test/platform/spacemit/test_ggml_bridge.cpp",
        "test/kernels/spacemit/test_spacemit_ops.cpp",
    ]

    for f in impl_files:
        local = os.path.join(local_base, f)
        remote = os.path.join(base, f)
        if os.path.exists(local):
            upload_file(client, local, remote)

    # Upload ggml files
    print("Uploading ggml...")
    ggml_files = [
        "ggml.c", "ggml.h", "ggml-alloc.c", "ggml-alloc.h",
        "ggml-backend.cpp", "ggml-backend.h",
        "ggml-quants.c", "ggml-quants.h",
        "ggml-impl.h", "ggml-backend-impl.h", "ggml-common.h",
        "ggml-threading.h", "ggml-cpu-impl.h", "ggml-cpu.h", "gguf.h",
        "binary-ops.h", "common.h", "traits.h",
        "ime_env.cpp", "ime_env.h", "ime.cpp", "ime.h", "ime_kernels.h",
        "ime1_kernels.cpp", "ime2_kernels.cpp",
        "rvv_kernels.cpp", "rvv_kernels.h",
        "spine_mem_pool.cpp", "spine_mem_pool.h",
        "spine_tcm.h", "spine_barrier.h",
        "repack.cpp", "repack.h",
    ]

    for f in ggml_files:
        local = os.path.join(local_base, "third_party/ggml-spacemit", f)
        remote = os.path.join(base, "third_party/ggml-spacemit", f)
        if os.path.exists(local):
            upload_file(client, local, remote)

    # Create CMakeLists with explicit -I flag to override system headers
    print("\nCreating CMakeLists...")
    cmake = f'''cmake_minimum_required(VERSION 3.18)
project(xllm_spacemit_final CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# Force local headers BEFORE any system paths
set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -march=rv64gcv -mabi=lp64d")
set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -march=rv64gcv -mabi=lp64d")

add_library(ggml-spacemit STATIC
    third_party/ggml-spacemit/ggml.c
    third_party/ggml-spacemit/ggml-alloc.c
    third_party/ggml-spacemit/ggml-backend.cpp
    third_party/ggml-spacemit/ggml-quants.c
    third_party/ggml-spacemit/ime_env.cpp
    third_party/ggml-spacemit/spine_mem_pool.cpp
    third_party/ggml-spacemit/repack.cpp
    third_party/ggml-spacemit/ime.cpp
)

if(ON)
    target_sources(ggml-spacemit PRIVATE third_party/ggml-spacemit/ime2_kernels.cpp)
    target_compile_definitions(ggml-spacemit PUBLIC GGML_USE_SPACEMIT_IME2)
endif()

if(ON)
    target_sources(ggml-spacemit PRIVATE third_party/ggml-spacemit/rvv_kernels.cpp)
    target_compile_definitions(ggml-spacemit PUBLIC GGML_USE_RVV)
endif()

target_compile_definitions(ggml-spacemit PUBLIC GGML_USE_SPACEMIT GGML_SPACEMIT_RISCV64)

# Critical: Use -I flag to force local headers, suppress system warnings
target_compile_options(ggml-spacemit PRIVATE
    -I${{CMAKE_SOURCE_DIR}}/third_party/ggml-spacemit
    -Wno-error
)
target_include_directories(ggml-spacemit PRIVATE ${{CMAKE_SOURCE_DIR}}/third_party/ggml-spacemit)
target_link_libraries(ggml-spacemit m pthread)

add_library(spacemit_bridge STATIC
    xllm/core/platform/spacemit/ggml_bridge.cpp
    xllm/core/platform/spacemit/ggml_backend.cpp
)
target_include_directories(spacemit_bridge PRIVATE ${{CMAKE_SOURCE_DIR}}/third_party/ggml-spacemit)
target_link_libraries(spacemit_bridge ggml-spacemit)

add_library(spacemit_kernels STATIC
    xllm/core/kernels/spacemit/matmul_ggml.cpp
    xllm/core/kernels/spacemit/rms_norm_ggml.cpp
)
target_include_directories(spacemit_kernels PRIVATE ${{CMAKE_SOURCE_DIR}}/third_party/ggml-spacemit)
target_link_libraries(spacemit_kernels spacemit_bridge)

add_executable(test_ggml_bridge test/platform/spacemit/test_ggml_bridge.cpp)
target_include_directories(test_ggml_bridge PRIVATE ${{CMAKE_SOURCE_DIR}}/third_party/ggml-spacemit)
target_link_libraries(test_ggml_bridge spacemit_bridge pthread)

add_executable(test_spacemit_ops test/kernels/spacemit/test_spacemit_ops.cpp)
target_include_directories(test_spacemit_ops PRIVATE ${{CMAKE_SOURCE_DIR}}/third_party/ggml-spacemit)
target_link_libraries(test_spacemit_ops spacemit_kernels pthread)

message(STATUS "SpacemiT Build Configuration:")
message(STATUS "  IME2: ON")
message(STATUS "  RVV: ON")
'''

    sftp = client.open_sftp()
    with sftp.file(f"{base}/CMakeLists.txt", 'w') as f:
        f.write(cmake)
    sftp.close()

    # Build with explicit include path
    print("\n=== Building ===")
    print("Configuring...")
    code, out, err = exec_cmd(client, "cmake -B build -DCMAKE_BUILD_TYPE=Release", base)

    if code != 0:
        print(f"Configure failed:\n{err[:500]}")
        return 1

    print("Compiling...")
    code, out, err = exec_cmd(client, "cmake --build build -j8", base)

    if code != 0:
        print(f"Build failed:\n{err[:1000]}")
        return 1

    print("✓ Build successful!")

    # Run tests
    print("\n=== Running Tests ===")

    code, out, err = exec_cmd(client, "./build/test_ggml_bridge", base)
    print("Test 1: GGML Bridge")
    print(out if code == 0 else err)
    print("✓ PASSED" if code == 0 else "✗ FAILED")

    code, out, err = exec_cmd(client, "./build/test_spacemit_ops", base)
    print("\nTest 2: SpacemiT Ops")
    print(out if code == 0 else err)
    print("✓ PASSED" if code == 0 else "✗ FAILED")

    client.close()

    print("\n" + "="*60)
    print("✓ SpacemiT Full Build Complete!")
    print("="*60)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
