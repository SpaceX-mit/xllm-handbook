#!/usr/bin/env python3
"""Quick deploy and build SpacemiT on K3"""

import paramiko
import os
import sys

def connect_k3():
    """Connect to K3 worker"""
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
    """Execute command on K3"""
    if workdir:
        cmd = f"cd {workdir} && {cmd}"
    stdin, stdout, stderr = client.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    return exit_code, output, error

def upload_file(client, local, remote):
    """Upload file via SFTP"""
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
    print("Deploying SpacemiT to K3 worker...")

    client = connect_k3()
    print("✓ Connected to K3")

    base = "/home/bianbu/xllm-spacemit-test"
    local_base = "/data/workspace2026-new/work0618/tech-analyais0713/xllm-handbook"

    # Clean and create directory
    print("\nPreparing remote directory...")
    exec_cmd(client, f"rm -rf {base} && mkdir -p {base}")

    # Upload all necessary files
    print("\nUploading files...")

    files = [
        # Platform
        "xllm/core/platform/spacemit/ggml_bridge.h",
        "xllm/core/platform/spacemit/ggml_bridge.cpp",
        "xllm/core/platform/spacemit/ggml_backend.h",
        "xllm/core/platform/spacemit/ggml_backend.cpp",
        # Kernels
        "xllm/core/kernels/spacemit/matmul_ggml.cpp",
        "xllm/core/kernels/spacemit/rms_norm_ggml.cpp",
        # Tests
        "test/platform/spacemit/test_ggml_bridge.cpp",
        "test/kernels/spacemit/test_spacemit_ops.cpp",
    ]

    for f in files:
        local = os.path.join(local_base, f)
        remote = os.path.join(base, f)
        if os.path.exists(local):
            upload_file(client, local, remote)
            print(f"  ✓ {f}")

    # Upload ggml-spacemit files
    print("\nUploading ggml-spacemit...")
    ggml_files = [
        "ggml.c", "ggml.h",
        "ggml-alloc.c", "ggml-alloc.h",
        "ggml-backend.cpp", "ggml-backend.h",
        "ime_env.cpp", "ime_env.h",
        "ime.cpp", "ime.h",
        "ime_kernels.h",
        "ime1_kernels.cpp",
        "ime2_kernels.cpp",
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
            print(f"  ✓ {f}")

    # Upload CMakeLists.txt
    print("\nUploading CMakeLists.txt...")
    cmake_local = os.path.join(local_base, "CMakeLists_standalone_fixed.txt")
    cmake_remote = os.path.join(base, "CMakeLists.txt")
    upload_file(client, cmake_local, cmake_remote)

    # Build
    print("\n=== Building on K3 ===")
    print("Configuring...")
    code, out, err = exec_cmd(client,
        "cmake -B build -DUSE_SPACEMIT=ON -DSPACEMIT_USE_IME2=ON -DCMAKE_BUILD_TYPE=Release",
        base)

    if code != 0:
        print(f"✗ Configure failed:\n{err}")
        return 1

    print("Building...")
    code, out, err = exec_cmd(client, "cmake --build build -j8", base)

    if code != 0:
        print(f"✗ Build failed:\n{err}")
        return 1

    print("✓ Build successful")

    # Run tests
    print("\n=== Running Tests ===")

    print("\n1. GGML Bridge Test...")
    code, out, err = exec_cmd(client, "./build/test_ggml_bridge", base)
    if code == 0:
        print("✓ GGML Bridge test PASSED")
        print(out)
    else:
        print("✗ GGML Bridge test FAILED")
        print(err)

    print("\n2. SpacemiT Ops Test...")
    code, out, err = exec_cmd(client, "./build/test_spacemit_ops", base)
    if code == 0:
        print("✓ SpacemiT Ops test PASSED")
        print(out)
    else:
        print("✗ SpacemiT Ops test FAILED")
        print(err)

    client.close()

    print("\n" + "="*60)
    print("✓ SpacemiT Deployment Complete!")
    print("="*60)
    print(f"\nLocation on K3: {base}")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
