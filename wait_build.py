#!/usr/bin/env python3
"""Wait for K3 build to complete and verify"""

import paramiko
import time
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

def exec_cmd(client, cmd):
    stdin, stdout, stderr = client.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    output = stdout.read().decode().strip()
    return exit_code, output

client = connect_k3()
print("Waiting for build to complete...")

# Wait for cmake to finish
max_wait = 600  # 10 minutes
waited = 0
while waited < max_wait:
    code, out = exec_cmd(client, "ps aux | grep 'cmake --build' | grep -v grep || echo 'DONE'")
    if "DONE" in out:
        print("\n✓ Build process completed")
        break

    # Show progress
    code, bin_out = exec_cmd(client, "ls /home/bianbu/llama.cpp-spacemit/build/bin/ 2>/dev/null | wc -l")
    print(f"  {waited}s - Binaries built: {bin_out.strip()}", end='\r')

    time.sleep(5)
    waited += 5
else:
    print("\n⚠ Build timeout after 10 minutes")

# Check build result
print("\n\n=== Build Results ===")
code, out = exec_cmd(client, "ls -lh /home/bianbu/llama.cpp-spacemit/build/bin/")
print(f"\nBinaries in build/bin/:\n{out}")

# Check for llama-cli
code, out = exec_cmd(client, "test -f /home/bianbu/llama.cpp-spacemit/build/bin/llama-cli && echo 'EXISTS' || echo 'NOT FOUND'")
if "EXISTS" in out:
    print("\n✓ llama-cli binary found")

    # Test execution
    code, out = exec_cmd(client, "cd /home/bianbu/llama.cpp-spacemit && ./build/bin/llama-cli --version 2>&1 | head -10")
    print(f"\nVersion info:\n{out}")
else:
    print("\n✗ llama-cli binary not found")

# Check for SpacemiT support
print("\n=== SpacemiT Support ===")
code, out = exec_cmd(client, "grep -r 'SpacemiT\\|SPACEMIT\\|spacemit' /home/bianbu/llama.cpp-spacemit/build/CMakeCache.txt 2>/dev/null | head -5")
if out:
    print(f"CMake configuration:\n{out}")
else:
    print("⚠ No SpacemiT flags found in CMake cache")

# Check SpacemiT source files
code, out = exec_cmd(client, "ls /home/bianbu/llama.cpp-spacemit/ggml/src/ggml-cpu/spacemit/*.cpp | head -5")
if out:
    print(f"\n✓ SpacemiT source files present:\n{out}")

client.close()

print("\n" + "="*60)
print("✓ Build verification complete")
print("="*60)
