#!/usr/bin/env python3
"""Deploy and build SpacemiT implementation on K3 - Standalone version"""

import paramiko
import os
import sys

class K3StandaloneBuilder:
    def __init__(self):
        self.host = "10.0.90.243"
        self.port = 22
        self.username = "bianbu"
        self.password = "bianbu"
        self.remote_base = "/home/bianbu/xllm-spacemit-test"
        self.local_base = "/data/workspace2026-new/work0618/tech-analyais0713/xllm-handbook"
        self.client = None
        self.sftp = None

    def connect(self):
        """Establish SSH connection"""
        print(f"Connecting to {self.username}@{self.host}...")
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            self.client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=10,
                allow_agent=False,
                look_for_keys=False
            )
            self.sftp = self.client.open_sftp()
            print("✓ Connected successfully")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def exec_command(self, command, workdir=None):
        """Execute remote command and return output"""
        if workdir:
            command = f"cd {workdir} && {command}"

        print(f"  $ {command}")
        stdin, stdout, stderr = self.client.exec_command(command)

        exit_code = stdout.channel.recv_exit_status()
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()

        if output:
            for line in output.split('\n'):
                print(f"    {line}")
        if error and exit_code != 0:
            for line in error.split('\n'):
                print(f"    ERROR: {line}")

        return exit_code, output, error

    def upload_file(self, local_path, remote_path):
        """Upload a file to K3"""
        try:
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote_path)
            self.exec_command(f"mkdir -p {remote_dir}")

            self.sftp.put(local_path, remote_path)
            return True
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
            return False

    def create_standalone_cmakelists(self):
        """Create a standalone CMakeLists.txt for testing"""
        content = """cmake_minimum_required(VERSION 3.18)
project(xllm_spacemit_test CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Options
option(USE_SPACEMIT "Enable SpacemiT support" ON)
option(SPACEMIT_USE_IME2 "Use IME2 for A100 clusters" ON)
option(SPACEMIT_USE_IME1 "Use IME1 for X100 clusters" OFF)
option(SPACEMIT_USE_RVV "Use RVV fallback" ON)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/xllm/core/platform/spacemit
    ${CMAKE_SOURCE_DIR}/third_party/ggml-spacemit
)

# ggml-spacemit library
add_library(ggml-spacemit STATIC
    third_party/ggml-spacemit/ggml.c
    third_party/ggml-spacemit/ggml-alloc.c
    third_party/ggml-spacemit/ggml-backend.cpp
    third_party/ggml-spacemit/ggml-quants.c
    third_party/ggml-spacemit/ime_env.cpp
    third_party/ggml-spacemit/spine_mem_pool.cpp
    third_party/ggml-spacemit/repack.cpp
)

# Add IME kernels based on options
if(SPACEMIT_USE_IME2)
    target_sources(ggml-spacemit PRIVATE
        third_party/ggml-spacemit/ime2_kernels.cpp
    )
    target_compile_definitions(ggml-spacemit PUBLIC GGML_USE_SPACEMIT_IME2)
endif()

if(SPACEMIT_USE_IME1)
    target_sources(ggml-spacemit PRIVATE
        third_party/ggml-spacemit/ime1_kernels.cpp
    )
    target_compile_definitions(ggml-spacemit PUBLIC GGML_USE_SPACEMIT_IME1)
endif()

if(SPACEMIT_USE_RVV)
    target_sources(ggml-spacemit PRIVATE
        third_party/ggml-spacemit/rvv_kernels.cpp
    )
    target_compile_definitions(ggml-spacemit PUBLIC GGML_USE_RVV)
endif()

target_compile_definitions(ggml-spacemit PUBLIC GGML_USE_SPACEMIT)

# SpacemiT bridge library
add_library(spacemit_bridge STATIC
    xllm/core/platform/spacemit/ggml_bridge.cpp
    xllm/core/platform/spacemit/ggml_backend.cpp
)

target_link_libraries(spacemit_bridge ggml-spacemit)

# SpacemiT kernels library
add_library(spacemit_kernels STATIC
    xllm/core/kernels/spacemit/matmul_ggml.cpp
    xllm/core/kernels/spacemit/rms_norm_ggml.cpp
)

target_link_libraries(spacemit_kernels spacemit_bridge)

# Test executables
add_executable(test_ggml_bridge
    test/platform/spacemit/test_ggml_bridge.cpp
)
target_link_libraries(test_ggml_bridge spacemit_bridge pthread)

add_executable(test_spacemit_ops
    test/kernels/spacemit/test_spacemit_ops.cpp
)
target_link_libraries(test_spacemit_ops spacemit_kernels pthread)

# Enable testing
enable_testing()
add_test(NAME ggml_bridge_test COMMAND test_ggml_bridge)
add_test(NAME spacemit_ops_test COMMAND test_spacemit_ops)

message(STATUS "SpacemiT configuration:")
message(STATUS "  USE_SPACEMIT: ${USE_SPACEMIT}")
message(STATUS "  SPACEMIT_USE_IME2: ${SPACEMIT_USE_IME2}")
message(STATUS "  SPACEMIT_USE_IME1: ${SPACEMIT_USE_IME1}")
message(STATUS "  SPACEMIT_USE_RVV: ${SPACEMIT_USE_RVV}")
"""

        local_path = os.path.join(self.local_base, "CMakeLists_standalone.txt")
        with open(local_path, 'w') as f:
            f.write(content)

        return local_path

    def deploy_all_files(self):
        """Deploy all necessary files"""
        print("\n=== Deploying Files to K3 ===\n")

        # Create clean remote directory
        print("Creating remote directory...")
        self.exec_command(f"rm -rf {self.remote_base}")
        self.exec_command(f"mkdir -p {self.remote_base}")

        files_to_deploy = [
            # Platform files
            "xllm/core/platform/spacemit/ggml_bridge.h",
            "xllm/core/platform/spacemit/ggml_bridge.cpp",
            "xllm/core/platform/spacemit/ggml_backend.h",
            "xllm/core/platform/spacemit/ggml_backend.cpp",

            # Kernel files
            "xllm/core/kernels/spacemit/matmul_ggml.cpp",
            "xllm/core/kernels/spacemit/rms_norm_ggml.cpp",

            # Test files
            "test/platform/spacemit/test_ggml_bridge.cpp",
            "test/kernels/spacemit/test_spacemit_ops.cpp",
        ]

        # Deploy main files
        print("\nDeploying SpacemiT implementation files...")
        for rel_path in files_to_deploy:
            local_path = os.path.join(self.local_base, rel_path)
            remote_path = os.path.join(self.remote_base, rel_path)

            if not os.path.exists(local_path):
                print(f"  ✗ Not found: {rel_path}")
                continue

            print(f"  Uploading: {rel_path}")
            if not self.upload_file(local_path, remote_path):
                return False

        # Deploy ggml-spacemit library
        print("\nDeploying ggml-spacemit library...")
        ggml_files = [
            "ggml.c", "ggml.h",
            "ggml-alloc.c", "ggml-alloc.h",
            "ggml-backend.cpp", "ggml-backend.h",
            "ggml-quants.c", "ggml-quants.h",
            "ggml-impl.h",
            "ime_env.cpp", "ime_env.h",
            "ime_kernels.h",
            "ime1_kernels.cpp", "ime1_kernels.h",
            "ime2_kernels.cpp", "ime2_kernels.h",
            "rvv_kernels.cpp", "rvv_kernels.h",
            "spine_mem_pool.cpp", "spine_mem_pool.h",
            "spine_tcm.h", "spine_barrier.h",
            "repack.cpp", "repack.h",
        ]

        for fname in ggml_files:
            local_path = os.path.join(self.local_base, "third_party/ggml-spacemit", fname)
            remote_path = os.path.join(self.remote_base, "third_party/ggml-spacemit", fname)

            if not os.path.exists(local_path):
                print(f"  ⚠ Not found: {fname}")
                continue

            print(f"  Uploading: {fname}")
            if not self.upload_file(local_path, remote_path):
                print(f"  ⚠ Failed to upload: {fname}")

        # Create and upload CMakeLists.txt
        print("\nCreating standalone CMakeLists.txt...")
        cmake_local = self.create_standalone_cmakelists()
        cmake_remote = os.path.join(self.remote_base, "CMakeLists.txt")

        print("  Uploading CMakeLists.txt...")
        if not self.upload_file(cmake_local, cmake_remote):
            return False

        print("\n✓ All files deployed")
        return True

    def build_project(self):
        """Build the project on K3"""
        print("\n=== Building Project on K3 ===\n")

        workdir = self.remote_base

        # Configure
        print("Configuring with CMake...")
        code, output, error = self.exec_command(
            "cmake -B build -DUSE_SPACEMIT=ON -DSPACEMIT_USE_IME2=ON -DCMAKE_BUILD_TYPE=Release",
            workdir
        )

        if code != 0:
            print("✗ CMake configuration failed")
            return False

        print("✓ CMake configuration successful\n")

        # Build
        print("Building...")
        code, output, error = self.exec_command(
            "cmake --build build -j8",
            workdir
        )

        if code != 0:
            print("✗ Build failed")
            return False

        print("\n✓ Build successful")
        return True

    def run_tests(self):
        """Run tests on K3"""
        print("\n=== Running Tests ===\n")

        workdir = self.remote_base

        tests = [
            ("test_ggml_bridge", "GGML Bridge Test"),
            ("test_spacemit_ops", "SpacemiT Ops Test"),
        ]

        all_passed = True
        for test_exe, test_name in tests:
            print(f"\nRunning {test_name}...")
            code, output, error = self.exec_command(
                f"./build/{test_exe}",
                workdir
            )

            if code == 0:
                print(f"  ✓ {test_name} PASSED")
            else:
                print(f"  ✗ {test_name} FAILED")
                all_passed = False

        return all_passed

    def check_ime_support(self):
        """Check IME support on K3"""
        print("\n=== Checking IME Support ===\n")

        # Check for IME device files
        code, output, error = self.exec_command("ls -l /dev/ime* 2>/dev/null || echo 'No IME devices found'")

        # Check for TCM support
        code2, output2, error2 = self.exec_command("cat /proc/cpuinfo | grep -i tcm || echo 'No TCM info'")

        # Check kernel modules
        code3, output3, error3 = self.exec_command("lsmod | grep ime || echo 'No IME modules loaded'")

        return True

    def create_test_script(self):
        """Create a simple inference test script"""
        print("\n=== Creating Test Script ===\n")

        script_content = """#!/bin/bash
# SpacemiT Test Script

echo "SpacemiT Integration Test"
echo "=========================="
echo ""

echo "1. Testing GGML Bridge..."
./build/test_ggml_bridge
if [ $? -eq 0 ]; then
    echo "✓ GGML Bridge test passed"
else
    echo "✗ GGML Bridge test failed"
    exit 1
fi

echo ""
echo "2. Testing SpacemiT Ops..."
./build/test_spacemit_ops
if [ $? -eq 0 ]; then
    echo "✓ SpacemiT Ops test passed"
else
    echo "✗ SpacemiT Ops test failed"
    exit 1
fi

echo ""
echo "=========================="
echo "✓ All tests passed!"
echo "SpacemiT integration is working correctly"
"""

        # Save locally first
        local_script = os.path.join(self.local_base, "run_spacemit_test.sh")
        with open(local_script, 'w') as f:
            f.write(script_content)

        # Upload to K3
        remote_script = os.path.join(self.remote_base, "run_test.sh")
        if self.upload_file(local_script, remote_script):
            # Make executable
            self.exec_command(f"chmod +x {remote_script}")
            print("✓ Test script created")
            return True

        return False

    def run_integration_test(self):
        """Run the integration test script"""
        print("\n=== Running Integration Test ===\n")

        code, output, error = self.exec_command(
            "./run_test.sh",
            self.remote_base
        )

        return code == 0

    def close(self):
        """Close connections"""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()

def main():
    builder = K3StandaloneBuilder()

    try:
        # Connect
        if not builder.connect():
            return 1

        # Deploy files
        if not builder.deploy_all_files():
            print("\n✗ Failed to deploy files")
            return 1

        # Check IME support
        builder.check_ime_support()

        # Build
        if not builder.build_project():
            print("\n✗ Build failed")
            return 1

        # Run tests
        if not builder.run_tests():
            print("\n⚠ Some tests failed")

        # Create and run integration test
        if builder.create_test_script():
            if builder.run_integration_test():
                print("\n" + "="*60)
                print("✓ SpacemiT Integration Complete!")
                print("="*60)
                print("\nAll tests passed successfully on K3 worker")
                print(f"Test location: {builder.remote_base}")
                return 0
            else:
                print("\n⚠ Integration test had issues")
                return 1

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        builder.close()

if __name__ == "__main__":
    sys.exit(main())
