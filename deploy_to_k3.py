#!/usr/bin/env python3
"""Deploy xLLM SpacemiT implementation to K3 worker"""

import paramiko
import os
import sys
from pathlib import Path

class K3Deployer:
    def __init__(self):
        self.host = "10.0.90.243"
        self.port = 22
        self.username = "bianbu"
        self.password = "bianbu"
        self.remote_base = "/home/bianbu/bianbu-agentos"
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

        print(f"  Executing: {command}")
        stdin, stdout, stderr = self.client.exec_command(command)

        exit_code = stdout.channel.recv_exit_status()
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()

        if output:
            print(f"  Output: {output}")
        if error:
            print(f"  Error: {error}")

        return exit_code, output, error

    def upload_file(self, local_path, remote_path):
        """Upload a file to K3"""
        try:
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote_path)
            self.exec_command(f"mkdir -p {remote_dir}")

            print(f"  Uploading: {local_path} -> {remote_path}")
            self.sftp.put(local_path, remote_path)
            return True
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
            return False

    def upload_directory(self, local_dir, remote_dir, patterns=None):
        """Upload directory recursively with optional file patterns"""
        uploaded = []
        skipped = []

        for root, dirs, files in os.walk(local_dir):
            # Skip hidden directories and build artifacts
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['build', '__pycache__']]

            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                remote_path = os.path.join(remote_dir, rel_path)

                # Check patterns if provided
                if patterns:
                    if not any(p in rel_path for p in patterns):
                        skipped.append(rel_path)
                        continue

                if self.upload_file(local_path, remote_path):
                    uploaded.append(rel_path)
                else:
                    skipped.append(rel_path)

        return uploaded, skipped

    def deploy_spacemit_files(self):
        """Deploy SpacemiT implementation files to K3"""
        print("\n=== Deploying SpacemiT Implementation ===\n")

        # Create target directories
        remote_xllm = f"{self.remote_base}/xllm"
        self.exec_command(f"mkdir -p {remote_xllm}")

        # Files to deploy
        files_to_deploy = [
            # SpacemiT platform files
            ("xllm/core/platform/spacemit/ggml_bridge.h", f"{remote_xllm}/core/platform/spacemit/ggml_bridge.h"),
            ("xllm/core/platform/spacemit/ggml_bridge.cpp", f"{remote_xllm}/core/platform/spacemit/ggml_bridge.cpp"),
            ("xllm/core/platform/spacemit/ggml_backend.h", f"{remote_xllm}/core/platform/spacemit/ggml_backend.h"),
            ("xllm/core/platform/spacemit/ggml_backend.cpp", f"{remote_xllm}/core/platform/spacemit/ggml_backend.cpp"),

            # SpacemiT kernels
            ("xllm/core/kernels/spacemit/matmul_ggml.cpp", f"{remote_xllm}/core/kernels/spacemit/matmul_ggml.cpp"),
            ("xllm/core/kernels/spacemit/rms_norm_ggml.cpp", f"{remote_xllm}/core/kernels/spacemit/rms_norm_ggml.cpp"),

            # Test files
            ("test/platform/spacemit/test_ggml_bridge.cpp", f"{remote_xllm}/test/platform/spacemit/test_ggml_bridge.cpp"),
            ("test/kernels/spacemit/test_spacemit_ops.cpp", f"{remote_xllm}/test/kernels/spacemit/test_spacemit_ops.cpp"),

            # Build configuration
            ("cmake/spacemit.cmake", f"{remote_xllm}/cmake/spacemit.cmake"),
            ("build_spacemit.sh", f"{remote_xllm}/build_spacemit.sh"),
        ]

        uploaded_count = 0
        for local_rel, remote_path in files_to_deploy:
            local_path = os.path.join(self.local_base, local_rel)
            if os.path.exists(local_path):
                if self.upload_file(local_path, remote_path):
                    uploaded_count += 1
            else:
                print(f"  ✗ File not found: {local_path}")

        print(f"\n✓ Uploaded {uploaded_count}/{len(files_to_deploy)} files")
        return uploaded_count > 0

    def deploy_third_party(self):
        """Deploy third_party/ggml-spacemit files"""
        print("\n=== Deploying ggml-spacemit Library ===\n")

        local_dir = os.path.join(self.local_base, "third_party/ggml-spacemit")
        remote_dir = f"{self.remote_base}/xllm/third_party/ggml-spacemit"

        if not os.path.exists(local_dir):
            print(f"✗ Directory not found: {local_dir}")
            return False

        # Upload key ggml-spacemit files
        patterns = ['.h', '.c', '.cpp', 'CMakeLists.txt']
        uploaded, skipped = self.upload_directory(local_dir, remote_dir, patterns)

        print(f"\n✓ Uploaded {len(uploaded)} files, skipped {len(skipped)}")
        return len(uploaded) > 0

    def check_build_tools(self):
        """Check if required build tools are available on K3"""
        print("\n=== Checking Build Tools ===\n")

        tools = {
            "cmake": "cmake --version",
            "gcc": "gcc --version",
            "g++": "g++ --version",
            "make": "make --version",
        }

        available = {}
        for tool, command in tools.items():
            code, output, error = self.exec_command(command)
            if code == 0:
                version = output.split('\n')[0] if output else "unknown"
                print(f"  ✓ {tool}: {version}")
                available[tool] = True
            else:
                print(f"  ✗ {tool}: not found")
                available[tool] = False

        return all(available.values())

    def build_on_k3(self):
        """Build the project on K3"""
        print("\n=== Building on K3 ===\n")

        workdir = f"{self.remote_base}/xllm"

        # Create build directory
        self.exec_command("mkdir -p build", workdir)

        # Configure with CMake
        print("\n  Configuring with CMake...")
        code, output, error = self.exec_command(
            "cmake -B build -DUSE_SPACEMIT=ON -DSPACEMIT_USE_IME2=ON",
            workdir
        )

        if code != 0:
            print("  ✗ CMake configuration failed")
            return False

        print("  ✓ CMake configuration successful")

        # Build
        print("\n  Building...")
        code, output, error = self.exec_command(
            "cmake --build build -j8",
            workdir
        )

        if code != 0:
            print("  ✗ Build failed")
            return False

        print("  ✓ Build successful")
        return True

    def run_tests(self):
        """Run tests on K3"""
        print("\n=== Running Tests ===\n")

        workdir = f"{self.remote_base}/xllm"

        tests = [
            "build/test/test_ggml_bridge",
            "build/test/test_spacemit_ops"
        ]

        results = {}
        for test in tests:
            test_name = os.path.basename(test)
            print(f"\n  Running {test_name}...")

            code, output, error = self.exec_command(test, workdir)

            if code == 0:
                print(f"  ✓ {test_name} passed")
                results[test_name] = "PASS"
            else:
                print(f"  ✗ {test_name} failed")
                results[test_name] = "FAIL"

        return results

    def download_model(self):
        """Download Qwen model on K3"""
        print("\n=== Downloading Model ===\n")

        workdir = f"{self.remote_base}/xllm"
        model_dir = f"{workdir}/models"

        # Create models directory
        self.exec_command(f"mkdir -p {model_dir}", workdir)

        # Check if model already exists
        model_file = "qwen2.5-0.5b-instruct-q4_0.gguf"
        code, output, error = self.exec_command(f"ls -lh {model_dir}/{model_file}", workdir)

        if code == 0:
            print(f"  ✓ Model already exists: {output}")
            return True

        # Download model
        print("  Downloading Qwen2.5-0.5B-Instruct Q4_0...")
        model_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"

        code, output, error = self.exec_command(
            f"wget -O {model_dir}/{model_file} {model_url}",
            workdir
        )

        if code == 0:
            print("  ✓ Model downloaded successfully")
            return True
        else:
            print("  ✗ Model download failed")
            return False

    def run_inference(self):
        """Run inference test with the model"""
        print("\n=== Running Inference Test ===\n")

        workdir = f"{self.remote_base}/xllm"
        model_file = f"{workdir}/models/qwen2.5-0.5b-instruct-q4_0.gguf"

        # Run xllm-cli
        print("  Running inference with SpacemiT backend...")
        code, output, error = self.exec_command(
            f'./bin/xllm-cli --model {model_file} --device spacemit --prompt "Hello, how are you?" --max-tokens 50',
            workdir
        )

        if code == 0:
            print("  ✓ Inference successful")
            print(f"\n  Output:\n{output}")
            return True
        else:
            print("  ✗ Inference failed")
            return False

    def close(self):
        """Close connections"""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        print("\n✓ Disconnected")

def main():
    deployer = K3Deployer()

    try:
        # Connect
        if not deployer.connect():
            return 1

        # Deploy files
        if not deployer.deploy_spacemit_files():
            print("\n✗ Failed to deploy SpacemiT files")
            return 1

        if not deployer.deploy_third_party():
            print("\n✗ Failed to deploy third_party files")
            return 1

        # Check build tools
        if not deployer.check_build_tools():
            print("\n✗ Required build tools not available")
            return 1

        # Build
        if not deployer.build_on_k3():
            print("\n✗ Build failed")
            return 1

        # Run tests
        results = deployer.run_tests()
        if not all(r == "PASS" for r in results.values()):
            print("\n⚠ Some tests failed")

        # Download model
        if not deployer.download_model():
            print("\n⚠ Model download failed")

        # Run inference
        deployer.run_inference()

        print("\n" + "="*50)
        print("✓ Deployment Complete!")
        print("="*50)

        return 0

    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        deployer.close()

if __name__ == "__main__":
    sys.exit(main())
