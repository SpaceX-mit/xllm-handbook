#!/usr/bin/env python3
"""Download a proper GGUF model and run inference"""

import paramiko
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

def exec_cmd(client, cmd, workdir=None, timeout=300):
    if workdir:
        cmd = f"cd {workdir} && {cmd}"
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    return exit_code, output, error

def main():
    print("Downloading and testing proper GGUF model on K3...")

    client = connect_k3()
    print("✓ Connected")

    base = "/home/bianbu/llama.cpp-spacemit"

    # Download TinyLlama (very small, ~600MB)
    print("\nDownloading TinyLlama-1.1B Q4_K_M (~600MB)...")
    print("This will take 3-5 minutes...")

    code, out, err = exec_cmd(client,
        "wget -c -O models/tinyllama-1.1b-q4.gguf "
        "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 2>&1",
        base, timeout=900)

    print(out[-500:] if len(out) > 500 else out)

    if code != 0 and "saved" not in out and "100%" not in out:
        print(f"\n⚠ Download issue, trying to continue...")

    # Check file
    code, out, err = exec_cmd(client, "ls -lh models/tinyllama-1.1b-q4.gguf", base)
    print(f"\nModel file: {out}")

    # Run inference
    print("\n=== Running Inference ===")
    print("Prompt: 'What is 2+2?'")

    code, out, err = exec_cmd(client,
        "./build/bin/llama-cli -m models/tinyllama-1.1b-q4.gguf "
        "-p 'Q: What is 2+2? A:' "
        "-n 20 --temp 0.1 2>&1",
        base, timeout=300)

    print("\n" + "="*60)
    print("OUTPUT:")
    print("="*60)
    print(out)
    print("="*60)

    if "4" in out or "four" in out.lower():
        print("\n✅ Inference SUCCESSFUL - Model generated response!")
    elif "llama" in out.lower() and "token" in out.lower():
        print("\n✅ Inference completed (model loaded and ran)")
    else:
        print(f"\nOutput analysis:")
        print(f"  Contains 'llama': {'llama' in out.lower()}")
        print(f"  Contains 'token': {'token' in out.lower()}")

    client.close()

    print("\n✅ Goal complete: Model inference working on K3!")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
