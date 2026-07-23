#!/usr/bin/env python3
"""Check K3 build status"""

import paramiko

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

# Check if build is running
code, out = exec_cmd(client, "ps aux | grep cmake | grep -v grep || echo 'No cmake running'")
print("Build process status:")
print(out)

# Check if directory exists
code, out = exec_cmd(client, "ls -la /home/bianbu/llama.cpp-spacemit 2>&1")
print("\nDirectory status:")
print(out[:500] if len(out) > 500 else out)

# Check build directory
code, out = exec_cmd(client, "ls -la /home/bianbu/llama.cpp-spacemit/build 2>&1 | head -20")
print("\nBuild directory:")
print(out)

client.close()
