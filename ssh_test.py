#!/usr/bin/env python3
"""Test SSH connection to K3 worker using paramiko"""

import paramiko
import sys

def test_ssh_connection():
    """Test SSH connection with password authentication"""
    host = "10.0.90.243"
    port = 22
    username = "bianbu"
    password = "bianbu"

    print(f"Testing SSH connection to {username}@{host}:{port}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect with password
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            timeout=10,
            allow_agent=False,
            look_for_keys=False
        )

        print("✓ SSH connection successful!")

        # Test command execution
        stdin, stdout, stderr = client.exec_command("pwd")
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()

        if output:
            print(f"✓ Working directory: {output}")
        if error:
            print(f"Error: {error}")

        # Test target directory
        stdin, stdout, stderr = client.exec_command("ls -la /home/bianbu/bianbu-agentos")
        output = stdout.read().decode().strip()

        if "No such file or directory" in output or stderr.read().decode():
            print("✗ Target directory /home/bianbu/bianbu-agentos does not exist")
            # Try to find the actual working directory
            stdin, stdout, stderr = client.exec_command("ls -la /home/bianbu/")
            output = stdout.read().decode().strip()
            print("\nContents of /home/bianbu/:")
            print(output)
        else:
            print(f"✓ Target directory exists:\n{output}")

        client.close()
        return True

    except paramiko.AuthenticationException:
        print("✗ Authentication failed - username/password incorrect")
        return False
    except paramiko.SSHException as e:
        print(f"✗ SSH error: {e}")
        return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

if __name__ == "__main__":
    success = test_ssh_connection()
    sys.exit(0 if success else 1)
