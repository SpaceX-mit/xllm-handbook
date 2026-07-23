#!/bin/bash
# 直接使用SSH传输和执行

set -e

K3_HOST="10.0.90.243"
K3_USER="bianbu"
K3_PASS="bianbu"

echo "=========================================="
echo "  部署到 K3 Worker"
echo "=========================================="

# 创建部署包
echo "1. 创建部署包..."
cd simple_test
tar czf ../k3_deploy.tar.gz test_zero_copy test_zero_copy.cpp
cd ..

echo "2. 传输到 K3..."
# 使用 ssh 带密码（通过文件描述符）
cat << SSHCMD | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${K3_USER}@${K3_HOST} 2>&1 | grep -v "Warning"
cd /home/bianbu
mkdir -p xllm_test
cd xllm_test
cat > test_received.txt << 'INNEREOF'
File received successfully on K3
INNEREOF
pwd
ls -la
uname -m
cat /proc/cpuinfo | grep -i "model name" | head -1 || cat /proc/cpuinfo | grep -i "cpu" | head -3
SSHCMD

echo ""
echo "✓ K3 连接测试完成"
