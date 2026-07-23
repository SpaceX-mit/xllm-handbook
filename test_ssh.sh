#!/bin/bash
# 测试SSH连接的多种方式

echo "测试1: 使用密码 'bianbu'"
sshpass -p 'bianbu' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null bianbu@10.0.90.243 'whoami && pwd' 2>&1 | head -5

echo ""
echo "测试2: 检查网络连通性"
ping -c 2 10.0.90.243 2>&1 | head -5

echo ""
echo "测试3: 尝试直接SSH（可能需要手动输入密码）"
echo "如果提示输入密码，请输入: bianbu"
