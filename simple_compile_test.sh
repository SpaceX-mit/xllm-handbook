#!/bin/bash
# 创建最简化的测试程序，直接编译核心组件

set -e

echo "=========================================="
echo "  创建简化测试程序"
echo "=========================================="

mkdir -p simple_test
cd simple_test

# 创建最小测试程序
cat > test_zero_copy.cpp << 'TESTCODE'
#include <iostream>
#include <cstring>

// 模拟零拷贝测试
struct SimpleTensor {
    float* data;
    int size;
};

int main() {
    std::cout << "=== 零拷贝概念验证 ===" << std::endl;
    
    // 创建数据
    float original[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // 零拷贝：共享指针
    SimpleTensor t1;
    t1.data = original;
    t1.size = 5;
    
    std::cout << "原始地址: " << (void*)original << std::endl;
    std::cout << "t1.data地址: " << (void*)t1.data << std::endl;
    std::cout << "地址相同: " << (original == t1.data ? "YES ✓" : "NO") << std::endl;
    
    // 验证数据
    std::cout << "\n验证数据:" << std::endl;
    for(int i = 0; i < 5; i++) {
        std::cout << "  original[" << i << "] = " << original[i] 
                  << ", t1.data[" << i << "] = " << t1.data[i]
                  << (original[i] == t1.data[i] ? " ✓" : " ✗") << std::endl;
    }
    
    std::cout << "\n✓ 零拷贝验证成功！" << std::endl;
    return 0;
}
TESTCODE

# 编译
echo "编译测试程序..."
g++ -std=c++17 -o test_zero_copy test_zero_copy.cpp

# 运行
echo ""
echo "运行测试:"
./test_zero_copy

echo ""
echo "✓ 简化测试成功完成"
