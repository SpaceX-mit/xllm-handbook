#!/bin/bash
# Build script for xLLM with SpacemiT support

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  xLLM SpacemiT A++ Build Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
BUILD_DIR=${BUILD_DIR:-build_spacemit}
INSTALL_DIR=${INSTALL_DIR:-${BUILD_DIR}/install}
BUILD_TYPE=${BUILD_TYPE:-Release}
NUM_JOBS=${NUM_JOBS:-$(nproc)}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Build directory: ${BUILD_DIR}"
echo "  Install directory: ${INSTALL_DIR}"
echo "  Build type: ${BUILD_TYPE}"
echo "  Parallel jobs: ${NUM_JOBS}"
echo ""

# Check if running on RISC-V
ARCH=$(uname -m)
if [[ "$ARCH" == "riscv64" ]]; then
    echo -e "${GREEN}✓ Native build on RISC-V${NC}"
    CROSS_COMPILE=OFF
else
    echo -e "${YELLOW}✓ Cross-compilation from ${ARCH} to riscv64${NC}"
    CROSS_COMPILE=ON
fi

# Check ggml-spacemit
if [ ! -d "third_party/ggml-spacemit" ]; then
    echo -e "${RED}✗ ggml-spacemit not found!${NC}"
    echo "Please run:"
    echo "  cp -r /path/to/llama.cpp/ggml/src/ggml-cpu/spacemit third_party/ggml-spacemit/"
    exit 1
fi

echo -e "${GREEN}✓ ggml-spacemit found${NC}"

# CMake configure
echo -e "${YELLOW}Configuring CMake...${NC}"

cmake -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DUSE_SPACEMIT=ON \
  -DSPACEMIT_USE_IME2=ON \
  -DSPACEMIT_USE_IME1=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_NPU=OFF \
  -DUSE_MLU=OFF \
  -DUSE_DCU=OFF \
  -DUSE_MUSA=OFF \
  -DUSE_ILU=OFF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ CMake configuration failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CMake configured${NC}"

# Build
echo -e "${YELLOW}Building xLLM...${NC}"

cmake --build ${BUILD_DIR} \
  --parallel ${NUM_JOBS} \
  --config ${BUILD_TYPE}

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build completed${NC}"

# Install
echo -e "${YELLOW}Installing...${NC}"

cmake --install ${BUILD_DIR}

echo -e "${GREEN}✓ Installed to ${INSTALL_DIR}${NC}"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Build Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo "  Binary: ${INSTALL_DIR}/bin/xllm"
echo "  Libraries: ${INSTALL_DIR}/lib/"
echo ""
echo "Next steps:"
echo "  1. Run tests: cd ${BUILD_DIR} && ctest"
echo "  2. Deploy to K3: sshpass -p 'bianbu' scp -r ${INSTALL_DIR} bianbu@10.0.90.243:/home/bianbu/"
echo ""
