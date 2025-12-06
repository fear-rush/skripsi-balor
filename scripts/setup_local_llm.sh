#!/bin/bash
# =============================================================================
# Local LLM Setup Script using llama.cpp
#
# This script sets up llama.cpp and downloads the quantized SQLCoder model
# for Text-to-SQL generation without cloud dependencies.
#
# Model: SQLCoder-7B-2 (4-bit quantized GGUF)
# Source: https://huggingface.co/QuantFactory/sqlcoder-7b-2-GGUF
#
# Requirements:
# - macOS, Linux, or WSL2 on Windows
# - At least 6GB RAM (8GB recommended)
# - At least 5GB disk space
# - CMake and C++ compiler
#
# Author: Firas
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LLAMA_CPP_DIR="${HOME}/.local/llama.cpp"
MODELS_DIR="${HOME}/.local/models"
MODEL_URL="https://huggingface.co/QuantFactory/sqlcoder-7b-2-GGUF/resolve/main/sqlcoder-7b-2.Q4_K_M.gguf"
MODEL_NAME="sqlcoder-7b-2.Q4_K_M.gguf"
SERVER_PORT=8080

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Local LLM Setup (llama.cpp + SQLCoder)${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo -e "Model: ${GREEN}SQLCoder-7B-2 (Q4_K_M, ~4.4GB)${NC}"
echo -e "Server: ${GREEN}llama-server on port ${SERVER_PORT}${NC}"
echo ""

# -----------------------------------------------------------------------------
# Check system requirements
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Checking system requirements...${NC}"

# Check OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM="Linux";;
    Darwin*)    PLATFORM="macOS";;
    CYGWIN*|MINGW*|MSYS*)
        echo -e "${RED}Windows detected. Please use WSL2.${NC}"
        PLATFORM="Windows";;
    *)
        echo -e "${RED}Unsupported OS: ${OS}${NC}"
        exit 1;;
esac
echo -e "  Platform: ${GREEN}${PLATFORM}${NC}"

# Check for required tools
MISSING_TOOLS=""

if ! command -v git &> /dev/null; then
    MISSING_TOOLS="$MISSING_TOOLS git"
fi

if ! command -v cmake &> /dev/null; then
    MISSING_TOOLS="$MISSING_TOOLS cmake"
fi

if ! command -v make &> /dev/null; then
    MISSING_TOOLS="$MISSING_TOOLS make"
fi

if [ -n "$MISSING_TOOLS" ]; then
    echo -e "${RED}  Missing required tools:${MISSING_TOOLS}${NC}"
    echo ""
    if [ "$PLATFORM" = "macOS" ]; then
        echo -e "  Install with: ${YELLOW}brew install${MISSING_TOOLS}${NC}"
    else
        echo -e "  Install with: ${YELLOW}sudo apt install${MISSING_TOOLS}${NC}"
    fi
    exit 1
fi

echo -e "  ${GREEN}All required tools found${NC}"

# Check memory
if [ "$PLATFORM" = "macOS" ]; then
    TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
else
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
fi
echo -e "  Total RAM: ${GREEN}${TOTAL_MEM}GB${NC}"

if [ "$TOTAL_MEM" -lt 6 ]; then
    echo -e "${YELLOW}  Warning: At least 6GB RAM recommended for Q4 model${NC}"
fi

echo ""

# -----------------------------------------------------------------------------
# Install/Update llama.cpp
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/6] Setting up llama.cpp...${NC}"

mkdir -p "$LLAMA_CPP_DIR"
mkdir -p "$MODELS_DIR"

if [ -d "$LLAMA_CPP_DIR/.git" ]; then
    echo -e "  llama.cpp found, updating..."
    cd "$LLAMA_CPP_DIR"
    git pull --quiet
else
    echo -e "  Cloning llama.cpp..."
    rm -rf "$LLAMA_CPP_DIR"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
fi

echo -e "  ${GREEN}llama.cpp repository ready${NC}"
echo ""

# -----------------------------------------------------------------------------
# Build llama.cpp
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/6] Building llama.cpp...${NC}"

cd "$LLAMA_CPP_DIR"

# Clean previous build
rm -rf build 2>/dev/null || true

# Build with cmake
mkdir -p build
cd build

# Configure based on platform
if [ "$PLATFORM" = "macOS" ]; then
    # Enable Metal acceleration on macOS
    echo -e "  Configuring with Metal acceleration..."
    cmake .. -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
else
    # Check for CUDA
    if command -v nvcc &> /dev/null; then
        echo -e "  Configuring with CUDA acceleration..."
        cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    else
        echo -e "  Configuring for CPU only..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
    fi
fi

# Build
echo -e "  Building (this may take a few minutes)..."
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if server binary exists
if [ -f "bin/llama-server" ]; then
    SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
    echo -e "  ${GREEN}Build successful!${NC}"
elif [ -f "llama-server" ]; then
    SERVER_BIN="$LLAMA_CPP_DIR/build/llama-server"
    echo -e "  ${GREEN}Build successful!${NC}"
else
    echo -e "${RED}  Build failed - llama-server not found${NC}"
    exit 1
fi

echo ""

# -----------------------------------------------------------------------------
# Download model
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/6] Downloading SQLCoder model (Q4_K_M)...${NC}"

MODEL_PATH="$MODELS_DIR/$MODEL_NAME"

if [ -f "$MODEL_PATH" ]; then
    echo -e "  ${GREEN}Model already downloaded${NC}"
    echo -e "  Path: $MODEL_PATH"
else
    echo -e "  Downloading from HuggingFace..."
    echo -e "  ${YELLOW}(~4.4GB, this may take 10-30 minutes)${NC}"
    echo ""

    # Download with progress
    if command -v wget &> /dev/null; then
        wget -O "$MODEL_PATH" "$MODEL_URL" --show-progress
    else
        curl -L -o "$MODEL_PATH" "$MODEL_URL" --progress-bar
    fi

    # Verify download
    if [ -f "$MODEL_PATH" ]; then
        SIZE=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
        echo -e "  ${GREEN}Download complete! (${SIZE})${NC}"
    else
        echo -e "${RED}  Download failed${NC}"
        exit 1
    fi
fi

echo ""

# -----------------------------------------------------------------------------
# Create startup script
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/6] Creating startup scripts...${NC}"

# Create server startup script
STARTUP_SCRIPT="$PROJECT_DIR/scripts/start_llm_server.sh"

cat > "$STARTUP_SCRIPT" << EOF
#!/bin/bash
# Start llama.cpp server with SQLCoder model

SERVER_BIN="$SERVER_BIN"
MODEL_PATH="$MODEL_PATH"
PORT=${SERVER_PORT}

echo "Starting llama.cpp server..."
echo "Model: \$MODEL_PATH"
echo "Port: \$PORT"
echo ""
echo "API Endpoints:"
echo "  - Health: http://localhost:\$PORT/health"
echo "  - Completion: http://localhost:\$PORT/completion"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

"\$SERVER_BIN" \\
    --model "\$MODEL_PATH" \\
    --ctx-size 2048 \\
    --host 0.0.0.0 \\
    --port \$PORT \\
    --threads \$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
EOF

chmod +x "$STARTUP_SCRIPT"
echo -e "  Created: ${GREEN}scripts/start_llm_server.sh${NC}"

# Create stop script
STOP_SCRIPT="$PROJECT_DIR/scripts/stop_llm_server.sh"

cat > "$STOP_SCRIPT" << EOF
#!/bin/bash
# Stop llama.cpp server

echo "Stopping llama.cpp server..."
pkill -f "llama-server" 2>/dev/null && echo "Server stopped" || echo "Server not running"
EOF

chmod +x "$STOP_SCRIPT"
echo -e "  Created: ${GREEN}scripts/stop_llm_server.sh${NC}"

echo ""

# -----------------------------------------------------------------------------
# Test server (optional)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[6/6] Testing setup...${NC}"

# Quick test without starting server
echo -e "  Verifying model file..."
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
    echo -e "  ${GREEN}Model OK (${SIZE})${NC}"
else
    echo -e "${RED}  Model file not found${NC}"
    exit 1
fi

echo -e "  Verifying server binary..."
if [ -f "$SERVER_BIN" ]; then
    echo -e "  ${GREEN}Server binary OK${NC}"
else
    echo -e "${RED}  Server binary not found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo -e "Model: ${BLUE}SQLCoder-7B-2 (Q4_K_M)${NC}"
echo -e "Location: ${BLUE}${MODEL_PATH}${NC}"
echo -e "Server API: ${BLUE}http://localhost:${SERVER_PORT}${NC}"
echo ""
echo -e "${YELLOW}To start the LLM server:${NC}"
echo -e "  ${GREEN}./scripts/start_llm_server.sh${NC}"
echo ""
echo -e "${YELLOW}To stop the LLM server:${NC}"
echo -e "  ${GREEN}./scripts/stop_llm_server.sh${NC}"
echo ""
echo -e "${YELLOW}To use with the chatbot:${NC}"
echo -e "  1. Start the LLM server in one terminal"
echo -e "  2. Run the chatbot with LLM support:"
echo -e "     ${GREEN}python -m src.chatbot_v2 models/intent_classifier_v2/best_model --use-llm${NC}"
echo ""
echo -e "${YELLOW}To test the LLM directly:${NC}"
echo -e "  ${GREEN}python -m src.llm_query_generator${NC}"
echo ""
