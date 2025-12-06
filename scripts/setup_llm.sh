#!/bin/bash
# Setup script for Qwen3-4B-Instruct-2507-GGUF with native llama.cpp (macOS Apple Silicon)
# Requires: brew install llama.cpp

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models"

echo "=========================================="
echo "Setting up Qwen3-4B-Instruct-2507 with llama.cpp"
echo "=========================================="

# Check if llama.cpp is installed
if ! command -v llama-server &> /dev/null; then
    echo "ERROR: llama.cpp not found!"
    echo "Install with: brew install llama.cpp"
    exit 1
fi

echo "llama.cpp version: $(llama-cli --version 2>&1 | head -1 || echo 'installed')"

# Create model directory
mkdir -p "${MODEL_DIR}"

# Check if model already exists
MODEL_FILE="${MODEL_DIR}/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at: ${MODEL_FILE}"
    echo "Size: $(du -h "${MODEL_FILE}" | cut -f1)"
else
    echo ""
    echo "Downloading Qwen3-4B-Instruct-2507 Q4_K_M (~2.6 GB)..."
    echo "This may take 5-15 minutes depending on your connection..."
    echo ""

    # Download using huggingface-cli or curl
    if command -v huggingface-cli &> /dev/null; then
        echo "Using huggingface-cli..."
        huggingface-cli download Qwen/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir "${MODEL_DIR}"
    else
        echo "Using curl..."
        curl -L -C - "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf?download=true" \
            -o "${MODEL_FILE}"
    fi


    echo ""
    echo "Download complete!"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Model location: ${MODEL_FILE}"
echo ""
echo "To start the server, run:"
echo "  llama-server -m ${MODEL_FILE} -c 4096 -ngl 99 --port 8080"
echo ""
echo "Or use the helper script:"
echo "  ./scripts/start_llm_server.sh"
echo ""
echo "API will be available at: http://localhost:8080"
echo ""
