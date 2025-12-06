#!/bin/bash
# Start llama.cpp server for Qwen3-4B-Instruct-2507
# Requires: brew install llama.cpp

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_FILE="${PROJECT_ROOT}/models/llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# Default settings
PORT="${PORT:-8080}"
CONTEXT="${CONTEXT:-4096}"
GPU_LAYERS="${GPU_LAYERS:-99}"  # 99 = offload all to Metal GPU

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model not found at ${MODEL_FILE}"
    echo "Run ./scripts/setup_llm.sh first to download the model."
    exit 1
fi

echo "=========================================="
echo "Starting Qwen3-4B-Instruct-2507 Server"
echo "=========================================="
echo ""
echo "Model: ${MODEL_FILE}"
echo "Port: ${PORT}"
echo "Context: ${CONTEXT}"
echo "GPU Layers: ${GPU_LAYERS}"
echo ""
echo "API Endpoint: http://localhost:${PORT}/v1/chat/completions"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start llama-server with Metal GPU acceleration
llama-server \
    --model "${MODEL_FILE}" \
    --ctx-size "${CONTEXT}" \
    --n-gpu-layers "${GPU_LAYERS}" \
    --port "${PORT}" \
    --host 0.0.0.0
