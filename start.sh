#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="${PROJECT_DIR}/frontend"

echo "=========================================="
echo "   Starting AgentSearch-UAV AI System    "
echo "=========================================="

cleanup() {
    echo ""
    echo "Stopping all services..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start Backend API
echo "Starting Backend API on port 5005..."
export PYTHONPATH="${PROJECT_DIR}"
if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -m api.main &
BACKEND_PID=$!

# 2. Start Frontend UI
echo "Starting Frontend UI on port 5173..."
cd "${FRONTEND_DIR}"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🚀 Services launched successfully:"
echo "   - Backend API : http://127.0.0.1:5005"
echo "   - Frontend UI : http://localhost:5173"
echo ""
echo "Press Ctrl+C to terminate both services."
echo "=========================================="

wait
