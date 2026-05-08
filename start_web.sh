#!/bin/bash

echo "========================================================"
echo "      FABRIC AI PRO WEB — Startup Script                "
echo "========================================================"

# Trap ctrl-c and call cleanup
trap cleanup INT

function cleanup() {
    echo -e "\n[STATUS] Shutting down FabricAI Pro Web services..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

echo "[STATUS] Starting FastAPI AI Engine..."
source venv/bin/activate
python backend_api.py &
BACKEND_PID=$!

echo "[STATUS] Starting Vite React Frontend..."
cd web_app
npm run dev -- --host &
FRONTEND_PID=$!

echo ""
echo "========================================================"
echo " SERVICES ARE RUNNING!                                  "
echo " 🌐 Frontend Web UI: http://localhost:5173              "
echo " 🧠 Backend AI Engine: http://localhost:8000            "
echo " Press CTRL+C to stop both services.                    "
echo "========================================================"

wait $BACKEND_PID $FRONTEND_PID
