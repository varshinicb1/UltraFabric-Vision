#!/bin/bash

# Kill existing processes on target ports
echo "Cleaning up ports..."
fuser -k 8000/tcp 2>/dev/null
fuser -k 5173/tcp 2>/dev/null
fuser -k 5174/tcp 2>/dev/null

# Start Backend
echo "Starting AI Backend (Enforcing GPU)..."
python3 backend_api.py &
BACKEND_PID=$!

# Start Main Dashboard
echo "Starting UltraFabric Dashboard..."
cd web_app
npm run dev -- --host &
DASHBOARD_PID=$!
cd ..

# Start Remote Firebase Streamer
echo "Starting Firebase Remote Cam..."
cd remote_cam
npm run dev -- --host &
REMOTE_PID=$!
cd ..

echo "------------------------------------------------"
echo "UltraFabric-Vision Platform Industrialized"
echo "Main Dashboard: http://localhost:5173"
echo "Firebase Streamer: http://localhost:5174"
echo "AI Backend: http://localhost:8000"
echo "------------------------------------------------"

# Keep script running
wait $BACKEND_PID $DASHBOARD_PID $REMOTE_PID
