import React, { useState, useRef, useEffect } from 'react';

function App() {
  const [status, setStatus] = useState('Disconnected');
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const defaultWs = `${protocol}//${window.location.hostname}:8000/ws/stream`;
  const [apiUrl, setApiUrl] = useState(defaultWs);
  const [fps, setFps] = useState(15);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    startCamera();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Camera error:", err);
      setStatus('Simulating (No Camera)');
      startSimulation();
    }
  };

  const startSimulation = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let offset = 0;
    
    setInterval(() => {
      // Draw fabric-like pattern
      ctx.fillStyle = '#f3f4f6';
      ctx.fillRect(0, 0, 640, 480);
      ctx.strokeStyle = '#d1d5db';
      ctx.lineWidth = 1;
      
      // Vertical lines
      for (let i = 0; i < 640; i += 20) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, 480);
        ctx.stroke();
      }
      
      // Horizontal lines (moving)
      offset = (offset + 2) % 40;
      for (let i = -40; i < 480; i += 40) {
        ctx.beginPath();
        ctx.moveTo(0, i + offset);
        ctx.lineTo(640, i + offset);
        ctx.stroke();
        
        // Add a "defect" occasionally
        if (i === 200 && offset > 10 && offset < 30) {
          ctx.fillStyle = 'rgba(153, 27, 27, 0.3)';
          ctx.beginPath();
          ctx.arc(320, 240, 30, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }, 100);
  };

  const connect = () => {
    if (wsRef.current) wsRef.current.close();
    if (timerRef.current) clearInterval(timerRef.current);
    
    setStatus('Connecting...');
    wsRef.current = new WebSocket(apiUrl);
    let reconnectTimer = null;
    
    wsRef.current.onopen = () => {
      setStatus(status.includes('Simulating') ? 'Streaming (Simulated)' : 'Streaming (Live)');
      startStreaming();
    };
    
    wsRef.current.onclose = () => {
      setStatus('Disconnected');
      if (timerRef.current) clearInterval(timerRef.current);
      reconnectTimer = setTimeout(() => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
          setStatus('Reconnecting...');
          connect();
        }
      }, 3000);
    };
    
    wsRef.current.onerror = () => {
      setStatus('Connection Error');
    };
  };

  const startStreaming = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const data = canvas.toDataURL('image/jpeg', 0.5);
        const base64Data = data.split(',')[1];
        wsRef.current.send(base64Data);
      }
    }, 1000 / fps);
  };

  return (
    <div className="remote-container">
      <header>
        <div className="logo-section">
          <span className="brand">UltraFabric</span>
          <span className="product">Remote Cam</span>
        </div>
        <div className={`status-badge ${status.toLowerCase().replace(/ /g, '-')}`}>
          {status}
        </div>
      </header>

      <main>
        <div className="preview-card">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
          <div className="overlay-info">
            <span>640x480</span>
            <span>{fps} FPS</span>
          </div>
        </div>

        <div className="controls-card">
          <div className="input-group">
            <label>UltraFabric API Endpoint</label>
            <input 
              type="text" 
              value={apiUrl} 
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="ws://localhost:8000/ws/stream"
            />
          </div>
          
          <div className="input-group">
            <label>Transmission FPS</label>
            <input 
              type="range" 
              min="1" max="30" 
              value={fps} 
              onChange={(e) => setFps(parseInt(e.target.value))}
            />
          </div>

          <button onClick={() => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.close();
              if (timerRef.current) clearInterval(timerRef.current);
              setStatus('Disconnected');
            } else {
              connect();
            }
          }} className={wsRef.current?.readyState === WebSocket.OPEN ? 'active' : ''}>
            {wsRef.current?.readyState === WebSocket.OPEN ? 'DISCONNECT' : 'CONNECT & STREAM'}
          </button>
        </div>
      </main>

      <footer>
        <div className="telemetry-bar">
          <span>TX: 0.0 MB/s</span>
          <span>Buffer: 0ms</span>
          <span>WebSocket Relay</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
