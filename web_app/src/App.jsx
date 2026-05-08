import React, { useState, useEffect, useRef } from 'react';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('analysis'); // Default to Batch Analysis
  const [frame, setFrame] = useState(null);
  const [rawFrame, setRawFrame] = useState(null);
  const [neuralGrid, setNeuralGrid] = useState(Array(14).fill(Array(14).fill(0)));
  const [attnHeads, setAttnHeads] = useState([]);
  const [selectedHead, setSelectedHead] = useState(0);
  const [logs, setLogs] = useState([]);
  const [batchResults, setBatchResults] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const [streamSource, setStreamSource] = useState('webcam'); // 'webcam' or 'url'
  const [cameraUrl, setCameraUrl] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  
  const [stats, setStats] = useState({
    status: 'Ready',
    score: 0,
    latency: 0,
    isAnomalous: false,
    fps: 0,
    engineState: 'Idle',
    gpuUtil: '0%',
    gpuMem: '0 MB / 0 MB'
  });

  const videoRef = useRef(null);
  const socketRef = useRef(null);
  const logEndRef = useRef(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // WebSocket Setup for Live Stream
  useEffect(() => {
    if (activeTab !== 'live' || !isStreaming) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let socketUrl = streamSource === 'webcam' 
      ? `${protocol}//${window.location.hostname}:8000/ws/stream`
      : streamSource === 'monitor'
      ? `${protocol}//${window.location.hostname}:8000/ws/subscribe`
      : `${protocol}//${window.location.hostname}:8000/ws/remote_stream?url=${encodeURIComponent(cameraUrl)}`;

    socketRef.current = new WebSocket(socketUrl);
    
    socketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.image_data) setFrame(data.image_data);
      if (data.raw_data) setRawFrame(data.raw_data);
      setNeuralGrid(data.neural_grid || []);
      setAttnHeads(data.attn_heads || []);
      
      setStats(prev => ({
        ...prev,
        status: data.is_anomalous ? 'DEFECT DETECTED' : 'SYSTEM CLEAR',
        score: data.score,
        latency: data.latency_ms,
        isAnomalous: data.is_anomalous,
        fps: data.latency_ms > 0 ? (1000 / data.latency_ms).toFixed(1) : 0,
        engineState: 'Active',
        gpuUtil: data.gpu_util || 'N/A',
        gpuMem: data.gpu_mem || 'N/A'
      }));

      if (data.trace) {
        const timestamp = new Date().toLocaleTimeString();
        setLogs(prev => {
          const newLogs = data.trace.map(t => `[${timestamp}] ${t}`);
          return [...prev.slice(-15), ...newLogs];
        });
      }
    };

    if (streamSource === 'webcam') {
      navigator.mediaDevices.getUserMedia({ video: { width: 448, height: 448 } })
        .then(stream => { if (videoRef.current) videoRef.current.srcObject = stream; })
        .catch(err => {
          console.error("Camera error:", err);
          setLogs(prev => [...prev, `[ERROR] Webcam Access Denied: ${err.message}`]);
        });
    }

    return () => { 
      if (socketRef.current) socketRef.current.close(); 
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, [activeTab, isStreaming, streamSource, cameraUrl]);

  // Webcam Frame Loop
  useEffect(() => {
    if (activeTab !== 'live' || streamSource !== 'webcam' || !isStreaming) return;
    const timer = setInterval(() => {
      if (videoRef.current && videoRef.current.readyState >= 2 && socketRef.current?.readyState === WebSocket.OPEN) {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = 224; canvas.height = 224;
          canvas.getContext('2d').drawImage(videoRef.current, 0, 0, 224, 224);
          const base64 = canvas.toDataURL('image/jpeg', 0.8);
          setRawFrame(base64);
          socketRef.current.send(base64);
        } catch (err) {
          console.error("Frame extraction error:", err);
        }
      }
    }, 200);
    return () => clearInterval(timer);
  }, [activeTab, streamSource, isStreaming]);

  const handleFileUpload = async (e, isBatch = false) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    setIsUploading(true);
    const formData = new FormData();
    
    if (isBatch) {
      files.forEach(file => formData.append('files', file));
      try {
        const res = await fetch('http://localhost:8000/api/batch_upload', { method: 'POST', body: formData });
        const data = await res.json();
        setBatchResults(data.results);
        if (data.results && data.results.length > 0) {
          const first = data.results[0];
          setSelectedResult(first);
          setNeuralGrid(first.neural_grid);
          setAttnHeads(first.attn_heads || []);
          setStats(prev => ({
            ...prev,
            status: first.is_anomalous ? 'DEFECT DETECTED' : 'SYSTEM CLEAR',
            score: first.score,
            latency: first.latency_ms,
            isAnomalous: first.is_anomalous
          }));
        }
        setActiveTab('analysis');
      } catch (err) { console.error("Batch upload failed:", err); }
    } else {
      formData.append('file', files[0]);
      try {
        const res = await fetch('http://localhost:8000/api/upload_image', { method: 'POST', body: formData });
        const data = await res.json();
        setBatchResults(prev => [data, ...prev]);
        setSelectedResult(data);
        setNeuralGrid(data.neural_grid);
        setAttnHeads(data.attn_heads || []);
        setStats(prev => ({
          ...prev,
          status: data.is_anomalous ? 'DEFECT DETECTED' : 'SYSTEM CLEAR',
          score: data.score,
          latency: data.latency_ms,
          isAnomalous: data.is_anomalous
        }));
        setActiveTab('analysis');
      } catch (err) { console.error("Upload failed:", err); }
    }
    setIsUploading(false);
  };

  const safeNeuralGrid = Array.isArray(neuralGrid) && neuralGrid.length > 0 ? neuralGrid : Array(14).fill(Array(14).fill(0));
  const safeAttnHeads = Array.isArray(attnHeads) && attnHeads.length > 0 ? attnHeads : Array(6).fill(safeNeuralGrid);

  return (
    <div className="dashboard">
      <header className="header">
        <div className="header-left">
          <img src="/rvce_logo.png" alt="RVCE" className="rvce-logo" />
          <div className="title-group">
            <h1>UltraFabric-Vision</h1>
            <p>Real-Time Universal Cloth Defect Detection using Transformer Architectures</p>
          </div>
        </div>
        <div className="header-right">
          <div className="academic-context">
            <span className="project-id">EC367P</span>
            <div className="guide-info">
              <strong>Guide:</strong> Dr. Jyoti Shetty<br/>
              Associate Professor, Dept. of CSE
            </div>
          </div>
        </div>
      </header>

      <nav className="sub-header">
        <div className="nav-tabs">
          <button className={`nav-link ${activeTab === 'live' ? 'active' : ''}`} onClick={() => setActiveTab('live')}>Live Stream Monitoring</button>
          <button className={`nav-link ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}>Offline Batch Analysis</button>
        </div>
        <div className="sys-status">
          <div className="pulse-indicator"></div>
          <span>Engine Status: <strong>{stats.engineState}</strong></span>
        </div>
      </nav>

      <main className="main-content">
        <div className="primary-view">
          {activeTab === 'live' ? (
            <div className="live-grid">
              <div className="stream-controls">
                <div className="source-selector">
                  <select value={streamSource} onChange={(e) => { setStreamSource(e.target.value); setIsStreaming(false); }}>
                    <option value="webcam">Local Webcam</option>
                    <option value="monitor">Remote Monitor (Firebase)</option>
                    <option value="url">Camera URL (IP/RTSP)</option>
                  </select>
                </div>
                {streamSource === 'url' && (
                  <input 
                    type="text" 
                    placeholder="Enter RTSP/HTTP Stream URL..." 
                    className="url-input"
                    value={cameraUrl}
                    onChange={(e) => setCameraUrl(e.target.value)}
                  />
                )}
                <button className={`stream-toggle ${isStreaming ? 'stop' : 'start'}`} onClick={() => setIsStreaming(!isStreaming)}>
                  {isStreaming ? 'STOP STREAM' : 'INITIALIZE ENGINE'}
                </button>
              </div>

              <div className="stream-container">
                <div className="panel-label">RAW INPUT FEED</div>
                <div className="viewport">
                  {rawFrame ? <img src={rawFrame} className="display-feed" alt="Raw Feed" /> : <div className="placeholder">Awaiting Signal...</div>}
                  <video ref={videoRef} autoPlay playsInline style={{ display: 'none' }} />
                </div>
              </div>
              <div className="stream-container">
                <div className="panel-label">NEURAL ATTENTION OUTPUT</div>
                <div className="viewport">
                  {frame ? <img src={frame} className="display-feed" alt="AI Feedback" /> : <div className="placeholder">Ready for Inference</div>}
                  <div className={`status-overlay ${stats.isAnomalous ? 'danger' : 'safe'}`}>
                    {isStreaming ? stats.status : 'ENGINE IDLE'}
                  </div>
                </div>
              </div>
              <div className="trace-container">
                <div className="panel-label">TRANSFORMER BACKEND TRACE</div>
                <div className="terminal">
                  {logs.length === 0 && <div className="log-line muted">Awaiting trace telemetry...</div>}
                  {logs.map((l, i) => (
                    <div key={i} className={`log-line ${l.includes('ERROR') ? 'error' : ''}`}>
                      <span className="cursor-char">{'>'}</span> {l}
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </div>
              </div>
            </div>
          ) : (
            <div className="analysis-grid">
              <div className="upload-hero">
                <h2>Batch Intelligence Processor</h2>
                <p>Industrial off-line inference using high-fidelity DINO-v2 feature matching.</p>
                <div className="actions">
                  <label className="btn-primary">
                    Upload Single Specimen
                    <input type="file" hidden onChange={(e) => handleFileUpload(e, false)} accept="image/*" />
                  </label>
                  <label className="btn-secondary">
                    Batch Process Folder
                    <input type="file" hidden multiple onChange={(e) => handleFileUpload(e, true)} accept="image/*" />
                  </label>
                </div>
              </div>

              {selectedResult && (
                <div className="focused-analysis">
                  <div className="comparison-view">
                    <div className="comp-item">
                      <label className="comp-label">RAW SPECIMEN</label>
                      <img src={selectedResult.raw_data} alt="Original" />
                    </div>
                    <div className="comp-item">
                      <label className="comp-label">NEURAL ANALYTIC</label>
                      <img src={selectedResult.image_data} alt="AI Heatmap" />
                    </div>
                  </div>
                  <div className="focus-stats">
                    <h3>Analysis Intelligence Report: {selectedResult.filename}</h3>
                    <div className="grid-stats">
                      <div className="s-card">
                        <label>Anomaly Score</label>
                        <div className={`val ${selectedResult.is_anomalous ? 'red' : 'green'}`}>{selectedResult.score.toFixed(2)}%</div>
                      </div>
                      <div className="s-card">
                        <label>Inference Time</label>
                        <div className="val">{selectedResult.latency_ms.toFixed(1)}ms</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="batch-scroller">
                {batchResults.map((res, i) => (
                  <div key={i} className={`batch-card ${selectedResult === res ? 'active' : ''}`} onClick={() => { 
                    setSelectedResult(res); 
                    setNeuralGrid(res.neural_grid); 
                    setAttnHeads(res.attn_heads || []);
                    setStats(prev => ({
                      ...prev,
                      status: res.is_anomalous ? 'DEFECT DETECTED' : 'SYSTEM CLEAR',
                      score: res.score,
                      latency: res.latency_ms,
                      isAnomalous: res.is_anomalous
                    }));
                  }}>
                    <img src={res.image_data} alt="thumb" />
                    <div className="info">
                      <span className="fname">{res.filename}</span>
                      <span className={`status ${res.is_anomalous ? 'bad' : 'good'}`}>{res.is_anomalous ? 'DEFECT' : 'NORMAL'}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <aside className="telemetry-sidebar">
          <div className="telemetry-card">
            <div className="card-head">Engine Telemetry</div>
            <div className="telemetry-grid">
              <div className="t-item">
                <label>THROUGHPUT</label>
                <span>{stats.fps} FPS</span>
              </div>
              <div className="t-item">
                <label>LATENCY</label>
                <span>{(stats.latency || 0).toFixed(1)} ms</span>
              </div>
              <div className="t-item">
                <label>GPU LOAD</label>
                <span>{stats.gpuUtil}</span>
              </div>
              <div className="t-item">
                <label>GPU MEMORY</label>
                <span>{stats.gpuMem}</span>
              </div>
            </div>
          </div>

          <div className="telemetry-card">
            <div className="card-head">Transformer Activation Trace</div>
            
            <div className="attention-multi-grid">
              {safeAttnHeads.slice(0, 6).map((head, hi) => (
                <div key={hi} className="head-container">
                  <div className="head-label">HEAD {hi + 1}</div>
                  <div className="mini-attention-grid">
                    {(Array.isArray(head) ? head.flat() : []).map((v, i) => (
                      <div 
                        key={i} 
                        className="attention-cell" 
                        style={{ 
                          backgroundColor: v > 15 ? `rgba(59, 130, 246, ${Math.min(v/100, 0.8)})` : 'transparent'
                        }} 
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="global-attention-section">
               <div className="head-label">ENSEMBLE FUSION (LATENT SPACE)</div>
               <div className="attention-grid large">
                {(Array.isArray(safeNeuralGrid) ? safeNeuralGrid.flat() : []).map((v, i) => (
                  <div 
                    key={i} 
                    className="attention-cell" 
                    style={{ 
                      backgroundColor: v > 15 ? `rgba(239, 68, 68, ${Math.min(v/80, 0.9)})` : 'rgba(0,0,0,0.05)'
                    }} 
                  />
                ))}
              </div>
            </div>

            <div className="attention-legend">
              <span className="dot blue"></span> <span>Attention Weights</span>
              <span className="dot red"></span> <span>Anomaly Gradient</span>
            </div>
          </div>

          <div className="telemetry-card credit-card">
            <div className="card-head">Project Team</div>
            <div className="team-list">
              <div className="student">VARSHINI C B <span>1RV23EE056</span></div>
              <div className="student">NAVI DEEPAK GURUPAD <span>1RV24EC410</span></div>
              <div className="student">AYUSH <span>1RV24CD400</span></div>
              <div className="student">SANJANA T M <span>1RV23EC131</span></div>
            </div>
            <div className="college-footer">
              RV College of Engineering® • Bangalore<br/>
              Interdisciplinary Project 2025-26
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
