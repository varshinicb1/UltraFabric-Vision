import React, { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('analysis');
  const [frame, setFrame] = useState(null);
  const [rawFrame, setRawFrame] = useState(null);
  const [neuralGrid, setNeuralGrid] = useState(Array(14).fill(Array(14).fill(0)));
  const [attnHeads, setAttnHeads] = useState([]);
  const [selectedHead, setSelectedHead] = useState(0);
  const [logs, setLogs] = useState([]);
  const [batchResults, setBatchResults] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const [streamSource, setStreamSource] = useState('webcam');
  const [cameraUrl, setCameraUrl] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [urlError, setUrlError] = useState('');
  // Inference mode: 'accurate' = full ensemble (best accuracy); 'fast' = single
  // detector (low latency). Sent to the backend via ?mode= on every request.
  const [mode, setMode] = useState('accurate');
  // Batch video analysis
  const [videoBatch, setVideoBatch] = useState('B001');
  const [videoMeters, setVideoMeters] = useState(5);
  const [videoReport, setVideoReport] = useState(null);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [batchHistory, setBatchHistory] = useState([]);
  const API = 'http://localhost:8000';
  
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
  const terminalRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const frameTimerRef = useRef(null);

  const isValidStreamUrl = (url) => {
    try {
      const u = new URL(url);
      return ['http:', 'https:', 'rtsp:', 'rtmp:', 'ftp:'].includes(u.protocol);
    } catch { return false; }
  };

  const addLog = useCallback((msg) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-15), `[${timestamp}] ${msg}`]);
  }, []);

  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
    if (socketRef.current) { socketRef.current.close(); socketRef.current = null; }
    if (frameTimerRef.current) { clearInterval(frameTimerRef.current); frameTimerRef.current = null; }
    if (reconnectTimerRef.current) { clearTimeout(reconnectTimerRef.current); reconnectTimerRef.current = null; }
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  const handleToggleStream = useCallback(() => {
    if (isStreaming) {
      stopStreaming();
      return;
    }
    if (streamSource === 'url') {
      if (!cameraUrl.trim()) {
        setUrlError('Please enter a camera stream URL');
        return;
      }
      if (!isValidStreamUrl(cameraUrl.trim())) {
        setUrlError('Invalid URL. Use rtsp://, http://, or https://');
        return;
      }
      setUrlError('');
    }
    setLogs([]);
    setIsStreaming(true);
  }, [isStreaming, streamSource, cameraUrl, stopStreaming]);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  // WebSocket Setup for Live Stream
  useEffect(() => {
    if (activeTab !== 'live' || !isStreaming) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let socketUrl;
    if (streamSource === 'webcam') {
      socketUrl = `${protocol}//${window.location.hostname}:8000/ws/stream?mode=${mode}`;
    } else if (streamSource === 'monitor') {
      socketUrl = `${protocol}//${window.location.hostname}:8000/ws/subscribe`;
    } else {
      if (!cameraUrl) {
        addLog('ERROR: No camera URL provided');
        setIsStreaming(false);
        return;
      }
      socketUrl = `${protocol}//${window.location.hostname}:8000/ws/remote_stream?url=${encodeURIComponent(cameraUrl)}&mode=${mode}`;
    }

    addLog(`Connecting to ${streamSource}...`);
    let ws = new WebSocket(socketUrl);
    socketRef.current = ws;
    
    ws.onopen = () => {
      addLog('WebSocket connected');
      setStats(prev => ({ ...prev, engineState: 'Active' }));

      if (streamSource === 'webcam') {
        navigator.mediaDevices.getUserMedia({ video: { width: 448, height: 448 } })
          .then(stream => { if (videoRef.current) videoRef.current.srcObject = stream; })
          .catch(err => {
            addLog(`Webcam error: ${err.message}`);
          });
      }
    };
    
    ws.onmessage = (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
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
        data.trace.forEach(t => addLog(t));
      }
    };

    ws.onerror = () => {
      addLog('WebSocket connection error');
    };

    ws.onclose = () => {
      addLog('WebSocket disconnected');
      setStats(prev => ({ ...prev, engineState: 'Idle' }));
      // Auto-reconnect for URL/remote modes
      if (streamSource !== 'webcam') {
        reconnectTimerRef.current = setTimeout(() => {
          if (isStreaming) {
            addLog('Reconnecting...');
            setIsStreaming(false);
            setTimeout(() => setIsStreaming(true), 100);
          }
        }, 3000);
      }
    };

    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      ws.close();
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, [activeTab, isStreaming, streamSource, cameraUrl, mode]);

  // Webcam Frame Capture Loop
  useEffect(() => {
    if (activeTab !== 'live' || streamSource !== 'webcam' || !isStreaming) return;
    if (frameTimerRef.current) clearInterval(frameTimerRef.current);
    frameTimerRef.current = setInterval(() => {
      const ws = socketRef.current;
      const vid = videoRef.current;
      if (vid && vid.readyState >= 2 && ws?.readyState === WebSocket.OPEN) {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = 224; canvas.height = 224;
          canvas.getContext('2d').drawImage(vid, 0, 0, 224, 224);
          const base64 = canvas.toDataURL('image/jpeg', 0.8);
          setRawFrame(base64);
          ws.send(base64);
        } catch (err) {
          addLog(`Frame error: ${err.message}`);
        }
      }
    }, 200);
    return () => { if (frameTimerRef.current) clearInterval(frameTimerRef.current); };
  }, [activeTab, streamSource, isStreaming]);

  const handleFileUpload = async (e, isBatch = false) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    setIsUploading(true);
    const formData = new FormData();
    
    if (isBatch) {
      files.forEach(file => formData.append('files', file));
      try {
        const res = await fetch(`http://localhost:8000/api/batch_upload?mode=${mode}`, { method: 'POST', body: formData });
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
        const res = await fetch(`http://localhost:8000/api/upload_image?mode=${mode}`, { method: 'POST', body: formData });
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

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/batch_history`);
      const data = await res.json();
      setBatchHistory(data.batches || []);
    } catch { /* backend not up yet */ }
  }, []);

  useEffect(() => { if (activeTab === 'video') fetchHistory(); }, [activeTab, fetchHistory]);

  const handleVideoUpload = async (e) => {
    const fileList = e.target.files;
    if (!fileList || !fileList.length) return;
    setIsProcessingVideo(true);
    setVideoReport(null);
    const formData = new FormData();
    formData.append('file', fileList[0]);
    const qs = `batch=${encodeURIComponent(videoBatch)}&meters=${videoMeters}&segments=10&mode=${mode}`;
    try {
      const res = await fetch(`${API}/api/upload_video?${qs}`, { method: 'POST', body: formData });
      const data = await res.json();
      if (data.error) { console.error(data.error); }
      else { setVideoReport(data); fetchHistory(); }
    } catch (err) { console.error('Video inspection failed:', err); }
    setIsProcessingVideo(false);
    e.target.value = '';
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
              <strong>Guide:</strong> Dr. Jyothi Shetty<br/>
              Associate Professor, Dept. of CSE
            </div>
          </div>
        </div>
      </header>

      <nav className="sub-header">
        <div className="nav-tabs">
          <button className={`nav-link ${activeTab === 'live' ? 'active' : ''}`} onClick={() => setActiveTab('live')}>Live Stream Monitoring</button>
          <button className={`nav-link ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}>Offline Batch Analysis</button>
          <button className={`nav-link ${activeTab === 'video' ? 'active' : ''}`} onClick={() => setActiveTab('video')}>Batch Video Inspection</button>
        </div>
        <div className="sys-status">
          <div className="mode-toggle" role="group" aria-label="Inference mode"
               style={{ display: 'inline-flex', gap: 4, marginRight: 16, padding: 3,
                        background: 'rgba(148,163,184,0.15)', borderRadius: 8 }}>
            {['accurate', 'fast'].map(m => (
              <button key={m} onClick={() => setMode(m)}
                title={m === 'accurate' ? 'Full ensemble — highest accuracy' : 'Single detector — lowest latency'}
                style={{
                  cursor: 'pointer', border: 'none', borderRadius: 6,
                  padding: '4px 12px', fontSize: 12, fontWeight: 600,
                  textTransform: 'capitalize',
                  background: mode === m ? (m === 'fast' ? '#f59e0b' : '#3b82f6') : 'transparent',
                  color: mode === m ? '#fff' : '#94a3b8',
                }}>
                {m === 'fast' ? '⚡ Fast' : '◎ Accurate'}
              </button>
            ))}
          </div>
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
                  <select value={streamSource} onChange={(e) => {
                    setStreamSource(e.target.value);
                    setIsStreaming(false);
                    setUrlError('');
                    stopStreaming();
                  }}>
                    <option value="webcam">Local Webcam</option>
                    <option value="url">Camera URL (IP/RTSP)</option>
                    <option value="monitor">Remote Monitor</option>
                  </select>
                </div>
                {streamSource === 'url' && (
                  <div className="url-input-group">
                    <input 
                      type="text" 
                      placeholder="rtsp://192.168.1.100:554/stream or http://..." 
                      className={`url-input ${urlError ? 'has-error' : ''}`}
                      value={cameraUrl}
                      onChange={(e) => { setCameraUrl(e.target.value); setUrlError(''); }}
                      onKeyDown={(e) => { if (e.key === 'Enter' && cameraUrl) handleToggleStream(); }}
                    />
                    {urlError && <span className="url-error-msg">{urlError}</span>}
                  </div>
                )}
                <button className={`stream-toggle ${isStreaming ? 'stop' : 'start'}`} onClick={handleToggleStream}>
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
                <div className="terminal" ref={terminalRef}>
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
          ) : activeTab === 'analysis' ? (
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
          ) : (
            <div className="video-analysis" style={{ padding: 4 }}>
              <div className="upload-hero">
                <h2>Batch Video Inspection</h2>
                <p>Upload a recorded fabric-batch video (e.g. from the conveyor). The engine
                   scores every frame and reports which segment of the batch is defective.</p>
                <div className="actions" style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
                  <label style={{ fontSize: 12, color: '#94a3b8' }}>Batch No.
                    <input value={videoBatch} onChange={e => setVideoBatch(e.target.value)}
                      style={{ display: 'block', marginTop: 4, padding: '6px 8px', borderRadius: 6, border: '1px solid #334155', background: '#0f172a', color: '#e2e8f0', width: 120 }} />
                  </label>
                  <label style={{ fontSize: 12, color: '#94a3b8' }}>Batch length (m)
                    <input type="number" min="0.5" step="0.5" value={videoMeters} onChange={e => setVideoMeters(parseFloat(e.target.value) || 5)}
                      style={{ display: 'block', marginTop: 4, padding: '6px 8px', borderRadius: 6, border: '1px solid #334155', background: '#0f172a', color: '#e2e8f0', width: 120 }} />
                  </label>
                  <label className="btn-primary" style={{ opacity: isProcessingVideo ? 0.6 : 1 }}>
                    {isProcessingVideo ? 'Processing…' : 'Upload & Inspect Video'}
                    <input type="file" hidden accept="video/*" disabled={isProcessingVideo} onChange={handleVideoUpload} />
                  </label>
                  <span style={{ fontSize: 12, color: '#64748b' }}>Mode: <strong style={{ color: mode === 'fast' ? '#f59e0b' : '#3b82f6' }}>{mode}</strong></span>
                </div>
              </div>

              {batchHistory.length > 0 && (
                <div style={{ marginTop: 8, marginBottom: 4 }}>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>INSPECTED BATCHES ({batchHistory.length})</div>
                  <div style={{ display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 4 }}>
                    {batchHistory.map((b) => (
                      <div key={b.seq} onClick={() => setVideoReport(b)}
                        style={{ cursor: 'pointer', minWidth: 130, padding: '8px 10px', borderRadius: 8,
                                 border: `1px solid ${b.passed ? '#14532d' : '#7f1d1d'}`,
                                 background: b.passed ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)' }}>
                        <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 13 }}>{b.batch}</div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: b.passed ? '#22c55e' : '#ef4444' }}>
                          {b.passed ? '✓ PASS' : '✗ DEFECT'}
                        </div>
                        <div style={{ fontSize: 10, color: '#64748b' }}>
                          {b.passed ? 'clean' : `zones ${b.zones_with_defects.join(',')}`}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {isProcessingVideo && <div style={{ padding: 20, color: '#94a3b8' }}>Running inference on every frame… this runs on the GPU and may take a few seconds per batch.</div>}

              {videoReport && (
                <div className="video-report" style={{ marginTop: 16 }}>
                  <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 14 }}>
                    {[['Batch', videoReport.batch],
                      ['Result', (videoReport.passed ?? (videoReport.defect_frames === 0)) ? '✓ PASS' : '✗ DEFECT'],
                      ['Defect frames', `${videoReport.defect_frames}/${videoReport.processed_frames} (${(((videoReport.defect_rate ?? (videoReport.defect_frames/Math.max(1,videoReport.processed_frames)))*100)).toFixed(0)}%)`],
                      ['Defective zones', videoReport.zones_with_defects.length ? videoReport.zones_with_defects.join(', ') : 'none']].map(([k,v]) => (
                      <div key={k} className="s-card" style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: '10px 14px', minWidth: 130 }}>
                        <label style={{ fontSize: 11, color: '#64748b' }}>{k}</label>
                        <div style={{ fontWeight: 700, color: '#e2e8f0' }}>{String(v)}</div>
                      </div>
                    ))}
                  </div>

                  <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>DEFECT LOCATION MAP (along {videoReport.batch_length_m} m batch)</div>
                  <img src={videoReport.defect_map} alt="defect map" style={{ width: '100%', borderRadius: 8, border: '1px solid #1e293b' }} />

                  {videoReport.annotated_video && (
                    <div style={{ marginTop: 12 }}>
                      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>ANNOTATED VIDEO (boxes + position overlay)</div>
                      <video src={`${API}${videoReport.annotated_video}`} controls
                             style={{ width: '100%', maxWidth: 420, borderRadius: 8, border: '1px solid #1e293b', background: '#000' }} />
                      <div><a href={`${API}${videoReport.annotated_video}`} download
                             style={{ color: '#3b82f6', fontSize: 12 }}>⬇ Download annotated video</a></div>
                    </div>
                  )}

                  {videoReport.defect_events.length > 0 && (
                    <div style={{ marginTop: 14 }}>
                      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>DEFECT EVENTS</div>
                      {videoReport.defect_events.map((ev, i) => (
                        <div key={i} style={{ fontSize: 13, color: '#fca5a5', padding: '4px 0' }}>
                          ▸ Defect at <strong>{ev.start_m}–{ev.end_m} m</strong> (zones {ev.zones.join(', ')}), peak score {ev.max_score}
                        </div>
                      ))}
                    </div>
                  )}

                  <div style={{ marginTop: 16 }}>
                    <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>SEGMENT REPORT</div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                        <thead><tr style={{ color: '#64748b', textAlign: 'left' }}>
                          <th style={{ padding: 6 }}>Zone</th><th style={{ padding: 6 }}>Position (m)</th>
                          <th style={{ padding: 6 }}>Status</th><th style={{ padding: 6 }}>Defect frames</th><th style={{ padding: 6 }}>Max score</th>
                        </tr></thead>
                        <tbody>
                          {videoReport.segment_summary.map(z => (
                            <tr key={z.zone} style={{ borderTop: '1px solid #1e293b', color: z.status === 'DEFECT' ? '#fca5a5' : '#94a3b8' }}>
                              <td style={{ padding: 6 }}>{z.zone}</td><td style={{ padding: 6 }}>{z.position_m}</td>
                              <td style={{ padding: 6, fontWeight: z.status === 'DEFECT' ? 700 : 400 }}>{z.status}</td>
                              <td style={{ padding: 6 }}>{z.defect_frames}</td><td style={{ padding: 6 }}>{z.max_score}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
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
