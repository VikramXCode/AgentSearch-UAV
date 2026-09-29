import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, RefreshCw, Sparkles, Download, FileCode, Layers, Terminal } from 'lucide-react';

import Header from './components/Header';
import TargetSearchPanel from './components/TargetSearchPanel';
import ImageComparison from './components/ImageComparison';
import DetectedObjectList from './components/DetectedObjectList';
import AIExplanationPanel from './components/AIExplanationPanel';
import TechnicalProcessingLog from './components/TechnicalProcessingLog';
import ZoomModal from './components/ZoomModal';
import BenchmarkModal from './components/BenchmarkModal';
import HistoryModal from './components/HistoryModal';

const API_BASE = 'http://localhost:5005';

export default function App() {
  // Mode: 'image' or 'video'
  const [mode, setMode] = useState('image');

  // Connection state
  const [backendConnected, setBackendConnected] = useState(false);
  const [latency, setLatency] = useState(null);

  // Search input state
  const [query, setQuery] = useState('Find red cars');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedReferenceFile, setSelectedReferenceFile] = useState(null);
  const [referencePreviewUrl, setReferencePreviewUrl] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [videoProgress, setVideoProgress] = useState(null);
  const [error, setError] = useState(null);

  // Benchmarks & Samples
  const [samples, setSamples] = useState([]);
  const [sampleVideos, setSampleVideos] = useState([]);

  // Result state
  const [results, setResults] = useState(null);
  const [pipelineStages, setPipelineStages] = useState(null);
  const [toolsStatus, setToolsStatus] = useState(null);
  const [detectionMedia, setDetectionMedia] = useState(null);

  // Modals
  const [zoomModal, setZoomModal] = useState({ isOpen: false, imageUrl: null, title: '' });
  const [benchmarkModalOpen, setBenchmarkModalOpen] = useState(false);
  const [historyModalOpen, setHistoryModalOpen] = useState(false);

  // Telemetry logs
  const [logs, setLogs] = useState([
    {
      timestamp: new Date().toLocaleTimeString(),
      level: 'info',
      message: 'AgentSearch-UAV AI Command Center initialized',
    },
    {
      timestamp: new Date().toLocaleTimeString(),
      level: 'info',
      message: 'Establishing telemetry connection to Flask backend on port 5005...',
    },
  ]);

  const addLog = useCallback((message, level = 'info') => {
    setLogs((prev) => [
      ...prev,
      {
        timestamp: new Date().toLocaleTimeString(),
        level,
        message,
      },
    ]);
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  // Check Backend Health
  const checkBackendHealth = useCallback(async () => {
    const start = performance.now();
    try {
      const res = await fetch(`${API_BASE}/`, { method: 'GET' });
      if (res.ok) {
        const roundTrip = Math.round(performance.now() - start);
        setBackendConnected(true);
        setLatency(roundTrip);
      } else {
        setBackendConnected(false);
        setLatency(null);
      }
    } catch {
      setBackendConnected(false);
      setLatency(null);
    }
  }, []);

  // Handle selecting a sample image / video
  const handleSelectSample = useCallback(async (sample, sampleType = 'image') => {
    try {
      addLog(`Selected preset benchmark ${sampleType}: ${sample.name}`, 'info');
      const fullUrl = `${API_BASE}${sample.url}`;
      setPreviewUrl(fullUrl);

      const response = await fetch(fullUrl);
      const blob = await response.blob();
      const file = new File(
        [blob],
        sample.name,
        { type: sampleType === 'video' ? 'video/mp4' : (blob.type || 'image/png') }
      );
      setSelectedFile(file);
      setError(null);
    } catch (err) {
      addLog(`Failed to load benchmark payload: ${err.message}`, 'error');
    }
  }, [addLog]);

  // Fetch Sample Images and Videos from Backend
  const fetchSamples = useCallback(async () => {
    try {
      const [imgRes, vidRes] = await Promise.all([
        fetch(`${API_BASE}/samples`),
        fetch(`${API_BASE}/sample-videos`),
      ]);

      if (imgRes.ok) {
        const data = await imgRes.json();
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
          addLog(`Loaded ${data.samples.length} UAV benchmark aerial scenes`, 'info');
          // Auto-select first sample if user hasn't selected a file
          setSelectedFile((currFile) => {
            if (!currFile) {
              handleSelectSample(data.samples[0], 'image');
            }
            return currFile;
          });
        }
      }

      if (vidRes.ok) {
        const vidData = await vidRes.json();
        if (vidData.videos && vidData.videos.length > 0) {
          setSampleVideos(vidData.videos);
          addLog(`Loaded ${vidData.videos.length} UAV aerial surveillance video streams`, 'info');
        }
      }
    } catch (err) {
      console.warn('Could not fetch preset sample files:', err);
    }
  }, [addLog, handleSelectSample]);

  useEffect(() => {
    checkBackendHealth();
    fetchSamples();
    const interval = setInterval(checkBackendHealth, 10000);
    return () => clearInterval(interval);
  }, [checkBackendHealth, fetchSamples]);

  // Handle mode switch defaults
  const handleModeChange = (newMode) => {
    setMode(newMode);
    setResults(null);
    setDetectionMedia(null);
    setSelectedFile(null);
    setPreviewUrl(null);
    setSelectedReferenceFile(null);
    setReferencePreviewUrl(null);
    setError(null);
    if (newMode === 'video') {
      setQuery('car');
      addLog('Mission mode switched to: Continuous Aerial Video Stream Tracking', 'info');
    } else {
      setQuery('Find red cars');
      addLog('Mission mode switched to: Static Reconnaissance Scene Search', 'info');
    }
  };

  // Handle Loading Query from History Modal
  const handleSelectHistoryItem = (item) => {
    if (item.query) {
      setQuery(item.query);
      addLog(`Recalled mission query: "${item.query}" from knowledge agent memory`, 'info');
    }
  };

  // Run Search (Image or Video)
  const handleRunSearch = async () => {
    if (!selectedFile && !previewUrl) {
      setError(`Please upload or select a UAV ${mode} payload to engage detection.`);
      return;
    }
    if (!query.trim() && !selectedReferenceFile) {
      setError('Please provide a search directive query or a reference image.');
      return;
    }

    setIsSearching(true);
    setVideoProgress(null);
    setError(null);
    setResults(null);
    setPipelineStages(null);
    setToolsStatus(null);
    setDetectionMedia(null);

    addLog(`[Mission Ingest] Directive: "${query.trim()}" | Mode: ${mode.toUpperCase()}`, 'info');
    addLog(`[Payload Ingest] ${selectedFile ? selectedFile.name : 'UAV Stream'}`, 'info');

    try {
      const formData = new FormData();
      formData.append('query', query.trim());

      let endpoint = `${API_BASE}/v2/detect`;

      if (mode === 'video') {
        endpoint = `${API_BASE}/detect-video`;
        if (selectedFile) {
          formData.append('video', selectedFile);
        } else if (previewUrl) {
          const response = await fetch(previewUrl);
          const blob = await response.blob();
          formData.append('video', blob, 'uav_video.mp4');
        }
      } else {
        if (selectedFile) {
          formData.append('image', selectedFile);
        } else if (previewUrl) {
          const response = await fetch(previewUrl);
          const blob = await response.blob();
          formData.append('image', blob, 'uav_frame.png');
        }
      }

      if (selectedReferenceFile) {
        formData.append('reference_image', selectedReferenceFile);
      } else if (referencePreviewUrl) {
        const response = await fetch(referencePreviewUrl);
        const blob = await response.blob();
        formData.append('reference_image', blob, 'ref_image.png');
      }

      const startTime = performance.now();
      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.message || `Server responded with HTTP ${res.status}`);
      }

      let data = await res.json();

      // If video job is queued, poll for progress until completion
      if (mode === 'video' && data.job_id && data.status === 'processing') {
        const jobId = data.job_id;
        const progressEndpoint = data.progress_url
          ? `${API_BASE}${data.progress_url}`
          : `${API_BASE}/video-progress/${jobId}`;
        addLog(`[Video Stream Processing] Background job engaged: ${jobId}`, 'info');

        let isCompleted = false;
        let lastLoggedPct = -1;

        while (!isCompleted) {
          await new Promise((resolve) => setTimeout(resolve, 350));
          try {
            const pollRes = await fetch(progressEndpoint);
            if (!pollRes.ok) {
              // Try fallback progress endpoint if needed
              const fallbackUrl = progressEndpoint.includes('/v2/')
                ? `${API_BASE}/video-progress/${jobId}`
                : `${API_BASE}/v2/video-progress/${jobId}`;
              const fallbackRes = await fetch(fallbackUrl);
              if (!fallbackRes.ok) continue;
              const fallbackStatus = await fallbackRes.json();
              if (fallbackStatus && fallbackStatus.status) {
                if (fallbackStatus.status === 'completed') {
                  isCompleted = true;
                  data = fallbackStatus.result;
                  break;
                } else if (fallbackStatus.status === 'failed') {
                  throw new Error(fallbackStatus.error || 'Video tracking failed');
                }
                setVideoProgress({
                  percent: fallbackStatus.progress || 0,
                  currentFrame: fallbackStatus.current_frame || 0,
                  totalFrames: fallbackStatus.total_frames || 0,
                  fps: fallbackStatus.fps || 0,
                  stage: fallbackStatus.stage || 'Tracking targets in video stream...',
                });
              }
              continue;
            }

            const jobStatus = await pollRes.json();
            setVideoProgress({
              percent: jobStatus.progress || 0,
              currentFrame: jobStatus.current_frame || 0,
              totalFrames: jobStatus.total_frames || 0,
              fps: jobStatus.fps || 0,
              stage: jobStatus.stage || 'Tracking targets in video stream...',
            });

            // Periodically log milestone progress to tactical log
            const currentPct = Math.round(jobStatus.progress || 0);
            if (currentPct > 0 && currentPct % 25 === 0 && currentPct !== lastLoggedPct) {
              lastLoggedPct = currentPct;
              addLog(`[Video Progress] ${currentPct}% processed (${jobStatus.current_frame}/${jobStatus.total_frames} frames @ ${jobStatus.fps} FPS)`, 'info');
            }

            if (jobStatus.status === 'completed') {
              isCompleted = true;
              data = jobStatus.result;
            } else if (jobStatus.status === 'failed') {
              throw new Error(jobStatus.error || 'Video tracking failed');
            }
          } catch (pollErr) {
            if (pollErr.message && !pollErr.message.includes('fetch')) {
              throw pollErr;
            }
          }
        }
      }

      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

      addLog(`[Inference Complete] Pipeline executed in ${elapsed}s`, 'success');
      addLog(
        `[Consensus Reached] Verified ${data?.count ?? data?.objects_found ?? (data?.detections?.length || 0)} target object(s)`,
        'success'
      );

      if (data?.explanation?.reasoning) {
        if (Array.isArray(data.explanation.reasoning)) {
          data.explanation.reasoning.forEach((step) =>
            addLog(`[Agent Reason] ${step}`, 'agent')
          );
        } else {
          addLog(`[Agent Reason] ${data.explanation.reasoning}`, 'agent');
        }
      }

      setResults(data);
      setPipelineStages(data?.pipeline);
      setToolsStatus(data?.tools_status || data?.tools);

      // Set Detection Output media
      if (mode === 'video') {
        const vidPath = data?.output_video || data?.annotated_video_url;
        const videoOutput = vidPath
          ? (vidPath.startsWith('http') ? vidPath : `${API_BASE}${vidPath}`)
          : `${API_BASE}/result-video?t=${Date.now()}`;
        setDetectionMedia(videoOutput);
      } else {
        const imgPath = data?.annotated_image_url;
        const resultImageUrl = imgPath 
          ? (imgPath.startsWith('http') ? imgPath : `${API_BASE}${imgPath}?t=${Date.now()}`)
          : `${API_BASE}/result?t=${Date.now()}`;
        setDetectionMedia(resultImageUrl);
      }

      // Smoothly scroll down to output comparison section only when result is received
      setTimeout(() => {
        document.getElementById('section-comparison')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      console.error('Detection error:', err);
      setError(`Inference failed: ${err.message}. Verify that the Python backend is active.`);
      addLog(`[Error] Execution aborted: ${err.message}`, 'error');
    } finally {
      setIsSearching(false);
    }
  };

  // Export AgentState JSON
  const handleExportState = () => {
    if (!results) return;
    const jsonStr = JSON.stringify(results, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agentsearch_state_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    addLog('Exported current AgentState telemetry JSON file', 'info');
  };

  const handleOpenZoom = (imageUrl, title) => {
    setZoomModal({ isOpen: true, imageUrl, title });
  };

  const handleCloseZoom = () => {
    setZoomModal({ isOpen: false, imageUrl: null, title: '' });
  };

  return (
    <div className="min-h-screen bg-[#080b11] text-slate-200 flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-200">
      {/* Top Tactical Command Header */}
      <Header
        backendConnected={backendConnected}
        latency={latency}
        mode={mode}
        setMode={handleModeChange}
        onOpenBenchmark={() => setBenchmarkModalOpen(true)}
        onOpenHistory={() => setHistoryModalOpen(true)}
      />

      {/* Main Mission Control Center */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Backend Disconnected Warning Banner */}
        {!backendConnected && (
          <div className="rounded-xl p-4 bg-rose-950/40 border border-rose-800/60 text-rose-300 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 font-mono">
            <div className="flex items-center gap-2.5">
              <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0" />
              <div>
                <span className="text-xs font-bold text-rose-200 uppercase">
                  AGENTSEARCH CORE DISCONNECTED
                </span>
                <p className="text-[11px] text-rose-400/90 mt-0.5">
                  Backend unreachable at <code className="bg-rose-950/80 px-1 py-0.2 rounded text-rose-200">http://localhost:5005</code>.
                  Ensure <code className="text-rose-200 font-bold">python -m api.main</code> is running.
                </p>
              </div>
            </div>
            <button
              onClick={checkBackendHealth}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-900/60 hover:bg-rose-800/80 text-rose-200 border border-rose-700/60 transition-colors flex items-center gap-1.5 flex-shrink-0"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Retry Connection
            </button>
          </div>
        )}

        {/* Global Error Banner */}
        {error && (
          <div className="rounded-xl p-3.5 bg-amber-950/40 border border-amber-800/60 text-amber-300 flex items-center justify-between gap-3 font-mono text-xs">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
              <span>{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-[11px] text-amber-400 hover:text-amber-200 underline cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* SECTION 1 — TARGET SEARCH & INPUT DIRECTIVE */}
        <section id="section-search">
          <TargetSearchPanel
            query={query}
            setQuery={setQuery}
            selectedFile={selectedFile}
            setSelectedFile={setSelectedFile}
            previewUrl={previewUrl}
            setPreviewUrl={setPreviewUrl}
            selectedReferenceFile={selectedReferenceFile}
            setSelectedReferenceFile={setSelectedReferenceFile}
            referencePreviewUrl={referencePreviewUrl}
            setReferencePreviewUrl={setReferencePreviewUrl}
            isSearching={isSearching}
            onRunSearch={handleRunSearch}
            samples={samples}
            sampleVideos={sampleVideos}
            onSelectSample={handleSelectSample}
            mode={mode}
            setMode={handleModeChange}
            videoProgress={videoProgress}
          />
        </section>

        {/* SECTION 2 — INTERACTIVE IMAGE SLIDER & DETECTION RESULTS */}
        <section id="section-comparison">
          <ImageComparison
            originalMedia={previewUrl}
            resultMedia={detectionMedia}
            isSearching={isSearching}
            query={query}
            results={results}
            onOpenZoom={handleOpenZoom}
            mode={mode}
          />
        </section>

        {/* SECTION 3 — VERIFIED TARGETS & TRAJECTORY MATRIX */}
        {(results?.detections?.length > 0 || isSearching) && (
          <section id="section-matrix">
            <DetectedObjectList
              detections={results?.detections}
              isSearching={isSearching}
              mode={mode}
            />
          </section>
        )}

        {/* SECTION 4 — AI EXPLANATION & DECISION REASONING */}
        <section id="section-explanation">
          <AIExplanationPanel
            explanation={results?.explanation}
            results={results}
            isSearching={isSearching}
            mode={mode}
          />
        </section>

        {/* Advanced Developer & Telemetry Tools (Collapsed by default) */}
        <details className="group rounded-xl border border-slate-800/80 bg-slate-950/40 p-4 font-mono text-xs transition-all">
          <summary className="cursor-pointer text-slate-400 hover:text-slate-200 flex items-center justify-between select-none py-1">
            <div className="flex items-center gap-2 font-semibold text-slate-300">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <span>Advanced Telemetry & Diagnostics (Optional)</span>
            </div>
            <span className="text-[11px] text-slate-500 group-open:rotate-180 transition-transform">▼</span>
          </summary>

          <div className="mt-4 pt-4 border-t border-slate-800/80 space-y-4">
            {results && (
              <div className="flex items-center justify-end">
                <button
                  onClick={handleExportState}
                  className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-mono font-semibold text-cyan-300 flex items-center gap-2 transition-all shadow-sm"
                >
                  <FileCode className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Export AgentState Telemetry JSON</span>
                </button>
              </div>
            )}
            <TechnicalProcessingLog logs={logs} onClearLogs={clearLogs} />
          </div>
        </details>
      </main>

      {/* Command Center Footer */}
      <footer className="border-t border-slate-800/80 bg-[#090d15] py-4 text-center text-xs text-slate-400 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2.5">
          <div className="text-slate-400">
            AgentSearch-UAV • Autonomous Multi-Agent Aerial Target Search System
          </div>
          <div className="flex flex-wrap items-center justify-center gap-3 text-[11px] text-slate-400">
            <span>VisDrone-Adapted YOLO-World</span>
            <span>•</span>
            <span>Tiled SAHI Slicing</span>
            <span>•</span>
            <span>Real-ESRGAN Upscaling</span>
            <span>•</span>
            <span>CLIP Semantic Validation</span>
          </div>
        </div>
      </footer>

      {/* Fullscreen High-Res Zoom Inspection Modal */}
      <ZoomModal
        isOpen={zoomModal.isOpen}
        onClose={handleCloseZoom}
        imageUrl={zoomModal.imageUrl}
        title={zoomModal.title}
      />

      {/* VisDrone Benchmark Modal */}
      <BenchmarkModal
        isOpen={benchmarkModalOpen}
        onClose={() => setBenchmarkModalOpen(false)}
      />

      {/* Mission History Modal */}
      <HistoryModal
        isOpen={historyModalOpen}
        onClose={() => setHistoryModalOpen(false)}
        onSelectHistoryItem={handleSelectHistoryItem}
      />
    </div>
  );
}
