import React, { useState, useRef, useEffect } from 'react';
import {
  Download,
  Maximize2,
  Layers,
  CheckCircle2,
  Sparkles,
  Crosshair,
  Sliders,
  Columns,
  Eye,
  Film,
} from 'lucide-react';
import { motion } from 'framer-motion';

export default function ImageComparison({
  originalMedia,
  resultMedia,
  isSearching,
  query,
  results,
  onOpenZoom,
  mode = 'image',
}) {
  const [viewMode, setViewMode] = useState('split'); // 'split', 'side-by-side', 'original', 'result'
  const [sliderPos, setSliderPos] = useState(50);
  const [cursorCoords, setCursorCoords] = useState({ x: 0, y: 0, show: false });
  const containerRef = useRef(null);

  const detectionCount =
    results?.count ?? results?.objects_found ?? (results?.detections?.length || 0);

  const avgConfidence = results?.average_confidence
    ? (results.average_confidence * 100).toFixed(1)
    : results?.confidence
    ? (results.confidence * 100).toFixed(1)
    : '85.0';

  const downloadResult = () => {
    if (!resultMedia) return;
    const a = document.createElement('a');
    a.href = resultMedia;
    a.download = `agentsearch_uav_${(query || 'target').replace(/\s+/g, '_')}_result.${
      mode === 'video' ? 'mp4' : 'jpg'
    }`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleMouseMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top));

    const pct = (x / rect.width) * 100;
    if (viewMode === 'split' && e.buttons === 1) {
      setSliderPos(pct);
    }

    setCursorCoords({
      x: Math.round(x),
      y: Math.round(y),
      show: true,
    });
  };

  const handleMouseLeave = () => {
    setCursorCoords((prev) => ({ ...prev, show: false }));
  };

  return (
    <div className="cmd-panel rounded-xl p-5 sm:p-6 border border-slate-800/90 space-y-4">
      {/* Header bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400">
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm sm:text-base font-bold text-white tracking-tight font-mono uppercase">
                {mode === 'video'
                  ? 'Aerial Video Stream Tracking Telemetry'
                  : 'Surveillance Imagery Comparison Console'}
              </h2>
              {results && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/40">
                  {detectionCount} Verified Target(s)
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              {mode === 'video'
                ? 'Dual-stream visual inspection & spatial trajectory verification'
                : 'Raw aerial feed vs. query-verified bounding box detections'}
            </p>
          </div>
        </div>

        {/* View Mode Controls */}
        <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto justify-between sm:justify-end">
          {resultMedia && !isSearching && (
            <button
              onClick={downloadResult}
              className="px-2.5 py-1.5 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-mono text-slate-200 hover:text-white flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5 text-cyan-400" />
              <span>Export {mode === 'video' ? 'Video' : 'Payload'}</span>
            </button>
          )}

          {/* View Mode Switcher */}
          <div className="flex items-center p-0.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono">
            {mode !== 'video' && originalMedia && resultMedia && (
              <button
                onClick={() => setViewMode('split')}
                className={`px-2.5 py-1 rounded transition-all ${
                  viewMode === 'split'
                    ? 'bg-slate-800 text-cyan-300 font-bold border border-slate-700 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Slider View
              </button>
            )}
            <button
              onClick={() => setViewMode('side-by-side')}
              className={`px-2.5 py-1 rounded transition-all ${
                viewMode === 'side-by-side'
                  ? 'bg-slate-800 text-cyan-300 font-bold border border-slate-700 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Dual Panel
            </button>
            <button
              onClick={() => setViewMode('original')}
              className={`px-2.5 py-1 rounded transition-all ${
                viewMode === 'original'
                  ? 'bg-slate-800 text-cyan-300 font-bold border border-slate-700 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Raw
            </button>
            <button
              onClick={() => setViewMode('result')}
              className={`px-2.5 py-1 rounded transition-all ${
                viewMode === 'result'
                  ? 'bg-slate-800 text-cyan-300 font-bold border border-slate-700 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Verified
            </button>
          </div>
        </div>
      </div>

      {/* Main Visual Display Area */}
      {viewMode === 'split' && mode !== 'video' && originalMedia && resultMedia && !isSearching ? (
        <div className="space-y-2">
          {/* Interactive Split Slider */}
          <div
            ref={containerRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            className="relative aspect-video w-full rounded-lg overflow-hidden border border-slate-800 bg-black flex items-center justify-center select-none cursor-ew-resize shadow-inner"
          >
            {/* Base: Result Detection Image */}
            <img
              src={resultMedia}
              alt="Detected Target"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            />

            {/* Overlay: Original Image clipped by sliderPos */}
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none border-r-2 border-cyan-400"
              style={{ width: `${sliderPos}%` }}
            >
              <img
                src={originalMedia}
                alt="Original UAV"
                className="absolute inset-0 w-full h-full object-contain max-w-none"
                style={{ width: containerRef.current?.clientWidth || '100%', height: '100%' }}
              />
              <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/80 border border-slate-700 text-[10px] font-mono text-slate-300">
                RAW AERIAL FEED
              </div>
            </div>

            {/* Top Right Result Tag */}
            <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-black/80 border border-emerald-500/40 text-[10px] font-mono text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" />
              VERIFIED TARGETS ({detectionCount})
            </div>

            {/* Slider Handle Line */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 pointer-events-none shadow-[0_0_10px_rgba(6,182,212,0.8)]"
              style={{ left: `${sliderPos}%` }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-slate-900 border-2 border-cyan-400 flex items-center justify-center text-cyan-300 text-[9px] font-mono shadow-md">
                ◀▶
              </div>
            </div>

            {/* Reticle HUD coordinates readout */}
            {cursorCoords.show && (
              <div className="absolute bottom-2 left-2 px-2 py-1 rounded bg-black/85 border border-slate-800 text-[10px] font-mono text-cyan-400 flex items-center gap-2">
                <Crosshair className="w-3 h-3" />
                <span>COORD: X:{cursorCoords.x} Y:{cursorCoords.y}</span>
                <span className="text-slate-600">|</span>
                <span>SPLIT: {Math.round(sliderPos)}%</span>
              </div>
            )}
          </div>
          <p className="text-[11px] font-mono text-slate-400 text-center">
            Drag cursor horizontally across frame to inspect pre/post agent verification
          </p>
        </div>
      ) : (
        /* Dual Panel / Single Panel Layout */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Left Panel: Raw Media */}
          {(viewMode === 'side-by-side' || viewMode === 'original') && (
            <div className={`space-y-2 ${viewMode === 'original' ? 'lg:col-span-2' : ''}`}>
              <div className="flex items-center justify-between text-xs font-mono uppercase tracking-wider text-slate-400">
                <span className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
                  {mode === 'video' ? 'RAW SURVEILLANCE VIDEO' : 'RAW AERIAL INPUT'}
                </span>
                {originalMedia && mode !== 'video' && (
                  <button
                    onClick={() => onOpenZoom(originalMedia, 'Raw UAV Aerial Feed')}
                    className="hover:text-cyan-400 flex items-center gap-1 transition-colors text-[11px]"
                  >
                    <Maximize2 className="w-3 h-3" /> Inspect
                  </button>
                )}
              </div>

              <div className="relative aspect-video w-full rounded-lg overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center shadow-inner">
                {originalMedia ? (
                  mode === 'video' ? (
                    <video
                      src={originalMedia}
                      controls
                      className="w-full h-full object-contain"
                      loop
                      muted
                    />
                  ) : (
                    <>
                      <img
                        src={originalMedia}
                        alt="Raw Aerial UAV"
                        className="w-full h-full object-contain"
                      />
                      <div className="absolute inset-0 bg-black/40 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                        <button
                          onClick={() => onOpenZoom(originalMedia, 'Raw UAV Aerial Feed')}
                          className="px-3 py-1 rounded bg-slate-900/90 text-xs font-mono text-white border border-slate-700 pointer-events-auto cursor-pointer"
                        >
                          <Eye className="w-3.5 h-3.5 inline mr-1 text-cyan-400" /> Fullscreen View
                        </button>
                      </div>
                    </>
                  )
                ) : (
                  <div className="text-center p-6 text-slate-500 font-mono text-xs">
                    No aerial payload ingested yet
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Right Panel: Verified Detection / Tracked Output */}
          {(viewMode === 'side-by-side' || viewMode === 'result') && (
            <div className={`space-y-2 ${viewMode === 'result' ? 'lg:col-span-2' : ''}`}>
              <div className="flex items-center justify-between text-xs font-mono uppercase tracking-wider text-slate-400">
                <span className="flex items-center gap-1.5 text-emerald-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  {mode === 'video' ? 'TRACKED VIDEO OUTPUT' : 'VERIFIED TARGET DETECTION'}
                  {results && (
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-500/40">
                      {detectionCount} Targets ({avgConfidence}%)
                    </span>
                  )}
                </span>
                {resultMedia && mode !== 'video' && !isSearching && (
                  <button
                    onClick={() => onOpenZoom(resultMedia, `Verified Detections: ${query}`)}
                    className="hover:text-emerald-400 flex items-center gap-1 transition-colors text-[11px]"
                  >
                    <Maximize2 className="w-3 h-3" /> Inspect
                  </button>
                )}
              </div>

              <div className="relative aspect-video w-full rounded-lg overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center shadow-inner">
                {isSearching ? (
                  <div className="flex flex-col items-center justify-center p-6 text-center space-y-2.5">
                    <div className="w-10 h-10 rounded-full border-2 border-cyan-500/30 border-t-cyan-400 animate-spin" />
                    <div>
                      <p className="text-xs font-bold font-mono text-white uppercase tracking-wider">
                        {mode === 'video' ? 'Computing Trajectory Tracks' : 'Executing Vision Inference'}
                      </p>
                      <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                        YOLO-World • SAHI • Attribute Filter
                      </p>
                    </div>
                  </div>
                ) : resultMedia ? (
                  mode === 'video' ? (
                    <video
                      src={resultMedia}
                      controls
                      autoPlay
                      loop
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <>
                      <img
                        src={resultMedia}
                        alt="Detection Result"
                        className="w-full h-full object-contain"
                      />
                      <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/85 border border-emerald-500/40 text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" />
                        Verified: {detectionCount} {query}
                      </div>

                      <div className="absolute inset-0 bg-black/40 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                        <button
                          onClick={() => onOpenZoom(resultMedia, `Verified Detections: ${query}`)}
                          className="px-3 py-1 rounded bg-slate-900/90 text-xs font-mono text-white border border-slate-700 pointer-events-auto cursor-pointer"
                        >
                          <Eye className="w-3.5 h-3.5 inline mr-1 text-emerald-400" /> Inspect Coordinates
                        </button>
                      </div>
                    </>
                  )
                ) : (
                  <div className="text-center p-6 text-slate-500 font-mono text-xs">
                    {mode === 'video'
                      ? 'Tracked stream will render here upon completion'
                      : 'Target bounding boxes will render here upon completion'}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
