import React, { useState, useRef } from 'react';
import {
  Search,
  UploadCloud,
  Image as ImageIcon,
  Video as VideoIcon,
  X,
  Play,
  Loader2,
  CheckCircle2,
  ChevronRight,
  Film,
  Crosshair,
  Sliders,
  FileCheck,
  Sparkles,
} from 'lucide-react';
import { motion } from 'framer-motion';

const EXAMPLE_IMAGE_QUERIES = [
  { label: 'Red Cars', query: 'Find red cars', category: 'Attribute Search' },
  { label: 'Trucks', query: 'Find trucks', category: 'Class Detection' },
  { label: 'Pedestrians', query: 'Find pedestrians', category: 'Small Scale' },
  { label: 'White Buses', query: 'Find white buses', category: 'Attribute Search' },
  { label: 'Motorcycles', query: 'Find motorcycles', category: 'Dense Scene' },
];

const EXAMPLE_VIDEO_QUERIES = [
  { label: 'Track Cars', query: 'car', category: 'Multi-Target' },
  { label: 'Track Trucks', query: 'truck', category: 'Heavy Vehicle' },
  { label: 'Track Bus', query: 'bus', category: 'Transit' },
  { label: 'Track Vehicles', query: 'vehicle', category: 'Open Class' },
];

export default function TargetSearchPanel({
  query,
  setQuery,
  selectedFile,
  setSelectedFile,
  previewUrl,
  setPreviewUrl,
  isSearching,
  onRunSearch,
  samples = [],
  sampleVideos = [],
  onSelectSample,
  mode = 'image',
  setMode,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      handleFileSelected(file);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      handleFileSelected(file);
    }
  };

  const handleFileSelected = (file) => {
    const isImg = file.type.startsWith('image/');
    const isVid = file.type.startsWith('video/');

    if (mode === 'image' && !isImg) {
      alert('Please upload a valid image file (JPG, PNG, WEBP).');
      return;
    }
    if (mode === 'video' && !isVid) {
      alert('Please upload a valid video file (MP4, MOV, AVI, WEBM).');
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const clearFile = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const exampleQueries = mode === 'video' ? EXAMPLE_VIDEO_QUERIES : EXAMPLE_IMAGE_QUERIES;

  return (
    <div className="cmd-panel rounded-xl p-5 sm:p-6 border border-slate-800/90 relative overflow-hidden">
      {/* Top Header Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80 mb-5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400">
            <Crosshair className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm sm:text-base font-bold text-white tracking-tight font-mono uppercase">
                Mission Ingest & Query Directive
              </h2>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/80">
                {mode === 'video' ? 'Video Tracker' : 'Multi-Agent Vision'}
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Natural language prompt parser & UAV surveillance payload feed
            </p>
          </div>
        </div>

        {/* Quick Mode Indicator */}
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <span className="text-slate-500 uppercase text-[10px] tracking-wider">Payload Mode:</span>
          <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-700 text-cyan-400 font-semibold uppercase text-[11px]">
            {mode === 'video' ? 'Aerial Video Stream' : 'Aerial Static Frame'}
          </span>
        </div>
      </div>

      {/* Query Input Box */}
      <div className="space-y-2.5 mb-5">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <label className="flex items-center gap-1.5 uppercase text-[11px] tracking-wider text-slate-300">
            <Search className="w-3.5 h-3.5 text-cyan-400" />
            Natural Language Search Query
          </label>
          <span className="text-[10px] text-slate-500">Press ENTER or click Engage below</span>
        </div>

        <div className="relative flex items-center">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && onRunSearch()}
            placeholder={
              mode === 'video'
                ? 'Specify target object to track (e.g., "car", "truck", "bus", "vehicle")...'
                : 'Enter query with objects, attributes, or quantity (e.g., "Find red cars", "Find trucks")...'
            }
            className="w-full pl-10 pr-24 py-3 rounded-lg bg-slate-950 border border-slate-700/90 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30 text-sm font-mono transition-all shadow-inner"
          />
          <div className="absolute left-3.5 text-slate-500 pointer-events-none">
            <Search className="w-4 h-4 text-cyan-400" />
          </div>

          {query && (
            <button
              onClick={() => setQuery('')}
              className="absolute right-3 p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Quick Query Presets */}
        <div className="flex flex-wrap items-center gap-2 pt-0.5">
          <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
            Presets:
          </span>
          {exampleQueries.map((item, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setQuery(item.query)}
              className={`text-xs px-2.5 py-1 rounded border font-mono transition-all flex items-center gap-1.5 ${
                query.toLowerCase() === item.query.toLowerCase()
                  ? 'bg-slate-800 border-cyan-500/70 text-cyan-300 font-semibold shadow-sm'
                  : 'bg-slate-950/70 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 hover:bg-slate-900'
              }`}
            >
              <span>{item.label}</span>
              <span className="text-[9px] text-slate-400 uppercase">({item.category})</span>
            </button>
          ))}
        </div>
      </div>

      {/* Media Ingest Area & Benchmark Sample Presets */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-5">
        {/* Drag & Drop Upload Zone (2 cols) */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`lg:col-span-2 relative rounded-lg border border-dashed p-4 flex flex-col items-center justify-center cursor-pointer transition-all min-h-[170px] overflow-hidden hud-corner-tl hud-corner-br ${
            isDragging
              ? 'border-cyan-400 bg-cyan-950/20'
              : previewUrl
              ? 'border-slate-700 bg-slate-950/80'
              : 'border-slate-700/80 hover:border-slate-600 bg-slate-950/50 hover:bg-slate-950'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept={
              mode === 'video'
                ? 'video/mp4, video/mov, video/avi, video/webm'
                : 'image/png, image/jpeg, image/jpg, image/webp'
            }
            className="hidden"
          />

          {previewUrl ? (
            <div className="relative w-full h-full flex flex-col sm:flex-row items-center gap-4">
              <div className="relative w-32 h-28 sm:w-40 sm:h-32 rounded border border-slate-800 bg-black/80 flex items-center justify-center overflow-hidden flex-shrink-0 shadow-md">
                {mode === 'video' ? (
                  <video
                    src={previewUrl}
                    className="w-full h-full object-contain"
                    muted
                    autoPlay
                    loop
                  />
                ) : (
                  <img
                    src={previewUrl}
                    alt="UAV Source"
                    className="w-full h-full object-contain"
                  />
                )}
                <div className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-black/80 border border-slate-700 text-[9px] font-mono text-slate-300">
                  {mode === 'video' ? 'STREAM' : 'RAW'}
                </div>
              </div>

              <div className="flex-1 text-center sm:text-left min-w-0 space-y-1">
                <div className="flex items-center gap-2 justify-center sm:justify-start">
                  <span className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-950/40 text-emerald-400 border border-emerald-500/40">
                    <FileCheck className="w-3 h-3" /> Payload Loaded
                  </span>
                  <span className="text-[10px] font-mono text-slate-400 uppercase">
                    Ready for analysis
                  </span>
                </div>

                <p className="text-sm font-semibold font-mono text-slate-100 truncate">
                  {selectedFile?.name || (mode === 'video' ? 'aerial_video_stream.mp4' : 'uav_aerial_frame.png')}
                </p>

                <div className="flex items-center justify-center sm:justify-start gap-3 text-xs font-mono text-slate-400">
                  <span>Size: {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : 'Preset benchmark'}</span>
                  <span>•</span>
                  <span>Type: {mode === 'video' ? 'MP4/H.264' : 'RGB/Lossless'}</span>
                </div>

                <div className="pt-1.5 flex items-center gap-3 justify-center sm:justify-start text-xs font-mono">
                  <span className="text-cyan-400 hover:underline cursor-pointer">
                    Click to replace
                  </span>
                  <span className="text-slate-600">|</span>
                  <button
                    onClick={clearFile}
                    className="text-rose-400 hover:underline cursor-pointer"
                  >
                    Clear payload
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center text-center py-2">
              <div className="w-10 h-10 rounded bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400 mb-2.5">
                {mode === 'video' ? (
                  <Film className="w-5 h-5 animate-pulse" />
                ) : (
                  <UploadCloud className="w-5 h-5" />
                )}
              </div>
              <p className="text-xs sm:text-sm font-semibold font-mono text-slate-200">
                Drop UAV surveillance {mode === 'video' ? 'video' : 'imagery'} here, or <span className="text-cyan-400 underline">browse</span>
              </p>
              <p className="text-[11px] text-slate-400 font-mono mt-1">
                {mode === 'video'
                  ? 'Supports aerial surveillance video clips (MP4, MOV, AVI)'
                  : 'Supports aerial reconnaissance scenes (JPG, PNG, WEBP)'}
              </p>
            </div>
          )}
        </div>

        {/* Preset Sample Selector (1 col) */}
        <div className="cmd-panel-sub rounded-lg p-3.5 border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5">
                {mode === 'video' ? (
                  <VideoIcon className="w-3.5 h-3.5 text-cyan-400" />
                ) : (
                  <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
                )}
                <h3 className="text-xs font-bold text-slate-200 uppercase font-mono tracking-wider">
                  Benchmark UAV Samples
                </h3>
              </div>
              <span className="text-[9px] font-mono text-slate-400">VisDrone Presets</span>
            </div>

            <p className="text-[11px] text-slate-400 font-mono mb-2.5">
              Select verified aerial benchmark scenes:
            </p>

            <div className="space-y-1.5">
              {mode === 'video' ? (
                sampleVideos && sampleVideos.length > 0 ? (
                  sampleVideos.map((sample, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => onSelectSample(sample, 'video')}
                      className="w-full px-2.5 py-1.5 rounded bg-slate-950 hover:bg-slate-900 border border-slate-800 hover:border-cyan-500/50 flex items-center justify-between text-xs font-mono text-slate-300 hover:text-white transition-all text-left group"
                    >
                      <div className="flex items-center gap-2 truncate">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-500/80 group-hover:bg-cyan-400" />
                        <span className="truncate">{sample.name}</span>
                      </div>
                      <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-cyan-400 flex-shrink-0" />
                    </button>
                  ))
                ) : (
                  ['12 sec.mp4', 'traffic video clip 1.mp4'].map((name, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => onSelectSample({ name, url: `/sample-video?name=${name}` }, 'video')}
                      className="w-full px-2.5 py-1.5 rounded bg-slate-950 hover:bg-slate-900 border border-slate-800 hover:border-cyan-500/50 flex items-center justify-between text-xs font-mono text-slate-300 hover:text-white transition-all text-left group"
                    >
                      <div className="flex items-center gap-2 truncate">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-500/80 group-hover:bg-cyan-400" />
                        <span className="truncate">{name}</span>
                      </div>
                      <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-cyan-400 flex-shrink-0" />
                    </button>
                  ))
                )
              ) : samples && samples.length > 0 ? (
                samples.slice(0, 3).map((sample, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => onSelectSample(sample, 'image')}
                    className="w-full px-2.5 py-1.5 rounded bg-slate-950 hover:bg-slate-900 border border-slate-800 hover:border-cyan-500/50 flex items-center justify-between text-xs font-mono text-slate-300 hover:text-white transition-all text-left group"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-500/80 group-hover:bg-cyan-400" />
                      <span className="truncate">{sample.name}</span>
                    </div>
                    <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-cyan-400 flex-shrink-0" />
                  </button>
                ))
              ) : (
                ['uav3.png', 'cctv.jpg', 'red car.jpg'].map((name, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => onSelectSample({ name, url: `/sample-image?name=${name}` }, 'image')}
                    className="w-full px-2.5 py-1.5 rounded bg-slate-950 hover:bg-slate-900 border border-slate-800 hover:border-cyan-500/50 flex items-center justify-between text-xs font-mono text-slate-300 hover:text-white transition-all text-left group"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-500/80 group-hover:bg-cyan-400" />
                      <span className="truncate">{name}</span>
                    </div>
                    <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-cyan-400 flex-shrink-0" />
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Precision Engage Mission Button */}
      <button
        type="button"
        onClick={onRunSearch}
        disabled={isSearching || !previewUrl || !query.trim()}
        className={`w-full py-3.5 rounded-lg font-mono font-bold text-sm tracking-wider uppercase flex items-center justify-center gap-2.5 transition-all shadow-md ${
          isSearching
            ? 'bg-slate-900 text-slate-400 border border-slate-800 cursor-not-allowed'
            : !previewUrl || !query.trim()
            ? 'bg-slate-900/80 text-slate-500 border border-slate-800/80 cursor-not-allowed'
            : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 border border-cyan-400/80 shadow-[0_0_20px_rgba(6,182,212,0.25)] cursor-pointer'
        }`}
      >
        {isSearching ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
            <span>
              {mode === 'video'
                ? 'EXECUTING VIDEO FRAME TRACKER...'
                : 'EXECUTING MULTI-AGENT INFERENCE PIPELINE...'}
            </span>
          </>
        ) : (
          <>
            <Crosshair className="w-4 h-4" />
            <span>
              {mode === 'video'
                ? 'ENGAGE UAV VIDEO TARGET TRACKER'
                : 'ENGAGE MULTI-AGENT TARGET SEARCH PIPELINE'}
            </span>
          </>
        )}
      </button>
    </div>
  );
}
