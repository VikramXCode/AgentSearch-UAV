import React, { useState, useRef } from 'react';
import {
  Search,
  UploadCloud,
  Image as ImageIcon,
  Video as VideoIcon,
  X,
  Loader2,
  ChevronRight,
  Film,
  Crosshair,
  FileCheck,
  Sparkles,
} from 'lucide-react';

const EXAMPLE_IMAGE_QUERIES = [
  { label: 'Red Cars', query: 'Find red cars' },
  { label: 'Trucks', query: 'Find trucks' },
  { label: 'Pedestrians', query: 'Find pedestrians' },
  { label: 'White Buses', query: 'Find white buses' },
  { label: 'Motorcycles', query: 'Find motorcycles' },
];

const EXAMPLE_VIDEO_QUERIES = [
  { label: 'Track Cars', query: 'car' },
  { label: 'Track Pedestrians', query: 'person' },
  { label: 'Track Trucks', query: 'truck' },
  { label: 'Track Buses', query: 'bus' },
  { label: 'Track Vehicles', query: 'vehicle' },
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
  videoProgress = null,
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
    <div className="cmd-panel rounded-2xl p-5 sm:p-6 border border-slate-800/90 shadow-xl space-y-5">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Crosshair className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-base sm:text-lg font-bold text-white tracking-tight">
              Query & Media Input
            </h2>
            <p className="text-xs text-slate-400">
              Type what target to find and provide an aerial image or video
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-slate-400 text-[11px]">Mode:</span>
          <span className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700/80 text-cyan-300 font-semibold text-[11px]">
            {mode === 'video' ? 'Aerial Video Stream' : 'Aerial Static Image'}
          </span>
        </div>
      </div>

      {/* Query Search Bar */}
      <div className="space-y-2">
        <label className="flex items-center gap-1.5 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <Search className="w-3.5 h-3.5 text-cyan-400" />
          Search Directive
        </label>

        <div className="relative flex items-center">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && onRunSearch()}
            placeholder={
              mode === 'video'
                ? 'Specify target object to track (e.g. "car", "truck", "bus")...'
                : 'Enter target object or query (e.g. "Find red cars", "Find trucks")...'
            }
            className="w-full pl-11 pr-24 py-3.5 rounded-xl bg-slate-950 border border-slate-700/90 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 text-sm font-sans transition-all shadow-inner"
          />
          <div className="absolute left-3.5 text-slate-500 pointer-events-none">
            <Search className="w-4 h-4 text-cyan-400" />
          </div>

          {query && (
            <button
              onClick={() => setQuery('')}
              className="absolute right-3 p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Clear text"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Quick Query Presets */}
        <div className="flex flex-wrap items-center gap-1.5 pt-1">
          <span className="text-[11px] text-slate-400 font-medium mr-1">
            Presets:
          </span>
          {exampleQueries.map((item, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setQuery(item.query)}
              className={`text-xs px-2.5 py-1 rounded-lg border transition-all cursor-pointer ${
                query.toLowerCase() === item.query.toLowerCase()
                  ? 'bg-cyan-500/20 border-cyan-500/60 text-cyan-300 font-semibold shadow-sm'
                  : 'bg-slate-900/70 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 hover:bg-slate-900'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Media Ingest & Preset Samples */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Upload Drop Zone (2 cols) */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`lg:col-span-2 relative rounded-xl border border-dashed p-4 flex flex-col items-center justify-center cursor-pointer transition-all min-h-[160px] overflow-hidden ${
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
              <div className="relative w-32 h-24 sm:w-36 sm:h-28 rounded-lg border border-slate-800 bg-black/80 flex items-center justify-center overflow-hidden flex-shrink-0 shadow-md">
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
                    Ready to Search
                  </span>
                </div>

                <p className="text-sm font-semibold font-mono text-slate-100 truncate">
                  {selectedFile?.name || (mode === 'video' ? 'aerial_video_stream.mp4' : 'uav_aerial_frame.png')}
                </p>

                <div className="flex items-center justify-center sm:justify-start gap-3 text-xs font-mono text-slate-400">
                  <span>Size: {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : 'Preset Benchmark'}</span>
                  <span>•</span>
                  <span>Type: {mode === 'video' ? 'MP4' : 'Image'}</span>
                </div>

                <div className="pt-1 flex items-center gap-3 justify-center sm:justify-start text-xs font-mono">
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
              <div className="w-10 h-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400 mb-2">
                {mode === 'video' ? (
                  <Film className="w-5 h-5 animate-pulse" />
                ) : (
                  <UploadCloud className="w-5 h-5" />
                )}
              </div>
              <p className="text-xs sm:text-sm font-semibold text-slate-200">
                Drop UAV surveillance {mode === 'video' ? 'video' : 'imagery'} here, or <span className="text-cyan-400 underline">browse</span>
              </p>
              <p className="text-[11px] text-slate-400 mt-0.5">
                {mode === 'video'
                  ? 'Supports aerial surveillance video clips (MP4, MOV, AVI)'
                  : 'Supports aerial reconnaissance scenes (JPG, PNG, WEBP)'}
              </p>
            </div>
          )}
        </div>

        {/* Preset Sample Selector (1 col) */}
        <div className="cmd-panel-sub rounded-xl p-3.5 border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5">
                {mode === 'video' ? (
                  <VideoIcon className="w-3.5 h-3.5 text-cyan-400" />
                ) : (
                  <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
                )}
                <h3 className="text-xs font-bold text-slate-200 uppercase font-mono tracking-wider">
                  Treasure Media
                </h3>
              </div>
              <span className="text-[9px] font-mono text-cyan-500">treasure/</span>
            </div>

            <p className="text-[11px] text-slate-400 mb-2">
              Select media from treasure folder:
            </p>

            <div className="space-y-1.5">
              {mode === 'video' ? (
                sampleVideos && sampleVideos.length > 0 ? (
                  sampleVideos.map((sample, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => onSelectSample(sample, 'video')}
                      className={`w-full px-2.5 py-1.5 rounded-lg border flex items-center justify-between text-xs font-mono transition-all text-left group cursor-pointer ${
                        selectedFile?.name === sample.name
                          ? 'bg-cyan-950/40 border-cyan-500/60 text-cyan-300 font-semibold'
                          : 'bg-slate-950 hover:bg-slate-900 border-slate-800 hover:border-cyan-500/50 text-slate-300 hover:text-white'
                      }`}
                    >
                      <div className="flex items-center gap-2 truncate">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                        <span className="truncate">{sample.name}</span>
                      </div>
                      <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-cyan-400 flex-shrink-0" />
                    </button>
                  ))
                ) : (
                  <div className="text-[11px] text-slate-500 italic p-3 border border-dashed border-slate-800 rounded-lg text-center">
                    No videos in <code className="text-cyan-400">treasure/videos</code>. Drop .mp4 files there or upload above.
                  </div>
                )
              ) : samples && samples.length > 0 ? (
                samples.map((sample, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => onSelectSample(sample, 'image')}
                    className={`w-full px-2.5 py-1.5 rounded-lg border flex items-center justify-between text-xs font-mono transition-all text-left group cursor-pointer ${
                      selectedFile?.name === sample.name
                        ? 'bg-cyan-950/40 border-cyan-500/60 text-cyan-300 font-semibold'
                        : 'bg-slate-950 hover:bg-slate-900 border-slate-800 hover:border-cyan-500/50 text-slate-300 hover:text-white'
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                      <span className="truncate">{sample.name}</span>
                    </div>
                    <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-cyan-400 flex-shrink-0" />
                  </button>
                ))
              ) : (
                <div className="text-[11px] text-slate-500 italic p-3 border border-dashed border-slate-800 rounded-lg text-center">
                  No images in <code className="text-cyan-400">treasure/images</code>. Drop image files there or upload above.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Real-Time Video Processing Telemetry Card */}
      {isSearching && mode === 'video' && (
        <div className="rounded-xl border border-cyan-500/40 bg-slate-950/90 p-4 space-y-2.5 font-mono shadow-[0_0_20px_rgba(6,182,212,0.15)]">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2 text-cyan-300 font-semibold">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400 flex-shrink-0" />
              <span className="truncate">{videoProgress?.stage || 'Tracking video frames...'}</span>
            </div>
            <span className="text-cyan-400 font-bold text-sm flex-shrink-0 ml-2">
              {Math.round(videoProgress?.percent || 0)}%
            </span>
          </div>

          {/* Animated Tactical Progress Bar */}
          <div className="w-full h-2.5 rounded-full bg-slate-900 overflow-hidden border border-slate-800 p-0.5 shadow-inner">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-500 via-blue-500 to-emerald-400 transition-all duration-300 shadow-[0_0_12px_rgba(6,182,212,0.8)]"
              style={{ width: `${Math.min(100, Math.max(0, videoProgress?.percent || 0))}%` }}
            />
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>
              Frame <strong className="text-slate-200">{videoProgress?.currentFrame || 0}</strong> of <strong className="text-slate-200">{videoProgress?.totalFrames || 0}</strong>
            </span>
            <span>Speed: <strong className="text-cyan-300">{videoProgress?.fps || 0} FPS</strong></span>
            <span>Target: <strong className="text-white font-semibold">{query}</strong></span>
          </div>
        </div>
      )}

      {/* Search / Engage Action Button */}
      <button
        type="button"
        onClick={onRunSearch}
        disabled={isSearching || !previewUrl || !query.trim()}
        className={`w-full py-3.5 rounded-xl font-medium text-sm tracking-wide transition-all shadow-lg flex items-center justify-center gap-2.5 ${
          isSearching
            ? 'bg-slate-800 text-slate-400 border border-slate-700 cursor-not-allowed'
            : !previewUrl || !query.trim()
            ? 'bg-slate-900 text-slate-500 border border-slate-800 cursor-not-allowed'
            : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold border border-cyan-400/50 shadow-[0_0_25px_rgba(6,182,212,0.3)] cursor-pointer'
        }`}
      >
        {isSearching ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-cyan-300" />
            <span>
              {mode === 'video'
                ? `Tracking ${query} (${Math.round(videoProgress?.percent || 0)}%)...`
                : 'Searching targets in aerial image...'}
            </span>
          </>
        ) : (
          <>
            <Crosshair className="w-4 h-4" />
            <span>Search Targets</span>
          </>
        )}
      </button>
    </div>
  );
}
