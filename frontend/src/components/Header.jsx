import React, { useState, useEffect } from 'react';
import {
  Activity,
  BarChart3,
  History,
  Image as ImageIcon,
  Video as VideoIcon,
  Crosshair,
  Shield,
  Layers,
  Terminal,
} from 'lucide-react';

export default function Header({
  backendConnected,
  latency,
  mode = 'image',
  setMode,
  onOpenBenchmark,
  onOpenHistory,
}) {
  const [utcTime, setUtcTime] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="border-b border-slate-800/90 bg-[#090d15]/95 backdrop-blur-md sticky top-0 z-40 px-4 sm:px-6 lg:px-8 py-3 select-none">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3.5">
        {/* Left: Tactical Brand & Research Spec */}
        <div className="flex items-center gap-3 w-full md:w-auto justify-between md:justify-start">
          <div className="flex items-center gap-2.5">
            <div className="relative w-8 h-8 rounded-lg bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400 shadow-inner">
              <Crosshair className="w-4 h-4 text-cyan-400 animate-pulse" />
              <div className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-cyan-400 ring-2 ring-[#090d15]" />
            </div>

            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm sm:text-base font-bold tracking-tight text-white font-mono uppercase">
                  AgentSearch-UAV
                </span>
                <span className="px-1.5 py-0.5 text-[9px] font-mono font-semibold uppercase tracking-wider rounded bg-slate-800/90 text-slate-300 border border-slate-700/80">
                  CVPR Research v2.5
                </span>
              </div>
              <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono">
                <span className="hidden sm:inline">Adaptive Multi-Agent Vision System</span>
                <span className="hidden sm:inline text-slate-600">•</span>
                <span className="text-slate-400">{utcTime || 'UTC Live'}</span>
              </div>
            </div>
          </div>

          {/* Mobile Connection Pill */}
          <div className="md:hidden">
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-mono font-semibold border ${
                backendConnected
                  ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-400'
                  : 'bg-rose-950/40 border-rose-500/40 text-rose-400'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${backendConnected ? 'bg-emerald-400 animate-ping' : 'bg-rose-400'}`} />
              <span>{backendConnected ? 'ONLINE' : 'OFFLINE'}</span>
            </div>
          </div>
        </div>

        {/* Center: Mission Ingest Mode Selector */}
        <div className="flex items-center p-1 rounded-lg bg-slate-950 border border-slate-800 shadow-inner">
          <button
            onClick={() => setMode('image')}
            className={`px-3 py-1.5 rounded text-xs font-mono font-semibold flex items-center gap-2 transition-all ${
              mode === 'image'
                ? 'bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
            <span>Static Scene Ingest</span>
          </button>
          <button
            onClick={() => setMode('video')}
            className={`px-3 py-1.5 rounded text-xs font-mono font-semibold flex items-center gap-2 transition-all ${
              mode === 'video'
                ? 'bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <VideoIcon className="w-3.5 h-3.5 text-cyan-400" />
            <span>Video Stream Tracking</span>
          </button>
        </div>

        {/* Right: Technical Controls & System Telemetry */}
        <div className="hidden md:flex items-center gap-2.5">
          {/* Benchmark Action */}
          <button
            onClick={onOpenBenchmark}
            className="px-2.5 py-1.5 rounded bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-xs font-mono font-medium text-slate-300 hover:text-white transition-colors flex items-center gap-1.5"
            title="View standardized VisDrone benchmark comparison table"
          >
            <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
            <span>VisDrone Benchmark</span>
          </button>

          {/* History Action */}
          <button
            onClick={onOpenHistory}
            className="px-2.5 py-1.5 rounded bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-xs font-mono font-medium text-slate-300 hover:text-white transition-colors flex items-center gap-1.5"
            title="Inspect historical mission logs"
          >
            <History className="w-3.5 h-3.5 text-cyan-400" />
            <span>Mission Logs</span>
          </button>

          {/* Latency badge */}
          {backendConnected && latency !== null && (
            <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800/80 text-xs font-mono text-slate-400">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span>{latency}ms</span>
            </div>
          )}

          {/* Connection status indicator */}
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded border text-xs font-mono font-semibold transition-all ${
              backendConnected
                ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-400'
                : 'bg-rose-950/40 border-rose-500/40 text-rose-400'
            }`}
          >
            <span className="relative flex h-2 w-2">
              {backendConnected && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              )}
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  backendConnected ? 'bg-emerald-500' : 'bg-rose-500'
                }`}
              ></span>
            </span>
            <span className="text-[11px] tracking-wide">
              {backendConnected ? 'CORE ONLINE' : 'API OFFLINE'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
