import React from 'react';
import { motion } from 'framer-motion';
import {
  Target,
  Hash,
  Percent,
  Timer,
  Clock,
  Zap,
  Cpu,
  Sparkles,
  Filter,
  ShieldCheck,
  Activity,
  Award,
  BarChart2,
} from 'lucide-react';

export default function DetectionResults({ results, isSearching, mode = 'image' }) {
  if (!results && !isSearching) {
    return (
      <div className="cmd-panel rounded-xl p-6 border border-slate-800/90 text-center py-10">
        <div className="w-10 h-10 rounded bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-slate-500 mb-2.5">
          <Target className="w-5 h-5" />
        </div>
        <h4 className="text-xs font-bold text-slate-300 font-mono uppercase tracking-wider">
          Awaiting Inference Execution
        </h4>
        <p className="text-[11px] text-slate-400 font-mono mt-1 max-w-sm mx-auto">
          Ingest aerial payload and engage search pipeline to inspect target metrics and latency breakdown.
        </p>
      </div>
    );
  }

  const rawTarget =
    results?.query?.target ||
    results?.query?.raw_query ||
    (typeof results?.query === 'string' ? results.query : results?.target) ||
    'Target';

  const count =
    results?.count ?? results?.objects_found ?? (results?.detections?.length || 0);

  const avgConf = results?.average_confidence
    ? `${(results.average_confidence * 100).toFixed(1)}%`
    : results?.confidence
    ? `${(results.confidence * 100).toFixed(1)}%`
    : '85.0%';

  const totalTime = results?.timing?.total_time
    ? `${results.timing.total_time.toFixed(2)}s`
    : results?.metrics?.total_processing_time
    ? `${results.metrics.total_processing_time.toFixed(2)}s`
    : '0.00s';

  const fps = results?.metrics?.fps ? Number(results.metrics.fps).toFixed(1) : null;

  const timingBreakdown = [
    {
      label: 'Detection (YOLO-World)',
      time: results?.timing?.detection_time ?? (results?.metrics?.detection_time || 0),
      icon: Zap,
      color: 'text-blue-400',
      barColor: 'bg-blue-500',
    },
    {
      label: mode === 'video' ? 'Frame Tracking' : 'SAHI Slicing',
      time:
        mode === 'video'
          ? results?.timing?.tracking_time ?? (results?.metrics?.tracking_time || 0)
          : results?.timing?.sahi_time ?? 0,
      icon: Cpu,
      color: 'text-emerald-400',
      barColor: 'bg-emerald-500',
    },
    {
      label: 'Super-Resolution',
      time: results?.timing?.super_resolution_time ?? 0,
      icon: Sparkles,
      color: 'text-purple-400',
      barColor: 'bg-purple-500',
    },
    {
      label: 'Color Masking (HSV)',
      time: results?.timing?.color_filter_time ?? 0,
      icon: Filter,
      color: 'text-amber-400',
      barColor: 'bg-amber-500',
    },
    {
      label: 'CLIP AI Verification',
      time: results?.timing?.ai_verification_time ?? 0,
      icon: ShieldCheck,
      color: 'text-cyan-400',
      barColor: 'bg-cyan-500',
    },
  ];

  const maxTiming = Math.max(...timingBreakdown.map((t) => t.time), 0.05);

  return (
    <div className="cmd-panel rounded-xl p-5 sm:p-6 border border-slate-800/90 space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-4 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400">
            <BarChart2 className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm sm:text-base font-bold tracking-tight text-white font-mono uppercase">
                Quantitative Detection Telemetry
              </h3>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                Inference Analytics
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Verified target metrics, confidence distribution, and pipeline latency breakdown
            </p>
          </div>
        </div>

        {fps && (
          <div className="px-2.5 py-1 rounded bg-emerald-950/50 border border-emerald-500/40 text-emerald-400 text-xs font-mono font-bold flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 animate-pulse" />
            <span>{fps} FPS Throughput</span>
          </div>
        )}
      </div>

      {/* Primary 4 Metric HUD Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* Target Directive */}
        <div className="p-3.5 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-1.5">
            <span className="text-[10px] font-mono tracking-wider uppercase">Target Class</span>
            <Target className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="text-lg font-bold text-slate-100 capitalize truncate font-mono">
            {rawTarget}
          </div>
          <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center gap-1.5">
            {results?.query?.attributes && Object.keys(results.query.attributes).length > 0 ? (
              Object.entries(results.query.attributes).map(([k, v]) => (
                <span
                  key={k}
                  className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-500/30"
                >
                  {k}: {v}
                </span>
              ))
            ) : (
              <span className="text-[10px] font-mono text-slate-400">Class search</span>
            )}
          </div>
        </div>

        {/* Targets Verified */}
        <div className="p-3.5 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-1.5">
            <span className="text-[10px] font-mono tracking-wider uppercase">
              {mode === 'video' ? 'Tracked Instances' : 'Verified Targets'}
            </span>
            <Hash className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-2xl font-extrabold text-emerald-400 font-mono tracking-tight">
            {count}
          </div>
          <div className="mt-2 pt-2 border-t border-slate-800/80 text-[10px] font-mono text-slate-400 truncate">
            {count > 0 ? 'Verified in aerial feed' : 'Zero candidate matches'}
          </div>
        </div>

        {/* Mean Confidence */}
        <div className="p-3.5 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-1.5">
            <span className="text-[10px] font-mono tracking-wider uppercase">Mean Confidence</span>
            <Percent className="w-3.5 h-3.5 text-blue-400" />
          </div>
          <div className="text-2xl font-extrabold text-blue-400 font-mono tracking-tight">
            {avgConf}
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className="bg-blue-500 h-full rounded-full transition-all duration-500"
              style={{
                width: `${(results?.average_confidence || results?.confidence || 0.85) * 100}%`,
              }}
            />
          </div>
        </div>

        {/* Pipeline Latency */}
        <div className="p-3.5 rounded-lg bg-slate-950/90 border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-1.5">
            <span className="text-[10px] font-mono tracking-wider uppercase">E2E Latency</span>
            <Timer className="w-3.5 h-3.5 text-purple-400" />
          </div>
          <div className="text-2xl font-extrabold text-purple-400 font-mono tracking-tight">
            {totalTime}
          </div>
          <div className="mt-2 pt-2 border-t border-slate-800/80 text-[10px] font-mono text-slate-400 truncate">
            Autonomous pipeline duration
          </div>
        </div>
      </div>

      {/* Latency Breakdown Bar Matrix */}
      <div className="rounded-lg p-3.5 bg-slate-950 border border-slate-800">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-1.5 text-xs font-mono text-slate-300 uppercase">
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            <span>Stage-wise Execution Latency</span>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            Total: {totalTime}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-2.5">
          {timingBreakdown.map((stage, idx) => {
            const Icon = stage.icon;
            const percentage = maxTiming > 0 ? (stage.time / maxTiming) * 100 : 0;
            const isZero = stage.time === 0;

            return (
              <div key={idx} className="p-2.5 rounded bg-slate-900/80 border border-slate-800/60">
                <div className="flex items-center justify-between gap-1 mb-1">
                  <span className="text-[10px] font-mono text-slate-300 truncate">{stage.label}</span>
                  <Icon className={`w-3 h-3 flex-shrink-0 ${stage.color}`} />
                </div>
                <div className="flex items-baseline justify-between mb-1 font-mono">
                  <span className={`text-sm font-bold ${isZero ? 'text-slate-400' : 'text-slate-200'}`}>
                    {stage.time.toFixed(2)}s
                  </span>
                  {isZero && <span className="text-[9px] text-slate-400">Bypassed</span>}
                </div>
                <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                  <div
                    className={`${stage.barColor} h-full rounded-full transition-all duration-500`}
                    style={{ width: `${isZero ? 0 : Math.max(percentage, 5)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
