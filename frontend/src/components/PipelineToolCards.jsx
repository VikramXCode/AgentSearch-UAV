import React from 'react';
import { motion } from 'framer-motion';
import { Bot, Layers, Sparkles, BrainCircuit, Users2, CheckCircle2, FastForward, Loader2, CircleDot, Terminal } from 'lucide-react';

const defaultTools = [
  {
    id: 'yolo',
    name: 'YOLO-World (VisDrone)',
    subtitle: 'Fine-Tuned Aerial Backbone',
    icon: Bot,
    status: 'waiting',
    reason: 'Active for target classes in open-vocabulary aerial space',
  },
  {
    id: 'sahi',
    name: 'SAHI Image Slicing',
    subtitle: 'Adaptive Multi-Scale Slicer',
    icon: Layers,
    status: 'waiting',
    reason: 'Engages when target scale < 32px or dense cluster detected',
  },
  {
    id: 'sr',
    name: 'Super-Resolution (ESRGAN)',
    subtitle: '2x High-Frequency Enhancement',
    icon: Sparkles,
    status: 'waiting',
    reason: 'Triggered when input ground sampling distance is degraded',
  },
  {
    id: 'verification',
    name: 'Two-Stage Verification',
    subtitle: 'HSV Filter + CLIP ViT-B/32',
    icon: BrainCircuit,
    status: 'waiting',
    reason: 'Triggers on color/spatial attribute prompts to prune false positives',
  },
  {
    id: 'multi_agent',
    name: 'Multi-Agent Consensus',
    subtitle: 'LangGraph State Core',
    icon: Users2,
    status: 'waiting',
    reason: '7 specialized autonomous agents coordinating mission state',
  },
];

export default function PipelineToolCards({ toolsStatus, isSearching }) {
  const findToolStatus = (id) => {
    if (!toolsStatus) return null;
    if (Array.isArray(toolsStatus)) {
      return (
        toolsStatus.find(
          (t) =>
            t.id === id ||
            (id === 'sr' && (t.id === 'super_res' || t.id === 'sr')) ||
            (id === 'verification' && (t.id === 'verify' || t.id === 'verification'))
        ) || null
      );
    }
    if (typeof toolsStatus === 'object') {
      return (
        toolsStatus[id] ||
        (id === 'sr' && toolsStatus['super_res']) ||
        (id === 'verification' && (toolsStatus['verify'] || toolsStatus['verification'])) ||
        null
      );
    }
    return null;
  };

  const tools = defaultTools.map((tool) => {
    if (isSearching) {
      return {
        ...tool,
        status: 'processing',
        reason: 'Evaluating execution condition...',
      };
    }
    const matched = findToolStatus(tool.id);
    if (matched) {
      return {
        ...tool,
        status: matched.status || 'waiting',
        reason: matched.reason || tool.reason,
      };
    }
    return tool;
  });

  const getStatusBadge = (status) => {
    switch (status) {
      case 'used':
      case 'enabled':
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/40">
            <CheckCircle2 className="w-3 h-3" />
            Active / Used
          </span>
        );
      case 'skipped':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-700/60">
            <FastForward className="w-3 h-3" />
            Adaptive Bypass
          </span>
        );
      case 'processing':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-cyan-950/70 text-cyan-300 border border-cyan-400/50 animate-pulse">
            <Loader2 className="w-3 h-3 animate-spin" />
            Evaluating
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-slate-950 text-slate-400 border border-slate-800">
            <CircleDot className="w-3 h-3" />
            Standby
          </span>
        );
    }
  };

  return (
    <div className="cmd-panel rounded-xl p-5 sm:p-6 border border-slate-800/90 relative overflow-hidden">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-4 border-b border-slate-800/80 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400">
              <Terminal className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm sm:text-base font-bold tracking-tight text-white font-mono uppercase">
                  Autonomous Tool Execution Audit
                </h3>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  Dynamic Dispatch Log
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Transparent rationale for model activation, slicing, enhancement, and attribute verification
              </p>
            </div>
          </div>
        </div>

        <div className="text-xs font-mono text-slate-400 px-2.5 py-1 rounded bg-slate-950 border border-slate-800 self-start sm:self-auto">
          {tools.filter((t) => t.status === 'used' || t.status === 'enabled').length} of {tools.length} Modules Engaged
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3.5">
        {tools.map((tool) => {
          const Icon = tool.icon;
          const isUsed = tool.status === 'used' || tool.status === 'enabled';
          const isSkipped = tool.status === 'skipped';

          return (
            <div
              key={tool.id}
              className={`rounded-lg p-3.5 border transition-all flex flex-col justify-between ${
                isUsed
                  ? 'bg-slate-950/90 border-emerald-500/40 shadow-[0_2px_12px_rgba(16,185,129,0.06)]'
                  : isSkipped
                  ? 'bg-slate-950/50 border-slate-800/80 opacity-75'
                  : 'bg-slate-950/40 border-slate-800/70'
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2.5">
                  <div className="p-2 rounded bg-slate-900 border border-slate-800 text-cyan-400">
                    <Icon className="w-4 h-4" />
                  </div>
                  {getStatusBadge(tool.status)}
                </div>

                <h4 className="text-xs font-bold text-slate-100 font-mono mb-0.5">
                  {tool.name}
                </h4>
                <p className="text-[10px] text-slate-400 font-mono mb-3">
                  {tool.subtitle}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-800/80">
                <div className="text-[9px] uppercase font-mono tracking-wider text-slate-400 mb-1">
                  Trigger Condition:
                </div>
                <p className="text-[11px] text-slate-300 font-mono leading-relaxed">
                  {tool.reason}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
