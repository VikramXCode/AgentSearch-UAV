import React, { useState } from 'react';
import {
  Upload,
  Search,
  Scan,
  Sparkles,
  Grid,
  CheckCircle2,
  Users,
  Target,
  Clock,
  Check,
  FastForward,
  XCircle,
  Loader2,
  Activity,
  Film,
  Video,
  Layers,
  Cpu,
  Brain,
  ChevronRight,
  Info,
  GitBranch,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const IMAGE_AGENTS = [
  {
    id: 'input',
    num: '01',
    name: 'Aerial Ingest',
    role: 'Sensor Ingest Agent',
    icon: Upload,
    type: 'Core Ingest',
    spec: 'VisDrone aerial standard input resolution',
    details: 'Calibrates aspect ratio & normalizes dynamic range',
  },
  {
    id: 'query',
    num: '02',
    name: 'NLP Parser',
    role: 'Query Understanding Agent',
    icon: Search,
    type: 'Semantic Parser',
    spec: 'Extracts target classes, color attributes, quantities',
    details: 'Structures natural language prompt into typed query schema',
  },
  {
    id: 'strategy',
    num: '03',
    name: 'Strategy Engine',
    role: 'Adaptive Strategy Agent',
    icon: GitBranch,
    type: 'Decision Core',
    spec: 'Autonomous DAG execution router',
    details: 'Dynamically routes execution based on image resolution & prompt',
  },
  {
    id: 'yolo',
    num: '04',
    name: 'YOLO-World',
    role: 'Fine-Tuned VisDrone Detector',
    icon: Scan,
    type: 'Neural Detector',
    spec: 'Open-vocabulary aerial weights with tuned anchors',
    details: 'Generates multi-scale bounding box proposals across scene',
  },
  {
    id: 'sr',
    num: '05',
    name: 'Super-Res',
    role: 'Real-ESRGAN Upscaler',
    icon: Sparkles,
    type: 'Conditional Module',
    spec: '2x high-frequency aerial feature reconstruction',
    details: 'Engaged for low-resolution patches or small far-distance objects',
  },
  {
    id: 'sahi',
    num: '06',
    name: 'SAHI Slicing',
    role: 'Tiled Sliced Inference',
    icon: Grid,
    type: 'Conditional Module',
    spec: 'Multi-scale 640x640 overlapping image slices',
    details: 'Activated for dense clusters & tiny aerial object scales',
  },
  {
    id: 'verification',
    num: '07',
    name: 'Two-Stage Verify',
    role: 'Attribute Verification Agent',
    icon: CheckCircle2,
    type: 'Conditional Module',
    spec: 'Stage 1: HSV Color Space | Stage 2: CLIP ViT-B/32',
    details: 'Validates semantic attributes to eliminate false positive candidates',
  },
  {
    id: 'decision',
    num: '08',
    name: 'Consensus & NMS',
    role: 'Multi-Agent Consensus Agent',
    icon: Users,
    type: 'Ensemble Core',
    spec: 'IoU-based cross-agent spatial deduplication',
    details: 'Applies agent voting and non-maximum suppression',
  },
  {
    id: 'final',
    num: '09',
    name: 'Target Output',
    role: 'Annotated Payload Generator',
    icon: Target,
    type: 'Output Emitter',
    spec: 'Verified bounding boxes + metadata coordinates',
    details: 'Emits finalized detections with confidence telemetry',
  },
];

const VIDEO_AGENTS = [
  {
    id: 'video_in',
    num: '01',
    name: 'Video Demuxer',
    role: 'Frame Ingest Agent',
    icon: Film,
    type: 'Core Ingest',
    spec: 'Continuous aerial surveillance video stream decode',
    details: 'Extracts sequential frame buffers & timing timestamps',
  },
  {
    id: 'target_probe',
    num: '02',
    name: 'Target Anchor',
    role: 'Target Class Formulation',
    icon: Search,
    type: 'Query Anchor',
    spec: 'Initializes target embedding and tracking anchors',
    details: 'Establishes semantic identity for continuous multi-frame track',
  },
  {
    id: 'frame_detect',
    num: '03',
    name: 'Frame Detector',
    role: 'Anchor Frame Detection',
    icon: Scan,
    type: 'Periodic Detector',
    spec: 'YOLO-World anchor detection re-verification',
    details: 'Periodically re-anchors active trajectories to prevent track drift',
  },
  {
    id: 'motion_track',
    num: '04',
    name: 'Motion Matcher',
    role: 'Template & Optical Tracking',
    icon: Activity,
    type: 'Spatial Follower',
    spec: 'Sub-pixel template matching & velocity extrapolation',
    details: 'Follows target bounding boxes across sequential inter-frame motions',
  },
  {
    id: 'track_match',
    num: '05',
    name: 'Track Associator',
    role: 'Multi-Target ID Manager',
    icon: Users,
    type: 'Data Association',
    spec: 'Distance matrix Hungarian track assignment',
    details: 'Maintains consistent target IDs (TRK-1, TRK-2) across occlusions',
  },
  {
    id: 'video_out',
    num: '06',
    name: 'Video Encoder',
    role: 'Stream Render Agent',
    icon: Video,
    type: 'Payload Emitter',
    spec: 'H.264 real-time annotated video output stream',
    details: 'Renders tracking bounding boxes and trajectory paths onto stream',
  },
];

export default function LivePipelineGraph({
  stages,
  pipelineData,
  isSearching,
  isLoading,
  mode = 'image',
}) {
  const isRunning = isSearching || isLoading;
  const currentPipeline = stages || pipelineData || [];
  const agents = mode === 'video' ? VIDEO_AGENTS : IMAGE_AGENTS;
  const [selectedNode, setSelectedNode] = useState(null);

  const getStageStatus = (stageId, index) => {
    if (isRunning) {
      if (index === 0 || index === 1) return 'completed';
      if (index === 2 || index === 3) return 'processing';
      return 'waiting';
    }

    if (!currentPipeline || currentPipeline.length === 0) {
      return 'waiting';
    }

    if (Array.isArray(currentPipeline)) {
      const match = currentPipeline.find(
        (s) => s.id === stageId || (stageId === 'sr' && s.id === 'super_res')
      );
      if (match) return match.status.toLowerCase();
    }

    return 'completed';
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
      case 'used':
        return (
          <span className="inline-flex items-center gap-1 text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/40">
            <Check className="w-2.5 h-2.5" /> Done
          </span>
        );
      case 'processing':
        return (
          <span className="inline-flex items-center gap-1 text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded bg-cyan-950/70 text-cyan-300 border border-cyan-400/50 animate-pulse">
            <Loader2 className="w-2.5 h-2.5 animate-spin" /> Active
          </span>
        );
      case 'skipped':
        return (
          <span className="inline-flex items-center gap-1 text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-700/60">
            <FastForward className="w-2.5 h-2.5" /> Bypass
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center gap-1 text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-400 border border-rose-500/40">
            <XCircle className="w-2.5 h-2.5" /> Fail
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 text-[9px] font-mono font-medium px-1.5 py-0.5 rounded bg-slate-950 text-slate-400 border border-slate-800">
            <Clock className="w-2.5 h-2.5" /> Ready
          </span>
        );
    }
  };

  return (
    <div className="cmd-panel rounded-xl p-5 sm:p-6 border border-slate-800/90 relative overflow-hidden">
      {/* Header bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5 pb-4 border-b border-slate-800/80 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400">
              <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm sm:text-base font-bold text-white tracking-tight font-mono uppercase">
                  Multi-Agent Execution DAG Visualization
                </h2>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950/50 text-cyan-400 border border-cyan-500/30">
                  {mode === 'video' ? '6-Stage Tracker Flow' : '9-Stage Autonomous Graph'}
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Dynamic inter-agent signal flow, conditional module bypasses, and telemetry
              </p>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]" />
            <span className="text-slate-300">Executed</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            <span className="text-slate-300">Running</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-slate-600" />
            <span className="text-slate-400">Adaptive Bypass</span>
          </span>
        </div>
      </div>

      {/* SVG Connecting Flow Lines Overlay (Desktop) */}
      <div className="relative">
        <div className="hidden lg:block absolute top-[52px] left-8 right-8 h-1 pointer-events-none z-0">
          <svg className="w-full h-4 overflow-visible">
            <line
              x1="0"
              y1="2"
              x2="100%"
              y2="2"
              stroke="#1e293b"
              strokeWidth="2"
            />
            {isRunning && (
              <line
                x1="0"
                y1="2"
                x2="100%"
                y2="2"
                stroke="#06b6d4"
                strokeWidth="2"
                className="animate-signal-flow"
              />
            )}
          </svg>
        </div>

        {/* Grid of Autonomous Agent Nodes */}
        <div
          className={`grid grid-cols-2 sm:grid-cols-3 ${
            mode === 'video' ? 'lg:grid-cols-6' : 'lg:grid-cols-9'
          } gap-2.5 relative z-10`}
        >
          {agents.map((agent, idx) => {
            const Icon = agent.icon;
            const status = getStageStatus(agent.id, idx);
            const stageDetails = Array.isArray(currentPipeline)
              ? currentPipeline.find((s) => s.id === agent.id)?.details
              : null;

            const isCompleted = status === 'completed' || status === 'used';
            const isSkipped = status === 'skipped';
            const isProcessing = status === 'processing';
            const isSelected = selectedNode?.id === agent.id;

            return (
              <div
                key={agent.id}
                onClick={() => setSelectedNode(isSelected ? null : agent)}
                className={`relative rounded-lg p-3 flex flex-col justify-between border cursor-pointer transition-all min-h-[145px] select-none ${
                  isCompleted
                    ? 'bg-slate-950/90 border-emerald-500/40 hover:border-emerald-400 shadow-[0_2px_12px_rgba(16,185,129,0.08)]'
                    : isProcessing
                    ? 'bg-cyan-950/50 border-cyan-400/80 hover:border-cyan-300 shadow-[0_0_15px_rgba(6,182,212,0.25)]'
                    : isSkipped
                    ? 'bg-slate-950/60 border-slate-800/80 opacity-70 hover:opacity-100 hover:border-slate-700'
                    : 'bg-slate-950/40 border-slate-800/70 hover:border-slate-700'
                } ${isSelected ? 'ring-2 ring-cyan-400/80 bg-slate-900' : ''}`}
              >
                {/* Node Top Header */}
                <div className="w-full flex items-center justify-between gap-1 mb-2">
                  <span className="text-[10px] font-mono font-bold text-slate-400">
                    {agent.num}
                  </span>
                  {getStatusBadge(status)}
                </div>

                {/* Node Icon */}
                <div className="my-1 flex items-center justify-center">
                  <div
                    className={`w-9 h-9 rounded-lg flex items-center justify-center transition-all ${
                      isCompleted
                        ? 'bg-emerald-950/70 text-emerald-400 border border-emerald-500/40'
                        : isProcessing
                        ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-400 animate-pulse'
                        : isSkipped
                        ? 'bg-slate-900 text-slate-400 border border-slate-800'
                        : 'bg-slate-900 text-slate-400 border border-slate-800/80'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                </div>

                {/* Node Name & Type */}
                <div className="w-full text-center mt-1">
                  <h3 className="text-xs font-bold font-mono text-slate-100 truncate">
                    {agent.name}
                  </h3>
                  <p className="text-[9px] font-mono text-slate-400 truncate mt-0.5">
                    {agent.type}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Interactive Node Technical Inspector Drawer */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 pt-4 border-t border-slate-800/80"
          >
            <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs font-mono">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-cyan-400 font-bold uppercase">
                    [{selectedNode.num}] {selectedNode.name}
                  </span>
                  <span className="text-slate-500">•</span>
                  <span className="text-slate-300 font-semibold">{selectedNode.role}</span>
                </div>
                <p className="text-slate-400 text-[11px]">
                  <strong>Technical Spec:</strong> {selectedNode.spec}
                </p>
                <p className="text-slate-400 text-[11px]">
                  <strong>Function:</strong> {selectedNode.details}
                </p>
              </div>

              <button
                onClick={() => setSelectedNode(null)}
                className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-400 hover:text-white text-[11px] self-end sm:self-auto"
              >
                Close Inspector
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
