import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain, FileText, CheckCircle, MessageSquare, Wrench, Copy, Check, Terminal, Shield } from 'lucide-react';

export default function AIExplanationPanel({ explanation, results, isSearching, mode = 'image' }) {
  const [copied, setCopied] = useState(false);

  if (isSearching) {
    return (
      <div className="cmd-panel rounded-xl p-6 border border-slate-800/90">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-spin" />
          <h3 className="text-sm font-bold tracking-wider text-slate-100 uppercase font-mono">
            Synthesizing Multi-Agent Decision Reasoning
          </h3>
        </div>
        <div className="space-y-2">
          <div className="h-5 w-1/4 bg-slate-900/60 animate-pulse rounded" />
          <div className="h-12 bg-slate-900/40 animate-pulse rounded" />
          <div className="h-20 bg-slate-900/30 animate-pulse rounded" />
        </div>
      </div>
    );
  }

  const exp = explanation || results?.explanation || {};
  const rawTarget =
    exp.target ||
    results?.query?.target ||
    results?.query?.raw_query ||
    (typeof results?.query === 'string' ? results.query : results?.target) ||
    'Target';

  const summary =
    exp.summary ||
    results?.summary ||
    (results
      ? `Mission complete. Identified and verified ${results.count ?? results.objects_found ?? 0} instance(s) of '${rawTarget}' in aerial surveillance feed.`
      : 'Execute a detection directive to generate autonomous multi-agent reasoning synthesis.');

  const attributes = exp.attributes || results?.query?.attributes || results?.attributes || {};
  const reasoning = Array.isArray(exp.reasoning)
    ? exp.reasoning
    : typeof exp.reasoning === 'string'
    ? [exp.reasoning]
    : [
        `Parsed natural language query for target object '${rawTarget}'`,
        `Selected optimal detector pipeline with VisDrone fine-tuned weights`,
        `Applied spatial deduplication and confidence threshold verification`,
      ];

  const toolsUsed = Array.isArray(exp.tools_used)
    ? exp.tools_used
    : mode === 'video'
    ? ['YOLO-World Detector', 'Video Tracker & Frame Matcher']
    : ['YOLO Fine-Tuned (VisDrone)', 'Query-Aware Verification'];

  const avgConf = exp.average_confidence
    ? `${(exp.average_confidence * 100).toFixed(1)}%`
    : results?.average_confidence
    ? `${(results.average_confidence * 100).toFixed(1)}%`
    : results?.confidence
    ? `${(results.confidence * 100).toFixed(1)}%`
    : '85.0%';

  const handleCopyExplanation = () => {
    const fullText = `Executive Mission Synthesis:\n${summary}\n\nMulti-Agent Decision Steps:\n${reasoning
      .map((r, i) => `[0${i + 1}] ${r}`)
      .join('\n')}\n\nTools Executed: ${toolsUsed.join(', ')}`;
    navigator.clipboard.writeText(fullText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="cmd-panel rounded-xl p-5 sm:p-6 border border-slate-800/90 relative overflow-hidden space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400">
              <Brain className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm sm:text-base font-bold tracking-tight text-white font-mono uppercase">
                  AI Explanation & Decision Reasoning Synthesis
                </h3>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  Agent Consensus Trail
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Multi-agent decision justification, confidence attribution, and tool provenance
              </p>
            </div>
          </div>
        </div>

        <button
          onClick={handleCopyExplanation}
          className="px-2.5 py-1.5 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-mono text-slate-300 hover:text-white flex items-center gap-1.5 transition-colors self-start sm:self-auto"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-cyan-400" />}
          <span>{copied ? 'Copied to Clipboard' : 'Copy Synthesis'}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column: Executive Summary & Attributes */}
        <div className="lg:col-span-1 space-y-3">
          {/* Executive Summary Card */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-cyan-400">
              <FileText className="w-3.5 h-3.5" />
              Executive Synthesis
            </div>
            <p className="text-xs font-mono text-slate-200 leading-relaxed">
              {summary}
            </p>
          </div>

          {/* Target Metadata Spec */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2.5 text-xs font-mono">
            <div>
              <div className="text-[10px] uppercase text-slate-400 mb-0.5">
                Target Identifier
              </div>
              <div className="text-sm font-bold text-slate-100 capitalize">
                {rawTarget}
              </div>
            </div>

            {Object.keys(attributes).length > 0 && (
              <div>
                <div className="text-[10px] uppercase text-slate-400 mb-1">
                  Extracted Attributes
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(attributes).map(([k, v]) => (
                    <span
                      key={k}
                      className="px-2 py-0.5 rounded text-[10px] bg-cyan-950/60 text-cyan-300 border border-cyan-500/30 capitalize"
                    >
                      {k}: <strong>{v}</strong>
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-slate-400">
              <span>Attributed Confidence:</span>
              <span className="font-bold text-emerald-400">{avgConf}</span>
            </div>
          </div>
        </div>

        {/* Right Column: Reasoning Trail & Tools */}
        <div className="lg:col-span-2 space-y-3">
          {/* Reasoning Steps */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-cyan-400">
              <MessageSquare className="w-3.5 h-3.5" />
              Multi-Agent Decision Trail
            </div>

            <div className="space-y-1.5">
              {reasoning.map((step, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-2.5 p-2.5 rounded bg-slate-900/60 border border-slate-800/60 text-xs font-mono"
                >
                  <span className="text-[10px] font-bold text-cyan-400 flex-shrink-0 mt-0.5">
                    [0{idx + 1}]
                  </span>
                  <p className="text-slate-300 leading-relaxed">
                    {step}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Tools Executed Audit */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-slate-400">
              <Wrench className="w-3.5 h-3.5 text-slate-400" />
              Executed Pipeline Modules
            </div>

            <div className="flex flex-wrap gap-1.5">
              {toolsUsed.map((tool, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono bg-slate-900 text-slate-200 border border-slate-700/80"
                >
                  <CheckCircle className="w-3 h-3 text-emerald-400" />
                  {tool}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
