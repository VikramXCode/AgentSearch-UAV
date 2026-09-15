import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, ChevronDown, ChevronUp, Copy, Check, Trash2, Download, Filter } from 'lucide-react';

export default function TechnicalProcessingLog({ logs = [], onClearLogs }) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [copied, setCopied] = useState(false);
  const [filterLevel, setFilterLevel] = useState('ALL'); // 'ALL', 'AGENT', 'INFO', 'WARN', 'ERROR'
  const logContainerRef = useRef(null);

  useEffect(() => {
    if (isExpanded && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, isExpanded]);

  const handleCopyLogs = (e) => {
    e.stopPropagation();
    const text = logs.map((l) => `[${l.timestamp}] [${l.level.toUpperCase()}] ${l.message}`).join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadLogs = (e) => {
    e.stopPropagation();
    const text = logs.map((l) => `[${l.timestamp}] [${l.level.toUpperCase()}] ${l.message}`).join('\n');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agentsearch_uav_telemetry_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const filteredLogs = logs.filter((log) => {
    if (filterLevel === 'ALL') return true;
    return log.level.toUpperCase() === filterLevel;
  });

  return (
    <div className="cmd-panel rounded-xl border border-slate-800/90 overflow-hidden transition-all duration-200">
      {/* Header Bar */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="px-5 py-3.5 flex items-center justify-between cursor-pointer bg-slate-950/90 hover:bg-slate-900/60 transition-colors select-none"
      >
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400">
            <Terminal className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs sm:text-sm font-bold font-mono tracking-wider text-slate-100 uppercase">
                Real-Time Telemetry & Agent Hooks Log
              </span>
              <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-slate-800 text-slate-400 border border-slate-700">
                {logs.length} events
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-mono mt-0.5">
              Live multi-agent communication trace and model runtime stderr/stdout hooks
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </div>

      {/* Collapsible Terminal Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-slate-800/80 bg-slate-950"
          >
            {/* Toolbar */}
            <div className="px-4 py-2 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
              {/* Filter Tabs */}
              <div className="flex items-center gap-1">
                <span className="text-slate-400 text-[10px] uppercase mr-1">Filter:</span>
                {['ALL', 'AGENT', 'INFO', 'ERROR'].map((lvl) => (
                  <button
                    key={lvl}
                    onClick={() => setFilterLevel(lvl)}
                    className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                      filterLevel === lvl
                        ? 'bg-slate-800 text-cyan-300 font-bold border border-slate-700'
                        : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>

              {/* Actions */}
              <div className="flex items-center gap-1.5">
                <button
                  onClick={handleCopyLogs}
                  className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 text-[11px] flex items-center gap-1 transition-colors"
                >
                  {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3 text-cyan-400" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
                <button
                  onClick={handleDownloadLogs}
                  className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 text-[11px] flex items-center gap-1 transition-colors"
                >
                  <Download className="w-3 h-3" />
                  <span>Save</span>
                </button>
                {onClearLogs && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onClearLogs();
                    }}
                    className="px-2 py-1 rounded bg-slate-900 hover:bg-rose-950/40 text-slate-400 hover:text-rose-300 border border-slate-700 text-[11px] flex items-center gap-1 transition-colors"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span>Clear</span>
                  </button>
                )}
              </div>
            </div>

            {/* Log Stream */}
            <div
              ref={logContainerRef}
              className="p-4 max-h-64 overflow-y-auto font-mono text-xs space-y-1 scrollbar-thin"
            >
              {filteredLogs.length === 0 ? (
                <div className="text-slate-500 text-xs italic py-2">No telemetry events matching filter.</div>
              ) : (
                filteredLogs.map((log, idx) => {
                  const levelBadge =
                    log.level === 'error'
                      ? 'text-red-400 bg-red-950/40 border-red-800/50'
                      : log.level === 'warn'
                      ? 'text-amber-400 bg-amber-950/40 border-amber-800/50'
                      : log.level === 'agent'
                      ? 'text-purple-400 bg-purple-950/40 border-purple-800/50'
                      : 'text-cyan-400 bg-cyan-950/30 border-cyan-800/40';

                  return (
                    <div
                      key={idx}
                      className="flex items-start gap-2 py-0.5 px-1.5 rounded hover:bg-slate-900/50 transition-colors"
                    >
                      <span className="text-slate-500 text-[11px] flex-shrink-0 select-none">
                        [{log.timestamp}]
                      </span>
                      <span
                        className={`px-1 py-0.2 rounded border text-[9px] font-bold flex-shrink-0 uppercase ${levelBadge}`}
                      >
                        {log.level}
                      </span>
                      <span className="text-slate-300 break-all leading-relaxed text-[11px]">
                        {log.message}
                      </span>
                    </div>
                  );
                })
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
