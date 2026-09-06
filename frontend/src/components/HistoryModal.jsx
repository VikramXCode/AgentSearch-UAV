import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, History, RotateCcw, Clock, Target, CheckCircle2, AlertCircle } from 'lucide-react';

export default function HistoryModal({ isOpen, onClose, onSelectHistoryItem }) {
  const [historyList, setHistoryList] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetch('http://localhost:5001/history')
        .then((res) => res.json())
        .then((data) => {
          if (data.status === 'success') {
            setHistoryList(data.history || []);
          }
        })
        .catch((err) => console.warn('History fetch error:', err))
        .finally(() => setLoading(false));
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/85 backdrop-blur-md">
        <div className="absolute inset-0" onClick={onClose} />

        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 10 }}
          className="relative max-w-4xl w-full max-h-[85vh] bg-[#0d121c] border border-slate-700/80 rounded-xl overflow-hidden flex flex-col shadow-2xl z-10"
        >
          {/* Header */}
          <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded bg-slate-900 border border-slate-700 flex items-center justify-center text-cyan-400">
                <History className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm sm:text-base font-bold font-mono text-white flex items-center gap-2">
                  UAV Mission Knowledge History
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/30">
                    {historyList.length} Archived Missions
                  </span>
                </h3>
                <p className="text-xs text-slate-400 font-mono">
                  Persistent multi-agent memory records & verified reconnaissance history
                </p>
              </div>
            </div>

            <button
              onClick={onClose}
              className="p-1.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto p-5">
            {loading ? (
              <div className="text-center py-10 text-slate-400 font-mono text-xs">
                Querying knowledge agent memory records...
              </div>
            ) : historyList.length === 0 ? (
              <div className="text-center py-10 text-slate-500 font-mono text-xs">
                No archived surveillance missions recorded in memory.
              </div>
            ) : (
              <div className="space-y-2">
                {historyList.map((item, idx) => {
                  const dateStr = item.timestamp
                    ? new Date(item.timestamp).toLocaleString()
                    : 'Recent';
                  const isSuccess = item.status === 'Completed' || (item.objects_found && item.objects_found > 0);

                  return (
                    <div
                      key={idx}
                      className="p-3 rounded-lg bg-slate-950 border border-slate-800 hover:border-slate-700 transition-colors flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs font-mono"
                    >
                      <div className="space-y-1 flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-slate-100 capitalize truncate">
                            "{item.query || item.target || 'Target search'}"
                          </span>
                          {isSuccess ? (
                            <span className="inline-flex items-center gap-1 text-[9px] px-1.5 py-0.2 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/40">
                              <CheckCircle2 className="w-2.5 h-2.5" /> {item.objects_found ?? 0} found
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-[9px] px-1.5 py-0.2 rounded bg-slate-900 text-slate-400 border border-slate-800">
                              <AlertCircle className="w-2.5 h-2.5" /> 0 matches
                            </span>
                          )}
                        </div>

                        <div className="flex flex-wrap items-center gap-2.5 text-[11px] text-slate-400">
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3 text-slate-500" />
                            {dateStr}
                          </span>
                          <span>•</span>
                          <span className="text-slate-300">
                            Detector: {item.detector || 'YOLO-World'}
                          </span>
                          {item.confidence_score > 0 && (
                            <>
                              <span>•</span>
                              <span className="text-emerald-400">
                                Conf: {(item.confidence_score * 100).toFixed(1)}%
                              </span>
                            </>
                          )}
                        </div>
                      </div>

                      <button
                        onClick={() => {
                          if (onSelectHistoryItem) {
                            onSelectHistoryItem(item);
                            onClose();
                          }
                        }}
                        className="px-2.5 py-1.5 rounded text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-slate-700 transition-colors flex items-center gap-1.5 flex-shrink-0"
                      >
                        <RotateCcw className="w-3 h-3" />
                        <span>Load Query</span>
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
