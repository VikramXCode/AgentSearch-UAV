import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ListFilter, CheckCircle2, ShieldAlert, Car, Box, Crosshair, Search, Tag, Eye } from 'lucide-react';

export default function DetectedObjectList({ detections, isSearching, mode = 'image' }) {
  const [showBBoxes, setShowBBoxes] = useState(false);
  const [filterText, setFilterText] = useState('');

  if (isSearching) {
    return (
      <div className="cmd-panel rounded-xl p-6 border border-slate-800/90">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-spin" />
          <h3 className="text-sm font-bold tracking-wider text-slate-100 uppercase font-mono">
            Verified Object Data Matrix
          </h3>
        </div>
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-10 rounded bg-slate-900/60 animate-pulse border border-slate-800" />
          ))}
        </div>
      </div>
    );
  }

  const items = (detections || []).filter((item) => {
    if (!filterText) return true;
    const label = (item.class || item.label || '').toLowerCase();
    return label.includes(filterText.toLowerCase());
  });

  return (
    <div className="cmd-panel rounded-xl p-5 sm:p-6 border border-slate-800/90 relative overflow-hidden space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-slate-900 border border-slate-700/80 flex items-center justify-center text-cyan-400">
              <Crosshair className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm sm:text-base font-bold tracking-tight text-white font-mono uppercase">
                  Verified Targets Matrix
                </h3>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  {items.length} Records
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                {mode === 'video'
                  ? 'Spatial bounding boxes and continuous trajectory IDs'
                  : 'NMS deduplicated and attribute-verified aerial bounding boxes'}
              </p>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Quick Filter */}
          <div className="relative flex items-center">
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="Filter class..."
              className="pl-8 pr-3 py-1.5 rounded text-xs font-mono bg-slate-950 border border-slate-700/80 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 w-32 sm:w-40"
            />
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 pointer-events-none" />
          </div>

          <button
            onClick={() => setShowBBoxes(!showBBoxes)}
            className="px-2.5 py-1.5 rounded text-xs font-mono font-medium bg-slate-900 text-slate-300 border border-slate-700/80 hover:bg-slate-800 transition-colors flex items-center gap-1.5"
          >
            <Crosshair className="w-3 h-3 text-cyan-400" />
            <span>{showBBoxes ? 'Hide BBoxes' : 'Show BBoxes'}</span>
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="text-center py-8 rounded-lg bg-slate-950 border border-slate-800/80">
          <Box className="w-8 h-8 text-slate-400 mx-auto mb-1.5" />
          <p className="text-xs font-semibold text-slate-400 font-mono">No verified targets matching filter</p>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            Modify search directive or clear filter parameters.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-400 bg-slate-900/80">
                <th className="py-2.5 px-3">Target ID</th>
                <th className="py-2.5 px-3">Object Class</th>
                <th className="py-2.5 px-3">Confidence Score</th>
                <th className="py-2.5 px-3">Attributes</th>
                <th className="py-2.5 px-3">Verification</th>
                {showBBoxes && <th className="py-2.5 px-3">Bounding Box [x1, y1, x2, y2]</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {items.map((item, idx) => {
                const rawConf = typeof item.confidence === 'number' ? item.confidence : 0.85;
                const confPct = rawConf <= 1.0 ? (rawConf * 100).toFixed(1) : rawConf.toFixed(1);
                const label = item.class || item.label || 'Target';
                const attributes = item.attributes || {};
                const attrColor = attributes.color || (item.color ? item.color : null);
                const isVerified = item.verified !== false;
                const bbox = item.bbox;
                const trackId = item.track_id;

                return (
                  <tr
                    key={idx}
                    className="hover:bg-slate-900/60 transition-colors group"
                  >
                    <td className="py-2.5 px-3 text-slate-400">
                      {trackId !== undefined ? `TRK-${trackId}` : `TGT-${(idx + 1).toString().padStart(2, '0')}`}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className="font-semibold text-slate-200 capitalize">
                        {label}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-200">{confPct}%</span>
                        <div className="w-14 bg-slate-800 h-1 rounded-full overflow-hidden hidden sm:block">
                          <div
                            className="bg-cyan-400 h-full rounded-full"
                            style={{ width: `${confPct}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-2.5 px-3">
                      {attrColor ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] bg-cyan-950/60 text-cyan-300 border border-cyan-500/30 capitalize">
                          <span
                            className="w-1.5 h-1.5 rounded-full"
                            style={{
                              backgroundColor:
                                attrColor.toLowerCase() === 'red'
                                  ? '#ef4444'
                                  : attrColor.toLowerCase() === 'white'
                                  ? '#f8fafc'
                                  : attrColor.toLowerCase() === 'black'
                                  ? '#0f172a'
                                  : attrColor.toLowerCase() === 'blue'
                                  ? '#3b82f6'
                                  : attrColor.toLowerCase() === 'green'
                                  ? '#22c55e'
                                  : '#06b6d4',
                            }}
                          />
                          {attrColor}
                        </span>
                      ) : (
                        <span className="text-slate-400 text-[10px]">Standard</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3">
                      {isVerified ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950/60 text-emerald-400 border border-emerald-500/30">
                          <CheckCircle2 className="w-2.5 h-2.5" />
                          Verified
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-950/60 text-amber-400 border border-amber-500/30">
                          <ShieldAlert className="w-2.5 h-2.5" />
                          Candidate
                        </span>
                      )}
                    </td>
                    {showBBoxes && (
                      <td className="py-2.5 px-3 text-[10px] text-slate-400">
                        {bbox && Array.isArray(bbox)
                          ? `[${bbox.map((n) => Math.round(n)).join(', ')}]`
                          : 'N/A'}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
