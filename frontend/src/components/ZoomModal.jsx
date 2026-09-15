import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ZoomIn, Download, Crosshair } from 'lucide-react';

export default function ZoomModal({ isOpen, onClose, imageUrl, title }) {
  if (!isOpen || !imageUrl) return null;

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = imageUrl;
    a.download = `uav_surveillance_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/90 backdrop-blur-md">
        {/* Backdrop click */}
        <div className="absolute inset-0" onClick={onClose} />

        {/* Modal Container */}
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 10 }}
          className="relative max-w-6xl w-full max-h-[92vh] bg-[#0d121c] border border-slate-700/80 rounded-xl overflow-hidden flex flex-col shadow-2xl z-10"
        >
          {/* Header */}
          <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950">
            <div className="flex items-center gap-2">
              <Crosshair className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs sm:text-sm font-bold font-mono text-slate-100 uppercase tracking-wider">
                {title || 'UAV Surveillance Spatial Inspection'}
              </h3>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleDownload}
                className="px-2.5 py-1.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-mono flex items-center gap-1.5 transition-colors"
                title="Download full resolution payload"
              >
                <Download className="w-3.5 h-3.5 text-cyan-400" />
                <span>Export</span>
              </button>
              <button
                onClick={onClose}
                className="p-1.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                title="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Image Viewer */}
          <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-black/90 tactical-grid">
            <img
              src={imageUrl}
              alt={title || 'UAV Preview'}
              className="max-w-full max-h-[78vh] object-contain rounded border border-slate-800 shadow-2xl"
            />
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
