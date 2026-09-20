import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5173,
    proxy: {
      '/health': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      },
      '/detect': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      },
      '/result': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      },
      '/samples': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      },
      '/sample-image': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      },
      '/detect-video': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      },
      '/result-video': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      },
      '/history': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      },
      '/benchmark': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      },
    },
  },
})
