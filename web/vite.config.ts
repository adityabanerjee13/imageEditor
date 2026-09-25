import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// dev: the API and the files it serves come from uvicorn on :8000
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/files': 'http://127.0.0.1:8000',
    },
  },
})
