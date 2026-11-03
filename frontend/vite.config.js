import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    modulePreload: {
      resolveDependencies: (url, deps) => {
        // Exclude heavy plotly chunk from initial index.html modulepreload
        return deps.filter((dep) => !dep.includes('plotly'))
      },
    },
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('plotly')) {
            return 'plotly'
          }
          if (id.includes('node_modules')) {
            return 'vendor'
          }
        },
      },
    },
    chunkSizeWarningLimit: 5000,
  },
  test: {
    globals: true,
    environment: 'jsdom',
  },
})
