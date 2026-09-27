import { defineConfig } from 'vite'

export default defineConfig({
  // PGlite ships its own WASM + data files; pre-bundling breaks their URLs.
  optimizeDeps: { exclude: ['@electric-sql/pglite', '@electric-sql/pglite-pgvector'] },
  worker: { format: 'es' },
  build: { rollupOptions: { input: { main: 'index.html', eval: 'eval.html' } } },
  server: {
    headers: {
      // Enables SharedArrayBuffer / multi-threaded WASM where supported.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
})
