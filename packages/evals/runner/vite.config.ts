import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

const repoRoot = fileURLToPath(new URL('../../..', import.meta.url))

export default defineConfig({
  optimizeDeps: { exclude: ['@electric-sql/pglite', '@electric-sql/pglite-pgvector'] },
  worker: { format: 'es' },
  // Workspace packages and hoisted node_modules live above the runner root.
  server: { fs: { allow: [repoRoot] } },
})
