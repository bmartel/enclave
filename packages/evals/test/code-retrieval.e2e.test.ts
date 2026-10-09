/**
 * Code search: natural-language questions → notebook code cells (CPU, Node),
 * EmbeddingGemma 2 vs v1, with the search prompt and the code retrieval
 * prompt (`task: 'code-retrieval'`).
 *   ENCLAVE_E2E=1 npx vitest run test/code-retrieval.e2e.test.ts
 * Writes reports/code-retrieval.json.
 */
import { writeFileSync } from 'node:fs'
import { afterAll, describe, it } from 'vitest'
import { Knowledge, type EmbedTask } from 'enclave-ai'
import { evalRetrieval } from 'enclave-ai/eval'
import { createDb } from 'enclave-ai/pglite'
import { transformersEmbedder } from 'enclave-ai/transformers'
import { CORE_MIGRATIONS, migrate } from '../../core/src/store/migrate.js'
import { CODE_CORPUS, CODE_QUERIES } from '../src/code-retrieval.js'

if (process.env.HF_CACHE) {
  const { env } = await import('@huggingface/transformers')
  env.cacheDir = process.env.HF_CACHE
}

const CANDIDATES = [
  { label: 'embeddinggemma v1 q4', preset: 'embeddinggemma', dtype: 'q4' },
  { label: 'embeddinggemma-2 q4', preset: 'embeddinggemma-2', dtype: 'q4' },
]
const rows: Record<string, unknown>[] = []
afterAll(() => {
  if (rows.length) writeFileSync(new URL('../reports/code-retrieval.json', import.meta.url), JSON.stringify({ date: new Date().toISOString(), queries: CODE_QUERIES.length, rows }, null, 2))
})

describe.skipIf(!process.env.ENCLAVE_E2E)('code search', () => {
  for (const c of CANDIDATES) {
    it(c.label, async () => {
      const db = await createDb({ dataDir: 'memory://' })
      await migrate(db, 'core', CORE_MIGRATIONS)
      const kb = new Knowledge(db, transformersEmbedder({ preset: c.preset, dtype: c.dtype }))
      await kb.ingest(CODE_CORPUS, { collection: 'code' })
      const row: Record<string, unknown> = { label: c.label }
      for (const task of ['search', 'code-retrieval'] as EmbedTask[]) {
        for (const mode of ['vector', 'hybrid'] as const) {
          const at1 = await evalRetrieval(kb, CODE_QUERIES, { k: 1, mode, task })
          const at3 = await evalRetrieval(kb, CODE_QUERIES, { k: 3, mode, task })
          row[`${task}/${mode}`] = {
            recallAt1: +at1.recall.toFixed(3),
            recallAt3: +at3.recall.toFixed(3),
            mrr: +at3.mrr.toFixed(3),
            misses: at1.perQuery.flatMap((p, i) => (p.recall > 0 ? [] : [`${p.query} → ${p.retrieved.join(', ')} (want ${CODE_QUERIES[i]!.relevant[0]})`])),
          }
        }
      }
      rows.push(row)
      await db.close()
    }, 900_000)
  }
})
