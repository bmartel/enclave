/**
 * Embedding models served by Ollama: retrieval quality on the 91 labeled
 * queries (vector and hybrid) and relevance-floor calibration.
 *   ENCLAVE_E2E=1 npx vitest run test/local-embeddings.e2e.test.ts
 * Writes reports/local-embeddings.json.
 */
import { writeFileSync } from 'node:fs'
import { afterAll, describe, it } from 'vitest'
import { Knowledge, type Embedder } from 'enclave-ai'
import { evalRetrieval } from 'enclave-ai/eval'
import { localEmbedder } from 'enclave-ai/models/local'
import { createDb } from 'enclave-ai/pglite'
import { CORE_MIGRATIONS, migrate } from '../../core/src/store/migrate.js'
import { FULL_CORPUS } from '../src/fixtures/corpus.js'
import { CONVERSATIONAL_QUERIES, RETRIEVAL_QUERIES } from '../src/retrieval.js'

// Qwen's documented format: no space after "Query:".
const QWEN_QUERY = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
const CANDIDATES: { label: string; embedder: () => Embedder }[] = [
  { label: 'qwen3-embedding:0.6b', embedder: () => localEmbedder({ provider: 'ollama', model: 'qwen3-embedding:0.6b', dimensions: 1024, queryPrefix: QWEN_QUERY }) },
  { label: 'qwen3-embedding:4b@1024', embedder: () => localEmbedder({ provider: 'ollama', model: 'qwen3-embedding:4b', dimensions: 1024, queryPrefix: QWEN_QUERY }) },
  { label: 'bge-m3', embedder: () => localEmbedder({ provider: 'ollama', model: 'bge-m3', dimensions: 1024 }) },
  {
    label: 'embeddinggemma',
    embedder: () =>
      localEmbedder({ provider: 'ollama', model: 'embeddinggemma', dimensions: 768, queryPrefix: 'task: search result | query: ', documentPrefix: 'title: none | text: ' }),
  },
]

const rows: Record<string, unknown>[] = []
afterAll(() => {
  if (rows.length) writeFileSync(new URL('../reports/local-embeddings.json', import.meta.url), JSON.stringify({ date: new Date().toISOString(), queries: RETRIEVAL_QUERIES.length, rows }, null, 2))
})

describe.skipIf(!process.env.ENCLAVE_E2E)('Ollama embedders', () => {
  for (const c of CANDIDATES) {
    it(c.label, async () => {
      const db = await createDb({ dataDir: 'memory://' })
      await migrate(db, 'core', CORE_MIGRATIONS)
      const kb = new Knowledge(db, c.embedder())
      const ingestStart = performance.now()
      await kb.ingest(FULL_CORPUS, { collection: 'handbook' })
      const ingestMs = Math.round(performance.now() - ingestStart)
      const row: Record<string, unknown> = { label: c.label, ingestMs }
      for (const mode of ['vector', 'hybrid'] as const) {
        const started = performance.now()
        const r = await evalRetrieval(kb, RETRIEVAL_QUERIES, { k: 3, mode })
        const multi = RETRIEVAL_QUERIES.flatMap((q, i) => (q.tags.includes('multilingual') ? [r.perQuery[i]!.recall > 0] : []))
        row[mode] = {
          recallAt3: +r.recall.toFixed(3),
          mrr: +r.mrr.toFixed(3),
          msPerQuery: Math.round((performance.now() - started) / RETRIEVAL_QUERIES.length),
          multilingual: `${multi.filter(Boolean).length}/${multi.length}`,
          misses: r.perQuery.flatMap((p, i) => (p.recall > 0 ? [] : [`${p.query} → ${p.retrieved.join(', ')}`])),
        }
      }
      // Relevance floor: top similarity for on-topic vs conversational messages.
      const top = async (q: string) => (await kb.search(q, { limit: 1, mode: 'vector' }))[0]?.similarity ?? 0
      const on = await Promise.all(RETRIEVAL_QUERIES.map((q) => top(q.query)))
      const off = await Promise.all(CONVERSATIONAL_QUERIES.map(top))
      const sorted = [...on].sort((a, b) => a - b)
      // Highest floor that still keeps ~97% of on-topic questions.
      const floor = +sorted[Math.floor(sorted.length * 0.03)]!.toFixed(2)
      row.relevance = { floor, onKept: on.filter((v) => v >= floor).length, onTotal: on.length, offSilenced: off.filter((v) => v < floor).length, offTotal: off.length }
      rows.push(row)
      await db.close()
    }, 900_000)
  }
})
