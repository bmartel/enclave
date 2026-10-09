/**
 * EmbeddingGemma 2 against EmbeddingGemma 300M (v1) on the 91 labeled
 * queries (CPU, Node): recall@1/@3, MRR, nDCG, multilingual, the query task
 * prompt (search vs question answering), Matryoshka sizes, ingest and query
 * time, and the relevance floor (top similarity of on-topic vs conversational
 * messages).
 *   ENCLAVE_E2E=1 npx vitest run test/embedding-upgrade.e2e.test.ts
 * Writes reports/embedding-upgrade.json.
 */
import { writeFileSync } from 'node:fs'
import { afterAll, describe, it } from 'vitest'
import { Knowledge, type EmbedTask } from 'enclave-ai'
import { evalRetrieval, wilson } from 'enclave-ai/eval'
import { createDb } from 'enclave-ai/pglite'
import { transformersEmbedder } from 'enclave-ai/transformers'
import { CORE_MIGRATIONS, migrate } from '../../core/src/store/migrate.js'
import { FULL_CORPUS } from '../src/fixtures/corpus.js'
import { CONVERSATIONAL_QUERIES, RETRIEVAL_QUERIES } from '../src/retrieval.js'

if (process.env.HF_CACHE) {
  const { env } = await import('@huggingface/transformers')
  env.cacheDir = process.env.HF_CACHE
}

interface Candidate {
  label: string
  preset: string
  dimensions?: number
  dtype?: string
  tasks?: EmbedTask[]
}

const CANDIDATES: Candidate[] = (
  [
    { label: 'embeddinggemma v1 q8', preset: 'embeddinggemma', tasks: ['search', 'question-answering'] },
    { label: 'embeddinggemma v1 q4', preset: 'embeddinggemma', dtype: 'q4' },
    { label: 'embeddinggemma-2 q8', preset: 'embeddinggemma-2', tasks: ['search', 'question-answering'] },
    { label: 'embeddinggemma-2 q4', preset: 'embeddinggemma-2', dtype: 'q4', tasks: ['search', 'question-answering'] },
    { label: 'embeddinggemma-2 q8 @512', preset: 'embeddinggemma-2', dimensions: 512 },
    { label: 'embeddinggemma-2 q8 @256', preset: 'embeddinggemma-2', dimensions: 256 },
  ] as Candidate[]
).filter((c) => !process.env.ONLY || new RegExp(process.env.ONLY).test(c.label))

const rows: Record<string, unknown>[] = []
afterAll(() => {
  if (rows.length) writeFileSync(new URL('../reports/embedding-upgrade.json', import.meta.url), JSON.stringify({ date: new Date().toISOString(), queries: RETRIEVAL_QUERIES.length, rows }, null, 2))
})

describe.skipIf(!process.env.ENCLAVE_E2E)('EmbeddingGemma 2 vs v1', () => {
  for (const c of CANDIDATES) {
    it(c.label, async () => {
      const db = await createDb({ dataDir: 'memory://' })
      await migrate(db, 'core', CORE_MIGRATIONS)
      const embedder = transformersEmbedder({ preset: c.preset, ...(c.dimensions ? { dimensions: c.dimensions } : {}), ...(c.dtype ? { dtype: c.dtype } : {}) })
      const loadStart = performance.now()
      await embedder.load!()
      const loadMs = Math.round(performance.now() - loadStart)
      const kb = new Knowledge(db, embedder)
      const ingestStart = performance.now()
      const ingested = await kb.ingest(FULL_CORPUS, { collection: 'handbook' })
      const ingestMs = Math.round(performance.now() - ingestStart)
      const row: Record<string, unknown> = { label: c.label, loadMs, ingestMs, chunks: ingested.chunks, msPerChunk: +(ingestMs / ingested.chunks).toFixed(1) }
      for (const task of c.tasks ?? ['search']) {
        for (const mode of ['vector', 'hybrid'] as const) {
          const started = performance.now()
          const at3 = await evalRetrieval(kb, RETRIEVAL_QUERIES, { k: 3, mode, task })
          const ms = (performance.now() - started) / RETRIEVAL_QUERIES.length
          const at1 = await evalRetrieval(kb, RETRIEVAL_QUERIES, { k: 1, mode, task })
          const hits = at3.perQuery.filter((p) => p.recall > 0).length
          const ci = wilson(hits, RETRIEVAL_QUERIES.length)
          const multi = RETRIEVAL_QUERIES.flatMap((q, i) => (q.tags.includes('multilingual') ? [at3.perQuery[i]!.recall > 0] : []))
          row[`${task}/${mode}`] = {
            recallAt1: +at1.recall.toFixed(3),
            recallAt3: +at3.recall.toFixed(3),
            ci: [+ci.low.toFixed(2), +ci.high.toFixed(2)],
            mrr: +at3.mrr.toFixed(3),
            ndcg: +at3.ndcg.toFixed(3),
            msPerQuery: Math.round(ms),
            multilingual: `${multi.filter(Boolean).length}/${multi.length}`,
            misses: at3.perQuery.flatMap((p, i) => (p.recall > 0 ? [] : [`${p.query} → ${p.retrieved.join(', ')} (want ${RETRIEVAL_QUERIES[i]!.relevant.join(', ')})`])),
          }
          console.log(`${c.label.padEnd(28)} ${task.padEnd(18)} ${mode.padEnd(6)} R@1 ${at1.recall.toFixed(3)} R@3 ${at3.recall.toFixed(3)} MRR ${at3.mrr.toFixed(3)} multi ${multi.filter(Boolean).length}/${multi.length} ${Math.round(ms)} ms/q`)
        }
      }
      // Relevance floor: the highest floor that keeps ~97% of on-topic questions.
      const top = async (q: string) => (await kb.search(q, { limit: 1, mode: 'vector' }))[0]?.similarity ?? 0
      const on: number[] = []
      for (const q of RETRIEVAL_QUERIES) on.push(await top(q.query))
      const off: number[] = []
      for (const q of CONVERSATIONAL_QUERIES) off.push(await top(q))
      const sorted = [...on].sort((a, b) => a - b)
      const floor = +sorted[Math.floor(sorted.length * 0.03)]!.toFixed(2)
      row.sims = { on: on.map((v) => +v.toFixed(3)), off: off.map((v) => +v.toFixed(3)) }
      row.relevance = { floor, onKept: on.filter((v) => v >= floor).length, onTotal: on.length, offSilenced: off.filter((v) => v < floor).length, offTotal: off.length, onMedian: +sorted[Math.floor(sorted.length / 2)]!.toFixed(3), offMax: +Math.max(...off).toFixed(3) }
      console.log(c.label, 'load', loadMs, 'ms; ingest', ingestMs, 'ms for', ingested.chunks, 'chunks; floor', JSON.stringify(row.relevance))
      rows.push(row)
      await db.close()
    }, 1_800_000)
  }
})
