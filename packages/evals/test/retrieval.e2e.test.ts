/**
 * Retrieval quality with real embedding models and the reranker (CPU, Node).
 *   pnpm --filter @enclave/evals test:retrieval
 * Writes reports/retrieval.json and asserts floors for the default setup.
 */
import { writeFileSync } from 'node:fs'
import { afterAll, describe, expect, it } from 'vitest'
import { Knowledge } from 'enclave-ai'
import { evalRetrieval, wilson } from 'enclave-ai/eval'
import { createDb } from 'enclave-ai/pglite'
import { transformersEmbedder, transformersReranker } from 'enclave-ai/transformers'
import { CORE_MIGRATIONS, migrate } from '../../core/src/store/migrate.js'
import { FULL_CORPUS } from '../src/fixtures/corpus.js'
import { RETRIEVAL_QUERIES } from '../src/retrieval.js'

const run = !!process.env.ENCLAVE_E2E
const EMBEDDINGS = ['embeddinggemma-2', 'embeddinggemma', 'granite-small-r2', 'granite-multilingual-r2', 'gte-small'] as const
const rows: Record<string, unknown>[] = []

async function knowledgeFor(preset: string, reranker?: string) {
  const db = await createDb({ dataDir: 'memory://' })
  await migrate(db, 'core', CORE_MIGRATIONS)
  const kb = new Knowledge(db, transformersEmbedder({ preset }), reranker ? { reranker: transformersReranker({ preset: reranker }) } : {})
  await kb.ingest(FULL_CORPUS, { collection: 'handbook' })
  return kb
}

async function measure(label: string, kb: Knowledge, mode: 'hybrid' | 'vector' | 'keyword' = 'hybrid') {
  const started = performance.now()
  const at3 = await evalRetrieval(kb, RETRIEVAL_QUERIES, { k: 3, mode })
  const ms = (performance.now() - started) / RETRIEVAL_QUERIES.length
  const hits = at3.perQuery.filter((p) => p.recall > 0).length
  const byTag: Record<string, string> = {}
  for (const tag of [...new Set(RETRIEVAL_QUERIES.flatMap((q) => q.tags))].sort()) {
    const idx = RETRIEVAL_QUERIES.flatMap((q, i) => (q.tags.includes(tag) ? [i] : []))
    byTag[tag] = `${idx.filter((i) => at3.perQuery[i]!.recall > 0).length}/${idx.length}`
  }
  const misses = at3.perQuery.flatMap((p, i) => (p.recall > 0 ? [] : [`${p.query} → ${p.retrieved.join(', ')} (want ${RETRIEVAL_QUERIES[i]!.relevant.join(', ')})`]))
  const ci = wilson(hits, RETRIEVAL_QUERIES.length)
  const row = { label, mode, recallAt3: +at3.recall.toFixed(3), ci: [+ci.low.toFixed(2), +ci.high.toFixed(2)], mrr: +at3.mrr.toFixed(3), ndcg: +at3.ndcg.toFixed(3), msPerQuery: Math.round(ms), byTag, misses }
  rows.push(row)
  console.log(`${label.padEnd(40)} recall@3 ${row.recallAt3} [${row.ci.join('–')}] MRR ${row.mrr} nDCG ${row.ndcg} ${row.msPerQuery}ms/q`)
  return row
}

afterAll(() => {
  if (rows.length) writeFileSync(new URL('../reports/retrieval.json', import.meta.url), JSON.stringify({ date: new Date().toISOString(), queries: RETRIEVAL_QUERIES.length, rows }, null, 2))
})

describe.skipIf(!run)(`retrieval benchmark (${RETRIEVAL_QUERIES.length} labeled queries)`, () => {
  for (const preset of EMBEDDINGS) {
    it(`${preset}: hybrid, vector, keyword, and hybrid + rerank`, async () => {
      const kb = await knowledgeFor(preset)
      const hybrid = await measure(`${preset} hybrid`, kb, 'hybrid')
      await measure(`${preset} vector`, kb, 'vector')
      if (preset !== 'embeddinggemma') return
      await measure('keyword only', kb, 'keyword')
      await measure(`${preset} hybrid + mxbai rerank`, await knowledgeFor(preset, 'mxbai-rerank-xsmall'), 'hybrid')
      await measure(`${preset} vector + mxbai rerank`, await knowledgeFor(preset, 'mxbai-rerank-xsmall'), 'vector')
      await measure(`${preset} hybrid + bge-m3 rerank`, await knowledgeFor(preset, 'bge-reranker-v2-m3'), 'hybrid')
      // Floors for the default embedding; raise them as quality improves.
      expect(hybrid.recallAt3).toBeGreaterThanOrEqual(0.85)
    }, 900_000)
  }
})
