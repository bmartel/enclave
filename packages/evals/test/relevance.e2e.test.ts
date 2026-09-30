/**
 * Relevance calibration: top cosine similarity per message for on-topic
 * questions vs conversational messages. Sets each preset's relevanceFloor.
 *   SIMS_OUT=reports/relevance.txt SIMS_PRESET=gte-small ENCLAVE_E2E=1 npx vitest run test/relevance.e2e.test.ts
 */
import { appendFileSync, writeFileSync } from 'node:fs'
import { it } from 'vitest'
import { Knowledge } from '@enclave/core'
import { createDb } from '@enclave/core/pglite'
import { transformersEmbedder } from '@enclave/core/transformers'
import { CORE_MIGRATIONS, migrate } from '../../core/src/store/migrate.js'
import { FULL_CORPUS } from '../src/fixtures/corpus.js'
import { CONVERSATIONAL_QUERIES, RETRIEVAL_QUERIES } from '../src/retrieval.js'

const OUT = process.env.SIMS_OUT ?? new URL('../reports/relevance.txt', import.meta.url).pathname
const log = (s: string) => appendFileSync(OUT, s + '\n')

const OFF = CONVERSATIONAL_QUERIES
const ON = [...RETRIEVAL_QUERIES.map((q) => q.query), 'What is the guest wifi password at the Toronto office?', 'Quick one: Toronto guest wifi password?', 'Summarize the vendor newsletter for me.', 'I have a 750 dollar conference registration to expense. Who has to approve it? Give me their name.']

it.skipIf(!process.env.ENCLAVE_E2E)('relevance calibration', async () => {
  writeFileSync(OUT, '')
  const db = await createDb({ dataDir: 'memory://' })
  await migrate(db, 'core', CORE_MIGRATIONS)
  const kb = new Knowledge(db, transformersEmbedder({ preset: process.env.SIMS_PRESET ?? 'embeddinggemma' }))
  await kb.ingest(FULL_CORPUS, { collection: 'handbook' })
  const dump = async (label: string, qs: string[]) => {
    log(`\n## ${label}`)
    for (const q of qs) {
      const hits = await kb.search(q, { limit: 5, mode: 'vector' })
      const top = hits[0]?.similarity ?? 0
      log(`${top.toFixed(3)} | ${hits.map((h) => `${h.documentId}:${h.similarity?.toFixed(2)}(${((h.similarity ?? 0) / top).toFixed(2)})`).join(' ')} | ${q.slice(0, 70)}`)
    }
  }
  await dump('OFF-TOPIC / conversational', OFF)
  await dump('ON-TOPIC', ON)
}, 600_000)
