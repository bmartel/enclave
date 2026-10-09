import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { PGlite } from '@electric-sql/pglite'
import { Knowledge, toOrQuery } from '../src/rag/knowledge.js'
import { migrate, CORE_MIGRATIONS } from '../src/store/migrate.js'
import { hashEmbedder } from '../src/testing.js'
import { memoryDb } from './helpers.js'

let db: PGlite
let kb: Knowledge

beforeAll(async () => {
  db = await memoryDb()
  await migrate(db, 'core', CORE_MIGRATIONS)
  kb = new Knowledge(db, hashEmbedder(128))
  await kb.ingest(
    [
      { id: 'vpn', title: 'VPN setup', content: 'Install the client and sign in with your badge number to reach the intranet.', metadata: { team: 'it' } },
      { id: 'pto', title: 'Vacation policy', content: 'Employees accrue fifteen vacation days per year. Requests go to your manager.', metadata: { team: 'hr' } },
      { id: 'expenses', title: 'Expenses', content: 'Submit receipts within thirty days. Meals are reimbursed up to a daily limit.', metadata: { team: 'finance' } },
    ],
    { collection: 'handbook' },
  )
})
afterAll(() => db.close())

describe('Knowledge', () => {
  it('finds the relevant document with hybrid search', async () => {
    const hits = await kb.search('how many vacation days do I get')
    expect(hits[0]?.documentId).toBe('pto')
    expect(hits[0]?.title).toBe('Vacation policy')
    expect(hits[0]?.similarity).toBeGreaterThan(0)
  })

  it('supports keyword-only and vector-only modes', async () => {
    expect((await kb.search('receipts', { mode: 'keyword' }))[0]?.documentId).toBe('expenses')
    expect((await kb.search('badge intranet client', { mode: 'vector' }))[0]?.documentId).toBe('vpn')
  })

  it('filters by metadata and collection', async () => {
    const hits = await kb.search('days', { filter: { team: 'finance' } })
    expect(hits.map((h) => h.documentId)).toEqual(['expenses'])
    expect(await kb.search('days', { collection: 'nope' })).toEqual([])
  })

  it('skips unchanged documents and replaces changed ones', async () => {
    const again = await kb.ingest({ id: 'vpn', title: 'VPN setup', content: 'Install the client and sign in with your badge number to reach the intranet.', metadata: { team: 'it' } }, { collection: 'handbook' })
    expect(again).toEqual({ documents: 0, chunks: 0, updated: 0, skipped: 1 })
    const changed = await kb.ingest({ id: 'vpn', title: 'VPN setup', content: 'Use the new zero trust agent instead.', metadata: { team: 'it' }, collection: 'handbook' })
    expect(changed.documents).toBe(1)
    expect((await kb.search('zero trust agent'))[0]?.content).toContain('zero trust')
  })

  it('embeds chunks from many documents in shared batches, in order', async () => {
    const inner = hashEmbedder(32)
    const calls: number[] = []
    const counting = { ...inner, id: inner.id, dimensions: inner.dimensions, embed: (texts: string[], kind: 'query' | 'document') => (calls.push(texts.length), inner.embed(texts, kind)) }
    const own = await memoryDb()
    await migrate(own, 'core', CORE_MIGRATIONS)
    const kb2 = new Knowledge(own, counting, { batchSize: 16 })
    const docs = Array.from({ length: 40 }, (_, i) => ({ id: `batch-${i}`, title: `Doc ${i}`, content: `Short note number ${i} about batching.` }))
    const progress: string[] = []
    const r = await kb2.ingest(docs, { collection: 'batching', onProgress: (p) => progress.push(p.document) })
    expect(r).toMatchObject({ documents: 40, chunks: 40 })
    expect(calls.every((n) => n <= 16)).toBe(true)
    expect(calls.length).toBeLessThanOrEqual(3) // 40 one-chunk docs in batches of 16, not 40 calls
    expect(progress).toEqual(docs.map((d) => d.id))
    expect((await kb2.search('note number 7 batching', { collection: 'batching', mode: 'keyword' }))[0]?.documentId).toBe('batch-7')
    // Unchanged documents are skipped without embedding.
    calls.length = 0
    expect((await kb2.ingest(docs, { collection: 'batching' })).skipped).toBe(40)
    expect(calls).toEqual([])
    await own.close()
  })

  it('updates metadata without re-embedding when only metadata changes', async () => {
    const before = await db.query<{ id: number }>(`select id from enclave.chunks where document_id = 'vpn'`)
    const moved = await kb.ingest({ id: 'vpn', title: 'VPN setup', content: 'Use the new zero trust agent instead.', metadata: { team: 'security' }, collection: 'handbook' })
    expect(moved).toEqual({ documents: 0, chunks: 0, updated: 1, skipped: 0 })
    const after = await db.query<{ id: number }>(`select id from enclave.chunks where document_id = 'vpn'`)
    expect(after.rows).toEqual(before.rows) // same chunk rows: nothing re-embedded
    expect((await kb.search('zero trust', { filter: { team: 'security' } })).map((h) => h.documentId)).toEqual(['vpn'])
    expect(await kb.search('zero trust', { filter: { team: 'it' } })).toEqual([])
    const back = await kb.ingest({ id: 'vpn', title: 'VPN setup', content: 'Use the new zero trust agent instead.', metadata: { team: 'it' }, collection: 'handbook' })
    expect(back.updated).toBe(1)
  })

  it('lists collections and removes documents', async () => {
    expect(await kb.collections()).toEqual([{ collection: 'handbook', documents: 3, chunks: 3 }])
    expect(await kb.remove('expenses')).toBe(true)
    expect((await kb.search('receipts')).some((h) => h.documentId === 'expenses')).toBe(false)
  })

  it('refuses a mismatched embedder, then reindexes', async () => {
    const other = new Knowledge(db, hashEmbedder(32))
    await expect(other.init()).rejects.toThrow(/reindex/)
    await other.reindex()
    expect((await other.search('vacation days'))[0]?.documentId).toBe('pto')
  })

  it('builds OR queries from free text', () => {
    expect(toOrQuery("What's the VPN? a")).toBe("'what' | 'the' | 'vpn'")
    expect(toOrQuery('!!')).toBe('')
  })
})

describe('reranking and embedder changes', () => {
  it('reorders candidates with a cross-encoder', async () => {
    const db2 = await memoryDb()
    await migrate(db2, 'core', CORE_MIGRATIONS)
    const calls: string[][] = []
    const kb2 = new Knowledge(db2, hashEmbedder(64), {
      reranker: {
        id: 'fake',
        async rerank(_q, docs) {
          calls.push(docs)
          // Prefer anything mentioning "badge".
          return docs.map((d) => (d.includes('badge') ? 0.9 : 0.1))
        },
      },
    })
    await kb2.ingest([
      { id: 'a', title: 'Parking', content: 'Parking garage access requires a badge at the gate.' },
      { id: 'b', title: 'Parking fees', content: 'Parking parking parking fees are monthly.' },
    ])
    const hits = await kb2.search('parking', { limit: 1 })
    expect(hits[0]).toMatchObject({ documentId: 'a', rerankScore: 0.9 })
    expect(calls[0]!.length).toBe(2)
    expect(calls[0]![0]).toMatch(/^Parking/) // title is passed to the reranker
    expect((await kb2.search('parking', { limit: 1, rerank: false }))[0]!.rerankScore).toBeUndefined()
    await db2.close()
  })

  it('auto-reindexes when the embedder changes and uses its document format', async () => {
    const db3 = await memoryDb()
    await migrate(db3, 'core', CORE_MIGRATIONS)
    await new Knowledge(db3, hashEmbedder(16)).ingest({ id: 'x', title: 'T', content: 'alpha beta' })
    const seen: string[] = []
    const base = hashEmbedder(32)
    const formatted = {
      ...base,
      id: 'formatted-32',
      formatDocument: (text: string, title?: string) => `title: ${title ?? 'none'} | text: ${text}`,
      embed: async (texts: string[], kind: 'query' | 'document') => {
        if (kind === 'document') seen.push(...texts)
        return base.embed(texts, kind)
      },
    }
    const progress: number[] = []
    const kb3 = new Knowledge(db3, formatted, { autoReindex: true, onReindexProgress: (d) => progress.push(d) })
    expect((await kb3.search('alpha'))[0]?.documentId).toBe('x')
    expect(seen).toEqual(['title: T | text: alpha beta'])
    expect(progress).toEqual([1])
    await db3.close()
  })
})

describe('background re-embedding', () => {
  const docs = [
    { id: 'pto', title: 'Vacation', content: 'Employees get 25 vacation days per year.' },
    { id: 'vpn', title: 'VPN', content: 'Use the zero trust agent to reach internal systems.' },
    { id: 'exp', title: 'Expenses', content: 'Submit receipts within 30 days of travel.' },
  ]
  const tagged = (id: string, dims: number) => {
    const base = hashEmbedder(dims)
    const calls: { kind: string; task?: string; n: number }[] = []
    const embedder = {
      ...base,
      id,
      embed: async (t: string[], kind: 'query' | 'document', o?: { task?: string }) => {
        calls.push({ kind, ...(o?.task ? { task: o.task } : {}), n: t.length })
        return base.embed(t, kind)
      },
    }
    return { calls, embedder }
  }
  const setup = async () => {
    const db = await memoryDb()
    await migrate(db, 'core', CORE_MIGRATIONS)
    await new Knowledge(db, hashEmbedder(16)).ingest(docs)
    return db
  }

  it('keeps keyword search while it re-embeds in idle steps, then swaps to vectors', async () => {
    const db = await setup()
    const { embedder, calls } = tagged('next-32', 32)
    const progress: number[] = []
    const kb = new Knowledge(db, embedder, { autoReindex: true, reindexInBackground: true, batchSize: 2, onReindexProgress: (d) => progress.push(d) })
    await kb.init()
    expect(calls).toEqual([]) // nothing re-embedded during init
    expect(kb.vectorsReady).toBe(false)
    expect(await kb.reindexStatus()).toMatchObject({ from: { id: 'hash-16', dimensions: 16 }, to: { id: 'next-32', dimensions: 32 }, done: 0, total: 3 })
    // Search works (keyword) and never compares a new-model query with old vectors.
    const hits = await kb.search('vacation days', { mode: 'vector' })
    expect(hits[0]?.documentId).toBe('pto')
    expect(hits[0]?.similarity).toBeNull()
    expect(calls).toEqual([])
    // A page written meanwhile gets the new embedder's vector only.
    await kb.ingest({ id: 'park', title: 'Parking', content: 'Visitors park on level two.' })
    expect(await kb.reindexStatus()).toMatchObject({ done: 1, total: 4 })
    expect(await kb.reindexStep()).toMatchObject({ done: 3, total: 4 })
    expect(progress).toEqual([3])
    expect(await kb.reindexStep()).toBeNull() // the last batch: swapped
    expect(kb.vectorsReady).toBe(true)
    expect(await kb.reindexStatus()).toBeNull()
    const { rows } = await db.query<{ dims: number }>('select vector_dims(embedding) as dims from enclave.chunks limit 1')
    expect(rows[0]?.dims).toBe(32)
    const qa = await kb.search('vacation days', { mode: 'vector', task: 'question-answering' })
    expect(qa[0]?.similarity).not.toBeNull()
    expect(calls.at(-1)).toEqual({ kind: 'query', task: 'question-answering', n: 1 })
    // Recorded: a new instance with the same embedder starts ready.
    const again = new Knowledge(db, tagged('next-32', 32).embedder, { autoReindex: true, reindexInBackground: true })
    await again.init()
    expect(again.vectorsReady).toBe(true)
    await db.close()
  })

  it('resumes after a restart, and goes back cleanly when the old embedder returns', async () => {
    const db = await setup()
    const first = new Knowledge(db, tagged('next-32', 32).embedder, { autoReindex: true, reindexInBackground: true, batchSize: 1 })
    await first.reindexStep()
    // Restart: the same move carries on where it was.
    const resumed = new Knowledge(db, tagged('next-32', 32).embedder, { autoReindex: true, reindexInBackground: true, batchSize: 1 })
    expect(await resumed.reindexStatus()).toMatchObject({ done: 1, total: 3 })
    await resumed.ingest({ id: 'park', title: 'Parking', content: 'Visitors park on level two.' })
    // Back to the old model: the half-made column goes, and the new page gets an old-model vector.
    const back = new Knowledge(db, hashEmbedder(16), { autoReindex: true, reindexInBackground: true })
    await back.init()
    expect(back.vectorsReady).toBe(true)
    expect((await back.search('parking visitors', { mode: 'vector' }))[0]?.documentId).toBe('park')
    const { rows } = await db.query<{ n: number }>(
      "select count(*)::int as n from information_schema.columns where table_name = 'chunks' and column_name = 'embedding_next'",
    )
    expect(rows[0]?.n).toBe(0)
    await db.close()
  })

  it('moves to a third embedder from the middle of a move', async () => {
    const db = await setup()
    await new Knowledge(db, tagged('next-32', 32).embedder, { autoReindex: true, reindexInBackground: true, batchSize: 1 }).reindexStep()
    const third = new Knowledge(db, hashEmbedder(24), { autoReindex: true, reindexInBackground: true })
    expect(await third.reindexStatus()).toMatchObject({ to: { dimensions: 24 }, done: 0, total: 3 })
    while (await third.reindexStep());
    expect((await third.search('receipts travel', { mode: 'vector' }))[0]?.documentId).toBe('exp')
    await db.close()
  })

  it('swaps at once when there is nothing to re-embed', async () => {
    const db = await memoryDb()
    await migrate(db, 'core', CORE_MIGRATIONS)
    await new Knowledge(db, hashEmbedder(16)).init()
    const kb = new Knowledge(db, hashEmbedder(32), { autoReindex: true, reindexInBackground: true })
    await kb.init()
    expect(kb.vectorsReady).toBe(true)
    await kb.ingest(docs)
    expect((await kb.search('receipts travel', { mode: 'vector' }))[0]?.documentId).toBe('exp')
    await db.close()
  })
})
