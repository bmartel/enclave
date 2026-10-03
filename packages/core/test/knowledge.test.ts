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
