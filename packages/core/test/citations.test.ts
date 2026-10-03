import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import type { PGlite } from '@electric-sql/pglite'
import { createEnclave, type AgentEvent, type Citation } from '../src/index.js'
import { knowledgeSkill } from '../src/skills/index.js'
import { hashEmbedder, mockModel } from '../src/testing.js'
import { collect, memoryDb } from './helpers.js'

let db: PGlite
beforeEach(async () => {
  db = await memoryDb()
})
afterEach(() => db.close())

const DOCS = [
  { id: 'wifi', title: 'Office wifi', content: 'The guest wifi password is sunflower. Staff use SSO.', collection: 'handbook', metadata: { team: 'it' } },
  { id: 'leave', title: 'Parental leave', content: 'Parental leave is sixteen weeks paid, with notice of four weeks.', collection: 'handbook', metadata: { team: 'hr' } },
  { id: 'garden', title: 'Garden log', content: 'Tomatoes need full sun and deep watering twice weekly.', collection: 'personal', metadata: {} },
]

async function setup(script: Parameters<typeof mockModel>[0], autoLimit = 3) {
  const model = mockModel(script)
  // Auto-retrieval accepts any hash-embedder match, so the tests control what is found.
  const ai = await createEnclave({
    db,
    model,
    embedder: hashEmbedder(),
    skills: [knowledgeSkill({ autoRetrieve: { minSimilarity: 0, relativeCutoff: 0, limit: autoLimit } })],
  })
  await ai.knowledge!.ingest(DOCS)
  return { ai, model }
}

const citationEvents = (events: AgentEvent[]) =>
  events.filter((e): e is Extract<AgentEvent, { type: 'citations' }> => e.type === 'citations').map((e) => e.citations)

describe('citations', () => {
  it('numbers auto-retrieved and searched passages in one sequence the UI can resolve', async () => {
    const { ai, model } = await setup([
      { toolCalls: [{ name: 'search_knowledge', input: { query: 'tomatoes watering sun garden' } }] },
      (req) => {
        // The tool output must continue the numbering of the auto-retrieved passages.
        const tool = req.messages.findLast((m) => m.role === 'tool')!
        return `Wifi is sunflower [1]. Water tomatoes twice weekly ${tool.content.match(/\[(\d+)\] Garden log/)![0].slice(0, 3)}.`
      },
    ], 1)
    const events = await collect(ai.thread('t').send('What is the guest wifi password?'))
    const lists = citationEvents(events)
    expect(lists.length).toBeGreaterThanOrEqual(2)
    const final = lists.at(-1)!
    expect(final.map((c) => c.n)).toEqual(final.map((_, i) => i + 1))
    expect(final[0]!.title).toBe('Office wifi')

    const garden = final.find((c) => c.documentId === 'garden')!
    const answer = events.filter((e) => e.type === 'text-delta').map((e) => (e as { delta: string }).delta).join('')
    expect(answer).toContain(`[${garden.n}]`)
    expect(garden.n).toBeGreaterThan(1)

    // The model saw the same numbers the UI got.
    const toolMsg = model.requests[1]!.messages.findLast((m) => m.role === 'tool')!
    expect(toolMsg.content).toContain(`[${garden.n}] Garden log`)
    expect(model.requests[0]!.context).toContain('[1] Office wifi')
  })

  it('stores citations on the final answer, returns them from result(), and never sends them to the model', async () => {
    const { ai, model } = await setup(['Sunflower [1].', 'You are welcome.'])
    const result = await ai.thread('t').send('guest wifi password').result()
    expect(result.citations[0]).toMatchObject({ n: 1, documentId: 'wifi', title: 'Office wifi' })

    const history = await ai.thread('t').messages()
    const answer = history.findLast((m) => m.role === 'assistant')!
    expect(answer.role === 'assistant' && answer.citations?.[0]?.documentId).toBe('wifi')

    await ai.thread('t').send('thanks').result()
    for (const m of model.requests[1]!.messages) expect('citations' in m).toBe(false)
  })

  it('scopes retrieval and search to chosen documents for one run', async () => {
    const { ai, model } = await setup([
      { toolCalls: [{ name: 'search_knowledge', input: { query: 'wifi password sunflower' } }] },
      'Only the garden note is in scope [1].',
    ])
    const result = await ai.thread('t').send('guest wifi password', { knowledge: { documentIds: ['garden'] } }).result()
    const cited = new Set(result.citations.map((c: Citation) => c.documentId))
    expect([...cited]).toEqual(['garden'])
    expect(model.requests[0]!.context ?? '').not.toContain('Office wifi')
  })

  it('scopes by collection and metadata', async () => {
    const { ai } = await setup(['ok'])
    const byCollection = await ai.thread('a').send('tomatoes', { knowledge: { collection: 'personal' } }).result()
    expect(byCollection.citations.every((c) => c.collection === 'personal')).toBe(true)
    const byTeam = await ai.knowledge!.search('leave notice wifi', { filter: { team: 'hr' }, documentIds: ['leave', 'wifi'] })
    expect(byTeam.map((h) => h.documentId)).toEqual(['leave'])
  })
})
