import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { z } from 'zod'
import type { PGlite } from '@electric-sql/pglite'
import { createEnclave, defineSkill, tool, trimHistory, type AgentEvent, type Message } from '../src/index.js'
import { knowledgeSkill, memorySkill, sqlSkill } from '../src/skills/index.js'
import { hashEmbedder, mockModel } from '../src/testing.js'
import { collect, memoryDb } from './helpers.js'

let db: PGlite
beforeEach(async () => {
  db = await memoryDb()
})
afterEach(() => db.close())

const types = (events: AgentEvent[]) => events.map((e) => e.type)

describe('agent loop with the sql skill', () => {
  it('creates a table (with approval), inserts, queries and answers', async () => {
    const model = mockModel([
      { text: 'Creating it.', toolCalls: [{ name: 'execute_sql', input: { sql: 'create table notes (id bigint primary key generated always as identity, body text)' } }] },
      { toolCalls: [{ name: 'execute_sql', input: { sql: "insert into notes (body) values ('a'), ('b')" } }] },
      { toolCalls: [{ name: 'execute_sql', input: { sql: 'select count(*)::int as n from notes' } }] },
      'You have 2 notes.',
    ])
    const approvals: string[] = []
    const ai = await createEnclave({
      db,
      model,
      skills: [sqlSkill()],
      onApproval: (call) => (approvals.push((call.input as { sql: string }).sql), true),
    })

    const thread = ai.thread('t1')
    const events = await collect(thread.send('Make a notes table with two notes and count them'))

    expect(approvals).toHaveLength(2) // create + insert; select needs no approval
    expect(types(events).filter((t) => t === 'approval-request')).toHaveLength(2)
    const results = events.filter((e) => e.type === 'tool-result')
    expect(results.every((r) => !r.isError)).toBe(true)
    expect(results.at(-1)!.output).toEqual([{ columns: ['n'], rows: [{ n: 2 }], rowCount: 1, affectedRows: 0 }])
    expect(events.at(-1)).toMatchObject({ type: 'finish', reason: 'stop', steps: 4 })

    // Live schema context reaches the model after the table exists.
    expect(model.requests[1]!.context).toContain('table notes')
    expect(model.requests[0]!.system).toContain('# Skill: sql')

    // Thread history is persisted and survives a fresh enclave on the same db.
    const reopened = await createEnclave({ db, model: mockModel([]), skills: [sqlSkill()] })
    const history = await reopened.thread('t1').messages()
    expect(history.map((m) => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant', 'tool', 'assistant', 'tool', 'assistant'])
    expect((await reopened.threads()).map((t) => t.id)).toEqual(['t1'])
  })

  it('reports denied approvals and SQL errors back to the model', async () => {
    const model = mockModel([
      { toolCalls: [{ name: 'execute_sql', input: { sql: 'drop table if exists x' } }] },
      { toolCalls: [{ name: 'execute_sql', input: { sql: 'select * from missing_table' } }] },
      'ok',
    ])
    const ai = await createEnclave({ db, model, skills: [sqlSkill()] }) // no approval handler => deny
    const { messages } = await ai.run('go').result()
    const tools = messages.filter((m) => m.role === 'tool')
    expect(tools[0]).toMatchObject({ isError: true, content: expect.stringContaining('declined') })
    expect(tools[1]).toMatchObject({ isError: true, content: expect.stringContaining('missing_table') })
  })

  it('enforces read-only mode and hides the internal schema', async () => {
    const model = mockModel([
      { toolCalls: [{ name: 'execute_sql', input: { sql: 'create table y (id int)' } }] },
      { toolCalls: [{ name: 'execute_sql', input: { sql: 'select * from enclave.messages' } }] },
      'done',
    ])
    const ai = await createEnclave({ db, model, skills: [sqlSkill({ readOnly: true })] })
    const { messages } = await ai.run('go').result()
    const tools = messages.filter((m) => m.role === 'tool')
    expect(tools[0]!.content).toContain('read-only')
    expect(tools[1]!.content).toContain('internal')
  })
})

describe('custom skills', () => {
  const tasks = defineSkill({
    name: 'tasks',
    description: 'Track tasks',
    instructions: 'Use add_task to record tasks.',
    migrations: ['create table tasks (id serial primary key, title text not null, done boolean default false)'],
    tools: {
      add_task: tool({
        description: 'Add a task',
        input: z.object({ title: z.string().min(1) }),
        execute: async ({ title }, ctx) => {
          ctx.emit({ progress: 'saving' })
          const { rows } = await ctx.db.query<{ id: number }>('insert into tasks (title) values ($1) returning id', [title])
          return { id: rows[0]!.id, title, secret: 'ui-only' }
        },
        toModelOutput: ({ id }) => ({ id }),
      }),
    },
    context: async ({ db }) => {
      const { rows } = await db.query<{ n: number }>('select count(*)::int as n from tasks')
      return `${rows[0]!.n} tasks`
    },
  })

  it('runs migrations once, streams custom events and shapes model output', async () => {
    const model = mockModel([
      { toolCalls: [{ name: 'add_task', input: { title: '' } }] }, // invalid → validation error
      { toolCalls: [{ name: 'add_task', input: { title: 'Ship it' } }] },
      'Added.',
    ])
    const ai = await createEnclave({ db, model, skills: [tasks] })
    const events = await collect(ai.run('add ship it'))

    const results = events.filter((e) => e.type === 'tool-result')
    expect(results[0]).toMatchObject({ isError: true })
    expect(JSON.stringify(results[0]!.output)).toMatch(/Invalid input/)
    expect(results[1]!.output).toEqual({ id: 1, title: 'Ship it', secret: 'ui-only' })
    expect(events).toContainEqual({ type: 'custom', skill: 'tasks', tool: 'add_task', data: { progress: 'saving' } })

    const toolMsg = events.flatMap((e) => (e.type === 'message' && e.message.role === 'tool' ? [e.message] : [])).at(-1)
    expect(toolMsg!.content).toBe('{"id":1}')
    expect(model.requests[2]!.context).toContain('1 tasks')

    // Re-registering on the same database does not re-run the migration.
    await expect(createEnclave({ db, model: mockModel([]), skills: [tasks] })).resolves.toBeDefined()
  })

  it('rejects duplicate tool names across skills', async () => {
    const clash = defineSkill({ name: 'clash', description: 'x', tools: { add_task: tasks.tools.add_task } })
    await expect(createEnclave({ db, model: mockModel([]), skills: [tasks, clash] })).rejects.toThrow(/collides/)
  })

  it('discloses lazy skills on demand and remembers activation per thread', async () => {
    const lazy = { ...tasks, name: 'tasks', lazy: true }
    const model = mockModel([
      { toolCalls: [{ name: 'add_task', input: { title: 'too early' } }] },
      { toolCalls: [{ name: 'activate_skill', input: { name: 'tasks' } }] },
      { toolCalls: [{ name: 'add_task', input: { title: 'now' } }] },
      'ok',
      'second turn',
    ])
    const ai = await createEnclave({ db, model, skills: [lazy] })
    const thread = ai.thread()
    await thread.send('add a task').result()

    expect(model.requests[0]!.tools.map((t) => t.name)).toEqual(['activate_skill'])
    expect(model.requests[0]!.system).toContain('- tasks: Track tasks')
    expect(model.requests[0]!.system).not.toContain('Use add_task')
    expect(model.requests[2]!.tools.map((t) => t.name)).toEqual(['add_task'])
    expect(model.requests[2]!.system).toContain('Use add_task')
    const history = await thread.messages()
    expect(history[2]).toMatchObject({ role: 'tool', isError: true, content: expect.stringContaining('Unknown tool') })

    await thread.send('again').result()
    expect(model.requests[4]!.tools.map((t) => t.name)).toEqual(['add_task'])
  })
})

describe('knowledge and memory skills', () => {
  it('answers from retrieved passages and stores memories', async () => {
    const model = mockModel([
      { toolCalls: [{ name: 'search_knowledge', input: { query: 'wifi password guest network' } }] },
      (req) => {
        const last = req.messages.at(-1)!
        expect(last.role).toBe('tool')
        expect(last.content).toContain('[1] Office wifi')
        return { toolCalls: [{ name: 'remember', input: { fact: 'The user works from the Berlin office.' } }] }
      },
      'The guest password is "sunflower" [1].',
      'noted',
    ])
    const ai = await createEnclave({ db, model, embedder: hashEmbedder(), skills: [knowledgeSkill(), memorySkill()] })
    await ai.knowledge!.ingest({ title: 'Office wifi', content: 'The guest network password is sunflower. Staff use SSO.', source: 'wiki/wifi' })

    const answer = await ai.thread('k').send('What is the guest wifi password?').text()
    expect(answer).toContain('[1]')

    await ai.thread('k').send('thanks').result()
    expect(model.requests[3]!.context).toContain('Berlin office')
  })
})

describe('trimHistory', () => {
  it('never starts on an orphaned tool result', () => {
    const h: Message[] = [
      { role: 'user', content: 'a' },
      { role: 'assistant', content: '', toolCalls: [{ id: '1', name: 'x', input: {} }] },
      { role: 'tool', toolCallId: '1', name: 'x', content: '' },
      { role: 'assistant', content: 'b' },
      { role: 'user', content: 'c' },
      { role: 'assistant', content: 'd' },
    ]
    expect(trimHistory(h, 4)).toEqual(h.slice(4))
    expect(trimHistory(h, 10)).toBe(h)
  })
})
