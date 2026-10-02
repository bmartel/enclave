import { z } from 'zod'
import { defineSkill, tool } from 'enclave-ai'

/**
 * Example bespoke skill: its own table, a live context line, a tool that
 * streams progress to the UI, and one that needs user approval.
 */
export const notesSkill = defineSkill({
  name: 'notes',
  description: 'Pin short notes the user wants to keep in view.',
  instructions: 'When the user asks you to pin, note or jot something, call pin_note. Use unpin_note to remove one.',
  migrations: [
    `create table notes (
       id bigint primary key generated always as identity,
       body text not null,
       created_at timestamptz not null default now()
     )`,
  ],
  tools: {
    pin_note: tool({
      description: 'Pin a short note.',
      input: z.object({ body: z.string().min(1).max(500) }),
      execute: async ({ body }, { db, emit }) => {
        const { rows } = await db.query<{ id: number }>('insert into notes (body) values ($1) returning id', [body])
        emit({ pinned: body })
        return { id: rows[0]!.id }
      },
    }),
    unpin_note: tool({
      description: 'Remove a pinned note by id.',
      input: z.object({ id: z.number().int() }),
      needsApproval: true,
      execute: async ({ id }, { db }) => {
        const { affectedRows } = await db.query('delete from notes where id = $1', [id])
        return { removed: affectedRows === 1 }
      },
    }),
  },
  context: async ({ db }) => {
    const { rows } = await db.query<{ id: number; body: string }>('select id, body from notes order by id desc limit 20')
    return rows.length ? `Pinned notes:\n${rows.map((n) => `- #${n.id} ${n.body}`).join('\n')}` : 'No pinned notes.'
  },
})
