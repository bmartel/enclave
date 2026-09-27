import { z } from 'zod'
import { defineSkill } from '../skill.js'
import { tool, type ToolContext } from '../tool.js'
import { uid } from '../util.js'

export interface MemorySkillOptions {
  /** Knowledge collection that stores memories. Default `memory`. */
  collection?: string
  /** Most recent memories injected into every step. Default 10; 0 disables. */
  recent?: number
}

function requireKnowledge(ctx: Pick<ToolContext, 'knowledge'>) {
  if (!ctx.knowledge) throw new Error('Memory needs an embedder: pass `embedder` to createEnclave().')
  return ctx.knowledge
}

/** Durable, searchable long-term memory about the user, kept on-device. */
export function memorySkill(options: MemorySkillOptions = {}) {
  const collection = options.collection ?? 'memory'
  const recent = options.recent ?? 10

  return defineSkill({
    name: 'memory',
    description: 'Remember durable facts and preferences about the user across conversations.',
    instructions: `Save stable facts, preferences and decisions the user shares with remember (one fact per call, written as a standalone sentence).
Use recall when earlier context might matter. Don't store secrets or transient details.`,
    tools: {
      remember: tool({
        description: 'Store one durable fact.',
        input: z.object({ fact: z.string().min(3).max(1000) }),
        execute: async ({ fact }, ctx) => {
          const id = uid('mem_')
          await requireKnowledge(ctx).ingest({
            id,
            content: fact,
            collection,
            metadata: { threadId: ctx.threadId ?? null, createdAt: new Date().toISOString() },
          })
          return { id, saved: true }
        },
      }),
      recall: tool({
        description: 'Search stored memories.',
        input: z.object({ query: z.string().min(1), limit: z.number().int().min(1).max(20).optional() }),
        execute: async ({ query, limit }, ctx) => {
          const hits = await requireKnowledge(ctx).search(query, { collection, limit: limit ?? 5 })
          return hits.map((h) => ({ id: h.documentId, fact: h.content }))
        },
      }),
      forget: tool({
        description: 'Delete a stored memory by id.',
        input: z.object({ id: z.string() }),
        execute: async ({ id }, ctx) => ({ deleted: await requireKnowledge(ctx).remove(id) }),
      }),
    },
    context: async ({ db, knowledge }) => {
      if (!knowledge || recent <= 0) return undefined
      await knowledge.init()
      const { rows } = await db.query<{ id: string; content: string }>(
        `select d.id, c.content from enclave.documents d join enclave.chunks c on c.document_id = d.id
         where d.collection = $1 order by d.created_at desc limit $2`,
        [collection, recent],
      )
      if (!rows.length) return undefined
      return `Known about the user:\n${rows.map((r) => `- (${r.id}) ${r.content}`).join('\n')}`
    },
  })
}
