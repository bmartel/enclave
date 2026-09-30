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

  /** Delete a memory by id; never a document from another collection. */
  const removeMemory = async (ctx: Pick<ToolContext, 'db' | 'knowledge'>, id: string) => {
    const knowledge = requireKnowledge(ctx)
    await knowledge.init()
    const { rows } = await ctx.db.query(`select 1 from enclave.documents where id = $1 and collection = $2`, [id, collection])
    return rows.length ? knowledge.remove(id) : false
  }

  return defineSkill({
    name: 'memory',
    description: 'Remember durable facts and preferences about the user across conversations.',
    instructions: `Save stable facts, preferences and decisions the user shares with remember (one fact per call, written as a standalone sentence). When the user asks you to remember something, call remember: saying you will is not enough.
- "Known about the user" lists what you remember; use it to answer questions about the user.
- Use recall when earlier context might matter. Call forget with the id when a fact is outdated or the user asks you to forget it.
- Never store secrets (passwords, keys, card numbers) or transient details. Tell the user you won't store secrets.`,
    tools: {
      remember: tool({
        description: 'Store one durable fact. When it updates a fact you already know, pass that memory id as replaces.',
        input: z.object({
          fact: z.string().min(3).max(1000),
          replaces: z.string().optional().describe('Id of an outdated memory this fact replaces, e.g. mem_abc.'),
        }),
        execute: async ({ fact, replaces }, ctx) => {
          if (looksLikeSecret(fact)) {
            throw new Error('Not saved: this looks like a password, key or other secret, and secrets are never stored in memory. Tell the user.')
          }
          const id = uid('mem_')
          await requireKnowledge(ctx).ingest({
            id,
            content: fact,
            collection,
            metadata: { threadId: ctx.threadId ?? null, createdAt: new Date().toISOString() },
          })
          // Updating in the same call: small models rarely follow up with forget.
          const replaced = replaces ? await removeMemory(ctx, replaces) : false
          return { id, saved: true, ...(replaces ? { replaced } : {}) }
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
        execute: async ({ id }, ctx) => ({ deleted: await removeMemory(ctx, id) }),
      }),
    },
    context: async ({ db, knowledge, messages }) => {
      if (!knowledge) return undefined
      const sections: string[] = []
      if (recent > 0) {
        await knowledge.init()
        const { rows } = await db.query<{ id: string; content: string }>(
          `select d.id, c.content from enclave.documents d join enclave.chunks c on c.document_id = d.id
           where d.collection = $1 order by d.created_at desc limit $2`,
          [collection, recent],
        )
        if (rows.length) sections.push(`Known about the user:\n${rows.map((r) => `- (${r.id}) ${r.content}`).join('\n')}`)
      }
      // Small models often say "I'll remember that" without calling the tool.
      // A request in the latest message gets an explicit, turn-local nudge.
      const latest = messages.at(-1)
      if (latest?.role === 'user') {
        if (FORGET_REQUEST.test(latest.content)) {
          sections.push('The latest message asks you to forget something: call forget with its id.')
        } else if (REMEMBER_REQUEST.test(latest.content)) {
          sections.push('The latest message asks you to remember something: call remember now (if it updates a fact listed above, pass that id as replaces), unless it is a secret.')
        }
      }
      return sections.length ? sections.join('\n\n') : undefined
    },
  })
}

/** Credentials and card numbers, which must never reach long-term memory. */
const SECRET_PATTERNS = [
  // "password is Tr0ub4dor&3", "api key: abc123…", "PIN = 4821"
  /\b(password|passcode|passphrase|pin|api[ _-]?key|secret|token|credentials?|cvv)\b\W{0,3}(is|was|:|=)\s*["'`]?(?=\S*[\d!@#$%^&*])\S{4,}/i,
  // Card numbers
  /\b(?:\d[ -]?){13,19}\b/,
  // Common key prefixes
  /\b(sk|pk|rk|ghp|gho|xox[abp])[-_][A-Za-z0-9_-]{10,}/,
  // Private key blocks
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
]

/**
 * A bare password: one token of 8+ characters mixing letters, digits and
 * symbols ("Tr0ub4dor&3"). A model stored exactly that, with no "password is"
 * in front. URLs and email addresses are not treated as secrets.
 */
const PASSWORD_LIKE = /^(?=.*[A-Za-z])(?=.*\d)(?=.*[^\w\s]).{8,}$/

export const looksLikeSecret = (text: string): boolean =>
  SECRET_PATTERNS.some((p) => p.test(text)) ||
  text
    .split(/\s+/)
    .map((token) => token.replace(/^["'`(]+|["'`).,;:!?]+$/g, ''))
    .some((token) => PASSWORD_LIKE.test(token) && !/^[a-z]+:\/\//i.test(token) && !/^[^@\s]+@[^@\s]+\.[a-z]+$/i.test(token))

/** "Remember that…", "please note…", "update what you remember" — not "do you remember…?" */
const REMEMBER_REQUEST = /\b(remember|don'?t forget|keep in mind|make a note|note that|update what you (know|remember))\b(?![^.?!]*\?)/i
const FORGET_REQUEST = /(?<!don'?t )\b(forget|delete what you (know|remember)|stop remembering)\b(?![^.?!]*\?)/i
