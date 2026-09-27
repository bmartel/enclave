import { z } from 'zod'
import type { Knowledge, SearchHit } from '../rag/knowledge.js'
import { defineSkill } from '../skill.js'
import { tool, type ToolContext } from '../tool.js'

export interface KnowledgeSkillOptions {
  /** Restrict search to these collections. Default: all. */
  collections?: string[]
  /** Hits returned per search. Default 6. */
  limit?: number
  /** Drop weak vector matches. Default 0.2 cosine similarity. */
  minSimilarity?: number
  /**
   * Classic RAG alongside the tool: before the model runs, search with the
   * latest user message and put the best passages in context. Makes small
   * in-browser models reliable (they often forget to call tools). Default true.
   */
  autoRetrieve?: boolean | { limit?: number; minRerankScore?: number; maxChars?: number }
}

function requireKnowledge(ctx: Pick<ToolContext, 'knowledge'>) {
  if (!ctx.knowledge) throw new Error('Knowledge search needs an embedder: pass `embedder` to createEnclave().')
  return ctx.knowledge
}

const label = (h: SearchHit) => [h.title, h.source].filter(Boolean).join(' — ') || h.documentId

export function formatPassages(hits: SearchHit[], maxChars = Infinity): string {
  let out = ''
  for (const [i, h] of hits.entries()) {
    const block = `[${i + 1}] ${label(h)}\n${h.content}\n\n`
    if (out && out.length + block.length > maxChars) break
    out += block
  }
  return out.trim()
}

/** Retrieval-augmented generation over the private, on-device knowledge base. */
export function knowledgeSkill(options: KnowledgeSkillOptions = {}) {
  const limit = options.limit ?? 6
  const collections = options.collections
  const collection = collections?.length
    ? z.enum(collections as [string, ...string[]]).optional().describe('Collection to search. Omit to search all.')
    : z.string().optional().describe('Collection to search. Omit to search all.')
  const auto = options.autoRetrieve === false ? undefined : typeof options.autoRetrieve === 'object' ? options.autoRetrieve : {}

  // Steps within one turn share the same user message: search once per message.
  const retrieved = new Map<string, Promise<SearchHit[]>>()
  const retrieve = (knowledge: Knowledge, query: string) => {
    let hits = retrieved.get(query)
    if (!hits) {
      if (retrieved.size > 32) retrieved.clear()
      hits = knowledge
        .search(query, {
          limit: auto?.limit ?? 3,
          minSimilarity: options.minSimilarity ?? 0.2,
          ...(collections ? { collection: collections } : {}),
        })
        .then((found) => found.filter((h) => h.rerankScore === undefined || h.rerankScore >= (auto?.minRerankScore ?? 0.15)))
      retrieved.set(query, hits)
      hits.catch(() => retrieved.delete(query))
    }
    return hits
  }

  return defineSkill({
    name: 'knowledge',
    description: 'Search private documents stored on this device.',
    instructions: `The user's private documents are stored on this device. You don't know their contents unless you read them.
- For any question that could be answered by those documents, use the passages in "Retrieved passages" or call search_knowledge.
- Ground answers in the passages and cite them inline as [1], [2] matching the passage numbers. Only cite document passages, never other tool results.
- If the passages don't contain the answer, say so plainly instead of guessing.
- Rephrase and search again with different keywords when a search misses.`,
    tools: {
      search_knowledge: tool({
        description: 'Hybrid semantic + keyword search over private documents. Returns numbered passages.',
        input: z.object({
          query: z.string().min(1).describe('What to look for, in natural language.'),
          collection,
          limit: z.number().int().min(1).max(20).optional(),
        }),
        execute: async ({ query, collection: only, limit: max }, ctx) => {
          const scope = only ?? collections
          return requireKnowledge(ctx).search(query, {
            limit: max ?? limit,
            minSimilarity: options.minSimilarity ?? 0.2,
            ...(scope ? { collection: scope } : {}),
          })
        },
        toModelOutput: (hits: SearchHit[]) => (hits.length ? formatPassages(hits) : 'No matching passages.'),
      }),
      list_collections: tool({
        description: 'List document collections with document and chunk counts.',
        input: z.object({}),
        execute: async (_input, ctx) => {
          const all = await requireKnowledge(ctx).collections()
          return collections ? all.filter((c) => collections.includes(c.collection)) : all
        },
      }),
    },
    context: async ({ knowledge, messages }) => {
      if (!knowledge) return undefined
      const available = (await knowledge.collections()).filter((c) => !collections || collections.includes(c.collection))
      if (!available.length) return 'The knowledge base is empty.'
      const sections = [
        `Knowledge base: ${available.map((c) => `${c.collection} (${c.documents} documents)`).join(', ')}.`,
      ]
      const lastUser = messages.findLast((m) => m.role === 'user')?.content.trim()
      if (auto && lastUser) {
        const hits = await retrieve(knowledge, lastUser)
        if (hits.length) {
          sections.push(`Retrieved passages for the latest request:\n${formatPassages(hits, auto.maxChars ?? 3600)}`)
        }
      }
      return sections.join('\n\n')
    },
  })
}
