import { z } from 'zod'
import type { Citation, CitationInput, Db, Embedder, JSONSchema, KnowledgeScope, ToolSpec } from './types.js'
import type { Knowledge } from './rag/knowledge.js'

/** Everything a tool can reach at execution time. */
export interface ToolContext {
  db: Db
  /** Present when the enclave was created with an embedder. */
  knowledge: Knowledge | undefined
  embedder: Embedder | undefined
  threadId: string | undefined
  /** Name of the skill that owns the executing tool. */
  skill: string
  signal: AbortSignal
  /** Stream arbitrary data to the UI (surfaces as a `custom` agent event). */
  emit(data: unknown): void
  /** Retrieval scope requested for this run (`send(…, { knowledge })`). */
  scope?: KnowledgeScope
  /** Register passages returned to the model; see SkillContext.cite. */
  cite?(passages: CitationInput[]): Citation[]
}

export interface ToolDef<I extends z.ZodType = z.ZodType, O = unknown> {
  description: string
  input: I
  execute(input: z.output<I>, ctx: ToolContext): O | Promise<O>
  /** Require a human decision before running. Denied calls are reported back to the model. */
  needsApproval?: boolean | ((input: z.output<I>, ctx: ToolContext) => boolean | Promise<boolean>)
  /**
   * Shape what the model sees. The full output still goes to the UI via the
   * `tool-result` event, so you can render 10k rows while sending the model 50.
   */
  toModelOutput?(output: O): unknown
}

/** Identity helper that gives `execute` a fully typed `input`. */
export function tool<I extends z.ZodType, O>(def: ToolDef<I, O>): ToolDef<I, O> {
  return def
}

const schemaCache = new WeakMap<z.ZodType, JSONSchema>()

export function toJSONSchema(schema: z.ZodType): JSONSchema {
  let json = schemaCache.get(schema)
  if (!json) {
    const { $schema: _ignored, ...rest } = z.toJSONSchema(schema, { io: 'input', unrepresentable: 'any' })
    json = rest
    schemaCache.set(schema, json)
  }
  return json
}

export function toToolSpec(name: string, def: ToolDef<any, any>): ToolSpec {
  return { name, description: def.description, inputSchema: toJSONSchema(def.input) }
}
