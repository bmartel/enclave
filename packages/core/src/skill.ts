import type { ToolDef } from './tool.js'
import type { Db, Embedder, Message } from './types.js'
import type { Knowledge } from './rag/knowledge.js'

export interface SkillContext {
  db: Db
  knowledge: Knowledge | undefined
  embedder: Embedder | undefined
  threadId: string | undefined
  /** Conversation so far (empty during setup). Lets `context` react to the latest request. */
  messages: readonly Message[]
}

/**
 * A skill is the unit of capability: instructions + tools + storage + live context.
 *
 * - `instructions` go into the system prompt (or are disclosed on demand when `lazy`).
 * - `tools` are exposed to the model; names must be unique across all skills.
 * - `migrations` are SQL strings applied once, in order, and tracked per skill.
 * - `context` runs before every model step and injects fresh state (e.g. current schema).
 */
export interface Skill {
  name: string
  description: string
  instructions?: string
  /**
   * Progressive disclosure: only the description is shown until the model calls
   * `activate_skill`. Keeps the prompt small when you ship many skills.
   */
  lazy?: boolean
  tools?: Record<string, ToolDef<any, any>>
  migrations?: string[]
  setup?(ctx: SkillContext): void | Promise<void>
  context?(ctx: SkillContext): string | undefined | Promise<string | undefined>
}

const SKILL_NAME = /^[a-z][a-z0-9_-]{0,63}$/
const TOOL_NAME = /^[a-zA-Z][a-zA-Z0-9_-]{0,63}$/

export function defineSkill<const S extends Skill>(skill: S): S {
  if (!SKILL_NAME.test(skill.name)) {
    throw new Error(`Invalid skill name "${skill.name}": use lowercase letters, digits, "-" or "_"`)
  }
  for (const name of Object.keys(skill.tools ?? {})) {
    if (!TOOL_NAME.test(name)) throw new Error(`Invalid tool name "${name}" in skill "${skill.name}"`)
  }
  return skill
}
