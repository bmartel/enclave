import { z } from 'zod'
import type { Skill, SkillContext } from './skill.js'
import { toToolSpec, type ToolContext, type ToolDef } from './tool.js'
import type {
  AgentEvent,
  ApprovalHandler,
  AssistantMessage,
  Db,
  Embedder,
  FinishReason,
  Message,
  Model,
  StepMetrics,
  ToolCall,
  ToolMessage,
  ToolSpec,
  Usage,
} from './types.js'
import type { Knowledge } from './rag/knowledge.js'
import { drain, safeStringify, truncate } from './util.js'

export const ACTIVATE_SKILL = 'activate_skill'

export const DEFAULT_SYSTEM = `You are an AI assistant built into this application. You run on the user's device, and their data stays here.
- Answer general questions (writing, translation, explanations, reasoning) directly from your own knowledge.
- Use tools for anything about the user's own data, or anything that requires an action. You are authorized to use every tool you are given: when asked to create, change or look something up, do it by calling the tool instead of describing how.
- If a tool returns an error, read it, fix your call, and try again.
Be concise.`

interface ToolEntry {
  skill: Skill
  def: ToolDef<any, any>
}

export class SkillRegistry {
  readonly skills = new Map<string, Skill>()
  readonly tools = new Map<string, ToolEntry>()

  add(skill: Skill): void {
    if (this.skills.has(skill.name)) throw new Error(`Skill "${skill.name}" is already registered`)
    for (const name of Object.keys(skill.tools ?? {})) {
      if (name === ACTIVATE_SKILL) throw new Error(`Tool name "${ACTIVATE_SKILL}" is reserved`)
      const owner = this.tools.get(name)
      if (owner) throw new Error(`Tool "${name}" in skill "${skill.name}" collides with skill "${owner.skill.name}"`)
    }
    this.skills.set(skill.name, skill)
    for (const [name, def] of Object.entries(skill.tools ?? {})) this.tools.set(name, { skill, def })
  }

  /** Skills whose instructions and tools are currently visible to the model. */
  visible(active: ReadonlySet<string>): Skill[] {
    return [...this.skills.values()].filter((s) => !s.lazy || active.has(s.name))
  }

  dormant(active: ReadonlySet<string>): Skill[] {
    return [...this.skills.values()].filter((s) => s.lazy && !active.has(s.name))
  }
}

export interface AgentRuntime {
  model: Model
  db: Db
  knowledge: Knowledge | undefined
  embedder: Embedder | undefined
  registry: SkillRegistry
  system: string
  maxSteps: number
  maxHistory: number
  maxToolOutputChars: number
  onApproval: ApprovalHandler | undefined
}

export interface RunState {
  /** Mutated in place: new assistant/tool messages are appended. */
  history: Message[]
  threadId: string | undefined
  /** Lazy skills the model has activated. Mutated in place. */
  activeSkills: Set<string>
  signal: AbortSignal
  onApproval?: ApprovalHandler
}

/**
 * The agent loop: prompt → stream → execute tools → repeat until the model
 * stops calling tools or `maxSteps` is reached.
 */
export async function* runAgent(rt: AgentRuntime, state: RunState): AsyncGenerator<AgentEvent, void> {
  const usage: Usage = { inputTokens: 0, outputTokens: 0 }
  const skillCtx: SkillContext = {
    db: rt.db,
    knowledge: rt.knowledge,
    embedder: rt.embedder,
    threadId: state.threadId,
    messages: state.history,
  }

  for (let step = 1; step <= rt.maxSteps; step++) {
    if (state.signal.aborted) {
      yield { type: 'finish', reason: 'aborted', steps: step - 1, usage }
      return
    }
    yield { type: 'step-start', step }

    const visible = rt.registry.visible(state.activeSkills)
    const dormant = rt.registry.dormant(state.activeSkills)
    const tools = toolSpecs(visible, dormant)
    const context = await buildContext(visible, skillCtx)
    const system = buildSystem(rt.system, visible, dormant)
    const budget = contextBudget(rt.model.contextWindow, system, context, tools)
    const toolOutputChars = budget ? Math.min(rt.maxToolOutputChars, Math.max(800, Math.floor(budget * 0.3))) : rt.maxToolOutputChars

    let text = ''
    let reasoning = ''
    const calls: ToolCall[] = []
    let reason: FinishReason = 'stop'
    let providerData: AssistantMessage['providerData']
    let stepUsage: Usage | undefined
    let metrics: StepMetrics | undefined
    const stepStarted = performance.now()

    try {
      for await (const chunk of rt.model.stream({
        system,
        context,
        messages: fitHistory(state.history, rt.maxHistory, budget),
        tools,
        signal: state.signal,
      })) {
        switch (chunk.type) {
          case 'text':
            text += chunk.delta
            yield { type: 'text-delta', delta: chunk.delta }
            break
          case 'reasoning':
            reasoning += chunk.delta
            yield { type: 'reasoning-delta', delta: chunk.delta }
            break
          case 'tool-call':
            calls.push(chunk.call)
            break
          case 'finish':
            reason = chunk.reason
            providerData = chunk.providerData
            stepUsage = chunk.usage
            metrics = chunk.metrics
            if (chunk.usage) {
              usage.inputTokens += chunk.usage.inputTokens
              usage.outputTokens += chunk.usage.outputTokens
            }
        }
      }
    } catch (error) {
      if (state.signal.aborted) {
        yield { type: 'finish', reason: 'aborted', steps: step, usage }
        return
      }
      throw error
    }

    const assistant: AssistantMessage = { role: 'assistant', content: text }
    if (calls.length) assistant.toolCalls = calls
    if (reasoning) assistant.reasoning = reasoning
    if (providerData) assistant.providerData = providerData
    state.history.push(assistant)
    yield { type: 'message', message: assistant }
    yield {
      type: 'step-finish',
      step,
      reason,
      durationMs: performance.now() - stepStarted,
      ...(stepUsage ? { usage: stepUsage } : {}),
      ...(metrics ? { metrics } : {}),
    }

    if (!calls.length) {
      yield { type: 'finish', reason, steps: step, usage }
      return
    }

    for (const call of calls) {
      yield { type: 'tool-call', call }
      const result: ToolMessage = yield* executeCall(rt, state, call, toolOutputChars)
      state.history.push(result)
      yield { type: 'message', message: result }
    }
  }

  yield { type: 'finish', reason: 'max-steps', steps: rt.maxSteps, usage }
}

async function* executeCall(
  rt: AgentRuntime,
  state: RunState,
  call: ToolCall,
  maxOutputChars: number,
): AsyncGenerator<AgentEvent, ToolMessage> {
  const started = performance.now()
  const reply = (_output: unknown, modelOutput: unknown, isError: boolean): ToolMessage => ({
    role: 'tool',
    toolCallId: call.id,
    name: call.name,
    content: truncate(safeStringify(modelOutput), maxOutputChars),
    ...(isError ? { isError } : {}),
  })
  const fail = function* (message: string): Generator<AgentEvent, ToolMessage> {
    const output = { error: message }
    yield { type: 'tool-result', call, output, isError: true, durationMs: performance.now() - started }
    return reply(output, output, true)
  }

  if (call.name === ACTIVATE_SKILL) return yield* activateSkill(rt, state, call, started, reply, fail)

  const entry = rt.registry.tools.get(call.name)
  if (!entry || !rt.registry.visible(state.activeSkills).includes(entry.skill)) {
    return yield* fail(`Unknown tool "${call.name}".`)
  }

  const parsed = entry.def.input.safeParse(call.input)
  if (!parsed.success) return yield* fail(`Invalid input: ${z.prettifyError(parsed.error)}`)

  return yield* drain<AgentEvent, ToolMessage>(async (emit) => {
    const ctx: ToolContext = {
      db: rt.db,
      knowledge: rt.knowledge,
      embedder: rt.embedder,
      threadId: state.threadId,
      skill: entry.skill.name,
      signal: state.signal,
      emit: (data) => emit({ type: 'custom', skill: entry.skill.name, tool: call.name, data }),
    }

    const gate = entry.def.needsApproval
    const needsApproval = typeof gate === 'function' ? await gate(parsed.data, ctx) : !!gate
    if (needsApproval) {
      emit({ type: 'approval-request', call })
      const decide = state.onApproval ?? rt.onApproval
      const approved = decide ? await decide(call) : false
      if (!approved) {
        const output = { error: 'The user declined this action. Ask how they would like to proceed.' }
        emit({ type: 'tool-result', call, output, isError: true, durationMs: performance.now() - started })
        return reply(output, output, true)
      }
    }

    try {
      const output = await entry.def.execute(parsed.data, ctx)
      const modelOutput = entry.def.toModelOutput ? entry.def.toModelOutput(output) : output
      emit({ type: 'tool-result', call, output, isError: false, durationMs: performance.now() - started })
      return reply(output, modelOutput, false)
    } catch (error) {
      const output = { error: error instanceof Error ? error.message : String(error) }
      emit({ type: 'tool-result', call, output, isError: true, durationMs: performance.now() - started })
      return reply(output, output, true)
    }
  })
}

function* activateSkill(
  rt: AgentRuntime,
  state: RunState,
  call: ToolCall,
  started: number,
  reply: (output: unknown, modelOutput: unknown, isError: boolean) => ToolMessage,
  fail: (message: string) => Generator<AgentEvent, ToolMessage>,
): Generator<AgentEvent, ToolMessage> {
  const name = (call.input as { name?: unknown } | null)?.name
  const skill = typeof name === 'string' ? rt.registry.skills.get(name) : undefined
  if (!skill?.lazy) return yield* fail(`No activatable skill named ${JSON.stringify(name)}.`)
  state.activeSkills.add(skill.name)
  const output = {
    activated: skill.name,
    tools: Object.keys(skill.tools ?? {}),
    instructions: skill.instructions ?? '',
  }
  yield { type: 'tool-result', call, output, isError: false, durationMs: performance.now() - started }
  return reply(output, output, false)
}

function toolSpecs(visible: Skill[], dormant: Skill[]): ToolSpec[] {
  const specs = visible.flatMap((s) => Object.entries(s.tools ?? {}).map(([name, def]) => toToolSpec(name, def)))
  if (dormant.length) {
    specs.push({
      name: ACTIVATE_SKILL,
      description: 'Load an additional skill, making its instructions and tools available from the next step on.',
      inputSchema: {
        type: 'object',
        properties: { name: { type: 'string', enum: dormant.map((s) => s.name) } },
        required: ['name'],
        additionalProperties: false,
      },
    })
  }
  return specs
}

export function buildSystem(base: string, visible: Skill[], dormant: Skill[]): string {
  const parts = [base.trim()]
  for (const skill of visible) {
    if (skill.instructions) parts.push(`# Skill: ${skill.name}\n${skill.instructions.trim()}`)
  }
  if (dormant.length) {
    parts.push(
      `# More skills (call ${ACTIVATE_SKILL} to load one)\n` +
        dormant.map((s) => `- ${s.name}: ${s.description}`).join('\n'),
    )
  }
  return parts.join('\n\n')
}

async function buildContext(visible: Skill[], ctx: SkillContext): Promise<string | undefined> {
  const sections = await Promise.all(
    visible.map(async (s) => {
      const body = await s.context?.(ctx)
      return body ? `## ${s.name}\n${body.trim()}` : undefined
    }),
  )
  const present = sections.filter(Boolean)
  return present.length ? `# Current state\n${present.join('\n\n')}` : undefined
}

/** Rough, tokenizer-free estimate; errs on the side of fewer tokens per char. */
export const CHARS_PER_TOKEN = 3.2

/**
 * Characters of history that fit in the model's window after the system
 * prompt, live context, tool definitions and room for the reply.
 */
export function contextBudget(
  contextWindow: number | undefined,
  system: string,
  context: string | undefined,
  tools: ToolSpec[],
): number | undefined {
  if (!contextWindow) return undefined
  const reply = Math.min(2048, Math.floor(contextWindow * 0.25))
  const fixed = system.length + (context?.length ?? 0) + JSON.stringify(tools).length
  return Math.max(1500, Math.floor((contextWindow - reply) * CHARS_PER_TOKEN) - fixed)
}

const messageChars = (m: Message) =>
  m.content.length + (m.role === 'assistant' && m.toolCalls ? JSON.stringify(m.toolCalls).length : 0) + 16

/**
 * `trimHistory`, then drop whole older turns until the history fits
 * `maxChars`. If the current turn alone is too big, older tool results inside
 * it are shortened (on a copy; stored history is untouched).
 */
export function fitHistory(history: Message[], maxMessages: number, maxChars: number | undefined): Message[] {
  let window = trimHistory(history, maxMessages)
  if (maxChars === undefined) return window
  const total = (ms: Message[]) => ms.reduce((n, m) => n + messageChars(m), 0)

  while (total(window) > maxChars) {
    const next = window.findIndex((m, i) => i > 0 && m.role === 'user')
    if (next === -1) break
    window = window.slice(next)
  }
  if (total(window) <= maxChars) return window

  const lastTool = window.findLastIndex((m) => m.role === 'tool')
  const compacted = window.map((m, i) =>
    m.role === 'tool' && i < lastTool && m.content.length > 400
      ? { ...m, content: `${m.content.slice(0, 400)}\n…[shortened to fit context]` }
      : m,
  )
  return compacted
}

/**
 * Keep the most recent `max` messages, starting on a user message so that
 * tool results are never separated from the call that produced them.
 */
export function trimHistory(history: Message[], max: number): Message[] {
  if (history.length <= max) return history
  let start = history.length - max
  while (start < history.length && history[start]!.role !== 'user') start++
  if (start === history.length) {
    // No user message in the window: fall back to the last user message.
    start = history.findLastIndex((m) => m.role === 'user')
  }
  return history.slice(Math.max(start, 0))
}
