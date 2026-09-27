import type { Message, Model, ModelChunk, ModelRequest, ToolCall, ToolSpec } from '../types.js'
import { safeStringify } from '../util.js'

/**
 * A plain text-in/text-out chat model. `fromTextModel` layers tool calling on
 * top using the Hermes/Qwen `<tool_call>` convention, so any local model that
 * follows instructions can drive tools, including ones with no native support.
 */
export interface TextModel {
  readonly id: string
  streamText(request: {
    system: string
    messages: { role: 'user' | 'assistant'; content: string }[]
    signal?: AbortSignal
    /** Stop sequences; backends that support them should end generation early. */
    stop?: string[]
  }): AsyncIterable<string>
  readonly contextWindow?: number
}

/** Small models sometimes keep going and invent the tool's answer. Stop there. */
export const TOOL_STOP = ['<tool_response>']

export function fromTextModel(model: TextModel): Model {
  return {
    id: model.id,
    ...(model.contextWindow ? { contextWindow: model.contextWindow } : {}),
    async *stream(request: ModelRequest): AsyncGenerator<ModelChunk> {
      const parser = new TaggedStreamParser()
      for await (const delta of model.streamText({
        system: toolSystemPrompt(request),
        messages: toTextMessages(request.messages),
        ...(request.signal ? { signal: request.signal } : {}),
        ...(request.tools.length ? { stop: TOOL_STOP } : {}),
      })) {
        yield* parser.push(delta)
      }
      const tail = parser.flush()
      yield* tail
      const hasCalls = parser.callCount > 0
      yield { type: 'finish', reason: hasCalls ? 'tool-calls' : 'stop' }
    },
  }
}

export function toolSystemPrompt(request: Pick<ModelRequest, 'system' | 'context' | 'tools'>): string {
  const parts = [request.system]
  if (request.tools.length) parts.push(renderTools(request.tools))
  if (request.context) parts.push(request.context)
  return parts.join('\n\n')
}

function renderTools(tools: ToolSpec[]): string {
  const lines = tools.map((t) =>
    JSON.stringify({ type: 'function', function: { name: t.name, description: t.description, parameters: t.inputSchema } }),
  )
  return `# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
${lines.join('\n')}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
After emitting tool calls, stop and wait: results arrive in <tool_response> tags in the next user turn.`
}

/** Flatten tool traffic into user/assistant text turns, merging consecutive same-role turns. */
export function toTextMessages(messages: Message[]): { role: 'user' | 'assistant'; content: string }[] {
  const out: { role: 'user' | 'assistant'; content: string }[] = []
  const push = (role: 'user' | 'assistant', content: string) => {
    const last = out.at(-1)
    if (last?.role === role) last.content += `\n${content}`
    else out.push({ role, content })
  }
  for (const m of messages) {
    if (m.role === 'user') push('user', m.content)
    else if (m.role === 'assistant') {
      const calls = (m.toolCalls ?? []).map(
        (c) => `<tool_call>\n${JSON.stringify({ name: c.name, arguments: c.input })}\n</tool_call>`,
      )
      push('assistant', [m.content, ...calls].filter(Boolean).join('\n') || '(no output)')
    } else {
      push('user', `<tool_response>\n${safeStringify({ name: m.name, content: m.content })}\n</tool_response>`)
    }
  }
  return out
}

type Mode = 'text' | 'think' | 'tool' | 'discard'
const OPEN: Record<string, Mode> = { '<think>': 'think', '<tool_call>': 'tool', '<tool_response>': 'discard' }
const CLOSE: Record<'think' | 'tool', string> = { think: '</think>', tool: '</tool_call>' }

/**
 * Incremental parser that separates `<think>` reasoning and `<tool_call>` JSON
 * from visible text, holding back only as much as could be a partial tag.
 */
export class TaggedStreamParser {
  private buffer = ''
  private mode: Mode = 'text'
  /** Swallow whitespace at the start of the answer and after <think>/<tool_call> blocks. */
  private trimLeading = true
  callCount = 0;

  *push(delta: string): Generator<ModelChunk> {
    // Everything after a hallucinated <tool_response> is the model talking to itself.
    if (this.mode === 'discard') return
    this.buffer += delta
    while (this.buffer) {
      if (this.mode === 'discard') {
        this.buffer = ''
        return
      }
      if (this.mode === 'text') {
        let best: { index: number; tag: string } | undefined
        for (const tag of Object.keys(OPEN)) {
          const index = this.buffer.indexOf(tag)
          if (index !== -1 && (!best || index < best.index)) best = { index, tag }
        }
        if (best) {
          const before = this.visible(this.buffer.slice(0, best.index))
          if (before) yield { type: 'text', delta: before }
          this.mode = OPEN[best.tag]!
          this.buffer = this.buffer.slice(best.index + best.tag.length)
          continue
        }
        const keep = partialSuffix(this.buffer, Object.keys(OPEN))
        const emit = this.buffer.slice(0, this.buffer.length - keep)
        this.buffer = this.buffer.slice(emit.length)
        const shown = this.visible(emit)
        if (shown) yield { type: 'text', delta: shown }
        return
      }

      const close = CLOSE[this.mode]
      const index = this.buffer.indexOf(close)
      if (index !== -1) {
        const body = this.buffer.slice(0, index)
        if (this.mode === 'think') {
          if (body) yield { type: 'reasoning', delta: body }
        } else {
          yield this.toolCall(body)
        }
        this.buffer = this.buffer.slice(index + close.length)
        this.mode = 'text'
        this.trimLeading = true
        continue
      }
      if (this.mode === 'think') {
        const keep = partialSuffix(this.buffer, [close])
        const emit = this.buffer.slice(0, this.buffer.length - keep)
        if (emit) yield { type: 'reasoning', delta: emit }
        this.buffer = this.buffer.slice(emit.length)
      }
      // Tool mode: accumulate until the closing tag arrives.
      return
    }
  }

  *flush(): Generator<ModelChunk> {
    const rest = this.buffer
    this.buffer = ''
    if (!rest || this.mode === 'discard') return
    if (this.mode === 'tool') yield this.toolCall(rest)
    else if (this.mode === 'think') yield { type: 'reasoning', delta: rest }
    else {
      const shown = this.visible(rest)
      if (shown) yield { type: 'text', delta: shown }
    }
    this.mode = 'text'
  }

  private visible(text: string): string {
    if (!this.trimLeading) return text
    const trimmed = text.replace(/^\s+/, '')
    if (trimmed) this.trimLeading = false
    return trimmed
  }

  private toolCall(body: string): ModelChunk {
    const id = `call_${++this.callCount}_${Math.random().toString(36).slice(2, 8)}`
    let call: ToolCall
    const xml = parseXmlCall(body)
    if (xml) return { type: 'tool-call', call: { id, ...xml } }
    try {
      const parsed = JSON.parse(body.trim()) as { name?: unknown; arguments?: unknown; parameters?: unknown }
      if (typeof parsed.name !== 'string') throw new Error('missing "name"')
      let input = parsed.arguments ?? parsed.parameters ?? {}
      if (typeof input === 'string') input = JSON.parse(input)
      call = { id, name: parsed.name, input }
    } catch (error) {
      // Surface malformed calls as a call to a nonexistent tool so the agent
      // returns an error and the model can correct itself.
      call = { id, name: 'invalid_tool_call', input: { raw: body.trim(), error: String(error) } }
    }
    return { type: 'tool-call', call }
  }
}

/**
 * Qwen3-Coder / Qwen3.5 style:
 * <function=name><parameter=key>value</parameter></function>
 */
function parseXmlCall(body: string): { name: string; input: Record<string, unknown> } | undefined {
  const fn = body.match(/<function=([^>\s]+)>([\s\S]*?)(?:<\/function>|$)/)
  if (!fn) return undefined
  const input: Record<string, unknown> = {}
  for (const p of fn[2]!.matchAll(/<parameter=([^>\s]+)>\n?([\s\S]*?)\n?<\/parameter>/g)) {
    const raw = p[2]!
    try {
      input[p[1]!] = JSON.parse(raw)
    } catch {
      input[p[1]!] = raw
    }
  }
  return { name: fn[1]!, input }
}

function partialSuffix(text: string, tags: string[]): number {
  let longest = 0
  for (const tag of tags) {
    for (let n = Math.min(tag.length - 1, text.length); n > longest; n--) {
      if (text.endsWith(tag.slice(0, n))) {
        longest = n
        break
      }
    }
  }
  return longest
}
