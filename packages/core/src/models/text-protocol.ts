import type { Locality, Message, Model, ModelChunk, ModelRequest, StepMetrics, ToolCall, ToolSpec, Usage } from '../types.js'

export interface TextMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface TextRequest {
  system: string
  messages: TextMessage[]
  /** Offered tools, for backends that can constrain decoding to valid calls. */
  tools: ToolSpec[]
  /** What the model is responding to: a new user request, or tool results mid-turn. */
  after: 'user' | 'tool'
  signal?: AbortSignal
  /** Stop sequences; backends that support them should end generation early. */
  stop?: string[]
}

/** Backends may report usage/timing after the text stream. */
export interface TextStats {
  usage?: Usage
  metrics?: StepMetrics
}

/**
 * A plain text-in/text-out chat model. `fromTextModel` layers tool calling on
 * top using the Hermes/Qwen `<tool_call>` convention, so any local model that
 * follows instructions can drive tools, including ones with no native support.
 */
export interface TextModel {
  readonly id: string
  readonly contextWindow?: number
  readonly locality?: Locality
  streamText(request: TextRequest): AsyncIterable<string | TextStats>
}

export interface TextProtocolOptions {
  /**
   * Where per-step live context goes. `message` (default) keeps the system
   * prompt byte-stable and attaches context to the newest message, so engines
   * with prefix KV reuse (WebLLM) only prefill what is new each step.
   */
  contextPlacement?: 'system' | 'message'
  /**
   * What happens to reasoning from earlier turns:
   * - `current-turn` (default): dropped, as Qwen3's chat template does. The
   *   KV cache is rebuilt at every new user turn.
   * - `all`: kept, so the KV cache survives across turns, at the cost of context.
   * - `auto`: kept until the conversation reaches `compactAt` of the context
   *   window, then all older reasoning is dropped at once (one rebuild) and
   *   accumulation starts again. Needs the model's `contextWindow`.
   */
  reasoningHistory?: 'current-turn' | 'all' | 'auto'
  /** Share of the context window that triggers compaction in `auto`. Default 0.6. */
  compactAt?: number
}

/** Small models sometimes keep going and invent the tool's answer. Stop there. */
export const TOOL_STOP = ['<tool_response>']

export function fromTextModel(model: TextModel, options: TextProtocolOptions = {}): Model {
  const provider = `text:${model.id}`
  const placement = options.contextPlacement ?? 'message'
  // Exactly what each message looked like when first sent. Replaying it
  // verbatim is what lets the engine keep its KV cache between steps.
  const sent = new WeakMap<Message, string>()
  // Assistant turns whose reasoning has been compacted away (auto mode). Once
  // compacted a turn stays compacted, so its rendering, and the cache, stay stable.
  const compacted = new WeakSet<Message>()
  const mode = options.reasoningHistory ?? 'current-turn'
  const compactChars = model.contextWindow ? model.contextWindow * 3.2 * (options.compactAt ?? 0.6) : Infinity

  return {
    id: model.id,
    ...(model.contextWindow ? { contextWindow: model.contextWindow } : {}),
    ...(model.locality ? { locality: model.locality } : {}),
    async *stream(request: ModelRequest): AsyncGenerator<ModelChunk> {
      const parser = new TaggedStreamParser()
      const inlineContext = placement === 'message' ? request.context : undefined
      const system = toolSystemPrompt(placement === 'system' ? request : { ...request, context: undefined })
      const render = () =>
        renderMessages(request.messages, {
          provider,
          sent,
          context: inlineContext,
          keepReasoning: mode === 'current-turn' ? false : mode === 'all' ? true : (m) => !compacted.has(m),
        })
      let rendered = render()
      let didCompact = false
      // Only compact when a new user turn starts: within a turn the cache is
      // always reused, and the agent's history budget guards the window.
      const newTurn = request.messages.at(-1)?.role === 'user'
      if (mode === 'auto' && newTurn && size(system, rendered) > compactChars) {
        didCompact = true
        const lastUser = request.messages.findLastIndex((m) => m.role === 'user')
        request.messages.forEach((m, i) => {
          if (i < lastUser && m.role === 'assistant') compacted.add(m)
        })
        rendered = render()
      }
      const promptChars = size(system, rendered)
      let raw = ''
      let stats: TextStats = {}
      for await (const piece of model.streamText({
        system,
        messages: rendered,
        tools: request.tools,
        after: request.messages.at(-1)?.role === 'tool' ? 'tool' : 'user',
        ...(request.signal ? { signal: request.signal } : {}),
        ...(request.tools.length ? { stop: TOOL_STOP } : {}),
      })) {
        if (typeof piece !== 'string') {
          stats = piece
          continue
        }
        raw += piece
        yield* parser.push(piece)
      }
      yield* parser.flush()
      yield {
        type: 'finish',
        reason: parser.callCount > 0 ? 'tool-calls' : 'stop',
        ...(stats.usage ? { usage: stats.usage } : {}),
        metrics: { ...stats.metrics, promptChars, ...(didCompact ? { compacted: true } : {}) },
        providerData: { provider, data: { raw } },
      }
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
</tool_call>`
}

const THINK_BLOCK = /<think>[\s\S]*?<\/think>\s*/g

interface RenderOptions {
  /** Replay assistant turns produced by this provider verbatim. */
  provider?: string
  /** Cache of as-sent renderings; new entries are recorded for the trailing message(s). */
  sent?: WeakMap<Message, string>
  /** Live context attached to the newest message. */
  context?: string | undefined
  /** Keep reasoning from earlier turns (per message when a function). */
  keepReasoning?: boolean | ((message: Message) => boolean)
}

/**
 * Flatten tool traffic into user/assistant text turns (Qwen/Hermes layout).
 * Reasoning is kept for the current turn and dropped from earlier ones,
 * matching how Qwen3 was trained.
 */
export function renderMessages(messages: Message[], options: RenderOptions = {}): TextMessage[] {
  const lastUser = messages.findLastIndex((m) => m.role === 'user')
  // Messages after the last assistant turn form the new input this step.
  const trailingStart = messages.findLastIndex((m) => m.role === 'assistant') + 1
  const out: TextMessage[] = []
  const push = (role: TextMessage['role'], content: string) => {
    const last = out.at(-1)
    if (last?.role === role) last.content += `\n${content}`
    else out.push({ role, content })
  }

  messages.forEach((m, i) => {
    const cached = options.sent?.get(m)
    if (cached !== undefined) return push(m.role === 'assistant' ? 'assistant' : 'user', cached)

    let text: string
    if (m.role === 'user') text = m.content
    else if (m.role === 'tool') text = `<tool_response>\n${m.content}\n</tool_response>`
    else {
      const raw = m.providerData?.provider === options.provider ? (m.providerData?.data as { raw?: string })?.raw : undefined
      if (raw !== undefined) {
        const keep = typeof options.keepReasoning === 'function' ? options.keepReasoning(m) : !!options.keepReasoning
        text = i < lastUser && !keep ? raw.replace(THINK_BLOCK, '') : raw
      } else {
        const calls = (m.toolCalls ?? []).map(
          (c) => `<tool_call>\n${JSON.stringify({ name: c.name, arguments: c.input ?? {} })}\n</tool_call>`,
        )
        text = [m.content, ...calls].filter(Boolean).join('\n') || '(no output)'
      }
    }

    if (i === trailingStart && m.role !== 'assistant' && options.context) {
      text = `<context>\n${options.context}\n</context>\n\n${text}`
    }
    if (i >= trailingStart) options.sent?.set(m, text)
    push(m.role === 'assistant' ? 'assistant' : 'user', text)
  })
  return out
}

const size = (system: string, messages: TextMessage[]) =>
  messages.reduce((n, m) => n + m.content.length + 16, system.length)

/** @deprecated Use `renderMessages`. */
export const toTextMessages = (messages: Message[]) => renderMessages(messages)

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
