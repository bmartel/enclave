import Anthropic from '@anthropic-ai/sdk'
import type {
  BetaContentBlockParam,
  BetaMessageParam,
  BetaTextBlockParam,
  BetaToolResultBlockParam,
  BetaToolUnion,
} from '@anthropic-ai/sdk/resources/beta/messages/messages'
import type { FinishReason, Message, Model, ModelChunk, ModelRequest } from '../types.js'

export interface AnthropicOptions {
  /** Defaults to `claude-opus-5`. */
  model?: string
  apiKey?: string
  /** Bring your own client (custom baseURL, proxy, auth). */
  client?: Anthropic
  /** Default 64,000; streaming keeps long outputs safe from HTTP timeouts. */
  maxTokens?: number
  effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max'
  /** Stream readable thinking summaries as `reasoning` events. Default true. */
  showThinking?: boolean
  /**
   * Re-run a refused request on a fallback model inside the same call
   * (server-side refusal fallbacks). Default true.
   */
  refusalFallbacks?: boolean
}

const PROVIDER = 'anthropic'
const FALLBACK_BETA = 'server-side-fallback-2026-07-01'

/**
 * Claude via the Messages API. Works from the browser with a user-supplied key
 * (`dangerouslyAllowBrowser`) or through your own proxy via `client`.
 */
export function anthropic(options: AnthropicOptions = {}): Model {
  const model = options.model ?? 'claude-opus-5'
  const client =
    options.client ??
    new Anthropic({ ...(options.apiKey ? { apiKey: options.apiKey } : {}), dangerouslyAllowBrowser: true })
  const fallbacks = options.refusalFallbacks ?? true

  return {
    id: `anthropic:${model}`,
    async *stream(request: ModelRequest): AsyncGenerator<ModelChunk> {
      const system: BetaTextBlockParam[] = [
        { type: 'text', text: request.system, cache_control: { type: 'ephemeral' } },
      ]
      // Volatile context goes after the cached prefix so it never invalidates it.
      if (request.context) system.push({ type: 'text', text: request.context })

      const tools: BetaToolUnion[] = request.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.inputSchema as Anthropic.Beta.BetaTool.InputSchema,
        eager_input_streaming: true,
      }))

      const stream = client.beta.messages.stream(
        {
          model,
          max_tokens: options.maxTokens ?? 64_000,
          system,
          messages: toAnthropicMessages(request.messages),
          ...(tools.length ? { tools } : {}),
          thinking: { type: 'adaptive', display: options.showThinking === false ? 'omitted' : 'summarized' },
          ...(options.effort ? { output_config: { effort: options.effort } } : {}),
          cache_control: { type: 'ephemeral' },
          ...(fallbacks ? { betas: [FALLBACK_BETA], fallbacks: 'default' as const } : {}),
        },
        request.signal ? { signal: request.signal } : undefined,
      )

      for await (const event of stream) {
        if (event.type !== 'content_block_delta') continue
        if (event.delta.type === 'text_delta') yield { type: 'text', delta: event.delta.text }
        else if (event.delta.type === 'thinking_delta') yield { type: 'reasoning', delta: event.delta.thinking }
      }

      const final = await stream.finalMessage()
      const truncated = final.stop_reason === 'max_tokens'
      // A tool_use cut off by max_tokens may carry partial input: never run it.
      const content = truncated ? final.content.filter((b) => b.type !== 'tool_use') : final.content
      for (const block of content) {
        if (block.type === 'tool_use') yield { type: 'tool-call', call: { id: block.id, name: block.name, input: block.input } }
      }

      yield {
        type: 'finish',
        reason: mapStop(final.stop_reason),
        usage: {
          inputTokens:
            final.usage.input_tokens +
            (final.usage.cache_read_input_tokens ?? 0) +
            (final.usage.cache_creation_input_tokens ?? 0),
          outputTokens: final.usage.output_tokens,
        },
        providerData: { provider: PROVIDER, data: { model: final.model, content } },
      }
    },
  }
}

function toAnthropicMessages(messages: Message[]): BetaMessageParam[] {
  const out: BetaMessageParam[] = []
  let pendingResults: BetaToolResultBlockParam[] = []
  const flushResults = () => {
    // All results for one assistant turn go back in a single user message.
    if (pendingResults.length) out.push({ role: 'user', content: pendingResults })
    pendingResults = []
  }

  for (const m of messages) {
    if (m.role === 'tool') {
      pendingResults.push({
        type: 'tool_result',
        tool_use_id: m.toolCallId,
        content: m.content,
        ...(m.isError ? { is_error: true } : {}),
      })
      continue
    }
    flushResults()
    if (m.role === 'user') {
      out.push({ role: 'user', content: m.content })
    } else if (m.providerData?.provider === PROVIDER) {
      // Replay verbatim so signed thinking blocks stay valid.
      const data = m.providerData.data as { content: BetaContentBlockParam[] }
      out.push({ role: 'assistant', content: data.content })
    } else {
      const content: BetaContentBlockParam[] = []
      if (m.content) content.push({ type: 'text', text: m.content })
      for (const c of m.toolCalls ?? []) {
        content.push({ type: 'tool_use', id: c.id, name: c.name, input: (c.input ?? {}) as Record<string, unknown> })
      }
      out.push({ role: 'assistant', content: content.length ? content : [{ type: 'text', text: '(no output)' }] })
    }
  }
  flushResults()
  return out
}

function mapStop(reason: string | null): FinishReason {
  switch (reason) {
    case 'tool_use':
      return 'tool-calls'
    case 'max_tokens':
      return 'length'
    case 'refusal':
      return 'refusal'
    default:
      return 'stop'
  }
}
