import type { FinishReason, Message, Model, ModelChunk, ModelRequest, ToolCall } from '../types.js'
import { fromTextModel } from './text-protocol.js'
import { sseData } from './sse.js'
import { localityOfUrl } from '../privacy/index.js'

export interface OpenAICompatibleOptions {
  /** e.g. `http://localhost:11434/v1` (Ollama), `http://localhost:1234/v1` (LM Studio), `https://api.openai.com/v1`. */
  baseURL: string
  model: string
  apiKey?: string
  headers?: Record<string, string>
  temperature?: number
  maxTokens?: number
  /** Merged into every request body. */
  extraBody?: Record<string, unknown>
  /**
   * `native` uses the endpoint's `tools` support. `prompt` describes tools in
   * the system prompt and parses `<tool_call>` tags, for servers or models
   * without function calling. Default `native`.
   */
  toolMode?: 'native' | 'prompt'
  fetch?: typeof fetch
}

/**
 * Any OpenAI-compatible `/chat/completions` endpoint: Ollama, LM Studio,
 * llama.cpp server, vLLM, OpenRouter, OpenAI, or a private gateway.
 */
export function openaiCompatible(options: OpenAICompatibleOptions): Model {
  const id = `openai-compatible:${options.model}`
  const locality = localityOfUrl(options.baseURL)
  if (options.toolMode === 'prompt') {
    return fromTextModel({
      id,
      locality,
      async *streamText({ system, messages, signal }) {
        const body = {
          messages: [{ role: 'system', content: system }, ...messages],
        }
        for await (const chunk of streamCompletions(options, body, signal)) {
          const delta = chunk.choices?.[0]?.delta
          if (delta?.content) yield delta.content
        }
      },
    })
  }

  return {
    id,
    locality,
    async *stream(request: ModelRequest): AsyncGenerator<ModelChunk> {
      const body: Record<string, unknown> = {
        messages: toOpenAIMessages(request),
        stream_options: { include_usage: true },
      }
      if (request.tools.length) {
        body.tools = request.tools.map((t) => ({
          type: 'function',
          function: { name: t.name, description: t.description, parameters: t.inputSchema },
        }))
      }

      const pending = new Map<number, { id: string; name: string; args: string }>()
      let reason: FinishReason = 'stop'
      let usage: { inputTokens: number; outputTokens: number } | undefined

      for await (const chunk of streamCompletions(options, body, request.signal)) {
        if (chunk.usage) {
          usage = { inputTokens: chunk.usage.prompt_tokens ?? 0, outputTokens: chunk.usage.completion_tokens ?? 0 }
        }
        const choice = chunk.choices?.[0]
        if (!choice) continue
        const delta = choice.delta ?? {}
        const thinking = delta.reasoning_content ?? delta.reasoning
        if (thinking) yield { type: 'reasoning', delta: thinking }
        if (delta.content) yield { type: 'text', delta: delta.content }
        for (const tc of delta.tool_calls ?? []) {
          const index = tc.index ?? 0
          const entry = pending.get(index) ?? { id: '', name: '', args: '' }
          if (tc.id) entry.id = tc.id
          if (tc.function?.name) entry.name += tc.function.name
          if (tc.function?.arguments) entry.args += tc.function.arguments
          pending.set(index, entry)
        }
        if (choice.finish_reason) reason = mapFinish(choice.finish_reason)
      }

      for (const [index, entry] of [...pending].sort(([a], [b]) => a - b)) {
        const call: ToolCall = {
          id: entry.id || `call_${index}_${Math.random().toString(36).slice(2, 8)}`,
          name: entry.name,
          input: parseArgs(entry.args),
        }
        yield { type: 'tool-call', call }
      }
      if (pending.size && reason === 'stop') reason = 'tool-calls'
      yield { type: 'finish', reason, ...(usage ? { usage } : {}) }
    },
  }
}

interface CompletionChunk {
  choices?: {
    delta?: {
      content?: string | null
      reasoning_content?: string | null
      reasoning?: string | null
      tool_calls?: { index?: number; id?: string; function?: { name?: string; arguments?: string } }[]
    }
    finish_reason?: string | null
  }[]
  usage?: { prompt_tokens?: number; completion_tokens?: number } | null
}

async function* streamCompletions(
  options: OpenAICompatibleOptions,
  body: Record<string, unknown>,
  signal: AbortSignal | undefined,
): AsyncGenerator<CompletionChunk> {
  const doFetch = options.fetch ?? fetch
  const response = await doFetch(`${options.baseURL.replace(/\/$/, '')}/chat/completions`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}),
      ...options.headers,
    },
    body: JSON.stringify({
      model: options.model,
      stream: true,
      ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
      ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
      ...options.extraBody,
      ...body,
    }),
    ...(signal ? { signal } : {}),
  })
  if (!response.ok || !response.body) {
    throw new Error(`Model request failed (${response.status}): ${await response.text().catch(() => '')}`)
  }
  for await (const data of sseData(response.body)) {
    if (data === '[DONE]') return
    yield JSON.parse(data) as CompletionChunk
  }
}

function toOpenAIMessages(request: ModelRequest): unknown[] {
  const system = request.context ? `${request.system}\n\n${request.context}` : request.system
  const out: unknown[] = [{ role: 'system', content: system }]
  for (const m of request.messages as Message[]) {
    if (m.role === 'user') out.push({ role: 'user', content: m.content })
    else if (m.role === 'assistant') {
      out.push({
        role: 'assistant',
        content: m.content || null,
        ...(m.toolCalls?.length
          ? {
              tool_calls: m.toolCalls.map((c) => ({
                id: c.id,
                type: 'function',
                function: { name: c.name, arguments: JSON.stringify(c.input ?? {}) },
              })),
            }
          : {}),
      })
    } else {
      out.push({ role: 'tool', tool_call_id: m.toolCallId, content: m.content })
    }
  }
  return out
}

function parseArgs(raw: string): unknown {
  if (!raw.trim()) return {}
  try {
    return JSON.parse(raw)
  } catch {
    // Let schema validation report the problem back to the model.
    return { __unparseable: raw }
  }
}

function mapFinish(reason: string): FinishReason {
  switch (reason) {
    case 'tool_calls':
    case 'function_call':
      return 'tool-calls'
    case 'length':
      return 'length'
    case 'content_filter':
      return 'refusal'
    default:
      return 'stop'
  }
}
