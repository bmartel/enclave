import type { Embedder, FinishReason, Message, Model, ModelChunk, ModelRequest } from '../types.js'
import { openaiCompatible } from './openai.js'
import { localityOfUrl } from '../privacy/index.js'

export const OLLAMA_URL = 'http://localhost:11434'
export const LMSTUDIO_URL = 'http://localhost:1234'

// ---------------------------------------------------------------------------
// Ollama (native /api/chat: supports num_ctx, thinking and tools)
// ---------------------------------------------------------------------------

export interface OllamaOptions {
  model: string
  baseURL?: string
  /**
   * Context window to allocate (`num_ctx`). Ollama's default is small; agents
   * need room for tools, schema and history. Default 16384.
   */
  contextWindow?: number
  /** Reasoning for thinking-capable models (Qwen3, DeepSeek R1, gpt-oss…). Default false. */
  think?: boolean | 'low' | 'medium' | 'high'
  temperature?: number
  /** How long Ollama keeps the model in memory after a request, e.g. `30m`. */
  keepAlive?: string
  fetch?: typeof fetch
}

/**
 * A model served by Ollama on this machine. Data stays on the device; the
 * browser talks to localhost. Ollama allows localhost origins by default.
 */
export function ollama(options: OllamaOptions): Model {
  const baseURL = (options.baseURL ?? OLLAMA_URL).replace(/\/$/, '')
  const contextWindow = options.contextWindow ?? 16384
  const doFetch = options.fetch ?? fetch

  return {
    id: `ollama:${options.model}`,
    locality: localityOfUrl(baseURL),
    contextWindow,
    async *stream(request: ModelRequest): AsyncGenerator<ModelChunk> {
      const response = await doFetch(`${baseURL}/api/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model: options.model,
          stream: true,
          messages: toOllamaMessages(request),
          ...(request.tools.length
            ? {
                tools: request.tools.map((t) => ({
                  type: 'function',
                  function: { name: t.name, description: t.description, parameters: t.inputSchema },
                })),
              }
            : {}),
          think: options.think ?? false,
          ...(options.keepAlive ? { keep_alive: options.keepAlive } : {}),
          options: {
            num_ctx: contextWindow,
            ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
          },
        }),
        ...(request.signal ? { signal: request.signal } : {}),
      }).catch((error: unknown) => {
        throw new Error(`Can't reach Ollama at ${baseURL}. Is it running? (${String(error)})`)
      })
      if (!response.ok || !response.body) {
        throw new Error(`Ollama request failed (${response.status}): ${await response.text().catch(() => '')}`)
      }

      let calls = 0
      let reason: FinishReason = 'stop'
      let usage: { inputTokens: number; outputTokens: number } | undefined
      for await (const line of ndjson(response.body)) {
        const chunk = JSON.parse(line) as {
          message?: {
            content?: string
            thinking?: string
            tool_calls?: { id?: string; function: { name: string; arguments: unknown } }[]
          }
          done?: boolean
          done_reason?: string
          prompt_eval_count?: number
          eval_count?: number
          error?: string
        }
        if (chunk.error) throw new Error(`Ollama: ${chunk.error}`)
        const m = chunk.message
        if (m?.thinking) yield { type: 'reasoning', delta: m.thinking }
        if (m?.content) yield { type: 'text', delta: m.content }
        for (const tc of m?.tool_calls ?? []) {
          const input = typeof tc.function.arguments === 'string' ? safeJSON(tc.function.arguments) : tc.function.arguments
          calls++
          yield {
            type: 'tool-call',
            call: { id: tc.id ?? `call_${calls}_${Math.random().toString(36).slice(2, 8)}`, name: tc.function.name, input },
          }
        }
        if (chunk.done) {
          reason = chunk.done_reason === 'length' ? 'length' : calls ? 'tool-calls' : 'stop'
          usage = { inputTokens: chunk.prompt_eval_count ?? 0, outputTokens: chunk.eval_count ?? 0 }
        }
      }
      yield { type: 'finish', reason, ...(usage ? { usage } : {}) }
    },
  }
}

function toOllamaMessages(request: ModelRequest): unknown[] {
  const system = request.context ? `${request.system}\n\n${request.context}` : request.system
  const out: unknown[] = [{ role: 'system', content: system }]
  for (const m of request.messages as Message[]) {
    if (m.role === 'user') out.push({ role: 'user', content: m.content })
    else if (m.role === 'assistant') {
      out.push({
        role: 'assistant',
        content: m.content,
        ...(m.toolCalls?.length
          ? { tool_calls: m.toolCalls.map((c) => ({ function: { name: c.name, arguments: c.input ?? {} } })) }
          : {}),
      })
    } else {
      out.push({ role: 'tool', tool_name: m.name, content: m.content })
    }
  }
  return out
}

// ---------------------------------------------------------------------------
// LM Studio (OpenAI-compatible server with native tool calling)
// ---------------------------------------------------------------------------

export interface LMStudioOptions {
  model: string
  baseURL?: string
  /** Informs the agent's context budgeting. Set the actual size when loading the model in LM Studio. */
  contextWindow?: number
  temperature?: number
  fetch?: typeof fetch
}

/** A model served by LM Studio. Enable CORS in LM Studio's server settings. */
export function lmstudio(options: LMStudioOptions): Model {
  const baseURL = `${(options.baseURL ?? LMSTUDIO_URL).replace(/\/$/, '')}/v1`
  const inner = openaiCompatible({
    baseURL,
    model: options.model,
    ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
    ...(options.fetch ? { fetch: options.fetch } : {}),
  })
  return {
    id: `lmstudio:${options.model}`,
    locality: localityOfUrl(baseURL),
    ...(options.contextWindow ? { contextWindow: options.contextWindow } : {}),
    async *stream(request) {
      try {
        yield* inner.stream(request)
      } catch (error) {
        if (error instanceof TypeError) {
          throw new Error(`Can't reach LM Studio at ${baseURL}. Start the server and enable CORS. (${error.message})`)
        }
        throw error
      }
    },
  }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

export interface LocalModelInfo {
  provider: 'ollama' | 'lmstudio'
  id: string
  label: string
  kind: 'llm' | 'embedding'
  sizeBytes?: number
  parameterSize?: string
  quantization?: string
  contextLength?: number
  capabilities?: string[]
  loaded?: boolean
}

export interface DiscoverOptions {
  ollama?: string | false
  lmstudio?: string | false
  timeoutMs?: number
  fetch?: typeof fetch
}

/**
 * List models installed in local Ollama / LM Studio servers. Unreachable
 * servers are skipped, so this is safe to call on every page load.
 */
export async function discoverLocalModels(options: DiscoverOptions = {}): Promise<LocalModelInfo[]> {
  const doFetch = options.fetch ?? fetch
  const timeout = options.timeoutMs ?? 1500
  const get = async (url: string, init?: RequestInit) => {
    const response = await doFetch(url, { ...init, signal: AbortSignal.timeout(timeout) })
    if (!response.ok) throw new Error(String(response.status))
    return response.json()
  }

  const fromOllama = async (base: string): Promise<LocalModelInfo[]> => {
    const tags = (await get(`${base}/api/tags`)) as {
      models: { name: string; size?: number; details?: { parameter_size?: string; quantization_level?: string; family?: string } }[]
    }
    return Promise.all(
      tags.models.map(async (m) => {
        let capabilities: string[] | undefined
        let contextLength: number | undefined
        try {
          const show = (await get(`${base}/api/show`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ model: m.name }),
          })) as { capabilities?: string[]; model_info?: Record<string, unknown> }
          capabilities = show.capabilities
          const ctx = Object.entries(show.model_info ?? {}).find(([k]) => k.endsWith('.context_length'))?.[1]
          if (typeof ctx === 'number') contextLength = ctx
        } catch {
          /* older Ollama */
        }
        const embedding = capabilities ? capabilities.includes('embedding') && !capabilities.includes('completion') : /embed/i.test(m.name)
        return {
          provider: 'ollama' as const,
          id: m.name,
          label: m.name,
          kind: embedding ? ('embedding' as const) : ('llm' as const),
          ...(m.size ? { sizeBytes: m.size } : {}),
          ...(m.details?.parameter_size ? { parameterSize: m.details.parameter_size } : {}),
          ...(m.details?.quantization_level ? { quantization: m.details.quantization_level } : {}),
          ...(contextLength ? { contextLength } : {}),
          ...(capabilities ? { capabilities } : {}),
        }
      }),
    )
  }

  const fromLMStudio = async (base: string): Promise<LocalModelInfo[]> => {
    try {
      // REST API with richer metadata.
      const rich = (await get(`${base}/api/v0/models`)) as {
        data: { id: string; type?: string; state?: string; max_context_length?: number; quantization?: string; capabilities?: string[] }[]
      }
      return rich.data.map((m) => ({
          provider: 'lmstudio' as const,
          id: m.id,
          label: m.id,
          kind: m.type === 'embeddings' ? ('embedding' as const) : ('llm' as const),
          ...(m.max_context_length ? { contextLength: m.max_context_length } : {}),
          ...(m.quantization ? { quantization: m.quantization } : {}),
          ...(m.capabilities ? { capabilities: m.capabilities } : {}),
        loaded: m.state === 'loaded',
      }))
    } catch {
      const basic = (await get(`${base}/v1/models`)) as { data: { id: string }[] }
      return basic.data.map((m) => ({
        provider: 'lmstudio' as const,
        id: m.id,
        label: m.id,
        kind: /embed/i.test(m.id) ? ('embedding' as const) : ('llm' as const),
      }))
    }
  }

  const jobs: Promise<LocalModelInfo[]>[] = []
  if (options.ollama !== false) jobs.push(fromOllama((options.ollama || OLLAMA_URL).replace(/\/$/, '')).catch(() => []))
  if (options.lmstudio !== false) jobs.push(fromLMStudio((options.lmstudio || LMSTUDIO_URL).replace(/\/$/, '')).catch(() => []))
  return (await Promise.all(jobs)).flat()
}

/** Build a `Model` for a discovered entry. */
export function localModel(info: LocalModelInfo, overrides: { contextWindow?: number; think?: boolean } = {}): Model {
  const contextWindow = overrides.contextWindow ?? Math.min(info.contextLength ?? 16384, 32768)
  return info.provider === 'ollama'
    ? ollama({ model: info.id, contextWindow, ...(overrides.think !== undefined ? { think: overrides.think } : {}) })
    : lmstudio({ model: info.id, contextWindow })
}

// ---------------------------------------------------------------------------
// Embeddings from a local server
// ---------------------------------------------------------------------------

export interface LocalEmbedderOptions {
  provider: 'ollama' | 'lmstudio'
  model: string
  dimensions: number
  baseURL?: string
  queryPrefix?: string
  documentPrefix?: string
  fetch?: typeof fetch
}

/** Embeddings computed by Ollama or LM Studio via their OpenAI-compatible `/v1/embeddings`. */
export function localEmbedder(options: LocalEmbedderOptions): Embedder {
  const base = (options.baseURL ?? (options.provider === 'ollama' ? OLLAMA_URL : LMSTUDIO_URL)).replace(/\/$/, '')
  const doFetch = options.fetch ?? fetch
  return {
    id: `${options.provider}:${options.model}@${options.dimensions}`,
    locality: localityOfUrl(base),
    dimensions: options.dimensions,
    async embed(texts, kind) {
      const prefix = (kind === 'query' ? options.queryPrefix : options.documentPrefix) ?? ''
      const response = await doFetch(`${base}/v1/embeddings`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: options.model, input: texts.map((t) => prefix + t) }),
      })
      if (!response.ok) throw new Error(`Embedding request failed (${response.status}): ${await response.text()}`)
      const body = (await response.json()) as { data: { index: number; embedding: number[] }[] }
      return body.data.sort((a, b) => a.index - b.index).map((d) => d.embedding)
    },
  }
}

// ---------------------------------------------------------------------------

async function* ndjson(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let newline: number
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        if (line) yield line
      }
    }
    if (buffer.trim()) yield buffer.trim()
  } finally {
    reader.releaseLock()
  }
}

function safeJSON(raw: string): unknown {
  try {
    return JSON.parse(raw)
  } catch {
    return { __unparseable: raw }
  }
}
