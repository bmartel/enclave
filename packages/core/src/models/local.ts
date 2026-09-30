import type { Embedder, FinishReason, Message, Model, ModelChunk, ModelRequest, StepMetrics } from '../types.js'
import { openaiCompatible } from './openai.js'
import { TurnContext } from './turn-context.js'
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
  /**
   * Reasoning for thinking-capable models (Qwen3, DeepSeek R1, gpt-oss…).
   * `'auto'` (default) reasons on new user requests and answers directly after
   * tool results, as with WebLLM; models without the thinking capability
   * never get the flag. `true`/`false`/effort levels are sent as given.
   */
  think?: boolean | 'low' | 'medium' | 'high' | 'auto'
  temperature?: number
  /** How long Ollama keeps the model in memory after a request. Default `30m`. */
  keepAlive?: string
  fetch?: typeof fetch
}

/**
 * A model served by Ollama on this machine. Data stays on the device; the
 * browser talks to localhost. Ollama allows localhost origins by default.
 *
 * Prompts are laid out for Ollama's prompt cache, which is reused only while
 * the start of the prompt is unchanged: the system prompt holds instructions
 * only, live context rides on the newest user message, and earlier messages
 * are replayed exactly as first sent.
 */
export function ollama(options: OllamaOptions): Model {
  const baseURL = (options.baseURL ?? OLLAMA_URL).replace(/\/$/, '')
  const contextWindow = options.contextWindow ?? 16384
  const doFetch = options.fetch ?? fetch
  const thinkMode = options.think ?? 'auto'
  const turn = new TurnContext()
  let capabilities: Promise<string[]> | undefined
  const capabilitiesOf = () =>
    (capabilities ??= doFetch(`${baseURL}/api/show`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: options.model }),
    })
      .then(async (r) => (r.ok ? ((await r.json()) as { capabilities?: string[] }).capabilities ?? [] : []))
      .catch((): string[] => []))

  return {
    id: `ollama:${options.model}`,
    locality: localityOfUrl(baseURL),
    contextWindow,
    async *stream(request: ModelRequest): AsyncGenerator<ModelChunk> {
      const messages = request.messages as Message[]
      let think: boolean | string = thinkMode === 'auto' ? messages.at(-1)?.role === 'user' : thinkMode
      if (thinkMode === 'auto' && !(await capabilitiesOf()).includes('thinking')) think = false
      const wire = toOllamaMessages(request, turn)
      const promptChars = JSON.stringify(wire).length
      const started = performance.now()
      let firstTokenMs: number | undefined

      const response = await doFetch(`${baseURL}/api/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model: options.model,
          stream: true,
          messages: wire,
          ...(request.tools.length
            ? {
                tools: request.tools.map((t) => ({
                  type: 'function',
                  function: { name: t.name, description: t.description, parameters: t.inputSchema },
                })),
              }
            : {}),
          think,
          keep_alive: options.keepAlive ?? '30m',
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
      let metrics: StepMetrics | undefined
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
          prompt_eval_duration?: number
          eval_count?: number
          eval_duration?: number
          error?: string
        }
        if (chunk.error) throw new Error(`Ollama: ${chunk.error}`)
        const m = chunk.message
        if (firstTokenMs === undefined && (m?.thinking || m?.content || m?.tool_calls?.length)) {
          firstTokenMs = Math.round(performance.now() - started)
        }
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
          const decodeRate = chunk.eval_count && chunk.eval_duration ? Math.round(chunk.eval_count / (chunk.eval_duration / 1e9)) : undefined
          // Ollama's prompt_eval_count is the whole prompt, cached tokens
          // included, so it can't show cache reuse or prefill speed; the time
          // spent reading the prompt can (a reused cache makes it a fraction).
          metrics = {
            promptChars,
            ...(chunk.prompt_eval_duration !== undefined ? { prefillMs: Math.round(chunk.prompt_eval_duration / 1e6) } : {}),
            ...(firstTokenMs !== undefined ? { timeToFirstTokenMs: firstTokenMs } : {}),
            ...(decodeRate ? { decodeTokensPerSec: decodeRate } : {}),
          }
        }
      }
      yield { type: 'finish', reason, ...(usage ? { usage } : {}), ...(metrics ? { metrics } : {}) }
    },
  }
}

function toOllamaMessages(request: ModelRequest, turn: TurnContext): unknown[] {
  const { userContent, changedContext } = turn.layout(request)
  const out: { role: string; content: string; [key: string]: unknown }[] = [{ role: 'system', content: request.system }]
  for (const [i, m] of (request.messages as Message[]).entries()) {
    if (m.role === 'user') out.push({ role: 'user', content: userContent(m, i) })
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
  if (changedContext) out.at(-1)!.content += changedContext
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
  /**
   * Reasoning for thinking-capable models. `'auto'` (default) reasons on new
   * user requests and answers directly after tool results, as with WebLLM and
   * Ollama. LM Studio honours `reasoning_effort: 'none'`; `chat_template_kwargs`
   * and `reasoning.effort` were ignored in testing (0.4.25).
   */
  think?: boolean | 'auto'
  fetch?: typeof fetch
}

/** A model served by LM Studio. Enable CORS in LM Studio's server settings. */
export function lmstudio(options: LMStudioOptions): Model {
  const baseURL = `${(options.baseURL ?? LMSTUDIO_URL).replace(/\/$/, '')}/v1`
  const inner = openaiCompatible({
    baseURL,
    model: options.model,
    // LM Studio's llama.cpp and MLX engines reuse their cache by prompt prefix.
    contextPlacement: 'message',
    requestBody: (request) => {
      const think = options.think ?? 'auto'
      const off = think === false || (think === 'auto' && request.messages.at(-1)?.role !== 'user')
      return off ? { reasoning_effort: 'none' } : {}
    },
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
export function localModel(info: LocalModelInfo, overrides: { contextWindow?: number; think?: OllamaOptions['think'] } = {}): Model {
  const contextWindow = overrides.contextWindow ?? Math.min(info.contextLength ?? 16384, 32768)
  return info.provider === 'ollama'
    ? ollama({ model: info.id, contextWindow, ...(overrides.think !== undefined ? { think: overrides.think } : {}) })
    : lmstudio({ model: info.id, contextWindow, ...(overrides.think !== undefined && overrides.think !== 'low' && overrides.think !== 'medium' && overrides.think !== 'high' ? { think: overrides.think } : {}) })
}

// ---------------------------------------------------------------------------
// Embeddings from a local server
// ---------------------------------------------------------------------------

export interface LocalEmbedderOptions {
  provider: 'ollama' | 'lmstudio'
  model: string
  /**
   * Vector size to store. Smaller than the model's output means Matryoshka
   * truncation (keep the leading dimensions, re-normalize), which only
   * Matryoshka-trained models (Qwen3-Embedding, EmbeddingGemma) support.
   * pgvector's HNSW index takes at most 2000 dimensions.
   */
  dimensions: number
  baseURL?: string
  queryPrefix?: string
  documentPrefix?: string
  /** Cosine similarity below which passages are unrelated (see `Embedder.relevanceFloor`). */
  relevanceFloor?: number
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
    ...(options.relevanceFloor !== undefined ? { relevanceFloor: options.relevanceFloor } : {}),
    async embed(texts, kind) {
      const prefix = (kind === 'query' ? options.queryPrefix : options.documentPrefix) ?? ''
      const response = await doFetch(`${base}/v1/embeddings`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: options.model, input: texts.map((t) => prefix + t) }),
      })
      if (!response.ok) throw new Error(`Embedding request failed (${response.status}): ${await response.text()}`)
      const body = (await response.json()) as { data: { index: number; embedding: number[] }[] }
      return body.data.sort((a, b) => a.index - b.index).map((d) => truncate(d.embedding, options.dimensions))
    },
  }
}

// ---------------------------------------------------------------------------
// Measured presets (packages/evals: 65 production cases × 3 repeats, and the
// 91-query retrieval benchmark; M2 Max 32 GB, Ollama 0.35, default GPU limit)
// ---------------------------------------------------------------------------

export interface OllamaLLMPreset {
  tag: string
  role: 'quality' | 'fast'
  /** Download size in GB. */
  sizeGB: number
  /** Eval suite pass rate. */
  passRate: number
  /** Median seconds per eval case on the reference machine. */
  p50Seconds: number
  note?: string
}

/** Best first. Pick the first one that is installed (`recommendOllamaModel`). */
export const OLLAMA_LLM_PRESETS: OllamaLLMPreset[] = [
  { tag: 'qwen3.6:27b-q4_K_M', role: 'quality', sizeGB: 17, passRate: 0.985, p50Seconds: 15, note: 'Ties qwen3.8:27b on quality, 25% faster at p90.' },
  { tag: 'qwen3.8:27b-q4_K_M', role: 'quality', sizeGB: 17, passRate: 0.99, p50Seconds: 17 },
  {
    tag: 'qwen3.5:9b',
    role: 'fast',
    sizeGB: 6.6,
    passRate: 0.928,
    p50Seconds: 6,
    note: 'Twice claimed an action that had not happened; prefer a 27B where actions matter.',
  },
]

const QWEN_EMBED_QUERY = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'

/** Ollama embedders with the prompts and relevance floors they were measured with. */
export const OLLAMA_EMBEDDING_PRESETS: Record<string, Omit<LocalEmbedderOptions, 'provider' | 'fetch' | 'baseURL'>> = {
  // recall@3 0.995, MRR 0.941, 21 ms/query, multilingual 10/10
  embeddinggemma: { model: 'embeddinggemma', dimensions: 768, queryPrefix: 'task: search result | query: ', documentPrefix: 'title: none | text: ', relevanceFloor: 0.32 },
  // recall@3 0.995, MRR 0.908, 68 ms/query; truncated to 1024 (Matryoshka) for pgvector's HNSW limit
  'qwen3-embedding:4b': { model: 'qwen3-embedding:4b', dimensions: 1024, queryPrefix: QWEN_EMBED_QUERY, relevanceFloor: 0.43 },
  // recall@3 0.973, MRR 0.886. The exact instruction format matters: a paraphrase scored 0.912.
  'qwen3-embedding:0.6b': { model: 'qwen3-embedding:0.6b', dimensions: 1024, queryPrefix: QWEN_EMBED_QUERY, relevanceFloor: 0.36 },
  // recall@3 0.956, MRR 0.879, multilingual 9/10
  'bge-m3': { model: 'bge-m3', dimensions: 1024, relevanceFloor: 0.49 },
}

/** An Ollama embedder from a measured preset. Default `embeddinggemma`, the best measured. */
export function ollamaEmbedder(model = 'embeddinggemma', overrides: Partial<LocalEmbedderOptions> = {}): Embedder {
  const preset = OLLAMA_EMBEDDING_PRESETS[model]
  if (!preset) throw new Error(`No preset for ${model}; use localEmbedder() with its dimensions and prompts.`)
  return localEmbedder({ provider: 'ollama', ...preset, ...overrides })
}

/** The best measured preset among installed models (from `discoverLocalModels`). */
export function recommendOllamaModel(installed: LocalModelInfo[], role?: 'quality' | 'fast'): OllamaLLMPreset | undefined {
  const names = new Set(installed.filter((m) => m.provider === 'ollama').flatMap((m) => [m.id, m.id.replace(/:latest$/, '')]))
  return OLLAMA_LLM_PRESETS.find((p) => (!role || p.role === role) && names.has(p.tag))
}

// ---------------------------------------------------------------------------

/** Matryoshka truncation: the leading `dims` components, re-normalized to unit length. */
function truncate(vector: number[], dims: number): number[] {
  if (vector.length <= dims) return vector
  const head = vector.slice(0, dims)
  const norm = Math.hypot(...head) || 1
  return head.map((v) => v / norm)
}

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
