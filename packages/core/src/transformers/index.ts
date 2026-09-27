import { fromTextModel } from '../models/text-protocol.js'
import type { Embedder, EmbedKind, Model, Reranker } from '../types.js'
import {
  findEmbedding,
  findLLM,
  findReranker,
  type EmbeddingPreset,
  type RerankerPreset,
} from '../web/catalog.js'
import { backendFor } from './protocol.js'
import type { DevicePreference, DtypeSpec, LoadProgress } from './runtime.js'

export type { LoadProgress, DevicePreference, DtypeSpec, TransformersEnv } from './runtime.js'
export { TransformersRuntime } from './runtime.js'

interface CommonOptions {
  /** Worker whose entry calls `serveTransformers()`. Omit to run in this thread. */
  worker?: Worker
  device?: DevicePreference
  /** Override the preset's dtype (e.g. `q8`, `q4`, `q4f16`, `fp16`, `fp32`). */
  dtype?: DtypeSpec
  onProgress?(progress: LoadProgress): void
}

/**
 * Point Transformers.js (in `worker`, or this thread) at your own model host
 * and ONNX Runtime WASM. Call before creating embedders/rerankers/LLMs.
 * `selfHostedTransformers()` from `@enclave/core/privacy` builds the settings.
 */
export function configureTransformers(env: import('./runtime.js').TransformersEnv, worker?: Worker): Promise<void> {
  return backendFor(worker).configure(env)
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

export interface TransformersEmbedderOptions extends CommonOptions {
  /** A preset id from `EMBEDDING_PRESETS` or a full preset object. Default `embeddinggemma`. */
  preset?: string | EmbeddingPreset
  /** Output size. Must be one of the preset's Matryoshka sizes when smaller than native. */
  dimensions?: number
  /** Start downloading immediately instead of on first use. */
  preload?: boolean
}

/**
 * Local embeddings. Presets encode each model's correct pooling, prefixes and
 * per-backend dtype (WebGPU f16 → WebGPU f32 → WASM), so vectors match the
 * model's published quality.
 */
export function transformersEmbedder(options: TransformersEmbedderOptions = {}): Embedder {
  const preset = resolvePreset(options.preset ?? 'embeddinggemma', findEmbedding, 'embedding')
  const dimensions = options.dimensions ?? preset.dimensions
  if (dimensions > preset.dimensions || (dimensions < preset.dimensions && !preset.matryoshka?.includes(dimensions))) {
    throw new Error(
      `${preset.id} supports ${preset.matryoshka?.join(', ') ?? preset.dimensions} dimensions, not ${dimensions}`,
    )
  }
  const backend = backendFor(options.worker)
  if (options.onProgress) subscribe(backend, preset.model, options.onProgress)
  const config = {
    model: preset.model,
    method: preset.method,
    ...(preset.pooling ? { pooling: preset.pooling } : {}),
    dimensions,
    maxTokens: preset.maxTokens,
    dtype: options.dtype ?? preset.dtype,
    ...(options.device ? { device: options.device } : {}),
  }

  const embedder: Embedder = {
    id: `transformers:${preset.model}@${dimensions}`,
    locality: 'device',
    dimensions,
    embed(texts: string[], kind: EmbedKind) {
      const prefixed = kind === 'query' && preset.queryPrefix ? texts.map((t) => preset.queryPrefix + t) : texts
      return backend.embed(config, prefixed)
    },
    formatDocument(text: string, title?: string) {
      if (preset.documentTemplate) {
        return preset.documentTemplate.replace('{title}', title?.trim() || 'none').replace('{text}', text)
      }
      return title ? `${title}\n\n${text}` : text
    },
    load: () => backend.load('embed', config),
  }
  if (options.preload) void embedder.load!().catch(() => undefined)
  return embedder
}

// ---------------------------------------------------------------------------
// Reranking
// ---------------------------------------------------------------------------

export interface TransformersRerankerOptions extends CommonOptions {
  /** A preset id from `RERANKER_PRESETS` or a full preset object. Default `mxbai-rerank-xsmall`. */
  preset?: string | RerankerPreset
  preload?: boolean
}

/** Cross-encoder reranking: reads query and passage together for precise ordering. */
export function transformersReranker(options: TransformersRerankerOptions = {}): Reranker {
  const preset = resolvePreset(options.preset ?? 'mxbai-rerank-xsmall', findReranker, 'reranker')
  const backend = backendFor(options.worker)
  if (options.onProgress) subscribe(backend, preset.model, options.onProgress)
  const config = {
    model: preset.model,
    dtype: options.dtype ?? preset.dtype,
    ...(options.device ? { device: options.device } : {}),
  }
  const reranker: Reranker = {
    id: `transformers:${preset.model}`,
    locality: 'device',
    rerank: (query, documents) => backend.rerank(config, query, documents),
    load: () => backend.load('rerank', config),
  }
  if (options.preload) void reranker.load!().catch(() => undefined)
  return reranker
}

// ---------------------------------------------------------------------------
// Text generation
// ---------------------------------------------------------------------------

export interface TransformersLLMOptions extends CommonOptions {
  /** A browser LLM preset id with `runtime: 'transformers'`, or a Hugging Face ONNX repo. */
  model: string
  contextWindow?: number
  maxNewTokens?: number
  /** Qwen3-style thinking. Default false here (WASM/CPU is slow); `browserLLM` enables it for hybrid presets. */
  thinking?: boolean
}

export interface TransformersLLM extends Model {
  load(): Promise<void>
}

/**
 * In-browser LLM on ONNX Runtime Web. Runs on WebGPU, or on WASM where
 * WebGPU is missing. WebLLM is usually faster on WebGPU; use this for
 * broader model coverage or CPU-only devices.
 */
export function transformersLLM(options: TransformersLLMOptions): TransformersLLM {
  const preset = findLLM(options.model)
  if (preset && preset.runtime !== 'transformers') {
    throw new Error(`Preset "${preset.id}" runs on WebLLM; use webllm() for it`)
  }
  const model = preset?.model ?? options.model
  const backend = backendFor(options.worker)
  if (options.onProgress) subscribe(backend, model, options.onProgress)
  const config = {
    model,
    dtype: options.dtype ?? { webgpu: 'q4f16', webgpuF32: 'q4', wasm: 'q4' },
    maxNewTokens: options.maxNewTokens ?? 2048,
    thinking: options.thinking ?? false,
    ...(options.device ? { device: options.device } : {}),
  }
  const contextWindow = options.contextWindow ?? preset?.contextWindow ?? 8192

  const text = fromTextModel({
    id: `transformers:${model}`,
    locality: 'device',
    async *streamText({ system, messages, signal, stop }) {
      // Bridge the callback-based stream into an async iterator.
      const queue: string[] = []
      let wake: (() => void) | undefined
      let done = false
      let failure: unknown
      backend
        .generate(
          config,
          { messages: [{ role: 'system', content: system }, ...messages], ...(stop ? { stop } : {}) },
          (t) => {
            queue.push(t)
            wake?.()
          },
          signal,
        )
        .catch((e) => (failure = e))
        .finally(() => {
          done = true
          wake?.()
        })
      while (true) {
        while (queue.length) yield queue.shift()!
        if (done) break
        await new Promise<void>((r) => (wake = r))
        wake = undefined
      }
      if (failure) throw failure
      signal?.throwIfAborted()
    },
  })
  return { ...text, contextWindow, load: () => backend.load('generate', config) }
}

// ---------------------------------------------------------------------------

function resolvePreset<T extends { id: string }>(
  value: string | T,
  find: (id: string) => T | undefined,
  kind: string,
): T {
  if (typeof value !== 'string') return value
  const preset = find(value)
  if (!preset) throw new Error(`Unknown ${kind} preset "${value}"`)
  return preset
}

function subscribe(
  backend: ReturnType<typeof backendFor>,
  model: string,
  listener: (p: LoadProgress) => void,
): () => void {
  return backend.onProgress((p) => {
    if (p.model === model) listener(p)
  })
}
