import { ModelLoadError, type ModelLoadReason } from '../models/load-error.js'
import {
  TransformersRuntime,
  type EmbedConfig,
  type GenerateConfig,
  type GenerateRequest,
  type LoadKind,
  type LoadProgress,
  type RerankConfig,
  type TranscribeConfig,
  type TranscribeEvent,
  type TranscribeRequest,
  type TransformersEnv,
} from './runtime.js'
import type { Transcript } from '../types.js'

type AnyConfig = EmbedConfig | RerankConfig | GenerateConfig | TranscribeConfig

export type Request =
  | { id: number; op: 'load'; kind: LoadKind; config: AnyConfig }
  | { id: number; op: 'embed'; config: EmbedConfig; texts: string[] }
  | { id: number; op: 'rerank'; config: RerankConfig; query: string; documents: string[] }
  | { id: number; op: 'generate'; config: GenerateConfig; request: GenerateRequest }
  | { id: number; op: 'transcribe'; config: TranscribeConfig; audio: Float32Array; request: TranscribeRequest }
  | { id: number; op: 'abort' }
  | { id: number; op: 'configure'; env: TransformersEnv }
  | { id: number; op: 'cached' | 'clear' | 'cachedFiles'; config: AnyConfig }

type WithoutId<T> = T extends unknown ? Omit<T, 'id'> : never

export type Response =
  | { type: 'progress'; progress: LoadProgress }
  | { type: 'text'; id: number; text: string }
  | { type: 'transcribe-event'; id: number; event: TranscribeEvent }
  | { type: 'result'; id: number; value: unknown }
  | { type: 'error'; id: number; error: string; name?: string; reason?: ModelLoadReason; model?: string }

/**
 * Where Transformers.js work runs: a worker (recommended) or this thread.
 * Every client built on the same worker shares one runtime and one model cache.
 */
export interface Backend {
  configure(env: TransformersEnv): Promise<void>
  isCached(config: AnyConfig): Promise<boolean>
  clearCache(config: AnyConfig): Promise<void>
  cachedFiles(config: AnyConfig): Promise<{ file: string; cached: boolean }[]>
  load(kind: LoadKind, config: AnyConfig): Promise<void>
  embed(config: EmbedConfig, texts: string[]): Promise<number[][]>
  rerank(config: RerankConfig, query: string, documents: string[]): Promise<number[]>
  generate(config: GenerateConfig, request: GenerateRequest, onText: (t: string) => void, signal?: AbortSignal): Promise<void>
  transcribe(
    config: TranscribeConfig,
    audio: Float32Array,
    request: TranscribeRequest,
    onEvent: (event: TranscribeEvent) => void,
    signal?: AbortSignal,
  ): Promise<Transcript>
  onProgress(listener: (p: LoadProgress) => void): () => void
}

const backends = new WeakMap<object, Backend>()
let inThread: Backend | undefined

export function backendFor(worker?: Worker): Backend {
  if (!worker) return (inThread ??= localBackend())
  let backend = backends.get(worker)
  if (!backend) backends.set(worker, (backend = workerBackend(worker)))
  return backend
}

function localBackend(): Backend {
  const listeners = new Set<(p: LoadProgress) => void>()
  const runtime = new TransformersRuntime((p) => listeners.forEach((l) => l(p)))
  return {
    configure: (env) => runtime.configure(env),
    isCached: (config) => runtime.isCached(config),
    clearCache: (config) => runtime.clearCache(config),
    cachedFiles: (config) => runtime.cachedFiles(config),
    load: (kind, config) => runtime.load(kind, config),
    embed: (config, texts) => runtime.embed(config, texts),
    rerank: (config, query, documents) => runtime.rerank(config, query, documents),
    generate: (config, request, onText, signal) => runtime.generate(config, request, onText, signal),
    transcribe: (config, audio, request, onEvent, signal) => runtime.transcribe(config, audio, request, onEvent, signal),
    onProgress(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
}

function workerBackend(worker: Worker): Backend {
  let nextId = 1
  const listeners = new Set<(p: LoadProgress) => void>()
  const pending = new Map<
    number,
    { resolve(v: any): void; reject(e: Error): void; onText?(t: string): void; onEvent?(e: TranscribeEvent): void }
  >()

  worker.addEventListener('message', (event: MessageEvent<Response>) => {
    const msg = event.data
    if (!msg || typeof msg !== 'object' || !('type' in msg)) return
    if (msg.type === 'progress') return listeners.forEach((l) => l(msg.progress))
    const entry = pending.get(msg.id)
    if (!entry) return
    if (msg.type === 'text') return entry.onText?.(msg.text)
    if (msg.type === 'transcribe-event') return entry.onEvent?.(msg.event)
    pending.delete(msg.id)
    if (msg.type === 'error') entry.reject(reviveError(msg))
    else entry.resolve(msg.value)
  })

  const call = <T>(
    message: WithoutId<Request>,
    onText?: (t: string) => void,
    signal?: AbortSignal,
    stream?: { onEvent(e: TranscribeEvent): void; rejectOnAbort: boolean },
  ): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      const id = nextId++
      if (stream?.rejectOnAbort && signal?.aborted) return reject(abortError(signal))
      pending.set(id, { resolve, reject, ...(onText ? { onText } : {}), ...(stream ? { onEvent: stream.onEvent } : {}) })
      signal?.addEventListener(
        'abort',
        () => {
          worker.postMessage({ id, op: 'abort' } satisfies Request)
          // Settle now; the worker stops at its next checkpoint and its reply is ignored.
          if (stream?.rejectOnAbort && pending.delete(id)) reject(abortError(signal))
        },
        { once: true },
      )
      worker.postMessage({ ...message, id } as Request)
    })

  return {
    configure: (env) => call({ op: 'configure', env }),
    isCached: (config) => call({ op: 'cached', config }),
    clearCache: (config) => call({ op: 'clear', config }),
    cachedFiles: (config) => call({ op: 'cachedFiles', config }),
    load: (kind, config) => call({ op: 'load', kind, config }),
    embed: (config, texts) => call({ op: 'embed', config, texts }),
    rerank: (config, query, documents) => call({ op: 'rerank', config, query, documents }),
    generate: (config, request, onText, signal) => call({ op: 'generate', config, request }, onText, signal),
    transcribe: (config, audio, request, onEvent, signal) =>
      call({ op: 'transcribe', config, audio, request }, undefined, signal, { onEvent, rejectOnAbort: true }),
    onProgress(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
}

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new DOMException('The operation was aborted.', 'AbortError')
}

/** Where `serveTransformers` listens. Defaults to the worker global scope. */
export interface ServeScope {
  postMessage(message: Response): void
  addEventListener(type: 'message', listener: (event: MessageEvent<Request>) => void): void
}

export interface ServeTransformersOptions {
  /** Message endpoint (default `self`). Tests pass one end of a `MessageChannel`. */
  scope?: ServeScope
  /** Runtime to serve (default a new `TransformersRuntime` reporting load progress to the client). */
  runtime?: (onProgress: (progress: LoadProgress) => void) => TransformersRuntime
}

/**
 * Worker entry for every Transformers.js-backed component (embedder,
 * reranker, LLM, transcriber). One worker can serve them all:
 *
 * ```ts
 * // ml.worker.ts
 * import { serveTransformers } from 'enclave-ai/transformers/worker'
 * serveTransformers()
 * ```
 */
export function serveTransformers(options: ServeTransformersOptions = {}): void {
  const scope = options.scope ?? (self as unknown as ServeScope)
  const report = (progress: LoadProgress) => scope.postMessage({ type: 'progress', progress })
  const runtime = options.runtime ? options.runtime(report) : new TransformersRuntime(report)
  const aborts = new Map<number, AbortController>()

  scope.addEventListener('message', async ({ data }) => {
    if (!data || typeof data !== 'object' || !('op' in data)) return
    if (data.op === 'abort') return aborts.get(data.id)?.abort()
    const { id } = data
    try {
      let value: unknown
      switch (data.op) {
        case 'configure':
          value = await runtime.configure(data.env)
          break
        case 'cached':
          value = await runtime.isCached(data.config)
          break
        case 'clear':
          value = await runtime.clearCache(data.config)
          break
        case 'cachedFiles':
          value = await runtime.cachedFiles(data.config)
          break
        case 'load':
          value = await runtime.load(data.kind, data.config)
          break
        case 'embed':
          value = await runtime.embed(data.config, data.texts)
          break
        case 'rerank':
          value = await runtime.rerank(data.config, data.query, data.documents)
          break
        case 'generate': {
          const controller = new AbortController()
          aborts.set(id, controller)
          try {
            await runtime.generate(data.config, data.request, (text) => scope.postMessage({ type: 'text', id, text }), controller.signal)
          } finally {
            aborts.delete(id)
          }
          break
        }
        case 'transcribe': {
          const controller = new AbortController()
          aborts.set(id, controller)
          try {
            value = await runtime.transcribe(
              data.config,
              data.audio,
              data.request,
              (event) => scope.postMessage({ type: 'transcribe-event', id, event }),
              controller.signal,
            )
          } finally {
            aborts.delete(id)
          }
        }
      }
      scope.postMessage({ type: 'result', id, value })
    } catch (error) {
      scope.postMessage({
        type: 'error',
        id,
        error: error instanceof Error ? error.message : String(error),
        ...(error instanceof Error ? { name: error.name } : {}),
        ...(error instanceof ModelLoadError ? { reason: error.reason, model: error.model } : {}),
      })
    }
  })
}

/** Rebuild a worker error on this side, keeping its name and load details. */
export function reviveError(msg: { error: string; name?: string; reason?: ModelLoadReason; model?: string }): Error {
  if (msg.name === 'ModelLoadError' && msg.reason && msg.model) {
    const revived = new ModelLoadError(msg.reason, msg.model)
    if (msg.reason === 'unknown') revived.message = msg.error
    return revived
  }
  return Object.assign(new Error(msg.error), msg.name ? { name: msg.name } : {})
}
