import {
  TransformersRuntime,
  type EmbedConfig,
  type GenerateConfig,
  type GenerateRequest,
  type LoadProgress,
  type RerankConfig,
} from './runtime.js'

export type Request =
  | { id: number; op: 'load'; kind: 'embed' | 'rerank' | 'generate'; config: EmbedConfig | RerankConfig | GenerateConfig }
  | { id: number; op: 'embed'; config: EmbedConfig; texts: string[] }
  | { id: number; op: 'rerank'; config: RerankConfig; query: string; documents: string[] }
  | { id: number; op: 'generate'; config: GenerateConfig; request: GenerateRequest }
  | { id: number; op: 'abort' }

type WithoutId<T> = T extends unknown ? Omit<T, 'id'> : never

export type Response =
  | { type: 'progress'; progress: LoadProgress }
  | { type: 'text'; id: number; text: string }
  | { type: 'result'; id: number; value: unknown }
  | { type: 'error'; id: number; error: string }

/**
 * Where Transformers.js work runs: a worker (recommended) or this thread.
 * Every client built on the same worker shares one runtime and one model cache.
 */
export interface Backend {
  load(kind: 'embed' | 'rerank' | 'generate', config: EmbedConfig | RerankConfig | GenerateConfig): Promise<void>
  embed(config: EmbedConfig, texts: string[]): Promise<number[][]>
  rerank(config: RerankConfig, query: string, documents: string[]): Promise<number[]>
  generate(config: GenerateConfig, request: GenerateRequest, onText: (t: string) => void, signal?: AbortSignal): Promise<void>
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
    load: (kind, config) => runtime.load(kind, config),
    embed: (config, texts) => runtime.embed(config, texts),
    rerank: (config, query, documents) => runtime.rerank(config, query, documents),
    generate: (config, request, onText, signal) => runtime.generate(config, request, onText, signal),
    onProgress(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
}

function workerBackend(worker: Worker): Backend {
  let nextId = 1
  const listeners = new Set<(p: LoadProgress) => void>()
  const pending = new Map<number, { resolve(v: any): void; reject(e: Error): void; onText?(t: string): void }>()

  worker.addEventListener('message', (event: MessageEvent<Response>) => {
    const msg = event.data
    if (!msg || typeof msg !== 'object' || !('type' in msg)) return
    if (msg.type === 'progress') return listeners.forEach((l) => l(msg.progress))
    const entry = pending.get(msg.id)
    if (!entry) return
    if (msg.type === 'text') return entry.onText?.(msg.text)
    pending.delete(msg.id)
    if (msg.type === 'error') entry.reject(new Error(msg.error))
    else entry.resolve(msg.value)
  })

  const call = <T>(message: WithoutId<Request>, onText?: (t: string) => void, signal?: AbortSignal): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      const id = nextId++
      pending.set(id, { resolve, reject, ...(onText ? { onText } : {}) })
      signal?.addEventListener('abort', () => worker.postMessage({ id, op: 'abort' } satisfies Request), { once: true })
      worker.postMessage({ ...message, id } as Request)
    })

  return {
    load: (kind, config) => call({ op: 'load', kind, config }),
    embed: (config, texts) => call({ op: 'embed', config, texts }),
    rerank: (config, query, documents) => call({ op: 'rerank', config, query, documents }),
    generate: (config, request, onText, signal) => call({ op: 'generate', config, request }, onText, signal),
    onProgress(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
}

/**
 * Worker entry for every Transformers.js-backed component (embedder,
 * reranker, LLM). One worker can serve all three:
 *
 * ```ts
 * // ml.worker.ts
 * import { serveTransformers } from '@enclave/core/transformers/worker'
 * serveTransformers()
 * ```
 */
export function serveTransformers(): void {
  const scope = self as unknown as {
    postMessage(message: Response): void
    addEventListener(type: 'message', listener: (event: MessageEvent<Request>) => void): void
  }
  const runtime = new TransformersRuntime((progress) => scope.postMessage({ type: 'progress', progress }))
  const aborts = new Map<number, AbortController>()

  scope.addEventListener('message', async ({ data }) => {
    if (!data || typeof data !== 'object' || !('op' in data)) return
    if (data.op === 'abort') return aborts.get(data.id)?.abort()
    const { id } = data
    try {
      let value: unknown
      switch (data.op) {
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
        }
      }
      scope.postMessage({ type: 'result', id, value })
    } catch (error) {
      scope.postMessage({ type: 'error', id, error: error instanceof Error ? error.message : String(error) })
    }
  })
}
