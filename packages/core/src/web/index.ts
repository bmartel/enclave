import { createEnclave, type Enclave, type EnclaveOptions } from '../enclave.js'
import { webllm } from '../models/webllm.js'
import { createDb } from '../store/pglite.js'
import { createWorkerDb } from '../store/pglite-worker.js'
import { transformersEmbedder, transformersLLM, transformersReranker, type LoadProgress } from '../transformers/index.js'
import type { Db, Embedder, Model, Reranker } from '../types.js'
import {
  findEmbedding,
  findLLM,
  findReranker,
  recommendEmbedding,
  recommendLLM,
  recommendReranker,
  resolveLLM,
  type EmbeddingPreset,
  type LLMChoice,
  type RerankerPreset,
} from './catalog.js'
import { detectDevice, persistStorage, type DeviceProfile } from './device.js'

export * from './catalog.js'
export * from './device.js'

export interface WebProgress {
  stage: 'device' | 'database' | 'embedding' | 'reranker' | 'llm'
  text: string
  /** 0..1 when known. */
  progress?: number
}

export interface WebWorkers {
  /** Entry calls `servePGlite()` (`@enclave/core/pglite-worker`). */
  db?: Worker
  /** Entry calls `serveTransformers()` (`@enclave/core/transformers/worker`). Hosts embeddings, reranker and Transformers.js LLMs. */
  ml?: Worker
  /** Entry calls `serveWebLLM()` (`@enclave/core/models/webllm-worker`). */
  llm?: Worker
}

export interface BrowserLLMOptions {
  workers?: WebWorkers
  device?: DeviceProfile
  thinking?: boolean
  temperature?: number
  contextWindow?: number
  onProgress?(p: WebProgress): void
}

/**
 * Build an in-browser model from a catalog preset id (or a resolved choice):
 * WebLLM for `runtime: 'webllm'`, Transformers.js otherwise.
 */
export async function browserLLM(
  selection: string | LLMChoice,
  options: BrowserLLMOptions = {},
): Promise<Model & { load(): Promise<void> }> {
  const device = options.device ?? (await detectDevice())
  let choice: LLMChoice
  if (typeof selection === 'string') {
    const preset = findLLM(selection)
    if (!preset) {
      // A raw WebLLM model id.
      return webllm({
        model: selection,
        ...(options.workers?.llm ? { worker: options.workers.llm } : {}),
        ...(options.contextWindow ? { contextWindow: options.contextWindow } : {}),
        ...(options.thinking !== undefined ? { thinking: options.thinking } : {}),
        ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
        ...(options.onProgress ? { onProgress: llmProgress(options.onProgress) } : {}),
      })
    }
    choice = resolveLLM(preset, device)
  } else {
    choice = selection
  }
  const contextWindow = options.contextWindow ?? choice.contextWindow
  // Hybrid-reasoning models call tools far more reliably with thinking on.
  const thinking = options.thinking ?? choice.preset.thinking === 'hybrid'

  if (choice.preset.runtime === 'webllm') {
    return webllm({
      model: choice.modelId,
      contextWindow,
      thinking,
      ...(options.workers?.llm ? { worker: options.workers.llm } : {}),
      ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
      ...(options.onProgress ? { onProgress: llmProgress(options.onProgress) } : {}),
    })
  }
  return transformersLLM({
    model: choice.modelId,
    contextWindow,
    ...(choice.dtype ? { dtype: choice.dtype } : {}),
    thinking,
    ...(options.workers?.ml ? { worker: options.workers.ml } : {}),
    ...(options.onProgress ? { onProgress: fileProgress('llm', options.onProgress) } : {}),
  })
}

export interface WebEnclaveOptions extends Omit<EnclaveOptions, 'db' | 'model' | 'embedder' | 'reranker'> {
  workers?: WebWorkers
  /** Default `idb://enclave`. Use `opfs-ahp://name` with a db worker for the fastest storage. */
  dataDir?: string
  /** Or bring your own PGlite. */
  db?: Db
  /** `auto` (default): best browser model for this device. Or a preset id, WebLLM id, or any `Model` (Ollama, LM Studio…). */
  llm?: 'auto' | string | Model
  /** `auto` (default), an `EMBEDDING_PRESETS` id, or any `Embedder`. */
  embedding?: 'auto' | string | Embedder
  /** `auto` (default), a `RERANKER_PRESETS` id, any `Reranker`, or `false`. */
  reranker?: 'auto' | string | Reranker | false
  /** Override the GPU memory budget used to pick models, in MB. */
  gpuBudgetMB?: number
  /** Largest first download `auto` may choose, in MB. Default 2500. */
  maxDownloadMB?: number
  /**
   * Thinking mode for hybrid-reasoning browser models (Qwen3). Default: on for
   * those models. It is slower but makes tool use reliable; turn off for plain chat.
   */
  thinking?: boolean
  /** Ask the browser to keep data and model caches from eviction. Default true. */
  persist?: boolean
  /** Download the LLM during setup instead of on the first message. Default false. */
  preloadLLM?: boolean
  onProgress?(p: WebProgress): void
}

export interface WebEnclave extends Enclave {
  device: DeviceProfile
  plan: {
    llm: LLMChoice | { id: string }
    embedding: EmbeddingPreset | { id: string }
    reranker: RerankerPreset | { id: string } | undefined
  }
  /** Switch the chat model: a browser preset id, WebLLM id, or any `Model`. */
  useModel(selection: string | Model): Promise<Model>
}

/**
 * Web-first setup in one call: profiles the device, picks the best browser
 * LLM, embedding model and reranker that fit, opens PGlite, and wires workers.
 *
 * ```ts
 * const ai = await createWebEnclave({
 *   workers: {
 *     db: new Worker(new URL('./db.worker.ts', import.meta.url), { type: 'module' }),
 *     ml: new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' }),
 *     llm: new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' }),
 *   },
 *   skills: [knowledgeSkill(), sqlSkill()],
 * })
 * ```
 */
export async function createWebEnclave(options: WebEnclaveOptions = {}): Promise<WebEnclave> {
  const report = options.onProgress ?? (() => undefined)
  const workers = options.workers ?? {}

  report({ stage: 'device', text: 'Checking device capabilities…' })
  const device = await detectDevice(options.gpuBudgetMB !== undefined ? { gpuBudgetMB: options.gpuBudgetMB } : {})
  if (options.persist ?? true) void persistStorage()

  report({ stage: 'database', text: 'Opening database…' })
  const db =
    options.db ??
    (workers.db
      ? await createWorkerDb(workers.db, { dataDir: options.dataDir ?? 'idb://enclave' })
      : await createDb({ dataDir: options.dataDir ?? 'idb://enclave' }))

  // Embeddings
  let embedder: Embedder
  let embeddingPlan: WebEnclave['plan']['embedding']
  if (options.embedding && typeof options.embedding === 'object') {
    embedder = options.embedding
    embeddingPlan = { id: embedder.id }
  } else {
    const preset =
      !options.embedding || options.embedding === 'auto' ? recommendEmbedding(device) : findEmbedding(options.embedding)
    if (!preset) throw new Error(`Unknown embedding preset "${String(options.embedding)}"`)
    embedder = transformersEmbedder({
      preset,
      preload: true,
      ...(workers.ml ? { worker: workers.ml } : {}),
      onProgress: fileProgress('embedding', report),
    })
    embeddingPlan = preset
  }

  // Reranker
  let reranker: Reranker | undefined
  let rerankerPlan: WebEnclave['plan']['reranker']
  if (options.reranker && typeof options.reranker === 'object') {
    reranker = options.reranker
    rerankerPlan = { id: reranker.id }
  } else if (options.reranker !== false) {
    const preset =
      !options.reranker || options.reranker === 'auto'
        ? recommendReranker(device, embeddingPlan as EmbeddingPreset)
        : findReranker(options.reranker)
    if (options.reranker && options.reranker !== 'auto' && !preset) {
      throw new Error(`Unknown reranker preset "${options.reranker}"`)
    }
    if (preset) {
      reranker = transformersReranker({
        preset,
        preload: true,
        ...(workers.ml ? { worker: workers.ml } : {}),
        onProgress: fileProgress('reranker', report),
      })
      rerankerPlan = preset
    }
  }

  // LLM
  const llmOptions: BrowserLLMOptions = {
    workers,
    device,
    ...(options.thinking !== undefined ? { thinking: options.thinking } : {}),
    onProgress: report,
  }
  let model: Model
  let llmPlan: WebEnclave['plan']['llm']
  if (options.llm && typeof options.llm === 'object') {
    model = options.llm
    llmPlan = { id: model.id }
  } else if (!options.llm || options.llm === 'auto') {
    const choice = recommendLLM(device, options.maxDownloadMB !== undefined ? { maxDownloadMB: options.maxDownloadMB } : {})
    if (!choice) throw new Error('No browser model fits this device; pass `llm` (e.g. an ollama() model).')
    model = await browserLLM(choice, llmOptions)
    llmPlan = choice
  } else {
    model = await browserLLM(options.llm, llmOptions)
    const preset = findLLM(options.llm)
    llmPlan = preset ? resolveLLM(preset, device) : { id: model.id }
  }
  if (options.preloadLLM) await (model as { load?(): Promise<void> }).load?.()

  const { workers: _w, dataDir: _d, db: _db, llm: _l, embedding: _e, reranker: _r, gpuBudgetMB: _g, maxDownloadMB: _m, thinking: _t, persist: _p, preloadLLM: _pl, onProgress: _o, ...rest } = options
  const enclave = await createEnclave({
    ...rest,
    db,
    model,
    embedder,
    ...(reranker ? { reranker } : {}),
    knowledge: {
      // Switching embedding presets re-embeds existing documents instead of failing.
      autoReindex: true,
      onReindexProgress: (done, total) =>
        report({ stage: 'embedding', text: `Re-embedding documents ${done}/${total}`, progress: done / total }),
      ...options.knowledge,
    },
  })

  return Object.assign(enclave, {
    device,
    plan: { llm: llmPlan, embedding: embeddingPlan, reranker: rerankerPlan },
    async useModel(selection: string | Model) {
      const next = typeof selection === 'string' ? await browserLLM(selection, llmOptions) : selection
      enclave.setModel(next)
      return next
    },
  })
}

function llmProgress(report: (p: WebProgress) => void) {
  return (r: { progress: number; text: string }) => report({ stage: 'llm', text: r.text, progress: r.progress })
}

function fileProgress(stage: WebProgress['stage'], report: (p: WebProgress) => void) {
  const files = new Map<string, { loaded: number; total: number }>()
  return (p: LoadProgress) => {
    if (p.status === 'progress' && p.file && p.total) {
      files.set(p.file, { loaded: p.loaded ?? 0, total: p.total })
      let loaded = 0
      let total = 0
      for (const f of files.values()) {
        loaded += f.loaded
        total += f.total
      }
      report({
        stage,
        text: `Downloading ${p.model.split('/').pop()} ${Math.round(loaded / 2 ** 20)}/${Math.round(total / 2 ** 20)} MB`,
        progress: loaded / total,
      })
    } else if (p.status === 'ready') {
      report({ stage, text: `${p.model.split('/').pop()} ready`, progress: 1 })
    }
  }
}
