import type { InitProgressReport, MLCEngineInterface } from '@mlc-ai/web-llm'
import type { Model } from '../types.js'
import { findLLM } from '../web/catalog.js'
import { fromTextModel } from './text-protocol.js'

export interface WebLLMOptions {
  /**
   * A WebLLM model id (e.g. `Qwen3-4B-q4f16_1-MLC`) or a catalog preset id
   * (e.g. `qwen3-4b`, resolved to the f16 build). Prefer `recommendLLM()`
   * from `@enclave/core/web` to pick one that fits the device.
   */
  model: string
  /**
   * Run inference in a dedicated worker (recommended). Its entry calls
   * `serveWebLLM()` from `@enclave/core/models/webllm-worker`. Models created
   * on the same worker share one engine: switching models unloads the old one.
   */
  worker?: Worker
  /** Tokens of context. WebLLM builds default to 4096; agents need more. Default 8192. */
  contextWindow?: number
  temperature?: number
  maxTokens?: number
  /**
   * Qwen3 thinking mode. Default true: in our measurements Qwen3 4B called
   * tools 4/4 times with thinking and 0/4 without, on the same agent prompt.
   * Turn off for faster plain chat.
   */
  thinking?: boolean
  onProgress?(report: InitProgressReport): void
}

export interface WebLLMModel extends Model {
  readonly modelId: string
  readonly contextWindow: number
  /** Download (first run) and compile the model. Called lazily on first use. */
  load(): Promise<void>
  unload(): Promise<void>
}

interface EngineSlot {
  engine: Promise<MLCEngineInterface>
  /** `modelId@contextWindow` currently loaded, or being loaded. */
  loaded: string | undefined
  loading: Promise<void>
  progress: Set<(report: InitProgressReport) => void>
}

const slots = new WeakMap<object, EngineSlot>()
const inThreadKey = {}

function slotFor(worker: Worker | undefined): EngineSlot {
  const key = worker ?? inThreadKey
  let slot = slots.get(key)
  if (!slot) {
    const progress = new Set<(report: InitProgressReport) => void>()
    const initProgressCallback = (report: InitProgressReport) => progress.forEach((l) => l(report))
    slot = {
      progress,
      loaded: undefined,
      loading: Promise.resolve(),
      engine: import('@mlc-ai/web-llm').then((lib) =>
        worker
          ? new lib.WebWorkerMLCEngine(worker, { initProgressCallback })
          : new lib.MLCEngine({ initProgressCallback }),
      ),
    }
    slots.set(key, slot)
  }
  return slot
}

/** Resolve a catalog preset id to a WebLLM model id; pass real ids through. */
export function resolveWebLLMId(model: string, shaderF16 = true): string {
  const preset = findLLM(model)
  if (!preset) return model
  if (preset.runtime !== 'webllm') throw new Error(`Preset "${model}" is not a WebLLM model`)
  return `${preset.model}-${shaderF16 ? 'q4f16_1' : 'q4f32_1'}-MLC`
}

/**
 * Fully offline inference on the user's GPU via WebGPU. Weights are cached by
 * the browser after the first download. Tool calling uses the `<tool_call>`
 * text protocol, which Qwen3 and Hermes models are trained on.
 */
export function webllm(options: WebLLMOptions): WebLLMModel {
  const modelId = resolveWebLLMId(options.model)
  const contextWindow = options.contextWindow ?? findLLM(options.model)?.contextWindow ?? 8192
  const key = `${modelId}@${contextWindow}`
  const slot = slotFor(options.worker)

  const load = async (): Promise<void> => {
    if (options.onProgress) slot.progress.add(options.onProgress)
    try {
      // Chain loads so concurrent callers never race `reload`.
      slot.loading = slot.loading.catch(() => undefined).then(async () => {
        if (slot.loaded === key) return
        const engine = await slot.engine
        slot.loaded = undefined
        await engine.reload(modelId, { context_window_size: contextWindow })
        slot.loaded = key
      })
      await slot.loading
    } finally {
      if (options.onProgress) slot.progress.delete(options.onProgress)
    }
  }

  const model = fromTextModel({
    id: `webllm:${modelId}`,
    contextWindow,
    async *streamText({ system, messages, signal, stop }) {
      await load()
      const engine = await slot.engine
      const chunks = await engine.chat.completions.create({
        stream: true,
        messages: [{ role: 'system', content: system }, ...messages],
        ...(stop?.length ? { stop } : {}),
        ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
        ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
        ...(isQwen3(modelId) ? { extra_body: { enable_thinking: options.thinking ?? true } } : {}),
      })
      const onAbort = () => engine.interruptGenerate()
      signal?.addEventListener('abort', onAbort, { once: true })
      try {
        for await (const chunk of chunks) {
          const delta = chunk.choices[0]?.delta?.content
          if (delta) yield delta
        }
      } finally {
        signal?.removeEventListener('abort', onAbort)
      }
      signal?.throwIfAborted()
    },
  })

  return {
    ...model,
    modelId,
    contextWindow,
    load,
    async unload() {
      if (slot.loaded !== key) return
      slot.loaded = undefined
      await (await slot.engine).unload()
    },
  }
}

function isQwen3(modelId: string): boolean {
  return /^Qwen3/i.test(modelId)
}

/** Whether the model's weights are already in the browser cache (loads offline). */
export async function isWebLLMCached(model: string): Promise<boolean> {
  const lib = await import('@mlc-ai/web-llm')
  return lib.hasModelInCache(resolveWebLLMId(model))
}

/** Free the disk space used by a downloaded model. */
export async function deleteWebLLMCache(model: string): Promise<void> {
  const lib = await import('@mlc-ai/web-llm')
  await lib.deleteModelAllInfoInCache(resolveWebLLMId(model))
}
