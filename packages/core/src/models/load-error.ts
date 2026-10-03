/** Why a model could not be loaded, as something an app can act on. */
export type ModelLoadReason = 'storage' | 'webgpu' | 'gpu-memory' | 'network' | 'unknown'

const MESSAGES: Record<ModelLoadReason, (model: string) => string> = {
  storage: (m) =>
    `There isn't enough browser storage to download ${m}. Free up disk space or clear data for other sites, then try again. ` +
    `If plenty of disk space is free, restart the browser: it can hold on to storage it no longer uses.`,
  webgpu: (m) => `${m} needs WebGPU, which isn't available in this browser. Use a recent Chrome or Edge, or choose a smaller model that runs on the CPU.`,
  'gpu-memory': (m) => `${m} doesn't fit in this device's GPU memory. Close other GPU-heavy tabs or choose a smaller model.`,
  network: (m) => `Couldn't download ${m}. Check the connection and try again. Once downloaded it works offline.`,
  unknown: (m) => `${m} failed to load.`,
}

/**
 * A model (language model, embedder or reranker) failed to download or
 * initialize. `reason` says why in terms an app can act on; `message` is
 * written for end users; `cause` is the original error.
 */
export class ModelLoadError extends Error {
  override readonly name = 'ModelLoadError'
  constructor(
    readonly reason: ModelLoadReason,
    readonly model: string,
    override readonly cause?: unknown,
  ) {
    const detail = reason === 'unknown' && cause ? ` ${String((cause as Error)?.message ?? cause)}` : ''
    super(MESSAGES[reason](model) + detail)
  }
}

/** Classify a load failure from the browser, WebLLM, ONNX Runtime or fetch. */
export function classifyLoadError(error: unknown): ModelLoadReason {
  if (error instanceof ModelLoadError) return error.reason
  const name = String((error as { name?: unknown } | null)?.name ?? '')
  const message = String((error as { message?: unknown } | null)?.message ?? error ?? '')
  if (name === 'QuotaExceededError' || /quota ?exceeded|no space|not enough (storage|space)|ENOSPC/i.test(message)) return 'storage'
  if (/out of memory|\bOOM\b|device (was )?lost|failed to allocate|allocation failed|exceeds the max(imum)? (buffer|storage)/i.test(message))
    return 'gpu-memory'
  if (/webgpu|navigator\.gpu|requestAdapter|no (suitable |compatible )?(gpu )?adapter|shader-f16 (is )?not supported/i.test(message))
    return 'webgpu'
  if (name === 'NetworkError' || /failed to fetch|networkerror|network error|load failed|ERR_(INTERNET|NETWORK|CONNECTION)|status (?:code )?(?:4|5)\d\d/i.test(message))
    return 'network'
  return 'unknown'
}

/** Wrap any load failure as a ModelLoadError (aborts and existing ModelLoadErrors pass through). */
export function toModelLoadError(error: unknown, model: string): unknown {
  if (error instanceof ModelLoadError) return error
  if ((error as { name?: string } | null)?.name === 'AbortError') return error
  return new ModelLoadError(classifyLoadError(error), model, error)
}
