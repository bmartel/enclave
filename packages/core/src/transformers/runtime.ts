/**
 * Model execution for Transformers.js. Runs inside a worker (see `serveTransformers`)
 * or in-thread. One instance caches every loaded model and serializes inference,
 * so embeddings, reranking and generation can share a single worker and GPU.
 */
import { toModelLoadError } from '../models/load-error.js'

export type DtypeSpec = string | { webgpu: string; webgpuF32: string; wasm: string }
export type DevicePreference = 'auto' | 'webgpu' | 'wasm'

interface LoadConfig {
  model: string
  dtype?: DtypeSpec
  device?: DevicePreference
}

export interface EmbedConfig extends LoadConfig {
  method: 'sentence_embedding' | 'pipeline'
  pooling?: 'mean' | 'cls' | 'last_token'
  /** Truncate to this many dimensions (Matryoshka) before normalizing. */
  dimensions?: number
  maxTokens?: number
}

export type RerankConfig = LoadConfig

export interface GenerateConfig extends LoadConfig {
  maxNewTokens?: number
  thinking?: boolean
}

export interface GenerateRequest {
  messages: { role: 'system' | 'user' | 'assistant'; content: string }[]
  stop?: string[]
}

/** Where Transformers.js fetches models and the ONNX Runtime WASM from. */
export interface TransformersEnv {
  remoteHost?: string
  remotePathTemplate?: string
  allowRemoteModels?: boolean
  /** Self-hosted ONNX Runtime WASM (otherwise loaded from cdn.jsdelivr.net). */
  wasmPaths?: string | { mjs: string; wasm: string }
}

export interface LoadProgress {
  model: string
  status: string
  file?: string
  progress?: number
  loaded?: number
  total?: number
}

type TJS = typeof import('@huggingface/transformers')

interface Backend {
  device: 'webgpu' | 'wasm' | undefined
  f16: boolean
}

function applyEnv(lib: TJS, env: TransformersEnv): void {
  if (env.remoteHost !== undefined) lib.env.remoteHost = env.remoteHost
  if (env.remotePathTemplate !== undefined) lib.env.remotePathTemplate = env.remotePathTemplate
  if (env.allowRemoteModels !== undefined) lib.env.allowRemoteModels = env.allowRemoteModels
  const wasm = (lib.env.backends.onnx as { wasm?: { wasmPaths?: unknown } }).wasm
  if (env.wasmPaths !== undefined && wasm) wasm.wasmPaths = env.wasmPaths
}

/** Files Transformers.js probes for but tolerates missing. */
const OPTIONAL_FILES = new Set([
  'generation_config.json',
  'preprocessor_config.json',
  'processor_config.json',
  'special_tokens_map.json',
  'chat_template.jinja',
  'chat_template.json',
])

export class TransformersRuntime {
  private lib: Promise<TJS> | undefined
  private backend: Promise<Backend> | undefined
  private readonly loaded = new Map<string, Promise<any>>()
  private queue: Promise<unknown> = Promise.resolve()

  private env: TransformersEnv = {}

  constructor(private readonly onProgress?: (progress: LoadProgress) => void) {}

  /** Apply hosting settings. Takes effect for models loaded afterwards. */
  async configure(env: TransformersEnv): Promise<void> {
    this.env = { ...this.env, ...env }
    if (this.lib) applyEnv(await this.lib, this.env)
  }

  private tjs(): Promise<TJS> {
    return (this.lib ??= import('@huggingface/transformers').then((lib) => {
      applyEnv(lib, this.env)
      return lib
    }))
  }

  private detect(): Promise<Backend> {
    return (this.backend ??= (async () => {
      const gpu = (globalThis.navigator as { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } } | undefined)?.gpu
      const inBrowser = typeof window !== 'undefined' || typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== 'undefined'
      try {
        const adapter = await gpu?.requestAdapter()
        if (adapter) return { device: 'webgpu', f16: adapter.features.has('shader-f16') }
      } catch {
        /* fall through */
      }
      // In Node, leave the device unset so onnxruntime-node picks the CPU.
      return { device: inBrowser ? 'wasm' : undefined, f16: false }
    })())
  }

  private async resolve(config: LoadConfig): Promise<{ device?: 'webgpu' | 'wasm'; dtype: string }> {
    const detected = await this.detect()
    const device = config.device === 'wasm' ? 'wasm' : config.device === 'webgpu' ? 'webgpu' : detected.device
    const spec = config.dtype ?? 'q8'
    const dtype =
      typeof spec === 'string' ? spec : device === 'webgpu' ? (detected.f16 ? spec.webgpu : spec.webgpuF32) : spec.wasm
    return { ...(device ? { device } : {}), dtype }
  }

  private progress(model: string) {
    return (p: Record<string, unknown>) =>
      this.onProgress?.({
        model,
        status: String(p.status ?? ''),
        ...(typeof p.file === 'string' ? { file: p.file } : {}),
        ...(typeof p.progress === 'number' ? { progress: p.progress } : {}),
        ...(typeof p.loaded === 'number' ? { loaded: p.loaded } : {}),
        ...(typeof p.total === 'number' ? { total: p.total } : {}),
      })
  }

  private cached<T>(key: string, load: () => Promise<T>): Promise<T> {
    let entry = this.loaded.get(key) as Promise<T> | undefined
    if (!entry) {
      entry = load().catch((error) => {
        this.loaded.delete(key)
        // Keys are `${kind}:${model}:…`: the second part names the model.
        throw toModelLoadError(error, key.split(':')[1] ?? key)
      })
      this.loaded.set(key, entry)
    }
    return entry
  }

  /** Serialize GPU work: one inference at a time across all models. */
  private exclusive<T>(task: () => Promise<T>): Promise<T> {
    const run = this.queue.then(task, task)
    this.queue = run.catch(() => undefined)
    return run
  }

  private async embedder(config: EmbedConfig) {
    const tjs = await this.tjs()
    const opts = { ...(await this.resolve(config)), progress_callback: this.progress(config.model) }
    return this.cached(`embed:${config.model}:${config.method}:${opts.dtype}:${opts.device}`, async () => {
      if (config.method === 'pipeline') {
        const extractor = await tjs.pipeline('feature-extraction', config.model, opts as never)
        return { kind: 'pipeline' as const, extractor }
      }
      const [tokenizer, model] = await Promise.all([
        tjs.AutoTokenizer.from_pretrained(config.model, { progress_callback: opts.progress_callback }),
        tjs.AutoModel.from_pretrained(config.model, opts as never),
      ])
      return { kind: 'sentence' as const, tokenizer, model }
    })
  }

  /**
   * Preload a model. Serialized with inference: creating several ONNX Runtime
   * WebGPU sessions concurrently in one worker can stall initialization.
   */
  load(kind: 'embed' | 'rerank' | 'generate', config: EmbedConfig | RerankConfig | GenerateConfig): Promise<void> {
    return this.exclusive(async () => {
      if (kind === 'embed') await this.embedder(config as EmbedConfig)
      else if (kind === 'rerank') await this.reranker(config)
      else await this.generator(config)
    })
  }

  /**
   * Whether every file this model needs (for the resolved dtype/device) is in
   * the cache. Transformers.js also lists optional files that many repos don't
   * have (e.g. generation_config.json); those are ignored here.
   */
  async isCached(config: LoadConfig): Promise<boolean> {
    try {
      const files = await this.cachedFiles(config)
      return files.length > 0 && files.every((f) => f.cached || OPTIONAL_FILES.has(f.file))
    } catch {
      return false
    }
  }

  /** Per-file cache state, for diagnostics and download UIs. */
  async cachedFiles(config: LoadConfig): Promise<{ file: string; cached: boolean }[]> {
    const tjs = await this.tjs()
    const { device, dtype } = await this.resolve(config)
    const result = await tjs.ModelRegistry.is_cached_files(config.model, { dtype: dtype as never, ...(device ? { device } : {}) })
    return result.files
  }

  /** Remove the model's cached files and unload it. */
  async clearCache(config: LoadConfig): Promise<void> {
    const tjs = await this.tjs()
    const { device, dtype } = await this.resolve(config)
    for (const [key, entry] of this.loaded) {
      if (!key.includes(`:${config.model}:`)) continue
      this.loaded.delete(key)
      const value = await entry.catch(() => undefined)
      await value?.model?.dispose?.()
      await value?.extractor?.dispose?.()
    }
    await tjs.ModelRegistry.clear_cache(config.model, { dtype: dtype as never, ...(device ? { device } : {}) })
  }

  embed(config: EmbedConfig, texts: string[]): Promise<number[][]> {
    return this.exclusive(async () => {
      const loaded = await this.embedder(config)
      let tensor: any
      if (loaded.kind === 'pipeline') {
        tensor = await loaded.extractor(texts, {
          pooling: config.pooling ?? 'mean',
          normalize: true,
        } as never)
      } else {
        const inputs = await loaded.tokenizer(texts, {
          padding: true,
          truncation: true,
          max_length: Math.min(config.maxTokens ?? 2048, 2048),
        })
        const output = await loaded.model(inputs)
        tensor = output.sentence_embedding
      }
      const native = tensor.dims.at(-1) as number
      if (config.dimensions && config.dimensions < native) tensor = tensor.slice(null, [0, config.dimensions])
      return tensor.normalize(2, -1).tolist() as number[][]
    })
  }

  private async reranker(config: RerankConfig) {
    const tjs = await this.tjs()
    const opts = { ...(await this.resolve(config)), progress_callback: this.progress(config.model) }
    return this.cached(`rerank:${config.model}:${opts.dtype}:${opts.device}`, async () => {
      const [tokenizer, model] = await Promise.all([
        tjs.AutoTokenizer.from_pretrained(config.model, { progress_callback: opts.progress_callback }),
        tjs.AutoModelForSequenceClassification.from_pretrained(config.model, opts as never),
      ])
      return { tokenizer, model }
    })
  }

  /** Cross-encoder relevance scores in [0, 1], one per document. */
  rerank(config: RerankConfig, query: string, documents: string[]): Promise<number[]> {
    if (!documents.length) return Promise.resolve([])
    return this.exclusive(async () => {
      const { tokenizer, model } = await this.reranker(config)
      const inputs = await tokenizer(new Array(documents.length).fill(query), {
        text_pair: documents,
        padding: true,
        truncation: true,
        max_length: 512,
      })
      const { logits } = await model(inputs)
      return (logits.sigmoid().tolist() as number[][]).map((row) => row[0]!)
    })
  }

  private async generator(config: GenerateConfig) {
    const tjs = await this.tjs()
    const opts = { ...(await this.resolve(config)), progress_callback: this.progress(config.model) }
    return this.cached(`generate:${config.model}:${opts.dtype}:${opts.device}`, async () => {
      const [tokenizer, model] = await Promise.all([
        tjs.AutoTokenizer.from_pretrained(config.model, { progress_callback: opts.progress_callback }),
        tjs.AutoModelForCausalLM.from_pretrained(config.model, opts as never),
      ])
      return { tokenizer, model }
    })
  }

  /** Stream generated text. Resolves when generation ends or `signal` aborts. */
  generate(
    config: GenerateConfig,
    request: GenerateRequest,
    onText: (text: string) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    return this.exclusive(async () => {
      const tjs = await this.tjs()
      const { tokenizer, model } = await this.generator(config)
      const inputs = tokenizer.apply_chat_template(request.messages as never, {
        add_generation_prompt: true,
        return_dict: true,
        enable_thinking: config.thinking ?? false,
      } as never) as Record<string, unknown>

      const stopper = new tjs.InterruptableStoppingCriteria()
      const onAbort = () => stopper.interrupt()
      signal?.addEventListener('abort', onAbort, { once: true })
      let produced = ''
      const streamer = new tjs.TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text: string) => {
          produced += text
          onText(text)
          if (request.stop?.some((s) => produced.includes(s))) stopper.interrupt()
        },
      } as never)
      try {
        await model.generate({
          ...inputs,
          max_new_tokens: config.maxNewTokens ?? 2048,
          do_sample: false,
          streamer,
          stopping_criteria: stopper,
        } as never)
      } finally {
        signal?.removeEventListener('abort', onAbort)
      }
    })
  }

  async dispose(): Promise<void> {
    const entries = [...this.loaded.values()]
    this.loaded.clear()
    for (const entry of entries) {
      const value = await entry.catch(() => undefined)
      await value?.model?.dispose?.()
      await value?.extractor?.dispose?.()
    }
  }
}
