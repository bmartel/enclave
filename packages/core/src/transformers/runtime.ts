/**
 * Model execution for Transformers.js. Runs inside a worker (see `serveTransformers`)
 * or in-thread. One instance caches every loaded model and serializes inference,
 * so embeddings, reranking and generation can share a single worker and GPU.
 */
import { classifyLoadError, ModelLoadError, toModelLoadError } from '../models/load-error.js'
import type { Transcript, TranscriptSegment } from '../types.js'
import { toSegments, transcribeChunked, type AsrOutput } from './transcribe.js'

/** One dtype for the whole model, or one per ONNX file (e.g. `{ encoder_model: 'fp32', decoder_model_merged: 'q4' }`). */
export type Dtype = string | Record<string, string>
/** A dtype, or one per backend: WebGPU with shader-f16, WebGPU without, WASM. */
export type DtypeSpec = Dtype | { webgpu: Dtype; webgpuF32: Dtype; wasm: Dtype }
export type DevicePreference = 'auto' | 'webgpu' | 'wasm'

function perBackend(spec: DtypeSpec): spec is { webgpu: Dtype; webgpuF32: Dtype; wasm: Dtype } {
  return typeof spec === 'object' && 'wasm' in spec && 'webgpu' in spec && 'webgpuF32' in spec
}

/** Stable cache-key form of a dtype. */
/**
 * Split texts, in order, into groups whose padded size (count × longest)
 * stays within `budget` tokens. A text alone over budget is a group of one.
 */
export function groupByTokens(texts: string[], count: (text: string) => number, budget: number): string[][] {
  const groups: string[][] = []
  let group: string[] = []
  let longest = 0
  for (const text of texts) {
    const n = count(text)
    const next = Math.max(longest, n)
    if (group.length && next * (group.length + 1) > budget) {
      groups.push(group)
      group = []
      longest = 0
    }
    group.push(text)
    longest = Math.max(longest, n)
  }
  if (group.length) groups.push(group)
  return groups
}

interface TjsCacheEnv {
  allowLocalModels: boolean
  useBrowserCache: boolean
  cacheKey?: string
  remoteHost: string
  remotePathTemplate: string
}

/** The URL Transformers.js caches a model's config.json under (its remote URL, `main` revision). */
export function cachedConfigUrl(env: Pick<TjsCacheEnv, 'remoteHost' | 'remotePathTemplate'>, model: string): string {
  const path = env.remotePathTemplate.replaceAll('{model}', model).replaceAll('{revision}', 'main')
  return [env.remoteHost.replace(/\/+$/, ''), path.replace(/^\/+|\/+$/g, ''), 'config.json'].join('/')
}

/** Matryoshka truncation, then L2 normalization, as plain arrays. */
function finish(tensor: any, dimensions?: number): number[][] {
  const native = tensor.dims.at(-1) as number
  if (dimensions && dimensions < native) tensor = tensor.slice(null, [0, dimensions])
  return tensor.normalize(2, -1).tolist() as number[][]
}

function dtypeKey(dtype: Dtype): string {
  return typeof dtype === 'string'
    ? dtype
    : Object.keys(dtype)
        .sort()
        .map((k) => `${k}=${dtype[k]}`)
        .join(',')
}

export interface LoadConfig {
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
  /** Load only the text model of a multimodal repo (its vision and audio configs dropped). */
  textOnly?: boolean
  /** On WebGPU, split batches so batch size × padded length stays within this. */
  webgpuMaxBatchTokens?: number
}

export type RerankConfig = LoadConfig

export interface GenerateConfig extends LoadConfig {
  maxNewTokens?: number
  thinking?: boolean
}

export type TranscribeConfig = LoadConfig

/** Serializable subset of `TranscribeOptions` (no callbacks or signal). */
export interface TranscribeRequest {
  language?: string
  task?: 'transcribe' | 'translate'
  chunkSeconds?: number
  strideSeconds?: number
}

export type TranscribeEvent = { type: 'segment'; segment: TranscriptSegment } | { type: 'progress'; fraction: number }

export interface GenerateRequest {
  messages: { role: 'system' | 'user' | 'assistant'; content: string }[]
  stop?: string[]
}

/** Where Transformers.js fetches models and the ONNX Runtime WASM from. */
export interface TransformersEnv {
  remoteHost?: string
  remotePathTemplate?: string
  allowRemoteModels?: boolean
  /**
   * Where ONNX Runtime's WASM comes from. Transformers.js otherwise points it at
   * cdn.jsdelivr.net. `'bundled'` uses the copy your bundler emitted with
   * ONNX Runtime (Vite, webpack and Rollup copy `ort-wasm-*.wasm` next to it),
   * so nothing is fetched from a third party; a URL or `{ mjs, wasm }` names a
   * self-hosted copy (`{ wasm }` alone keeps the bundled JavaScript glue).
   */
  wasmPaths?: string | { mjs?: string; wasm: string } | 'bundled'
}

export interface LoadProgress {
  model: string
  /**
   * Transformers.js's status (`initiate`, `download`, `progress`, `done`,
   * `ready`…), or `fallback`: WebGPU failed for this model (`error` says how)
   * and it now runs on WASM (`device`). Apps may remember that and pass
   * `device: 'wasm'` from then on.
   */
  status: string
  device?: 'webgpu' | 'wasm'
  error?: string
  file?: string
  progress?: number
  loaded?: number
  total?: number
}

type TJS = typeof import('@huggingface/transformers')

export interface Backend {
  device: 'webgpu' | 'wasm' | undefined
  f16: boolean
}

function applyEnv(lib: TJS, env: TransformersEnv): void {
  if (env.remoteHost !== undefined) lib.env.remoteHost = env.remoteHost
  if (env.remotePathTemplate !== undefined) lib.env.remotePathTemplate = env.remotePathTemplate
  if (env.allowRemoteModels !== undefined) lib.env.allowRemoteModels = env.allowRemoteModels
  const wasm = (lib.env.backends.onnx as { wasm?: { wasmPaths?: unknown } }).wasm
  // Unset, ONNX Runtime loads the WASM its bundler emitted (import.meta.url).
  if (env.wasmPaths === 'bundled' && wasm) delete wasm.wasmPaths
  else if (env.wasmPaths !== undefined && wasm) wasm.wasmPaths = env.wasmPaths
  const fetcher = lib.env.fetch as ((input: string | URL, init?: any) => Promise<any>) & { shared?: boolean }
  if (typeof fetcher === 'function' && !fetcher.shared) lib.env.fetch = shareSizeProbes(fetcher)
}

/**
 * Transformers.js asks for each file's size (a one-byte Range request) before
 * downloading it, so every file was requested twice. The probe starts the
 * download instead and is answered from its headers; the download that
 * follows gets that response.
 */
export function shareSizeProbes(base: (input: string | URL, init?: any) => Promise<any>) {
  const started = new Map<string, Promise<Response>>()
  /** Probes that found nothing: the request that follows gets the same answer. */
  const missing = new Map<string, { status: number; statusText: string }>()
  const shared = async (input: string | URL, init?: any): Promise<any> => {
    const url = String(input)
    const headers = new Headers(init?.headers)
    const range = headers.get('Range')
    if (range === 'bytes=0-0' && /^https?:/.test(url) && (init?.method ?? 'GET') === 'GET') {
      headers.delete('Range')
      let download = started.get(url)
      if (!download) {
        const { cache: _cache, ...rest } = init ?? {}
        download = base(input, { ...rest, headers }) as Promise<Response>
        started.set(url, download)
        download.catch(() => started.delete(url))
      }
      const response = await download
      if (!response.ok) {
        started.delete(url)
        response.body?.cancel().catch(() => undefined)
        if (response.status === 404) missing.set(url, { status: response.status, statusText: response.statusText })
        return new Response(null, { status: response.status, statusText: response.statusText })
      }
      const size = response.headers.get('content-length')
      const type = response.headers.get('content-type')
      return new Response(null, {
        status: size ? 206 : 200,
        headers: { ...(type ? { 'content-type': type } : {}), ...(size ? { 'content-range': `bytes 0-0/${size}` } : {}) },
      })
    }
    const download = !range ? started.get(url) : undefined
    if (download) {
      started.delete(url)
      return download
    }
    const gone = !range ? missing.get(url) : undefined
    if (gone) {
      missing.delete(url)
      return new Response(null, gone)
    }
    return base(input, init)
  }
  return Object.assign(shared, { shared: true })
}

/**
 * Whether a failure came from the WebGPU backend (creating a session or a
 * compute pipeline, a lost device, GPU memory), so the same model may still
 * run on WASM. Some mobile GPUs (Adreno 7xx in Chrome 154) report an adapter
 * but fail with "Failed to create a WebGPU compute pipeline: A valid external
 * Instance reference no longer exists".
 */
export function isWebGpuFailure(error: unknown): boolean {
  if (error instanceof ModelLoadError && (error.reason === 'webgpu' || error.reason === 'gpu-memory')) return true
  let e: unknown = error
  for (let depth = 0; e && depth < 4; depth++) {
    const message = String((e as { message?: unknown })?.message ?? e)
    if (/webgpu|\bGPU(Device|Adapter|Buffer|Queue)\b|compute pipeline|external Instance|device (was )?lost|\bdawn\b|\bjsep\b/i.test(message)) return true
    const reason = classifyLoadError(e)
    if (reason === 'webgpu' || reason === 'gpu-memory') return true
    e = (e as { cause?: unknown }).cause
  }
  return false
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
  /** WebGPU failed in this runtime: models not pinned to a device run on WASM from now on. */
  private webgpuFailed = false

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

  protected detect(): Promise<Backend> {
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

  protected async resolve(config: LoadConfig): Promise<{ device?: 'webgpu' | 'wasm'; dtype: Dtype }> {
    const detected = await this.detect()
    const device =
      config.device === 'wasm'
        ? 'wasm'
        : config.device === 'webgpu'
          ? 'webgpu'
          : this.webgpuFailed && detected.device === 'webgpu'
            ? 'wasm'
            : detected.device
    const spec = config.dtype ?? 'q8'
    const dtype = !perBackend(spec)
      ? spec
      : device === 'webgpu'
        ? detected.f16
          ? spec.webgpu
          : spec.webgpuF32
        : spec.wasm
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

  /**
   * Run `task` (a load or an inference) and, when it fails on WebGPU for a
   * model whose device was chosen automatically, unload that model, switch
   * this runtime to WASM, report `fallback` and run it once more. Errors on
   * WASM, or with `device: 'webgpu'` asked for, are thrown as they are.
   */
  private async withFallback<T>(config: LoadConfig, task: () => Promise<T>): Promise<T> {
    const { device } = await this.resolve(config)
    try {
      return await task()
    } catch (error) {
      if (device !== 'webgpu' || config.device === 'webgpu' || !isWebGpuFailure(error)) throw error
      this.webgpuFailed = true
      await this.unload(config.model, device)
      const cause = (error as { cause?: { message?: string } })?.cause?.message ?? (error as Error)?.message ?? String(error)
      this.onProgress?.({ model: config.model, status: 'fallback', device: 'wasm', error: String(cause) })
      return task()
    }
  }

  /** Drop loaded models (all of `model`'s, or only those on `device`). */
  private async unload(model: string, device?: string): Promise<void> {
    for (const [key, entry] of [...this.loaded]) {
      if (!key.includes(`:${model}:`) || (device && !key.endsWith(`:${device}`))) continue
      this.loaded.delete(key)
      const value = await entry.catch(() => undefined)
      for (const part of [value?.model, value?.extractor, value?.speech]) {
        try {
          await part?.dispose?.()
        } catch {
          /* a broken GPU session may fail to release */
        }
      }
    }
  }

  private async embedder(config: EmbedConfig) {
    const tjs = await this.tjs()
    const opts = { ...(await this.resolve(config)), progress_callback: this.progress(config.model) }
    return this.cached(`embed:${config.model}:${config.method}:${dtypeKey(opts.dtype)}:${opts.device}`, async () => {
      if (config.method === 'pipeline') {
        const extractor = await tjs.pipeline('feature-extraction', config.model, opts as never)
        return { kind: 'pipeline' as const, extractor }
      }
      const modelConfig = await this.modelConfig(config, opts.progress_callback)
      const [tokenizer, model] = await Promise.all([
        tjs.AutoTokenizer.from_pretrained(config.model, { progress_callback: opts.progress_callback }),
        tjs.AutoModel.from_pretrained(config.model, (modelConfig ? { ...opts, config: modelConfig } : opts) as never),
      ])
      return { kind: 'sentence' as const, tokenizer, model, device: opts.device }
    })
  }

  /**
   * The config a text-only load uses: the repo's, without its vision and audio
   * encoders (else they would be loaded and downloaded too). Undefined otherwise.
   */
  private async modelConfig(config: LoadConfig, progress_callback?: unknown): Promise<Record<string, unknown> | undefined> {
    if (!(config as EmbedConfig).textOnly) return undefined
    const tjs = await this.tjs()
    // A cached config.json is read from the cache (no network).
    const loaded = (await tjs.AutoConfig.from_pretrained(config.model, { progress_callback } as never)) as unknown as Record<string, unknown>
    loaded.vision_config = null
    loaded.audio_config = null
    return loaded
  }

  /**
   * Preload a model. Serialized with inference: creating several ONNX Runtime
   * WebGPU sessions concurrently in one worker can stall initialization.
   */
  load(kind: LoadKind, config: EmbedConfig | RerankConfig | GenerateConfig | TranscribeConfig): Promise<void> {
    return this.exclusive(() =>
      this.withFallback(config, async () => {
        if (kind === 'embed') await this.embedder(config as EmbedConfig)
        else if (kind === 'rerank') await this.reranker(config)
        else if (kind === 'transcribe') await this.speechModel(config)
        else await this.generator(config)
      }),
    )
  }

  /**
   * Whether every file this model needs (for the resolved dtype/device) is in
   * the cache. Transformers.js also lists optional files that many repos don't
   * have (e.g. generation_config.json); those are ignored here.
   */
  async isCached(config: LoadConfig): Promise<boolean> {
    try {
      // Nothing is cached without its config.json: say so without asking the
      // model host (listing the files probes each one over the network).
      if (!(await this.configCached(config.model))) return false
      const files = await this.cachedFiles(config)
      return files.length > 0 && files.every((f) => f.cached || OPTIONAL_FILES.has(f.file))
    } catch {
      return false
    }
  }

  /**
   * Whether the model's config.json is in the cache (no network). In browsers
   * Transformers.js disables local models, so `local_files_only` always throws
   * there: look the file up in its Cache Storage, under its remote URL.
   */
  private async configCached(model: string): Promise<boolean> {
    const tjs = await this.tjs()
    const env = tjs.env as unknown as TjsCacheEnv
    try {
      if (!env.allowLocalModels && env.useBrowserCache && typeof caches !== 'undefined') {
        return !!(await (await caches.open(env.cacheKey ?? 'transformers-cache')).match(cachedConfigUrl(env, model)))
      }
      await tjs.AutoConfig.from_pretrained(model, { local_files_only: true } as never)
      return true
    } catch {
      return false
    }
  }

  /** Per-file cache state, for diagnostics and download UIs. */
  async cachedFiles(config: LoadConfig): Promise<{ file: string; cached: boolean }[]> {
    const tjs = await this.tjs()
    const { device, dtype } = await this.resolve(config)
    const modelConfig = await this.modelConfig(config)
    const result = await tjs.ModelRegistry.is_cached_files(config.model, { dtype: dtype as never, ...(device ? { device } : {}), ...(modelConfig ? { config: modelConfig as never } : {}) })
    return result.files
  }

  /** Remove the model's cached files and unload it. */
  async clearCache(config: LoadConfig): Promise<void> {
    const tjs = await this.tjs()
    const { device, dtype } = await this.resolve(config)
    await this.unload(config.model)
    // Text-only models: their files, by the text-only config (else every encoder's, which is harmless too).
    const modelConfig = (await this.configCached(config.model)) ? await this.modelConfig(config).catch(() => undefined) : undefined
    await tjs.ModelRegistry.clear_cache(config.model, { dtype: dtype as never, ...(device ? { device } : {}), ...(modelConfig ? { config: modelConfig as never } : {}) })
  }

  embed(config: EmbedConfig, texts: string[]): Promise<number[][]> {
    return this.exclusive(() => this.withFallback(config, async () => {
      const loaded = await this.embedder(config)
      let tensor: any
      if (loaded.kind === 'pipeline') {
        tensor = await loaded.extractor(texts, {
          pooling: config.pooling ?? 'mean',
          normalize: true,
        } as never)
      } else {
        const budget = loaded.device === 'webgpu' ? config.webgpuMaxBatchTokens : undefined
        const max_length = Math.min(config.maxTokens ?? 2048, budget ?? 2048)
        const out: number[][] = []
        for (const group of budget ? groupByTokens(texts, (t) => Math.min(loaded.tokenizer.encode(t).length, max_length), budget) : [texts]) {
          const inputs = await loaded.tokenizer(group, { padding: true, truncation: true, max_length })
          const output = await loaded.model(inputs)
          out.push(...finish(output.sentence_embedding, config.dimensions))
        }
        return out
      }
      return finish(tensor, config.dimensions)
    }))
  }

  private async reranker(config: RerankConfig) {
    const tjs = await this.tjs()
    const opts = { ...(await this.resolve(config)), progress_callback: this.progress(config.model) }
    return this.cached(`rerank:${config.model}:${dtypeKey(opts.dtype)}:${opts.device}`, async () => {
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
    return this.exclusive(() => this.withFallback(config, async () => {
      const { tokenizer, model } = await this.reranker(config)
      const inputs = await tokenizer(new Array(documents.length).fill(query), {
        text_pair: documents,
        padding: true,
        truncation: true,
        max_length: 512,
      })
      const { logits } = await model(inputs)
      return (logits.sigmoid().tolist() as number[][]).map((row) => row[0]!)
    }))
  }

  private async generator(config: GenerateConfig) {
    const tjs = await this.tjs()
    const opts = { ...(await this.resolve(config)), progress_callback: this.progress(config.model) }
    return this.cached(`generate:${config.model}:${dtypeKey(opts.dtype)}:${opts.device}`, async () => {
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

  private speechModel(config: TranscribeConfig): Promise<SpeechModel> {
    return (async () => {
      const opts = await this.resolve(config)
      const entry = await this.cached(`transcribe:${config.model}:${dtypeKey(opts.dtype)}:${opts.device}`, async () => ({
        speech: await this.createSpeechModel(config),
      }))
      return entry.speech
    })()
  }

  /**
   * Load a Whisper model. Override to substitute a fake in tests; the result
   * is cached per model, dtype and device like every other model here.
   */
  protected async createSpeechModel(config: TranscribeConfig): Promise<SpeechModel> {
    const tjs = await this.tjs()
    const opts = { ...(await this.resolve(config)), progress_callback: this.progress(config.model) }
    const asr = (await tjs.pipeline('automatic-speech-recognition', config.model, opts as never)) as any
    const generation = (asr.model.generation_config ?? {}) as {
      is_multilingual?: boolean
      lang_to_id?: Record<string, number>
      decoder_start_token_id?: number
    }
    const multilingual = !!generation.is_multilingual && !!generation.lang_to_id
    return {
      multilingual,
      async recognize(samples, { language, task }) {
        return (await asr(samples, {
          return_timestamps: true,
          chunk_length_s: 0,
          ...(multilingual ? { task: task ?? 'transcribe', ...(language ? { language } : {}) } : {}),
        })) as AsrOutput
      },
      // Transformers.js doesn't detect the language (it assumes English), so
      // ask the decoder which language token follows <|startoftranscript|>.
      async detectLanguage(samples) {
        if (!multilingual || generation.decoder_start_token_id === undefined) return undefined
        const { input_features } = await asr.processor(samples)
        const start = new tjs.Tensor('int64', BigInt64Array.from([BigInt(generation.decoder_start_token_id)]), [1, 1])
        const { logits } = await asr.model({ input_features, decoder_input_ids: start })
        const vocab = logits.dims.at(-1) as number
        const data = logits.data as Float32Array
        const row = data.length - vocab
        let best: string | undefined
        let bestScore = -Infinity
        for (const [token, id] of Object.entries(generation.lang_to_id!)) {
          const score = data[row + id]!
          if (score > bestScore) {
            bestScore = score
            best = token
          }
        }
        return best?.replace(/^<\|(.+)\|>$/, '$1')
      },
      dispose: () => asr.dispose(),
    }
  }

  /**
   * Transcribe mono 16 kHz audio in overlapping windows. Each window is one
   * exclusive inference, so other models' work can interleave, segments and
   * progress stream out per window, and `signal` stops between windows.
   * Without a `language`, it is detected once from the first window with speech.
   */
  transcribe(
    config: TranscribeConfig,
    audio: Float32Array,
    request: TranscribeRequest,
    onEvent: (event: TranscribeEvent) => void,
    signal?: AbortSignal,
  ): Promise<Transcript> {
    let language = request.language
    return transcribeChunked(
      audio,
      (samples, window) =>
        this.exclusive(() =>
          this.withFallback(config, async () => {
            signal?.throwIfAborted()
            const speech = await this.speechModel(config)
            if (!language && speech.multilingual) language = await speech.detectLanguage?.(samples)
            const output = await speech.recognize(samples, {
              ...(language ? { language } : {}),
              ...(request.task ? { task: request.task } : {}),
            })
            return toSegments(output, window.offset, window.duration)
          }),
        ),
      {
        ...(request.chunkSeconds !== undefined ? { chunkSeconds: request.chunkSeconds } : {}),
        ...(request.strideSeconds !== undefined ? { strideSeconds: request.strideSeconds } : {}),
        ...(signal ? { signal } : {}),
        onSegment: (segment) => onEvent({ type: 'segment', segment }),
        onProgress: (fraction) => onEvent({ type: 'progress', fraction }),
      },
    ).then((transcript) => (language ? { ...transcript, language } : transcript))
  }

  async dispose(): Promise<void> {
    const entries = [...this.loaded.values()]
    this.loaded.clear()
    for (const entry of entries) {
      const value = await entry.catch(() => undefined)
      await value?.model?.dispose?.()
      await value?.extractor?.dispose?.()
      await value?.speech?.dispose?.()
    }
  }
}

export type LoadKind = 'embed' | 'rerank' | 'generate' | 'transcribe'

/** A loaded speech recognizer (Whisper through Transformers.js, or a test fake). */
export interface SpeechModel {
  /** Whether `language`/`task` apply (false for English-only checkpoints). */
  multilingual: boolean
  /** Recognize up to 30 s of mono 16 kHz audio, with timestamps relative to `samples`. */
  recognize(samples: Float32Array, options: { language?: string; task?: 'transcribe' | 'translate' }): Promise<AsrOutput>
  /** ISO 639-1 code of the speech. */
  detectLanguage?(samples: Float32Array): Promise<string | undefined>
  dispose?(): Promise<void>
}
