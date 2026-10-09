import type { EmbedTask } from '../types.js'
import type { DeviceProfile } from './device.js'

// ---------------------------------------------------------------------------
// LLMs that run in the browser
// ---------------------------------------------------------------------------

export interface BrowserLLMPreset {
  id: string
  label: string
  runtime: 'webllm' | 'transformers'
  /** WebLLM model id without the quantization suffix, or a Hugging Face ONNX repo. */
  model: string
  params: string
  /** GPU memory at a 4k context for the f16 build (WebLLM's own figure), MB. */
  vramMB: number
  /** Extra MB per additional 4k tokens of context (f16 KV cache). Estimate. */
  kvMBPer4k: number
  /** Approximate download, MB. */
  downloadMB: number
  /** Context we request by default when memory allows. */
  contextWindow: number
  /** Relative agentic/tool-use quality, used to rank. */
  quality: number
  thinking: 'hybrid' | 'none'
  /** Needs WebGPU adapters with >1 GB storage buffers. */
  needsLargeBuffers?: boolean
  notes?: string
}

export const BROWSER_LLMS: BrowserLLMPreset[] = [
  // WebLLM: fastest in-browser inference (TVM-compiled WebGPU kernels). Primary runtime.
  { id: 'qwen3-8b', label: 'Qwen3 8B', runtime: 'webllm', model: 'Qwen3-8B', params: '8B', vramMB: 5696, kvMBPer4k: 604, downloadMB: 4600, contextWindow: 16384, quality: 90, thinking: 'hybrid', needsLargeBuffers: true },
  { id: 'qwen3-4b', label: 'Qwen3 4B', runtime: 'webllm', model: 'Qwen3-4B', params: '4B', vramMB: 3432, kvMBPer4k: 604, downloadMB: 2300, contextWindow: 16384, quality: 80, thinking: 'hybrid' },
  { id: 'phi-4-mini', label: 'Phi-4 mini', runtime: 'webllm', model: 'Phi-4-mini-instruct', params: '3.8B', vramMB: 3438, kvMBPer4k: 537, downloadMB: 2200, contextWindow: 8192, quality: 66, thinking: 'none', needsLargeBuffers: true },
  { id: 'hermes-3-3b', label: 'Hermes 3 (Llama 3.2 3B)', runtime: 'webllm', model: 'Hermes-3-Llama-3.2-3B', params: '3B', vramMB: 2264, kvMBPer4k: 470, downloadMB: 1800, contextWindow: 8192, quality: 64, thinking: 'none', notes: 'Trained on the <tool_call> format.' },
  { id: 'qwen3-1.7b', label: 'Qwen3 1.7B', runtime: 'webllm', model: 'Qwen3-1.7B', params: '1.7B', vramMB: 2037, kvMBPer4k: 470, downloadMB: 1100, contextWindow: 16384, quality: 62, thinking: 'hybrid' },
  { id: 'llama-3.2-3b', label: 'Llama 3.2 3B', runtime: 'webllm', model: 'Llama-3.2-3B-Instruct', params: '3B', vramMB: 2264, kvMBPer4k: 470, downloadMB: 1800, contextWindow: 8192, quality: 58, thinking: 'none' },
  { id: 'qwen3.5-9b', label: 'Qwen3.5 9B (experimental)', runtime: 'webllm', model: 'Qwen3.5-9B', params: '9B', vramMB: 6433, kvMBPer4k: 200, downloadMB: 5100, contextWindow: 16384, quality: 50, thinking: 'hybrid', needsLargeBuffers: true, notes: 'WebLLM build limits history (max_history_size 1); may drop earlier turns. KV estimate.' },
  { id: 'qwen3.5-4b', label: 'Qwen3.5 4B (experimental)', runtime: 'webllm', model: 'Qwen3.5-4B', params: '4B', vramMB: 3868, kvMBPer4k: 160, downloadMB: 2600, contextWindow: 16384, quality: 55, thinking: 'hybrid', needsLargeBuffers: true, notes: 'WebLLM build limits history (max_history_size 1); may drop earlier turns.' },
  { id: 'qwen3.5-2b', label: 'Qwen3.5 2B (experimental)', runtime: 'webllm', model: 'Qwen3.5-2B', params: '2B', vramMB: 2245, kvMBPer4k: 120, downloadMB: 1400, contextWindow: 16384, quality: 45, thinking: 'hybrid', needsLargeBuffers: true, notes: 'WebLLM build limits history (max_history_size 1); may drop earlier turns.' },
  { id: 'qwen3-0.6b', label: 'Qwen3 0.6B', runtime: 'webllm', model: 'Qwen3-0.6B', params: '0.6B', vramMB: 1403, kvMBPer4k: 470, downloadMB: 500, contextWindow: 8192, quality: 40, thinking: 'hybrid' },
  { id: 'gemma-3-1b', label: 'Gemma 3 1B', runtime: 'webllm', model: 'gemma3-1b-it', params: '1B', vramMB: 711, kvMBPer4k: 110, downloadMB: 700, contextWindow: 8192, quality: 20, thinking: 'none', notes: 'Not trained for tool calling.' },

  // Transformers.js (ONNX Runtime Web): broader model coverage, also runs on WASM without WebGPU.
  { id: 'tjs-qwen3-1.7b', label: 'Qwen3 1.7B (Transformers.js)', runtime: 'transformers', model: 'onnx-community/Qwen3-1.7B-ONNX', params: '1.7B', vramMB: 2000, kvMBPer4k: 470, downloadMB: 1426, contextWindow: 8192, quality: 58, thinking: 'hybrid' },
  { id: 'tjs-granite-4-1b', label: 'Granite 4.0 1B (Transformers.js)', runtime: 'transformers', model: 'onnx-community/granite-4.0-1b-ONNX-web', params: '1B', vramMB: 1700, kvMBPer4k: 300, downloadMB: 1247, contextWindow: 8192, quality: 50, thinking: 'none', notes: 'Trained for tool calling.' },
  { id: 'tjs-lfm2-1.2b', label: 'LFM2 1.2B (Transformers.js)', runtime: 'transformers', model: 'onnx-community/LFM2-1.2B-ONNX', params: '1.2B', vramMB: 1200, kvMBPer4k: 100, downloadMB: 760, contextWindow: 8192, quality: 44, thinking: 'none' },
  { id: 'tjs-qwen3-0.6b', label: 'Qwen3 0.6B (Transformers.js)', runtime: 'transformers', model: 'onnx-community/Qwen3-0.6B-ONNX', params: '0.6B', vramMB: 900, kvMBPer4k: 470, downloadMB: 570, contextWindow: 8192, quality: 38, thinking: 'hybrid' },
  { id: 'tjs-qwen3.5-0.8b', label: 'Qwen3.5 0.8B (Transformers.js)', runtime: 'transformers', model: 'onnx-community/Qwen3.5-0.8B-Text-ONNX', params: '0.8B', vramMB: 900, kvMBPer4k: 60, downloadMB: 470, contextWindow: 8192, quality: 36, thinking: 'hybrid' },
]

export interface LLMChoice {
  preset: BrowserLLMPreset
  /** Concrete WebLLM id (with q4f16/q4f32 suffix) or HF repo. */
  modelId: string
  /** ONNX dtype for Transformers.js presets. */
  dtype?: 'q4f16' | 'q4'
  contextWindow: number
  estimatedMB: number
}

/** Resolve a preset to concrete build + context window for this device. */
export function resolveLLM(preset: BrowserLLMPreset, device: DeviceProfile): LLMChoice {
  const f32 = !device.shaderF16
  // f32 builds use roughly 25% more memory for weights and double the KV cache.
  const base = f32 ? preset.vramMB * 1.25 : preset.vramMB
  const kv = f32 ? preset.kvMBPer4k * 2 : preset.kvMBPer4k
  let contextWindow = preset.contextWindow
  while (contextWindow > 4096 && base + kv * (contextWindow / 4096 - 1) > device.gpuBudgetMB) contextWindow /= 2
  const estimatedMB = Math.round(base + kv * (contextWindow / 4096 - 1))
  if (preset.runtime === 'webllm') {
    return { preset, modelId: `${preset.model}-${f32 ? 'q4f32_1' : 'q4f16_1'}-MLC`, contextWindow, estimatedMB }
  }
  return { preset, modelId: preset.model, dtype: f32 ? 'q4' : 'q4f16', contextWindow, estimatedMB }
}

/** Below this, tool definitions + schema + a few turns no longer fit. */
export const MIN_AGENT_CONTEXT = 8192

export interface RankOptions {
  runtime?: BrowserLLMPreset['runtime']
  /** Skip models whose first download exceeds this. */
  maxDownloadMB?: number
}

/** Presets that fit this device, best first. */
export function rankLLMs(device: DeviceProfile, options: RankOptions = {}): LLMChoice[] {
  const { runtime, maxDownloadMB = Infinity } = options
  return BROWSER_LLMS.filter((p) => !runtime || p.runtime === runtime)
    .filter((p) => p.downloadMB <= maxDownloadMB)
    .filter((p) => device.webgpu || p.runtime === 'transformers')
    .filter((p) => !p.needsLargeBuffers || device.maxStorageBufferMB >= 1024)
    .map((p) => resolveLLM(p, device))
    .filter((c) => !device.webgpu || c.estimatedMB <= device.gpuBudgetMB)
    .sort((a, b) => agentScore(b) - agentScore(a))
}

/** Quality, discounted when the window is too small for comfortable tool use. */
function agentScore(choice: LLMChoice): number {
  return choice.preset.quality - (choice.contextWindow < MIN_AGENT_CONTEXT ? 15 : 0)
}

/** Default first-visit download ceiling for automatic model choice. */
export const DEFAULT_MAX_DOWNLOAD_MB = 2500

/**
 * The best browser LLM for this device: WebLLM when WebGPU is available,
 * otherwise a small Transformers.js model on WASM (slow, but works everywhere).
 * Capped at `maxDownloadMB` (default 2.5 GB) so first visits stay reasonable.
 */
export function recommendLLM(device: DeviceProfile, options: { maxDownloadMB?: number } = {}): LLMChoice | undefined {
  if (!device.webgpu) {
    const tiny = BROWSER_LLMS.find((p) => p.id === 'tjs-qwen3-0.6b')!
    return { ...resolveLLM(tiny, device), dtype: 'q4', contextWindow: 4096 }
  }
  const maxDownloadMB = options.maxDownloadMB ?? DEFAULT_MAX_DOWNLOAD_MB
  return rankLLMs(device, { runtime: 'webllm', maxDownloadMB })[0] ?? rankLLMs(device, { maxDownloadMB })[0]
}

export function findLLM(id: string): BrowserLLMPreset | undefined {
  return BROWSER_LLMS.find((p) => p.id === id)
}

// ---------------------------------------------------------------------------
// Embeddings (Transformers.js)
// ---------------------------------------------------------------------------

export interface EmbeddingPreset {
  id: string
  label: string
  model: string
  dimensions: number
  /**
   * `sentence_embedding`: the ONNX graph includes pooling (+ dense layers), call AutoModel.
   * `pipeline`: feature-extraction pipeline with the given pooling.
   */
  method: 'sentence_embedding' | 'pipeline'
  pooling?: 'mean' | 'cls' | 'last_token'
  /** Prefix for search queries (`task: 'search'`). */
  queryPrefix?: string
  /**
   * Query prefixes for other tasks the model was trained with (EmbeddingGemma's
   * `task: … | query: ` prompts). A task without one uses `queryPrefix`.
   */
  taskPrefixes?: Partial<Record<EmbedTask, string>>
  /** Document template; `{title}` and `{text}` are substituted. */
  documentTemplate?: string
  /**
   * A multimodal repo whose text model loads on its own (vision and audio
   * encoders left out of the config, so they are never downloaded).
   */
  textOnly?: boolean
  /**
   * Most tokens (batch × padded length) per WebGPU run. Some ONNX Runtime
   * WebGPU kernels exceed a dispatch limit beyond this; batches are split.
   */
  webgpuMaxBatchTokens?: number
  /** Supported Matryoshka output sizes (truncate + renormalize). */
  matryoshka?: number[]
  /** dtype per backend. `webgpuF32` is used when the adapter lacks shader-f16. */
  dtype: { webgpu: string; webgpuF32: string; wasm: string }
  downloadMB: number
  maxTokens: number
  languages: 'en' | 'multilingual'
  license: string
  /**
   * Best search mode for this model, measured on the evals retrieval benchmark
   * (91 labeled queries). Strong multilingual embedders do better on pure
   * vector search; smaller ones gain from keyword fusion.
   */
  searchMode: 'vector' | 'hybrid'
  /**
   * Cosine similarity below which queries and passages are unrelated for
   * this model (see `Embedder.relevanceFloor`). Measured on 95 on-topic and 18
   * conversational messages over the eval corpus.
   */
  relevanceFloor: number
}

/** EmbeddingGemma's query prompts per task (model cards of v1 and v2). */
const GEMMA_TASKS: Partial<Record<EmbedTask, string>> = {
  'question-answering': 'task: question answering | query: ',
  'fact-checking': 'task: fact checking | query: ',
  'code-retrieval': 'task: code retrieval | query: ',
  classification: 'task: classification | query: ',
  clustering: 'task: clustering | query: ',
  similarity: 'task: sentence similarity | query: ',
}

export const EMBEDDING_PRESETS: EmbeddingPreset[] = [
  {
    id: 'embeddinggemma-2',
    label: 'EmbeddingGemma 2 (best quality, 100+ languages)',
    model: 'onnx-community/embeddinggemma-2-ONNX',
    dimensions: 768,
    method: 'sentence_embedding',
    queryPrefix: 'task: search result | query: ',
    taskPrefixes: GEMMA_TASKS,
    documentTemplate: 'title: {title} | text: {text}',
    matryoshka: [768, 512, 256, 128],
    // The text model only (270M): its vision and audio encoders stay on the host.
    textOnly: true,
    // q4 on GPUs (fp32 activations; the card warns fp16 activations overflow). ONNX Runtime Web's
    // WASM backend has no GatherBlockQuantized kernel, so no quantized build runs there: the fp16
    // export does (0.9998 cosine to fp32 on its card; recall@3 1.000 on Vellum's evals), but at
    // 542 MB and ~460 ms per chunk, so devices without WebGPU get Granite (recommendEmbedding)
    // and this is only the fallback after WebGPU fails.
    dtype: { webgpu: 'q4', webgpuF32: 'q4', wasm: 'fp16' },
    webgpuMaxBatchTokens: 2048,
    downloadMB: 207, // model_q4 174 MB + tokenizer 32 MB
    maxTokens: 8192,
    languages: 'multilingual',
    license: 'apache-2.0',
    // q4 on the 91 queries: recall@3 0.984 vector vs 0.951 hybrid; multilingual 10/10 vs 8/10.
    // (v1 q4: 0.995, MRR 0.940 vs 0.903 here; on Vellum's evals v2 leads: recall@3 1.000 vs 0.972.)
    searchMode: 'vector',
    // Its cosines sit higher than v1's: keeps 89/91 on-topic, silences 5/18 conversational.
    relevanceFloor: 0.66,
  },
  {
    id: 'embeddinggemma',
    label: 'EmbeddingGemma 300M (previous version, 100+ languages)',
    model: 'onnx-community/embeddinggemma-300m-ONNX',
    dimensions: 768,
    method: 'sentence_embedding',
    queryPrefix: 'task: search result | query: ',
    taskPrefixes: GEMMA_TASKS,
    documentTemplate: 'title: {title} | text: {text}',
    matryoshka: [768, 512, 256, 128],
    // No fp16: EmbeddingGemma activations overflow in fp16.
    dtype: { webgpu: 'q4', webgpuF32: 'q4', wasm: 'q8' },
    downloadMB: 197,
    maxTokens: 2048,
    languages: 'multilingual',
    license: 'gemma',
    searchMode: 'vector', // recall@3 0.995 vs 0.951 hybrid; multilingual 10/10 vs 8/10
    relevanceFloor: 0.35, // keeps 91/95 on-topic, silences 10/18 conversational
  },
  {
    id: 'granite-small-r2',
    label: 'Granite Embedding Small R2 (fast, English)',
    model: 'onnx-community/granite-embedding-small-english-r2-ONNX',
    dimensions: 384,
    method: 'sentence_embedding',
    dtype: { webgpu: 'fp16', webgpuF32: 'fp32', wasm: 'q8' },
    downloadMB: 97,
    maxTokens: 8192,
    languages: 'en',
    license: 'apache-2.0',
    searchMode: 'hybrid', // MRR 0.822 hybrid vs 0.791 vector
    relevanceFloor: 0.78, // 92/95, 8/18
  },
  {
    id: 'granite-multilingual-r2',
    label: 'Granite Embedding 97M Multilingual R2 (fast, 200+ languages)',
    model: 'onnx-community/granite-embedding-97m-multilingual-r2-ONNX',
    dimensions: 384,
    method: 'pipeline',
    pooling: 'cls',
    dtype: { webgpu: 'fp16', webgpuF32: 'fp32', wasm: 'q8' },
    downloadMB: 195,
    maxTokens: 32768,
    languages: 'multilingual',
    license: 'apache-2.0',
    searchMode: 'hybrid', // MRR 0.852 hybrid vs 0.795 vector
    relevanceFloor: 0.76, // 94/95, 6/18
  },
  {
    id: 'qwen3-embedding-0.6b',
    label: 'Qwen3 Embedding 0.6B (highest quality, large)',
    model: 'onnx-community/Qwen3-Embedding-0.6B-ONNX',
    dimensions: 1024,
    method: 'pipeline',
    pooling: 'last_token',
    queryPrefix: 'Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery:',
    matryoshka: [1024, 768, 512, 256, 128],
    dtype: { webgpu: 'q4f16', webgpuF32: 'q4', wasm: 'q8' },
    downloadMB: 567,
    maxTokens: 32768,
    languages: 'multilingual',
    license: 'apache-2.0',
    searchMode: 'vector', // not yet benchmarked; strong embedder like EmbeddingGemma
    relevanceFloor: 0.4, // 92/95, 8/18
  },
  {
    id: 'gte-small',
    label: 'GTE Small (legacy, database.build default)',
    model: 'Supabase/gte-small',
    dimensions: 384,
    method: 'pipeline',
    pooling: 'mean',
    dtype: { webgpu: 'fp32', webgpuF32: 'fp32', wasm: 'q8' },
    downloadMB: 34,
    maxTokens: 512,
    languages: 'en',
    license: 'mit',
    searchMode: 'vector', // MRR 0.886 vector vs 0.861 hybrid
    relevanceFloor: 0.8, // 94/95, 6/18
  },
]

export function findEmbedding(id: string): EmbeddingPreset | undefined {
  return EMBEDDING_PRESETS.find((p) => p.id === id)
}

/** EmbeddingGemma 2 on GPUs, Granite (small and quick on CPU) otherwise. */
export function recommendEmbedding(device: DeviceProfile): EmbeddingPreset {
  return findEmbedding(device.webgpu && !device.mobile ? 'embeddinggemma-2' : 'granite-multilingual-r2')!
}

// ---------------------------------------------------------------------------
// Rerankers (cross-encoders, Transformers.js)
// ---------------------------------------------------------------------------

export interface RerankerPreset {
  id: string
  label: string
  model: string
  dtype: { webgpu: string; webgpuF32: string; wasm: string }
  downloadMB: number
  languages: 'en' | 'multilingual'
}

export const RERANKER_PRESETS: RerankerPreset[] = [
  {
    id: 'mxbai-rerank-xsmall',
    label: 'mxbai-rerank xsmall (balanced)',
    model: 'mixedbread-ai/mxbai-rerank-xsmall-v1',
    dtype: { webgpu: 'fp32', webgpuF32: 'fp32', wasm: 'q8' },
    downloadMB: 87,
    languages: 'en',
  },
  {
    id: 'ms-marco-minilm',
    label: 'MS MARCO MiniLM L6 (fastest)',
    model: 'Xenova/ms-marco-MiniLM-L-6-v2',
    dtype: { webgpu: 'fp16', webgpuF32: 'fp32', wasm: 'q8' },
    downloadMB: 23,
    languages: 'en',
  },
  {
    id: 'bge-reranker-v2-m3',
    label: 'BGE Reranker v2 M3 (multilingual, large)',
    model: 'onnx-community/bge-reranker-v2-m3-ONNX',
    dtype: { webgpu: 'q4f16', webgpuF32: 'q4', wasm: 'q8' },
    downloadMB: 571,
    languages: 'multilingual',
  },
]

export function findReranker(id: string): RerankerPreset | undefined {
  return RERANKER_PRESETS.find((p) => p.id === id)
}

/**
 * No reranker by default. On the evals retrieval benchmark, EmbeddingGemma (v1)
 * alone reached recall@3 0.995 / MRR 0.932; mxbai (English-only) lowered it to
 * 0.945 / 0.908 and halved multilingual recall; bge-reranker-v2-m3 raised MRR
 * to 0.973 at ~60x the latency and a 571 MB download. Opt in with
 * `reranker: 'bge-reranker-v2-m3'` when ranking precision matters most.
 */
export function recommendReranker(_device: DeviceProfile, _embedding?: EmbeddingPreset): RerankerPreset | undefined {
  return undefined
}

export function pickDtype<T extends string | Record<string, string> = string>(
  dtype: { webgpu: T; webgpuF32: T; wasm: T },
  device: Pick<DeviceProfile, 'webgpu' | 'shaderF16'>,
): T {
  if (!device.webgpu) return dtype.wasm
  return device.shaderF16 ? dtype.webgpu : dtype.webgpuF32
}

// ---------------------------------------------------------------------------
// Speech to text (Whisper, Transformers.js)
// ---------------------------------------------------------------------------

/** One dtype for the whole model, or one per ONNX file (`encoder_model`, `decoder_model_merged`). */
export type ModelDtype = string | Record<string, string>

export interface TranscriberPreset {
  id: string
  label: string
  /** Hugging Face ONNX repo. */
  model: string
  params: string
  /**
   * Approximate first download in bytes on WebGPU, the larger of the two
   * builds (weights + config + tokenizer, from the repo's file sizes).
   */
  sizeBytes: number
  /** Approximate first download in bytes on WASM (no WebGPU). */
  wasmSizeBytes: number
  /** `sizeBytes` in MB, like the other presets' `downloadMB`. */
  downloadMB: number
  languages: 'multilingual' | 'en'
  /**
   * Per backend: WebGPU with shader-f16, WebGPU without, WASM. The encoder
   * stays fp32 (quantized and fp16 Whisper encoders lose accuracy); the
   * decoder is q4 on WebGPU and q8 on WASM, as in Transformers.js' Whisper demos.
   */
  dtype: { webgpu: ModelDtype; webgpuF32: ModelDtype; wasm: ModelDtype }
}

const WHISPER_GPU = { encoder_model: 'fp32', decoder_model_merged: 'q4' }
const WHISPER_WASM = { encoder_model: 'fp32', decoder_model_merged: 'q8' }

export const TRANSCRIBER_PRESETS: TranscriberPreset[] = [
  {
    id: 'whisper-tiny',
    label: 'Whisper Tiny (fast, 99 languages)',
    model: 'onnx-community/whisper-tiny',
    params: '39M',
    sizeBytes: 122_388_197,
    wasmSizeBytes: 66_393_736,
    downloadMB: 122,
    languages: 'multilingual',
    dtype: { webgpu: WHISPER_GPU, webgpuF32: WHISPER_GPU, wasm: WHISPER_WASM },
  },
  {
    id: 'whisper-base',
    label: 'Whisper Base (more accurate, 99 languages)',
    model: 'onnx-community/whisper-base',
    params: '74M',
    sizeBytes: 208_840_059,
    wasmSizeBytes: 138_930_955,
    downloadMB: 209,
    languages: 'multilingual',
    dtype: { webgpu: WHISPER_GPU, webgpuF32: WHISPER_GPU, wasm: WHISPER_WASM },
  },
]

export function findTranscriber(id: string): TranscriberPreset | undefined {
  return TRANSCRIBER_PRESETS.find((p) => p.id === id)
}

/** Whisper Base on desktop GPUs, Whisper Tiny on phones and CPU-only devices. */
export function recommendTranscriber(device: Pick<DeviceProfile, 'webgpu' | 'mobile'>): TranscriberPreset {
  return findTranscriber(device.webgpu && !device.mobile ? 'whisper-base' : 'whisper-tiny')!
}
