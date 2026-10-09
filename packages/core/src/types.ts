import type { PGliteInterface } from '@electric-sql/pglite'
import type { TranscriberPreset } from './web/catalog.js'

/**
 * Where a component processes data: on this `device` (in the browser or a
 * loopback server), on the `local-network`, or on a `remote` service.
 */
export type Locality = 'device' | 'local-network' | 'remote'

/** Any PGlite instance (in-thread `PGlite` or `PGliteWorker`). */
export type Db = PGliteInterface

export type JSONSchema = Record<string, unknown>

// ---------------------------------------------------------------------------
// Conversation
// ---------------------------------------------------------------------------

export interface ToolCall {
  id: string
  name: string
  input: unknown
}

export interface UserMessage {
  role: 'user'
  content: string
  /** Added by the agent loop (e.g. a follow-through reminder), not typed by the user. UIs can hide it. */
  synthetic?: boolean
}

export interface AssistantMessage {
  role: 'assistant'
  content: string
  toolCalls?: ToolCall[]
  reasoning?: string
  /** Opaque provider payload (e.g. signed thinking blocks) replayed verbatim to the same provider. */
  providerData?: { provider: string; data: unknown }
  /**
   * The passages this answer could cite as [n], set on a run's final message.
   * For the UI and thread history; never sent to the model.
   */
  citations?: Citation[]
}

/** A passage the model was shown, numbered so it can cite it inline as `[n]`. */
export interface Citation {
  /** The number the model uses. Stable for the whole run: auto-retrieval and every search share one sequence. */
  n: number
  documentId: string
  chunkId?: number
  collection?: string
  title?: string | null
  source?: string | null
  content: string
  metadata?: Record<string, unknown>
}

/** A passage to register with `cite()`; the run assigns its number. */
export type CitationInput = Omit<Citation, 'n'>

/**
 * Narrow knowledge retrieval for one run, e.g. the sources a user picked:
 * `thread.send(question, { knowledge: { documentIds: [...] } })`.
 */
export interface KnowledgeScope {
  collection?: string | string[]
  /** Match documents whose metadata contains this object (jsonb `@>`). */
  filter?: Record<string, unknown>
  /** Only these documents. */
  documentIds?: string[]
}

export interface ToolMessage {
  role: 'tool'
  toolCallId: string
  name: string
  content: string
  isError?: boolean
}

export type Message = UserMessage | AssistantMessage | ToolMessage

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export interface ToolSpec {
  name: string
  description: string
  inputSchema: JSONSchema
}

export interface ModelRequest {
  /** Stable instructions. Safe to cache across steps and turns. */
  system: string
  /** Volatile per-step context (e.g. live schema). Adapters place it after the cacheable prefix. */
  context?: string
  messages: Message[]
  tools: ToolSpec[]
  signal?: AbortSignal
}

export interface Usage {
  inputTokens: number
  outputTokens: number
}

/** Per-step timing reported by backends that measure it (WebLLM). */
export interface StepMetrics {
  /** Tokens actually prefilled this step (low when the KV cache was reused). */
  prefillTokens?: number
  /** Size of the full prompt sent this step, in characters (context growth). */
  promptChars?: number
  /** Older reasoning was dropped this step to stay within the window (`reasoningHistory: 'auto'`). */
  compacted?: boolean
  timeToFirstTokenMs?: number
  /** Time spent reading the prompt this step (small when a cached prefix was reused). */
  prefillMs?: number
  prefillTokensPerSec?: number
  decodeTokensPerSec?: number
  /** Time spent compiling the tool-call grammar for this step. */
  grammarInitMs?: number
  /** True when the engine continued from its KV cache instead of re-reading the prompt. */
  kvCacheReused?: boolean
  /** Reasoning hit the thinking budget and the step was answered without thinking. */
  thinkingCutOff?: boolean
}

export type FinishReason = 'stop' | 'tool-calls' | 'length' | 'refusal' | 'error'

export type ModelChunk =
  | { type: 'text'; delta: string }
  | { type: 'reasoning'; delta: string }
  | { type: 'tool-call'; call: ToolCall }
  | {
      type: 'finish'
      reason: FinishReason
      usage?: Usage
      metrics?: StepMetrics
      providerData?: AssistantMessage['providerData']
    }

/** The single contract every model backend implements. */
export interface Model {
  readonly id: string
  /** Where prompts are processed. Undeclared counts as `remote`. */
  readonly locality?: Locality
  /**
   * Context window in tokens, when known. The agent uses it to budget history
   * and tool output so small local models don't overflow.
   */
  readonly contextWindow?: number
  stream(request: ModelRequest): AsyncIterable<ModelChunk>
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

export type EmbedKind = 'query' | 'document'

/**
 * What a query embedding is for. Task-prompted models (EmbeddingGemma) embed
 * the query with a different prompt per task; others ignore it. Documents are
 * always embedded the same way, so one index serves every query task.
 * Symmetric tasks (`classification`, `clustering`, `similarity`) embed both
 * sides as queries.
 */
export type EmbedTask = 'search' | 'question-answering' | 'fact-checking' | 'code-retrieval' | 'classification' | 'clustering' | 'similarity'

export interface EmbedOptions {
  /** Query task (default `search`). Ignored for documents. */
  task?: EmbedTask
}

export interface Embedder {
  /** Stable identifier; a change triggers a reindex requirement. */
  readonly id: string
  /** Where document text is processed. Undeclared counts as `remote`. */
  readonly locality?: Locality
  readonly dimensions: number
  embed(texts: string[], kind: EmbedKind, options?: EmbedOptions): Promise<number[][]>
  /**
   * Cosine similarity below which a query and a passage are almost certainly
   * unrelated. Calibrated per model (cosine scales differ widely); decides
   * when automatic retrieval should stay silent. Default 0.35.
   */
  readonly relevanceFloor?: number
  /** Model-specific document formatting (e.g. EmbeddingGemma's `title: … | text: …`). */
  formatDocument?(text: string, title?: string): string
  /** Start downloading/compiling the model ahead of first use. */
  load?(): Promise<void>
  isCached?(): Promise<boolean>
  clearCache?(): Promise<void>
}

/** A component whose weights are downloaded to, and cached in, the browser. */
export interface Downloadable {
  /** Download (first run) and initialize. */
  load(): Promise<void>
  /** True when every file needed to run is in the browser cache (works offline). */
  isCached(): Promise<boolean>
  /** Delete the cached files. */
  clearCache(): Promise<void>
}

// ---------------------------------------------------------------------------
// Speech to text
// ---------------------------------------------------------------------------

/** A stretch of speech; times in seconds from the start of the audio. */
export interface TranscriptSegment {
  start: number
  end: number
  text: string
}

export interface Transcript {
  text: string
  /** Spoken language: the ISO 639-1 code when detected, else the `language` option as given. */
  language?: string
  segments: TranscriptSegment[]
}

export interface TranscribeOptions {
  /** ISO 639-1 code (`en`) or English name (`english`). Omit to detect it from the first speech. */
  language?: string
  /** `translate` outputs English text. Default `transcribe`. */
  task?: 'transcribe' | 'translate'
  /** Checked between chunks; the promise then rejects with an `AbortError`. */
  signal?: AbortSignal
  /** Each segment as soon as its chunk is done, in order. */
  onSegment?(segment: TranscriptSegment): void
  /** Share of the audio processed, 0..1, after each chunk. */
  onProgress?(fraction: number): void
  /** Window length, seconds. Default 30 (Whisper's maximum). */
  chunkSeconds?: number
  /** Overlap between windows, seconds. Default 5. */
  strideSeconds?: number
}

/** On-device speech recognition. */
export interface Transcriber extends Downloadable {
  readonly id: string
  readonly locality: 'device'
  readonly preset: TranscriberPreset
  /** `audio` is mono 16 kHz PCM in [-1, 1] (see `audioToMono16k`). */
  transcribe(audio: Float32Array, options?: TranscribeOptions): Promise<Transcript>
}

/** Cross-encoder that scores (query, document) relevance; higher is better. */
export interface Reranker {
  readonly id: string
  readonly locality?: Locality
  rerank(query: string, documents: string[]): Promise<number[]>
  load?(): Promise<void>
  isCached?(): Promise<boolean>
  clearCache?(): Promise<void>
}

// ---------------------------------------------------------------------------
// Agent events
// ---------------------------------------------------------------------------

export type AgentEvent =
  | { type: 'step-start'; step: number }
  | { type: 'text-delta'; delta: string }
  | { type: 'reasoning-delta'; delta: string }
  | { type: 'tool-call'; call: ToolCall }
  | { type: 'approval-request'; call: ToolCall }
  | { type: 'tool-result'; call: ToolCall; output: unknown; isError: boolean; durationMs: number }
  | { type: 'custom'; skill: string; tool: string; data: unknown }
  /** The run's citation list grew; carries the full list so far. */
  | { type: 'citations'; citations: Citation[] }
  | { type: 'message'; message: Message }
  | { type: 'step-finish'; step: number; reason: FinishReason; durationMs: number; usage?: Usage; metrics?: StepMetrics }
  | { type: 'finish'; reason: FinishReason | 'max-steps' | 'aborted'; steps: number; usage: Usage }

export type ApprovalHandler = (call: ToolCall) => boolean | Promise<boolean>
