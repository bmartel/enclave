import type { PGliteInterface } from '@electric-sql/pglite'

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
  prefillTokensPerSec?: number
  decodeTokensPerSec?: number
  /** Time spent compiling the tool-call grammar for this step. */
  grammarInitMs?: number
  /** True when the engine continued from its KV cache instead of re-reading the prompt. */
  kvCacheReused?: boolean
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

export interface Embedder {
  /** Stable identifier; a change triggers a reindex requirement. */
  readonly id: string
  /** Where document text is processed. Undeclared counts as `remote`. */
  readonly locality?: Locality
  readonly dimensions: number
  embed(texts: string[], kind: EmbedKind): Promise<number[][]>
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
  | { type: 'message'; message: Message }
  | { type: 'step-finish'; step: number; reason: FinishReason; durationMs: number; usage?: Usage; metrics?: StepMetrics }
  | { type: 'finish'; reason: FinishReason | 'max-steps' | 'aborted'; steps: number; usage: Usage }

export type ApprovalHandler = (call: ToolCall) => boolean | Promise<boolean>
