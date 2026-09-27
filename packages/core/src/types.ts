import type { PGliteInterface } from '@electric-sql/pglite'

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

export type FinishReason = 'stop' | 'tool-calls' | 'length' | 'refusal' | 'error'

export type ModelChunk =
  | { type: 'text'; delta: string }
  | { type: 'reasoning'; delta: string }
  | { type: 'tool-call'; call: ToolCall }
  | {
      type: 'finish'
      reason: FinishReason
      usage?: Usage
      providerData?: AssistantMessage['providerData']
    }

/** The single contract every model backend implements. */
export interface Model {
  readonly id: string
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
  readonly dimensions: number
  embed(texts: string[], kind: EmbedKind): Promise<number[][]>
  /** Model-specific document formatting (e.g. EmbeddingGemma's `title: … | text: …`). */
  formatDocument?(text: string, title?: string): string
  /** Start downloading/compiling the model ahead of first use. */
  load?(): Promise<void>
}

/** Cross-encoder that scores (query, document) relevance; higher is better. */
export interface Reranker {
  readonly id: string
  rerank(query: string, documents: string[]): Promise<number[]>
  load?(): Promise<void>
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
  | { type: 'finish'; reason: FinishReason | 'max-steps' | 'aborted'; steps: number; usage: Usage }

export type ApprovalHandler = (call: ToolCall) => boolean | Promise<boolean>
