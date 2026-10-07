import type {
  Embedder,
  Model,
  ModelChunk,
  ModelRequest,
  ToolCall,
  Transcriber,
  TranscribeOptions,
  Transcript,
  TranscriptSegment,
} from './types.js'
import type { TranscriberPreset } from './web/catalog.js'

export interface MockTranscriber extends Transcriber {
  /** Every call, for assertions: sample count and the options without callbacks/signal. */
  requests: { samples: number; options: Pick<TranscribeOptions, 'language' | 'task' | 'chunkSeconds' | 'strideSeconds'> }[]
}

/**
 * Deterministic transcriber for tests: returns the given segments (or what
 * `script(audio, options)` returns), streaming each through `onSegment` and
 * reporting progress by segment end time. Honors `signal`; `load`,
 * `isCached` and `clearCache` are no-ops (always cached).
 */
export function mockTranscriber(
  script: TranscriptSegment[] | ((audio: Float32Array, options: TranscribeOptions) => Transcript | Promise<Transcript>),
): MockTranscriber {
  const preset: TranscriberPreset = {
    id: 'mock',
    label: 'Mock transcriber',
    model: 'mock',
    params: '0',
    sizeBytes: 0,
    wasmSizeBytes: 0,
    downloadMB: 0,
    languages: 'multilingual',
    dtype: { webgpu: 'fp32', webgpuF32: 'fp32', wasm: 'fp32' },
  }
  const requests: MockTranscriber['requests'] = []
  return {
    id: 'mock-transcriber',
    locality: 'device',
    preset,
    requests,
    async transcribe(audio, options = {}) {
      const { signal, onSegment, onProgress, ...rest } = options
      requests.push({ samples: audio.length, options: rest })
      signal?.throwIfAborted()
      const transcript =
        typeof script === 'function'
          ? await script(audio, options)
          : { text: script.map((s) => s.text.trim()).join(' '), segments: script.map((s) => ({ ...s })) }
      const duration = Math.max(audio.length / 16_000, transcript.segments.at(-1)?.end ?? 0) || 1
      onProgress?.(0)
      for (const segment of transcript.segments) {
        await Promise.resolve()
        signal?.throwIfAborted()
        onSegment?.(segment)
        onProgress?.(Math.min(segment.end / duration, 1))
      }
      onProgress?.(1)
      return {
        ...transcript,
        ...(transcript.language ?? options.language ? { language: transcript.language ?? options.language } : {}),
      }
    },
    load: async () => undefined,
    isCached: async () => true,
    clearCache: async () => undefined,
  }
}

export type ScriptedTurn =
  | string
  | { text?: string; reasoning?: string; toolCalls?: Omit<ToolCall, 'id'>[] }
  | ((request: ModelRequest) => ScriptedTurn)

export interface MockModel extends Model {
  /** Every request the model received, for assertions. */
  requests: ModelRequest[]
}

/**
 * Deterministic model for tests: each call to `stream` plays the next scripted
 * turn. Text is streamed in small pieces to exercise incremental consumers.
 */
export function mockModel(script: ScriptedTurn[]): MockModel {
  const requests: ModelRequest[] = []
  let turn = 0
  return {
    id: 'mock',
    locality: 'device',
    requests,
    async *stream(request): AsyncGenerator<ModelChunk> {
      requests.push(structuredClone({ ...request, signal: undefined }))
      let step = script[turn++]
      if (step === undefined) throw new Error(`mockModel: no scripted turn #${turn}`)
      while (typeof step === 'function') step = step(request)
      const t = typeof step === 'string' ? { text: step } : step
      if (t.reasoning) yield { type: 'reasoning', delta: t.reasoning }
      for (const piece of (t.text ?? '').match(/.{1,8}/gs) ?? []) yield { type: 'text', delta: piece }
      const calls = t.toolCalls ?? []
      for (const [i, c] of calls.entries()) yield { type: 'tool-call', call: { id: `call_${turn}_${i}`, ...c } }
      yield {
        type: 'finish',
        reason: calls.length ? 'tool-calls' : 'stop',
        usage: { inputTokens: 10, outputTokens: 5 },
      }
    },
  }
}

/**
 * Tiny deterministic embedder: hashed bag-of-words, L2-normalized. Texts that
 * share words land close together, which is enough to test retrieval logic
 * without downloading a model.
 */
export function hashEmbedder(dimensions = 64): Embedder {
  return {
    id: `hash-${dimensions}`,
    locality: 'device',
    dimensions,
    async embed(texts) {
      return texts.map((text) => {
        const v = new Array<number>(dimensions).fill(0)
        for (const word of text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
          let h = 2166136261
          for (let i = 0; i < word.length; i++) h = Math.imul(h ^ word.charCodeAt(i), 16777619)
          v[Math.abs(h) % dimensions]! += 1
        }
        const norm = Math.hypot(...v) || 1
        return v.map((x) => x / norm)
      })
    },
  }
}
