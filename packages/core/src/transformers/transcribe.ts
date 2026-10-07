/**
 * Pure helpers for long-form speech-to-text: windowing audio into overlapping
 * chunks, stitching per-chunk segments back together, and formatting times.
 * No Transformers.js or browser APIs here, so they run (and test) anywhere.
 */
import type { Transcript, TranscriptSegment } from '../types.js'

/** Whisper's input window. Longer chunks would be truncated by the feature extractor. */
export const MAX_CHUNK_SECONDS = 30

/** One window over the audio. */
export interface ChunkWindow {
  /** First sample (inclusive). */
  start: number
  /** Last sample (exclusive). */
  end: number
  /** Window start, seconds. */
  startTime: number
  /** Window end, seconds. */
  endTime: number
}

/**
 * Split `sampleCount` samples into windows of `chunkSeconds` that overlap by
 * `strideSeconds`. The last window ends at the last sample.
 */
export function chunkWindows(
  sampleCount: number,
  sampleRate: number,
  chunkSeconds = MAX_CHUNK_SECONDS,
  strideSeconds = 5,
): ChunkWindow[] {
  if (!(sampleRate > 0)) throw new Error(`Invalid sample rate ${sampleRate}`)
  if (!(chunkSeconds > 0)) throw new Error(`chunkSeconds must be positive, got ${chunkSeconds}`)
  if (!(strideSeconds >= 0) || strideSeconds >= chunkSeconds) {
    throw new Error(`strideSeconds must be in [0, chunkSeconds), got ${strideSeconds}`)
  }
  const size = Math.max(1, Math.round(chunkSeconds * sampleRate))
  const step = Math.max(1, Math.round((chunkSeconds - strideSeconds) * sampleRate))
  const windows: ChunkWindow[] = []
  for (let start = 0; start < sampleCount; start += step) {
    const end = Math.min(start + size, sampleCount)
    windows.push({ start, end, startTime: start / sampleRate, endTime: end / sampleRate })
    if (end >= sampleCount) break
  }
  return windows
}

/** What one window heard: its bounds and its segments, all in absolute seconds. */
export interface TranscriptChunk {
  start: number
  end: number
  segments: TranscriptSegment[]
}

/** A segment within this many seconds of a window's end is treated as cut off by it. */
const EDGE_SECONDS = 0.3
/** Tolerance when deciding a later window's segment was already heard. */
const OVERLAP_SLACK = 0.5

const normalize = (text: string) => text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim()

/** Same words, overlapping or within a second in time: the overlap transcribed twice. */
export function isBoundaryDuplicate(previous: TranscriptSegment | undefined, segment: TranscriptSegment): boolean {
  if (!previous) return false
  const text = normalize(segment.text)
  return text !== '' && normalize(previous.text) === text && segment.start < previous.end + 1
}

const words = (text: string) =>
  text
    .trim()
    .split(/\s+/)
    .map((raw) => ({ raw, norm: normalize(raw) }))
    .filter((w) => w.norm)

/**
 * The words of `tail` after its overlap with the end of `head`: finds the
 * longest run of at least two words that ends `head` and also appears in
 * `tail`. Returns undefined when they don't overlap, `''` when `tail` adds nothing.
 */
export function overlapRemainder(head: string, tail: string): string | undefined {
  const h = words(head).map((w) => w.norm)
  const t = words(tail)
  for (let k = Math.min(h.length, t.length); k >= 2; k--) {
    const run = h.slice(-k).join(' ')
    for (let j = 0; j + k <= t.length; j++) {
      if (
        t
          .slice(j, j + k)
          .map((w) => w.norm)
          .join(' ') === run
      ) {
        return t
          .slice(j + k)
          .map((w) => w.raw)
          .join(' ')
      }
    }
  }
  return undefined
}

/**
 * Incremental stitcher for overlapping windows. Kept segments are final, so
 * they can be streamed as soon as their window is done.
 * - A window owns the segments that start before the middle of its overlap
 *   with the next window, except one running into its end that the next
 *   window hears from its start (Whisper cuts such segments short).
 * - A later window's segment that starts before the last kept segment ended
 *   was (partly) heard already: only the words after the text overlap are
 *   kept, as a new segment from where the last one ended.
 */
export class SegmentMerger {
  readonly segments: TranscriptSegment[] = []

  /** Add a window's segments (absolute times). `next` is the following window's bounds. Returns the segments kept, in order. */
  add(chunk: TranscriptChunk, next?: { start: number; end: number }): TranscriptSegment[] {
    const kept: TranscriptSegment[] = []
    const sorted = [...chunk.segments].sort((a, b) => a.start - b.start || a.end - b.end)
    for (const segment of sorted) {
      let text = segment.text.trim()
      let start = segment.start
      if (!text) continue
      const last = this.segments.at(-1)
      if (last && start < last.end - OVERLAP_SLACK) {
        if (segment.end <= last.end + OVERLAP_SLACK) continue
        const recent = this.segments
          .slice(-3)
          .map((s) => s.text)
          .join(' ')
        const rest = overlapRemainder(recent, text)
        if (!rest) continue
        text = rest
        start = last.end
      }
      if (next) {
        const cut = (next.start + chunk.end) / 2
        if (start >= cut) continue
        if (segment.end >= chunk.end - EDGE_SECONDS && start >= next.start) continue
      }
      const clean = { start, end: segment.end, text }
      if (isBoundaryDuplicate(last, clean)) continue
      this.segments.push(clean)
      kept.push(clean)
    }
    return kept
  }
}

/**
 * Stitch per-window segments (absolute times, windows in order and
 * overlapping by a few seconds) into one timeline, keeping the overlaps once.
 */
export function mergeSegments(chunks: TranscriptChunk[]): TranscriptSegment[] {
  const merger = new SegmentMerger()
  chunks.forEach((chunk, i) => merger.add(chunk, chunks[i + 1]))
  return merger.segments.map((s) => ({ ...s }))
}

/** Join segment texts into one transcript string. */
export function joinSegments(segments: TranscriptSegment[]): string {
  return segments
    .map((s) => s.text.trim())
    .filter(Boolean)
    .join(' ')
}

/** `mm:ss`, or `h:mm:ss` from one hour. Fractions are floored; negatives and NaN read as 0. */
export function formatTimestamp(seconds: number): string {
  const total = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const pad = (n: number) => String(n).padStart(2, '0')
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`
}

/** Output of the Transformers.js ASR pipeline with `return_timestamps: true`. */
export interface AsrOutput {
  text: string
  chunks?: { timestamp: (number | null)[]; text: string }[]
}

/** Convert pipeline output for one window into absolute segments clamped to the window. */
export function toSegments(output: AsrOutput, offset: number, duration: number): TranscriptSegment[] {
  const chunks = output.chunks?.length
    ? output.chunks
    : output.text.trim()
      ? [{ timestamp: [0, duration], text: output.text }]
      : []
  return chunks
    .map((c) => {
      const [from, to] = c.timestamp
      const start = Math.min(Math.max(from ?? 0, 0), duration)
      const end = Math.min(Math.max(to ?? duration, start), duration)
      return { start: round(offset + start), end: round(offset + end), text: c.text.trim() }
    })
    .filter((s) => s.text)
}

const round = (n: number) => Math.round(n * 1000) / 1000

/** Peak amplitude at or below this counts as silence (about -46 dBFS). Whisper invents text for silence. */
export const SILENCE_PEAK = 0.005

export function isSilent(samples: Float32Array, threshold = SILENCE_PEAK): boolean {
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i]!
    if (v > threshold || v < -threshold) return false
  }
  return true
}

export interface ChunkedTranscribeOptions {
  sampleRate?: number
  chunkSeconds?: number
  strideSeconds?: number
  signal?: AbortSignal
  onSegment?(segment: TranscriptSegment): void
  onProgress?(fraction: number): void
}

/** Recognizes one window; returns segments with absolute times. */
export type WindowRecognizer = (
  samples: Float32Array,
  window: { offset: number; duration: number; index: number },
) => Promise<TranscriptSegment[]>

/**
 * Run a per-window recognizer over long audio. Segments stream out as each
 * window completes, progress is reported per window, and `signal` is checked
 * between windows. Silent windows are skipped.
 */
export async function transcribeChunked(
  audio: Float32Array,
  recognize: WindowRecognizer,
  options: ChunkedTranscribeOptions = {},
): Promise<Transcript> {
  const sampleRate = options.sampleRate ?? 16_000
  const chunkSeconds = Math.min(options.chunkSeconds ?? MAX_CHUNK_SECONDS, MAX_CHUNK_SECONDS)
  const strideSeconds = Math.min(options.strideSeconds ?? 5, chunkSeconds / 2)
  const windows = chunkWindows(audio.length, sampleRate, chunkSeconds, strideSeconds)
  const merger = new SegmentMerger()
  options.signal?.throwIfAborted()
  options.onProgress?.(0)
  for (const [index, w] of windows.entries()) {
    options.signal?.throwIfAborted()
    const samples = audio.subarray(w.start, w.end)
    if (!isSilent(samples)) {
      const segments = await recognize(samples, { offset: w.startTime, duration: w.endTime - w.startTime, index })
      options.signal?.throwIfAborted()
      const next = windows[index + 1]
      const kept = merger.add(
        { start: w.startTime, end: w.endTime, segments },
        next ? { start: next.startTime, end: next.endTime } : undefined,
      )
      for (const segment of kept) options.onSegment?.({ ...segment })
    }
    options.onProgress?.(w.end / audio.length)
  }
  if (!windows.length) options.onProgress?.(1)
  const segments = merger.segments.map((s) => ({ ...s }))
  return { text: joinSegments(segments), segments }
}
