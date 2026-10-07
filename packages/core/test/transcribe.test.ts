import { describe, expect, it } from 'vitest'
import { mockTranscriber } from '../src/testing.js'
import {
  audioToMono16k,
  chunkWindows,
  findTranscriber,
  formatTimestamp,
  mergeSegments,
  recommendTranscriber,
  transcribeChunked,
  transformersTranscriber,
  TRANSCRIBER_PRESETS,
  TransformersRuntime,
  type LoadProgress,
  type SpeechModel,
} from '../src/transformers/index.js'
import { serveTransformers } from '../src/transformers/protocol.js'
import type { TranscriberPreset as WebTranscriberPreset } from '../src/web/index.js'
import { overlapRemainder, toSegments } from '../src/transformers/transcribe.js'
import type { TranscriptSegment } from '../src/types.js'

const SR = 16_000
const tone = (seconds: number) => {
  const a = new Float32Array(Math.round(seconds * SR))
  for (let i = 0; i < a.length; i++) a[i] = 0.2 * Math.sin(i / 10)
  return a
}

describe('formatTimestamp', () => {
  it('formats mm:ss below an hour and h:mm:ss from one hour', () => {
    expect(formatTimestamp(0)).toBe('00:00')
    expect(formatTimestamp(5.9)).toBe('00:05')
    expect(formatTimestamp(65)).toBe('01:05')
    expect(formatTimestamp(3599.99)).toBe('59:59')
    expect(formatTimestamp(3600)).toBe('1:00:00')
    expect(formatTimestamp(3 * 3600 + 7 * 60 + 9)).toBe('3:07:09')
  })

  it('treats negative and non-finite input as zero', () => {
    expect(formatTimestamp(-3)).toBe('00:00')
    expect(formatTimestamp(Number.NaN)).toBe('00:00')
    expect(formatTimestamp(Infinity)).toBe('00:00')
  })
})

describe('chunkWindows', () => {
  it('covers the audio with overlapping windows', () => {
    const w = chunkWindows(70 * SR, SR, 30, 5)
    expect(w.map((x) => [x.startTime, x.endTime])).toEqual([
      [0, 30],
      [25, 55],
      [50, 70],
    ])
    expect(w[1]).toMatchObject({ start: 25 * SR, end: 55 * SR })
  })

  it('returns one window for short audio and none for empty audio', () => {
    expect(chunkWindows(10 * SR, SR, 30, 5)).toEqual([{ start: 0, end: 10 * SR, startTime: 0, endTime: 10 }])
    expect(chunkWindows(0, SR)).toEqual([])
  })

  it('does not add a window that only repeats the overlap', () => {
    expect(chunkWindows(30 * SR, SR, 30, 5)).toHaveLength(1)
    expect(chunkWindows(55 * SR, SR, 30, 5)).toHaveLength(2)
  })

  it('supports zero stride and rejects invalid settings', () => {
    expect(chunkWindows(60 * SR, SR, 30, 0).map((x) => x.startTime)).toEqual([0, 30])
    expect(() => chunkWindows(10, SR, 30, 30)).toThrow(/strideSeconds/)
    expect(() => chunkWindows(10, SR, 0, 0)).toThrow(/chunkSeconds/)
    expect(() => chunkWindows(10, 0)).toThrow(/sample rate/)
  })
})

describe('mergeSegments', () => {
  it('keeps the overlap once, preferring the window that heard a segment whole', () => {
    // Real whisper-base output around a 25-30 s overlap: window 1 cuts off at
    // 30 s; window 2 starts mid-sentence at 25 s.
    const merged = mergeSegments([
      {
        start: 0,
        end: 30,
        segments: [
          { start: 20.88, end: 25.32, text: 'At one end of it a colored poster had been tacked to the' },
          { start: 25.32, end: 26.32, text: 'wall.' },
          { start: 26.32, end: 29.72, text: 'It depicted simply an enormous face, more than a meter wide.' },
          { start: 29.72, end: 30, text: 'the face.' },
        ],
      },
      {
        start: 25,
        end: 55,
        segments: [
          { start: 25, end: 30.44, text: 'to the wall. It depicted simply an enormous face, more than a meter wide. The face of a man' },
          { start: 30.44, end: 36.76, text: 'of about 45, with a heavy black mustache.' },
        ],
      },
    ])
    expect(merged).toEqual([
      { start: 20.88, end: 25.32, text: 'At one end of it a colored poster had been tacked to the' },
      { start: 25.32, end: 26.32, text: 'wall.' },
      { start: 26.32, end: 30.44, text: 'It depicted simply an enormous face, more than a meter wide. The face of a man' },
      { start: 30.44, end: 36.76, text: 'of about 45, with a heavy black mustache.' },
    ])
  })

  it('appends what a later window heard after a segment the earlier window cut off', () => {
    const merged = mergeSegments([
      { start: 25, end: 55, segments: [{ start: 48.76, end: 54.96, text: 'seven flights up, and had an ulcer above his ankle.' }] },
      { start: 50, end: 57.5, segments: [{ start: 50, end: 56.24, text: 'had an ulcer above his ankle, went slowly, resting' }, { start: 56.24, end: 57.48, text: 'several times.' }] },
    ])
    expect(merged.map((s) => s.text)).toEqual([
      'seven flights up, and had an ulcer above his ankle.',
      'went slowly, resting',
      'several times.',
    ])
    expect(merged[1]).toMatchObject({ start: 54.96, end: 56.24 })
  })

  it('drops exact repeats across a boundary, empty text, and sorts by time', () => {
    const merged = mergeSegments([
      { start: 0, end: 30, segments: [{ start: 3, end: 6, text: ' second ' }, { start: 0, end: 3, text: 'first' }, { start: 26, end: 27, text: 'Hello there.' }] },
      { start: 25, end: 40, segments: [{ start: 26.2, end: 27.1, text: 'hello there' }, { start: 30, end: 31, text: '  ' }, { start: 33, end: 34, text: 'after' }] },
    ])
    expect(merged.map((s) => s.text)).toEqual(['first', 'second', 'Hello there.', 'after'])
  })

  it('finds the text overlap between the end of one text and another', () => {
    expect(overlapRemainder('and Winston, who was 39,', 'Winston who was 39 and had an ulcer')).toBe('and had an ulcer')
    expect(overlapRemainder('the end.', 'the end.')).toBe('')
    expect(overlapRemainder('nothing shared', 'completely different words')).toBeUndefined()
  })
})

describe('toSegments', () => {
  it('offsets pipeline timestamps, clamps them to the window and fills a missing end', () => {
    expect(
      toSegments(
        {
          text: ' a b',
          chunks: [
            { timestamp: [0, 2.5], text: ' a' },
            { timestamp: [2.5, null], text: ' b' },
            { timestamp: [3, 4], text: '   ' },
          ],
        },
        25,
        10,
      ),
    ).toEqual([
      { start: 25, end: 27.5, text: 'a' },
      { start: 27.5, end: 35, text: 'b' },
    ])
    expect(toSegments({ text: ' whole ' }, 5, 3)).toEqual([{ start: 5, end: 8, text: 'whole' }])
    expect(toSegments({ text: '' }, 5, 3)).toEqual([])
  })
})

describe('transcribeChunked', () => {
  const fakeRecognizer = async (samples: Float32Array, w: { offset: number; duration: number }) => [
    { start: w.offset + 1, end: w.offset + 2, text: `at ${w.offset}` },
  ]

  it('streams segments and progress per window with absolute times', async () => {
    const segments: TranscriptSegment[] = []
    const progress: number[] = []
    const result = await transcribeChunked(tone(70), fakeRecognizer, {
      onSegment: (s) => segments.push(s),
      onProgress: (f) => progress.push(f),
    })
    expect(result.segments.map((s) => s.start)).toEqual([1, 26, 51])
    expect(segments).toEqual(result.segments)
    expect(result.text).toBe('at 0 at 25 at 50')
    expect(progress[0]).toBe(0)
    expect(progress.at(-1)).toBe(1)
    expect(progress).toEqual([...progress].sort((a, b) => a - b))
  })

  it('skips silent windows and handles empty audio', async () => {
    const audio = new Float32Array(70 * SR)
    audio.set(tone(5), 60 * SR)
    const offsets: number[] = []
    await transcribeChunked(audio, async (s, w) => (offsets.push(w.offset), []))
    expect(offsets).toEqual([50])
    const progress: number[] = []
    expect(await transcribeChunked(new Float32Array(0), fakeRecognizer, { onProgress: (f) => progress.push(f) })).toEqual({
      text: '',
      segments: [],
    })
    expect(progress).toEqual([0, 1])
  })

  it('stops between windows when aborted', async () => {
    const controller = new AbortController()
    let calls = 0
    const run = transcribeChunked(
      tone(90),
      async (s, w) => {
        calls++
        return fakeRecognizer(s, w)
      },
      { signal: controller.signal, onSegment: () => controller.abort() },
    )
    await expect(run).rejects.toMatchObject({ name: 'AbortError' })
    expect(calls).toBe(1)
  })

  it('caps windows at 30 s', async () => {
    const durations: number[] = []
    await transcribeChunked(tone(100), async (s, w) => (durations.push(w.duration), []), { chunkSeconds: 60 })
    expect(Math.max(...durations)).toBe(30)
  })
})

describe('transcriber presets', () => {
  it('lists multilingual Whisper presets with per-backend dtypes and sizes', () => {
    expect(TRANSCRIBER_PRESETS.map((p) => p.id)).toEqual(['whisper-tiny', 'whisper-base'])
    for (const p of TRANSCRIBER_PRESETS) {
      expect(p.model).toMatch(/^onnx-community\/whisper-/)
      expect(p.languages).toBe('multilingual')
      expect(p.sizeBytes).toBeGreaterThan(p.wasmSizeBytes)
      expect(p.downloadMB).toBe(Math.round(p.sizeBytes / 1e6))
      for (const backend of ['webgpu', 'webgpuF32', 'wasm'] as const) {
        expect(p.dtype[backend]).toMatchObject({ encoder_model: 'fp32' })
      }
    }
    expect(findTranscriber('whisper-base')?.model).toBe('onnx-community/whisper-base')
    expect(findTranscriber('nope')).toBeUndefined()
    const fromWeb: WebTranscriberPreset = findTranscriber('whisper-tiny')!
    expect(fromWeb.id).toBe('whisper-tiny')
  })

  it('recommends base on desktop GPUs and tiny elsewhere', () => {
    expect(recommendTranscriber({ webgpu: true, mobile: false }).id).toBe('whisper-base')
    expect(recommendTranscriber({ webgpu: true, mobile: true }).id).toBe('whisper-tiny')
    expect(recommendTranscriber({ webgpu: false, mobile: false }).id).toBe('whisper-tiny')
  })

  it('rejects unknown presets', () => {
    expect(() => transformersTranscriber({ preset: 'whisper-huge' })).toThrow(/Unknown transcriber preset/)
  })
})

/** Runtime whose Whisper is a fake that "hears" the window offset. */
class FakeSpeechRuntime extends TransformersRuntime {
  loads = 0
  languages: (string | undefined)[] = []
  constructor(private readonly report: (p: LoadProgress) => void) {
    super(report)
  }
  protected override async createSpeechModel(config: { model: string }): Promise<SpeechModel> {
    this.loads++
    this.report({ model: config.model, status: 'progress', file: 'onnx/encoder_model.onnx', loaded: 1, total: 2 })
    this.report({ model: config.model, status: 'ready' })
    return {
      multilingual: true,
      detectLanguage: async () => 'fr',
      recognize: async (samples, options) => {
        this.languages.push(options.language)
        await new Promise((r) => setTimeout(r, 5))
        return { text: ' bonjour', chunks: [{ timestamp: [1, 2], text: ` bonjour ${samples.length / SR}` }] }
      },
    }
  }
}

describe('transcription through the worker protocol', () => {
  function connect() {
    const channel = new MessageChannel()
    let runtime!: FakeSpeechRuntime
    serveTransformers({
      scope: channel.port2 as never,
      runtime: (report) => (runtime = new FakeSpeechRuntime(report)),
    })
    channel.port1.start()
    channel.port2.start()
    return { worker: channel.port1 as unknown as Worker, runtime: () => runtime, close: () => channel.port1.close() }
  }

  it('streams segments and progress, detects the language once, and reports load progress', async () => {
    const { worker, runtime, close } = connect()
    const loads: LoadProgress[] = []
    const stt = transformersTranscriber({ worker, preset: 'whisper-base', onProgress: (p) => loads.push(p) })
    expect(stt).toMatchObject({ id: 'transformers:onnx-community/whisper-base', locality: 'device' })
    expect(stt.preset.id).toBe('whisper-base')
    const segments: TranscriptSegment[] = []
    const progress: number[] = []
    const audio = tone(70)
    const result = await stt.transcribe(audio.subarray(0, 60 * SR), {
      onSegment: (s) => segments.push(s),
      onProgress: (f) => progress.push(f),
    })
    expect(result.language).toBe('fr')
    expect(result.segments).toEqual([
      { start: 1, end: 2, text: 'bonjour 30' },
      { start: 26, end: 27, text: 'bonjour 30' },
      { start: 51, end: 52, text: 'bonjour 10' },
    ])
    expect(segments).toEqual(result.segments)
    expect(progress).toEqual([0, 0.5, 55 / 60, 1])
    expect(runtime().loads).toBe(1)
    expect(runtime().languages).toEqual(['fr', 'fr', 'fr'])
    expect(loads.map((p) => p.status)).toEqual(['progress', 'ready'])
    expect(loads[0]).toMatchObject({ model: 'onnx-community/whisper-base', file: 'onnx/encoder_model.onnx' })

    await stt.load()
    expect(runtime().loads).toBe(1)
    close()
  })

  it('passes language and task through and stops on abort', async () => {
    const { worker, runtime, close } = connect()
    const stt = transformersTranscriber({ worker })
    const english = await stt.transcribe(tone(10), { language: 'en', task: 'translate' })
    expect(english.language).toBe('en')
    expect(runtime().languages).toEqual(['en'])

    const controller = new AbortController()
    const run = stt.transcribe(tone(300), { signal: controller.signal, onSegment: () => controller.abort() })
    await expect(run).rejects.toMatchObject({ name: 'AbortError' })
    await new Promise((r) => setTimeout(r, 50))
    // The worker stopped after the window in flight.
    expect(runtime().languages.length).toBeLessThanOrEqual(3)

    const already = new AbortController()
    already.abort()
    await expect(stt.transcribe(tone(10), { signal: already.signal })).rejects.toMatchObject({ name: 'AbortError' })
    close()
  })
})

describe('mockTranscriber', () => {
  it('plays scripted segments through the Transcriber interface', async () => {
    const stt = mockTranscriber([
      { start: 0, end: 2, text: 'Hello' },
      { start: 2, end: 4, text: 'world' },
    ])
    const seen: string[] = []
    const progress: number[] = []
    const result = await stt.transcribe(new Float32Array(4 * SR), {
      language: 'en',
      onSegment: (s) => seen.push(s.text),
      onProgress: (f) => progress.push(f),
    })
    expect(result).toEqual({
      text: 'Hello world',
      language: 'en',
      segments: [
        { start: 0, end: 2, text: 'Hello' },
        { start: 2, end: 4, text: 'world' },
      ],
    })
    expect(seen).toEqual(['Hello', 'world'])
    expect(progress).toEqual([0, 0.5, 1, 1])
    expect(stt.requests).toEqual([{ samples: 4 * SR, options: { language: 'en' } }])
    expect(stt.locality).toBe('device')
    await expect(stt.load()).resolves.toBeUndefined()
    await expect(stt.isCached()).resolves.toBe(true)
    await expect(stt.clearCache()).resolves.toBeUndefined()
  })

  it('accepts a function and honors abort', async () => {
    const stt = mockTranscriber((audio) => ({
      text: `${audio.length} samples`,
      language: 'de',
      segments: [{ start: 0, end: 1, text: `${audio.length} samples` }],
    }))
    expect(await stt.transcribe(new Float32Array(3))).toMatchObject({ text: '3 samples', language: 'de' })
    const controller = new AbortController()
    controller.abort()
    await expect(stt.transcribe(new Float32Array(3), { signal: controller.signal })).rejects.toMatchObject({
      name: 'AbortError',
    })
  })
})

describe('audioToMono16k', () => {
  it('imports in Node and explains that it needs Web Audio', async () => {
    await expect(audioToMono16k(new ArrayBuffer(8))).rejects.toThrow(/OfflineAudioContext/)
  })

  it('downmixes and resamples with an OfflineAudioContext', async () => {
    const rendered: { channels: number; length: number; rate: number }[] = []
    class FakeBuffer {
      constructor(
        readonly numberOfChannels: number,
        readonly length: number,
        readonly sampleRate: number,
        private readonly data: Float32Array[],
      ) {}
      get duration() {
        return this.length / this.sampleRate
      }
      getChannelData(c: number) {
        return this.data[c]!
      }
    }
    class FakeOffline {
      destination = {}
      constructor(
        readonly channels: number,
        readonly length: number,
        readonly rate: number,
      ) {}
      decodeAudioData(_data: ArrayBuffer, ok: (b: FakeBuffer) => void) {
        // Decoding in a 16 kHz context yields 16 kHz stereo.
        ok(new FakeBuffer(2, 4, this.rate, [Float32Array.of(1, 1, 0, 0), Float32Array.of(0, 1, 0, -1)]))
      }
      createBufferSource() {
        return { buffer: null, connect() {}, start() {} }
      }
      async startRendering() {
        rendered.push({ channels: this.channels, length: this.length, rate: this.rate })
        return new FakeBuffer(1, this.length, this.rate, [new Float32Array(this.length).fill(0.25)])
      }
    }
    const g = globalThis as { OfflineAudioContext?: unknown }
    g.OfflineAudioContext = FakeOffline
    try {
      const input = new ArrayBuffer(8)
      expect(Array.from(await audioToMono16k(input))).toEqual([0.5, 1, 0, -0.5])
      expect(input.byteLength).toBe(8) // caller's buffer not detached
      const out = await audioToMono16k(new FakeBuffer(1, 48_000, 48_000, [new Float32Array(48_000)]) as unknown as AudioBuffer)
      expect(out).toHaveLength(SR)
      expect(rendered).toEqual([{ channels: 1, length: SR, rate: SR }])
    } finally {
      delete g.OfflineAudioContext
    }
  })
})
