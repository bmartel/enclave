/**
 * Real Whisper on CPU (onnxruntime-node). Downloads ~66 MB (whisper-tiny) on
 * first run. Speech comes from macOS `say`, so this suite runs on macOS only.
 *   pnpm --filter enclave-ai test:e2e
 */
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { transformersTranscriber } from '../src/transformers/index.js'

const run = !!process.env.ENCLAVE_E2E && process.platform === 'darwin'

/** Speak `text` to 16 kHz mono float32 PCM. */
function say(text: string, voice?: string): Float32Array {
  const file = join(mkdtempSync(join(tmpdir(), 'enclave-stt-')), 'speech.wav')
  execFileSync('say', [...(voice ? ['-v', voice] : []), '-o', file, '--data-format=LEF32@16000', text])
  const buf = readFileSync(file)
  for (let i = 12; i < buf.length; ) {
    const size = buf.readUInt32LE(i + 4)
    if (buf.toString('ascii', i, i + 4) === 'data') {
      return new Float32Array(buf.buffer.slice(buf.byteOffset + i + 8, buf.byteOffset + i + 8 + size))
    }
    i += 8 + size + (size % 2)
  }
  throw new Error('no data chunk')
}

const words = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? []

describe.skipIf(!run)('whisper-tiny (real weights)', () => {
  const stt = transformersTranscriber({ preset: 'whisper-tiny' })

  it('transcribes English with timestamps and detects the language', async () => {
    const result = await stt.transcribe(say('The quick brown fox jumps over the lazy dog.'))
    expect(result.language).toBe('en')
    expect(words(result.text)).toEqual(expect.arrayContaining(['quick', 'brown', 'fox', 'lazy', 'dog']))
    expect(result.segments[0]!.start).toBeLessThan(1)
    expect(await stt.isCached()).toBe(true)
  }, 300_000)

  it('detects French', async () => {
    const result = await stt.transcribe(say('Bonjour à tous. Nous allons parler de la reconnaissance vocale.', 'Thomas'))
    expect(result.language).toBe('fr')
    expect(words(result.text)).toEqual(expect.arrayContaining(['bonjour', 'reconnaissance']))
  }, 300_000)

  it('stitches long audio across windows without losing or repeating speech', async () => {
    const numbers = 'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen'.split(' ')
    const sentences = numbers.map((n) => `Sentence number ${n} talks about the weather in the mountains.`)
    const audio = say(sentences.join(' '))
    expect(audio.length / 16_000).toBeGreaterThan(40)
    const streamed: string[] = []
    const result = await stt.transcribe(audio, { language: 'en', onSegment: (s) => streamed.push(s.text) })
    expect(streamed.join(' ')).toBe(result.text)
    const text = words(result.text).join(' ')
    expect(text.match(/mountains/g)?.length).toBe(numbers.length)
    for (let i = 1; i < result.segments.length; i++) {
      expect(result.segments[i]!.start).toBeGreaterThanOrEqual(result.segments[i - 1]!.start)
    }
    expect(result.segments.at(-1)!.end).toBeGreaterThan(30)
  }, 300_000)
})
