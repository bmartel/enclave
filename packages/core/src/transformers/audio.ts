/**
 * Browser audio decoding for speech recognition. Uses Web Audio only inside
 * the function, so importing this module is safe in Node and workers.
 */

/** Sample rate Whisper expects. */
export const SAMPLE_RATE = 16_000

type OfflineCtor = new (channels: number, length: number, sampleRate: number) => OfflineAudioContext

function offlineContext(): OfflineCtor {
  const g = globalThis as { OfflineAudioContext?: OfflineCtor; webkitOfflineAudioContext?: OfflineCtor }
  const ctor = g.OfflineAudioContext ?? g.webkitOfflineAudioContext
  if (!ctor) {
    throw new Error('audioToMono16k needs the Web Audio API (OfflineAudioContext); decode audio on the main thread')
  }
  return ctor
}

function decode(ctx: BaseAudioContext, data: ArrayBuffer): Promise<AudioBuffer> {
  // The callback form also works in older Safari, where the promise form is missing.
  return new Promise((resolve, reject) => {
    const result = ctx.decodeAudioData(data, resolve, (e) => reject(e ?? new Error('Could not decode audio')))
    ;(result as Promise<AudioBuffer> | undefined)?.then?.(resolve, reject)
  })
}

function downmix(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0).slice()
  const out = new Float32Array(buffer.length)
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c)
    for (let i = 0; i < out.length; i++) out[i]! += data[i]!
  }
  const scale = 1 / buffer.numberOfChannels
  for (let i = 0; i < out.length; i++) out[i]! *= scale
  return out
}

/**
 * Decode an audio file (any format the browser can play) or take an
 * `AudioBuffer`, average its channels to mono, and resample to 16 kHz: the
 * input `Transcriber.transcribe` expects. Browser main thread only (Web Audio
 * is not available in workers).
 */
export async function audioToMono16k(input: Blob | ArrayBuffer | AudioBuffer): Promise<Float32Array> {
  const Offline = offlineContext()
  let buffer: AudioBuffer
  if (typeof (input as AudioBuffer).getChannelData === 'function') {
    buffer = input as AudioBuffer
  } else {
    // decodeAudioData detaches its argument: copy a caller's ArrayBuffer.
    const bytes = input instanceof ArrayBuffer ? input.slice(0) : await (input as Blob).arrayBuffer()
    // Decoding in a 16 kHz context resamples as part of decoding.
    buffer = await decode(new Offline(1, 1, SAMPLE_RATE), bytes)
  }
  if (buffer.sampleRate === SAMPLE_RATE) return downmix(buffer)

  const length = Math.max(1, Math.ceil(buffer.duration * SAMPLE_RATE))
  const ctx = new Offline(1, length, SAMPLE_RATE)
  const source = ctx.createBufferSource()
  source.buffer = buffer
  // Mono destination: Web Audio's standard down-mix (L+R)/2, 5.1 rules, etc.
  source.connect(ctx.destination)
  source.start()
  const rendered = await ctx.startRendering()
  return rendered.getChannelData(0).slice()
}
