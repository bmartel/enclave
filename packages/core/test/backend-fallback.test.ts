import { describe, expect, it } from 'vitest'
import { ModelLoadError } from '../src/models/load-error.js'
import { transformersTranscriber, TransformersRuntime, type LoadProgress, type SpeechModel } from '../src/transformers/index.js'
import { isWebGpuFailure, shareSizeProbes, type Backend } from '../src/transformers/runtime.js'
import { serveTransformers } from '../src/transformers/protocol.js'

const SR = 16_000
const ADRENO = 'Failed to create a WebGPU compute pipeline: A valid external Instance reference no longer exists.'

/** A runtime that sees a WebGPU adapter; its Whisper fails on WebGPU as `fail` says. */
class GpuRuntime extends TransformersRuntime {
  created: string[] = []
  constructor(
    private readonly fail: { load?: boolean; run?: boolean; wasm?: boolean },
    report: (p: LoadProgress) => void,
  ) {
    super(report)
  }
  protected override detect(): Promise<Backend> {
    return Promise.resolve({ device: 'webgpu', f16: true })
  }
  protected override async createSpeechModel(config: { model: string; device?: 'auto' | 'webgpu' | 'wasm' }): Promise<SpeechModel> {
    const { device } = await this.resolve(config)
    this.created.push(device ?? 'none')
    if (device === 'webgpu' && this.fail.load) throw new Error(ADRENO)
    if (device === 'wasm' && this.fail.wasm) throw new Error('no WASM either')
    return {
      multilingual: false,
      recognize: async () => {
        if (device === 'webgpu' && this.fail.run) throw new Error(ADRENO)
        return { text: ` on ${device}`, chunks: [{ timestamp: [0, 1], text: ` on ${device}` }] }
      },
    }
  }
}

function connect(fail: ConstructorParameters<typeof GpuRuntime>[0]) {
  const channel = new MessageChannel()
  let runtime!: GpuRuntime
  serveTransformers({ scope: channel.port2 as never, runtime: (report) => (runtime = new GpuRuntime(fail, report)) })
  channel.port1.start()
  channel.port2.start()
  return { worker: channel.port1 as unknown as Worker, runtime: () => runtime, close: () => channel.port1.close() }
}

const audio = () => new Float32Array(5 * SR).fill(0.1)

describe('WebGPU → WASM fallback', () => {
  it('retries a window on WASM when WebGPU fails at inference, and reports it', async () => {
    const { worker, runtime, close } = connect({ run: true })
    const events: LoadProgress[] = []
    const stt = transformersTranscriber({ worker, onProgress: (p) => events.push(p) })
    const result = await stt.transcribe(audio())
    expect(result.text).toBe('on wasm')
    expect(runtime().created).toEqual(['webgpu', 'wasm'])
    expect(events).toContainEqual(expect.objectContaining({ status: 'fallback', device: 'wasm', error: expect.stringContaining('compute pipeline') }))
    // Later work stays on WASM: no second WebGPU attempt.
    await stt.transcribe(audio())
    expect(runtime().created).toEqual(['webgpu', 'wasm'])
    close()
  })

  it('falls back when the model fails to load on WebGPU', async () => {
    const { worker, runtime, close } = connect({ load: true })
    const stt = transformersTranscriber({ worker })
    await stt.load!()
    expect(runtime().created).toEqual(['webgpu', 'wasm'])
    expect((await stt.transcribe(audio())).text).toBe('on wasm')
    close()
  })

  it('throws a clear error when WASM fails too', async () => {
    const { worker, close } = connect({ load: true, wasm: true })
    const stt = transformersTranscriber({ worker })
    await expect(stt.load!()).rejects.toThrow(/no WASM either/)
    close()
  })

  it('does not fall back when WebGPU was asked for', async () => {
    const { worker, runtime, close } = connect({ load: true })
    const stt = transformersTranscriber({ worker, device: 'webgpu' })
    await expect(stt.load!()).rejects.toThrow(/WebGPU/)
    expect(runtime().created).toEqual(['webgpu'])
    close()
  })

  it('runs on WASM straight away when told to', async () => {
    const { worker, runtime, close } = connect({ load: true })
    const stt = transformersTranscriber({ worker, device: 'wasm' })
    expect((await stt.transcribe(audio())).text).toBe('on wasm')
    expect(runtime().created).toEqual(['wasm'])
    close()
  })

  it('recognises WebGPU failures', () => {
    expect(isWebGpuFailure(new Error(ADRENO))).toBe(true)
    expect(isWebGpuFailure(new ModelLoadError('unknown', 'm', new Error(ADRENO)))).toBe(true)
    expect(isWebGpuFailure(new ModelLoadError('gpu-memory', 'm'))).toBe(true)
    expect(isWebGpuFailure(new Error('GPUDevice was lost'))).toBe(true)
    expect(isWebGpuFailure(new Error('Failed to fetch'))).toBe(false)
    expect(isWebGpuFailure(new ModelLoadError('storage', 'm'))).toBe(false)
  })
})

describe('shareSizeProbes', () => {
  const host = (files: Record<string, string>) => {
    const calls: { url: string; range: string | null }[] = []
    const fetch = async (input: string | URL, init?: any) => {
      const url = String(input)
      calls.push({ url, range: new Headers(init?.headers).get('Range') })
      const body = files[url.split('/').pop()!]
      return body === undefined ? new Response('missing', { status: 404 }) : new Response(body, { headers: { 'content-length': String(body.length), 'content-type': 'application/json' } })
    }
    return { calls, fetch }
  }
  const probe = { method: 'GET', headers: new Headers({ Range: 'bytes=0-0' }), cache: 'no-store' }

  it('answers the size probe from the download that follows: one request per file', async () => {
    const { calls, fetch } = host({ 'config.json': '{"a":1}' })
    const shared = shareSizeProbes(fetch)
    const head = await shared('https://hf.example/m/config.json', probe)
    expect(head.status).toBe(206)
    expect(head.headers.get('content-range')).toBe('bytes 0-0/7')
    const full = await shared('https://hf.example/m/config.json', { headers: new Headers() })
    expect(await full.text()).toBe('{"a":1}')
    expect(calls).toEqual([{ url: 'https://hf.example/m/config.json', range: null }])
    // Used once: a later request goes to the network again.
    await shared('https://hf.example/m/config.json', {})
    expect(calls).toHaveLength(2)
  })

  it('passes missing files and other requests through', async () => {
    const { calls, fetch } = host({})
    const shared = shareSizeProbes(fetch)
    expect((await shared('https://hf.example/m/x.json', probe)).status).toBe(404)
    // The optional file's request that follows the probe gets the same 404, without asking again.
    expect((await shared('https://hf.example/m/x.json', {})).status).toBe(404)
    expect(calls).toHaveLength(1)
    expect((await shared('https://hf.example/m/x.json', {})).status).toBe(404)
    expect(calls).toHaveLength(2)
    const ranged = { headers: new Headers({ Range: 'bytes=10-20' }) }
    await shared('https://hf.example/m/x.json', ranged)
    expect(calls[2]).toEqual({ url: 'https://hf.example/m/x.json', range: 'bytes=10-20' })
  })
})
