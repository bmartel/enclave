import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  DISCRETE_F32_BUDGET_MB,
  detectDevice,
  findEmbedding,
  installAdapterFallback,
  isDiscreteGpu,
  overrideDevice,
  pickDtype,
  recommendEmbedding,
  recommendLLM,
  requestGpuAdapter,
  type GPUAdapterLike,
} from '../src/web/index.js'

const GB = 2 ** 30

/** An adapter as Chrome reports it. */
function adapter(p: { f16?: boolean; vendor?: string; architecture?: string; description?: string; storageGB?: number; fallback?: boolean } = {}): GPUAdapterLike {
  const features = new Set(['subgroups', 'timestamp-query', ...(p.f16 === false ? [] : ['shader-f16'])])
  return {
    features,
    limits: { maxStorageBufferBindingSize: (p.storageGB ?? 2) * GB, maxBufferSize: (p.storageGB ?? 2) * GB },
    info: { vendor: p.vendor ?? 'apple', architecture: p.architecture ?? 'metal-3', description: p.description ?? '', isFallbackAdapter: !!p.fallback },
  }
}

/** The user's RTX 4070 in Edge on Windows 11: Dawn's only usable adapter is Vulkan, without shader-f16. */
const rtx4070Vulkan = () => adapter({ f16: false, vendor: 'nvidia', architecture: 'lovelace', storageGB: 4 })

const WINDOWS_EDGE = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0'
const MAC_CHROME = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36'

function stubNavigator(gpu: unknown, ua = MAC_CHROME) {
  vi.stubGlobal('navigator', { gpu, userAgent: ua, deviceMemory: 8, maxTouchPoints: 0, userAgentData: { mobile: false, platform: /Windows/.test(ua) ? 'Windows' : 'macOS' } })
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('WebGPU detection', () => {
  it('says why there is no WebGPU: no API (a worker or browser without navigator.gpu)', async () => {
    stubNavigator(undefined)
    const d = await detectDevice({ retryDelayMs: 0 })
    expect(d).toMatchObject({ webgpu: false, webgpuStatus: 'no-api', gpuBudgetMB: 0 })
  })

  it('tries without powerPreference when the high-performance request comes back null', async () => {
    const requestAdapter = vi.fn(async (o?: { powerPreference?: string }) => (o?.powerPreference ? null : adapter()))
    stubNavigator({ requestAdapter })
    const d = await detectDevice({ retryDelayMs: 0 })
    expect(d).toMatchObject({ webgpu: true, webgpuStatus: 'available', adapterRequest: 'default', shaderF16: true })
    expect(requestAdapter).toHaveBeenNthCalledWith(1, { powerPreference: 'high-performance' })
  })

  it('asks again when the first request finds no adapter (the GPU process starting)', async () => {
    let calls = 0
    stubNavigator({ requestAdapter: async () => (++calls < 3 ? null : adapter()) })
    const d = await detectDevice({ retryDelayMs: 0 })
    expect(d).toMatchObject({ webgpu: true, adapterRequest: 'retry' })
  })

  it("doesn't pass powerPreference on Windows (Chrome ignores it there with a warning)", async () => {
    const requestAdapter = vi.fn(async () => rtx4070Vulkan())
    stubNavigator({ requestAdapter }, WINDOWS_EDGE)
    await detectDevice({ retryDelayMs: 0 })
    expect(requestAdapter).toHaveBeenCalledWith()
    expect(requestAdapter).not.toHaveBeenCalledWith({ powerPreference: 'high-performance' })
  })

  it('tells "no adapter", "only a software adapter" and "an error" apart', async () => {
    stubNavigator({ requestAdapter: async () => null })
    expect((await detectDevice({ retryDelayMs: 0 })).webgpuStatus).toBe('no-adapter')
    stubNavigator({ requestAdapter: async () => adapter({ fallback: true }) })
    expect(await detectDevice({ retryDelayMs: 0 })).toMatchObject({ webgpu: false, webgpuStatus: 'fallback-adapter' })
    stubNavigator({
      requestAdapter: async () => {
        throw new Error('GPU process crashed')
      },
    })
    expect(await detectDevice({ retryDelayMs: 0 })).toMatchObject({ webgpu: false, webgpuStatus: 'error', webgpuError: 'GPU process crashed' })
  })

  it('WebGPU with small buffers is still WebGPU (a smaller budget, not "no WebGPU")', async () => {
    stubNavigator({ requestAdapter: async () => adapter({ storageGB: 0.125 }) })
    const d = await detectDevice({ retryDelayMs: 0 })
    expect(d).toMatchObject({ webgpu: true, maxStorageBufferMB: 128 })
    expect(d.gpuBudgetMB).toBe(2000)
  })
})

describe('a desktop card without shader-f16 (RTX 4070 on Vulkan, Windows)', () => {
  it('runs on the GPU: the same tiers from full-precision builds, EmbeddingGemma 2 for search', async () => {
    stubNavigator({ requestAdapter: async () => rtx4070Vulkan() }, WINDOWS_EDGE)
    const d = await detectDevice({ retryDelayMs: 0 })
    expect(d).toMatchObject({ webgpu: true, shaderF16: false, discreteGpu: true, maxStorageBufferMB: 4096 })
    expect(d.gpuBudgetMB).toBe(DISCRETE_F32_BUDGET_MB)
    const llm = recommendLLM(d)!
    expect(llm.modelId).toBe('Qwen3-4B-q4f32_1-MLC')
    expect(llm.contextWindow).toBe(8192)
    expect(llm.estimatedMB).toBeLessThanOrEqual(d.gpuBudgetMB)
    const embedding = recommendEmbedding(d)
    expect(embedding.id).toBe('embeddinggemma-2')
    expect(pickDtype(embedding.dtype, d)).toBe('q4')
  })

  it('a card that is not known to be discrete keeps the conservative budget', () => {
    expect(isDiscreteGpu({ vendor: 'amd', architecture: 'rdna-3', description: '' }, false)).toBe(false)
    expect(isDiscreteGpu({ vendor: 'amd', architecture: 'rdna-3', description: 'AMD Radeon RX 7800 XT' }, false)).toBe(true)
    expect(isDiscreteGpu({ vendor: 'intel', architecture: 'xe-lpg', description: '' }, false)).toBe(false)
    expect(isDiscreteGpu({ vendor: 'nvidia', architecture: 'lovelace', description: '' }, true)).toBe(false)
    expect(isDiscreteGpu({ vendor: 'apple', architecture: 'metal-3', description: '' }, false)).toBe(false)
  })

  it('a card with f16 keeps the f16 builds and its budget', async () => {
    stubNavigator({ requestAdapter: async () => adapter({ vendor: 'nvidia', architecture: 'lovelace', storageGB: 4 }) }, WINDOWS_EDGE)
    const d = await detectDevice({ retryDelayMs: 0 })
    expect(d.gpuBudgetMB).toBe(4096)
    expect(recommendLLM(d)!.modelId).toBe('Qwen3-4B-q4f16_1-MLC')
  })

  it('overrideDevice tries that path on another machine (development switch)', async () => {
    stubNavigator({ requestAdapter: async () => adapter() })
    const mac = await detectDevice({ retryDelayMs: 0 })
    const forced = overrideDevice(mac, { shaderF16: false, discreteGpu: true })
    expect(forced.gpuBudgetMB).toBe(DISCRETE_F32_BUDGET_MB)
    expect(recommendLLM(forced)!.modelId).toBe('Qwen3-4B-q4f32_1-MLC')
    expect(pickDtype(findEmbedding('embeddinggemma-2')!.dtype, forced)).toBe('q4')
  })
})

describe('installAdapterFallback (WebLLM worker)', () => {
  it('answers a null high-performance request with the default adapter', async () => {
    const a = rtx4070Vulkan()
    const gpu = { requestAdapter: vi.fn(async (o?: { powerPreference?: string }) => (o?.powerPreference ? null : a)) }
    expect(installAdapterFallback(gpu)).toBe(true)
    expect(installAdapterFallback(gpu)).toBe(false) // once
    expect(await gpu.requestAdapter({ powerPreference: 'high-performance' })).toBe(a)
  })

  it('requestGpuAdapter without navigator.gpu', async () => {
    expect(await requestGpuAdapter(undefined)).toEqual({ adapter: null, status: 'no-api' })
  })
})
