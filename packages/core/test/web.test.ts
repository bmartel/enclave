import { describe, expect, it } from 'vitest'
import { estimateGpuBudget, recommendEmbedding, recommendLLM, rankLLMs, resolveLLM, findLLM, type DeviceProfile } from '../src/web/index.js'
import { resolveWebLLMId } from '../src/models/webllm.js'

function device(p: Partial<DeviceProfile>): DeviceProfile {
  const base: DeviceProfile = {
    webgpu: true,
    shaderF16: true,
    adapter: undefined,
    maxStorageBufferMB: 2048,
    deviceMemoryGB: 8,
    mobile: false,
    gpuBudgetMB: 0,
    crossOriginIsolated: false,
    storage: undefined,
  }
  const d = { ...base, ...p }
  if (p.gpuBudgetMB === undefined) d.gpuBudgetMB = estimateGpuBudget(d)
  return d
}

describe('model selection', () => {
  it('picks Qwen3 4B with an 8k window on a typical 8 GB desktop with f16', () => {
    const choice = recommendLLM(device({}))!
    expect(choice.modelId).toBe('Qwen3-4B-q4f16_1-MLC')
    expect(choice.contextWindow).toBe(8192)
    expect(choice.estimatedMB).toBeLessThanOrEqual(4096)
  })

  it('uses the full 16k window when memory allows, and 8B on big budgets', () => {
    expect(recommendLLM(device({ gpuBudgetMB: 6000 }))).toMatchObject({ modelId: 'Qwen3-4B-q4f16_1-MLC', contextWindow: 16384 })
    expect(recommendLLM(device({ gpuBudgetMB: 12000 }))!.modelId).toBe('Qwen3-4B-q4f16_1-MLC') // 8B exceeds the download cap
    expect(recommendLLM(device({ gpuBudgetMB: 12000 }), { maxDownloadMB: 5000 })!.modelId).toBe('Qwen3-8B-q4f16_1-MLC')
  })

  it('falls back to q4f32 builds without shader-f16', () => {
    const choice = recommendLLM(device({ shaderF16: false }))!
    expect(choice.modelId).toMatch(/q4f32_1-MLC$/)
    expect(choice.estimatedMB).toBeLessThanOrEqual(4096)
  })

  it('keeps phones on small models and skips large-buffer models on weak adapters', () => {
    expect(recommendLLM(device({ mobile: true, deviceMemoryGB: 4 }))!.preset.id).toBe('qwen3-0.6b')
    const weak = rankLLMs(device({ maxStorageBufferMB: 512, gpuBudgetMB: 8000 }))
    expect(rankLLMs(device({ gpuBudgetMB: 8000 }), { maxDownloadMB: 1000 }).every((c) => c.preset.downloadMB <= 1000)).toBe(true)
    expect(weak.some((c) => c.preset.needsLargeBuffers)).toBe(false)
  })

  it('uses a Transformers.js model on WASM when there is no WebGPU', () => {
    const choice = recommendLLM(device({ webgpu: false }))!
    expect(choice.preset.runtime).toBe('transformers')
    expect(choice.dtype).toBe('q4')
    expect(recommendEmbedding(device({ webgpu: false })).id).toBe('granite-multilingual-r2')
    expect(recommendEmbedding(device({})).id).toBe('embeddinggemma')
  })

  it('resolves preset ids to concrete builds', () => {
    expect(resolveWebLLMId('qwen3-1.7b')).toBe('Qwen3-1.7B-q4f16_1-MLC')
    expect(resolveWebLLMId('Llama-3.2-1B-Instruct-q4f16_1-MLC')).toBe('Llama-3.2-1B-Instruct-q4f16_1-MLC')
    expect(() => resolveWebLLMId('tjs-qwen3-0.6b')).toThrow()
    expect(resolveLLM(findLLM('tjs-lfm2-1.2b')!, device({}))).toMatchObject({ modelId: 'onnx-community/LFM2-1.2B-ONNX', dtype: 'q4f16' })
  })
})
