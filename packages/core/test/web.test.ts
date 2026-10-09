import { describe, expect, it } from 'vitest'
import { estimateGpuBudget, findEmbedding, recommendEmbedding, recommendLLM, rankLLMs, resolveLLM, findLLM, type DeviceProfile } from '../src/web/index.js'
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
    expect(recommendEmbedding(device({})).id).toBe('embeddinggemma-2')
  })

  it('resolves preset ids to concrete builds', () => {
    expect(resolveWebLLMId('qwen3-1.7b')).toBe('Qwen3-1.7B-q4f16_1-MLC')
    expect(resolveWebLLMId('Llama-3.2-1B-Instruct-q4f16_1-MLC')).toBe('Llama-3.2-1B-Instruct-q4f16_1-MLC')
    expect(() => resolveWebLLMId('tjs-qwen3-0.6b')).toThrow()
    expect(resolveLLM(findLLM('tjs-lfm2-1.2b')!, device({}))).toMatchObject({ modelId: 'onnx-community/LFM2-1.2B-ONNX', dtype: 'q4f16' })
  })
})

describe('tool-call grammar', () => {
  it('builds an xgrammar structural tag per tool', async () => {
    const { toolCallStructuralTag } = await import('../src/models/webllm.js')
    const tag = toolCallStructuralTag([
      { name: 'execute_sql', description: '', inputSchema: { type: 'object', properties: { sql: { type: 'string' } }, required: ['sql'] } },
    ]) as any
    expect(tag.format.triggers).toEqual(['<tool_call>'])
    expect(tag.format.tags[0]).toEqual({
      type: 'tag',
      begin: '<tool_call>\n{"name": "execute_sql", "arguments": ',
      content: { type: 'json_schema', json_schema: { type: 'object', properties: { sql: { type: 'string' } }, required: ['sql'] } },
      end: '}\n</tool_call>',
    })
  })

  it('drops string length limits, which stop the model writing escapes like \\n', async () => {
    const { toolCallStructuralTag, grammarSchema } = await import('../src/models/webllm.js')
    const schema = {
      type: 'object',
      properties: {
        code: { type: 'string', minLength: 1, maxLength: 10000, description: 'A program' },
        lines: { type: 'array', minItems: 1, items: { type: 'string', minLength: 1 } },
        mode: { type: 'string', enum: ['a', 'b'] },
        nested: { anyOf: [{ type: 'string', minLength: 2 }, { type: 'null' }] },
        minLength: { type: 'number' },
      },
      required: ['code'],
    }
    expect(grammarSchema(schema)).toEqual({
      type: 'object',
      properties: {
        code: { type: 'string', description: 'A program' },
        lines: { type: 'array', minItems: 1, items: { type: 'string' } },
        mode: { type: 'string', enum: ['a', 'b'] },
        nested: { anyOf: [{ type: 'string' }, { type: 'null' }] },
        minLength: { type: 'number' },
      },
      required: ['code'],
    })
    const tag = toolCallStructuralTag([{ name: 'run_code', description: '', inputSchema: schema }]) as any
    expect(JSON.stringify(tag)).not.toMatch(/"minLength":\d/)
  })
})

describe('embedding presets', () => {
  it('embeds queries with the prompt of their task, documents with the template', async () => {
    const { queryPrefix } = await import('../src/transformers/index.js')
    for (const id of ['embeddinggemma-2', 'embeddinggemma']) {
      const p = findEmbedding(id)!
      expect(queryPrefix(p)).toBe('task: search result | query: ')
      expect(queryPrefix(p, 'question-answering')).toBe('task: question answering | query: ')
      expect(queryPrefix(p, 'code-retrieval')).toBe('task: code retrieval | query: ')
      expect(queryPrefix(p, 'similarity')).toBe('task: sentence similarity | query: ')
    }
    // Models without task prompts use their one query prefix for every task.
    expect(queryPrefix(findEmbedding('granite-small-r2')!, 'clustering')).toBe('')
    expect(queryPrefix(findEmbedding('qwen3-embedding-0.6b')!, 'clustering')).toMatch(/^Instruct:/)
  })

  it('loads only the text model of EmbeddingGemma 2: q4 on GPUs, the fp16 export on WASM (no quantized build runs there)', () => {
    const p = findEmbedding('embeddinggemma-2')!
    expect(p).toMatchObject({ textOnly: true, license: 'apache-2.0', maxTokens: 8192, matryoshka: [768, 512, 256, 128] })
    expect(p.dtype).toEqual({ webgpu: 'q4', webgpuF32: 'q4', wasm: 'fp16' })
    expect(p.webgpuMaxBatchTokens).toBeLessThan(2700)
  })
})

describe('token-budgeted batches', () => {
  it('keeps batch size × longest text within the budget, in order', async () => {
    const { groupByTokens } = await import('../src/transformers/runtime.js')
    const len = (t: string) => t.length
    expect(groupByTokens(['aa', 'bb', 'cc', 'dd'], len, 4)).toEqual([['aa', 'bb'], ['cc', 'dd']])
    // A long text raises the padded size of its group.
    expect(groupByTokens(['a', 'b', 'cccc', 'd'], len, 6)).toEqual([['a', 'b'], ['cccc'], ['d']])
    // Alone over budget: a group of one, never dropped.
    expect(groupByTokens(['xxxxxxxx', 'y'], len, 4)).toEqual([['xxxxxxxx'], ['y']])
    expect(groupByTokens([], len, 4)).toEqual([])
  })
})

describe('model cache checks', () => {
  it('finds config.json under the URL Transformers.js caches it by (browsers have no local models)', async () => {
    const { cachedConfigUrl } = await import('../src/transformers/runtime.js')
    expect(cachedConfigUrl({ remoteHost: 'https://huggingface.co/', remotePathTemplate: '{model}/resolve/{revision}/' }, 'onnx-community/embeddinggemma-2-ONNX')).toBe(
      'https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main/config.json',
    )
    // Self-hosted mirrors (enclave-mirror's layout).
    expect(cachedConfigUrl({ remoteHost: 'https://cdn.example.com/models/hf', remotePathTemplate: '/{model}/resolve/{revision}' }, 'org/m')).toBe(
      'https://cdn.example.com/models/hf/org/m/resolve/main/config.json',
    )
  })
})
