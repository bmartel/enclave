import { describe, expect, it } from 'vitest'
import { discoverLocalModels, lmstudio, localEmbedder, ollama } from '../src/models/local.js'
import { collect } from './helpers.js'

const ndjson = (lines: unknown[]) =>
  new Response(lines.map((l) => JSON.stringify(l)).join('\n') + '\n', { status: 200 })

describe('ollama', () => {
  it('streams thinking, text and tool calls over the native API with num_ctx', async () => {
    let body: any
    const model = ollama({
      model: 'qwen3:8b',
      contextWindow: 12000,
      think: true,
      fetch: async (url, init) => {
        expect(String(url)).toBe('http://localhost:11434/api/chat')
        body = JSON.parse(String(init!.body))
        return ndjson([
          { message: { role: 'assistant', thinking: 'hmm' } },
          { message: { role: 'assistant', content: 'Checking.' } },
          { message: { role: 'assistant', content: '', tool_calls: [{ function: { name: 'execute_sql', arguments: { sql: 'select 1' } } }] } },
          { done: true, done_reason: 'stop', prompt_eval_count: 50, eval_count: 9 },
        ])
      },
    })
    expect(model.contextWindow).toBe(12000)
    const chunks = await collect(
      model.stream({
        system: 'S',
        tools: [{ name: 'execute_sql', description: 'd', inputSchema: { type: 'object' } }],
        messages: [
          { role: 'user', content: 'q' },
          { role: 'assistant', content: '', toolCalls: [{ id: 'a', name: 'execute_sql', input: { sql: 'x' } }] },
          { role: 'tool', toolCallId: 'a', name: 'execute_sql', content: '[]' },
        ],
      }),
    )
    expect(body.options.num_ctx).toBe(12000)
    expect(body.think).toBe(true)
    expect(body.messages.at(-1)).toEqual({ role: 'tool', tool_name: 'execute_sql', content: '[]' })
    expect(body.messages[2].tool_calls[0].function).toEqual({ name: 'execute_sql', arguments: { sql: 'x' } })
    expect(chunks.map((c) => c.type)).toEqual(['reasoning', 'text', 'tool-call', 'finish'])
    expect(chunks.at(-1)).toEqual({ type: 'finish', reason: 'tool-calls', usage: { inputTokens: 50, outputTokens: 9 } })
  })

  it('explains unreachable servers', async () => {
    const model = ollama({ model: 'm', fetch: async () => { throw new TypeError('Failed to fetch') } })
    await expect(collect(model.stream({ system: '', messages: [], tools: [] }))).rejects.toThrow(/Is it running/)
  })
})

describe('lmstudio', () => {
  it('targets the OpenAI-compatible endpoint', async () => {
    let url = ''
    const model = lmstudio({
      model: 'qwen3-4b',
      contextWindow: 8192,
      fetch: async (u) => {
        url = String(u)
        return new Response('data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
      },
    })
    const chunks = await collect(model.stream({ system: '', messages: [{ role: 'user', content: 'x' }], tools: [] }))
    expect(url).toBe('http://localhost:1234/v1/chat/completions')
    expect(chunks[0]).toEqual({ type: 'text', delta: 'hi' })
    expect(model.contextWindow).toBe(8192)
  })
})

describe('discoverLocalModels', () => {
  it('lists Ollama and LM Studio models and tolerates missing servers', async () => {
    const fetchMock = async (url: string | URL | Request, init?: RequestInit) => {
      const u = String(url)
      if (u.endsWith('/api/tags')) {
        return Response.json({
          models: [
            { name: 'qwen3:8b', size: 5e9, details: { parameter_size: '8.2B', quantization_level: 'Q4_K_M' } },
            { name: 'embeddinggemma:latest', size: 6e8 },
          ],
        })
      }
      if (u.endsWith('/api/show')) {
        const { model } = JSON.parse(String(init!.body))
        return Response.json(
          model === 'qwen3:8b'
            ? { capabilities: ['completion', 'tools', 'thinking'], model_info: { 'qwen3.context_length': 40960 } }
            : { capabilities: ['embedding'] },
        )
      }
      if (u.endsWith('/api/v0/models')) {
        return Response.json({ data: [{ id: 'google/gemma-3-4b', type: 'vlm', state: 'loaded', max_context_length: 131072 }, { id: 'text-embedding-nomic', type: 'embeddings' }] })
      }
      throw new TypeError('unreachable')
    }
    const models = await discoverLocalModels({ fetch: fetchMock as typeof fetch })
    expect(models).toEqual([
      expect.objectContaining({ provider: 'ollama', id: 'qwen3:8b', kind: 'llm', contextLength: 40960, parameterSize: '8.2B', capabilities: ['completion', 'tools', 'thinking'] }),
      expect.objectContaining({ provider: 'ollama', id: 'embeddinggemma:latest', kind: 'embedding' }),
      expect.objectContaining({ provider: 'lmstudio', id: 'google/gemma-3-4b', kind: 'llm', loaded: true, contextLength: 131072 }),
      expect.objectContaining({ provider: 'lmstudio', id: 'text-embedding-nomic', kind: 'embedding' }),
    ])

    const none = await discoverLocalModels({ fetch: (async () => { throw new TypeError('down') }) as typeof fetch })
    expect(none).toEqual([])
  })

  it('embeds via a local server', async () => {
    const e = localEmbedder({
      provider: 'ollama',
      model: 'embeddinggemma',
      dimensions: 2,
      queryPrefix: 'q: ',
      fetch: (async (_u: unknown, init?: RequestInit) => {
        const { input } = JSON.parse(String(init!.body))
        expect(input).toEqual(['q: hello'])
        return Response.json({ data: [{ index: 0, embedding: [1, 0] }] })
      }) as typeof fetch,
    })
    expect(await e.embed(['hello'], 'query')).toEqual([[1, 0]])
  })
})
