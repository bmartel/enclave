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
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: 'tool-calls', usage: { inputTokens: 50, outputTokens: 9 } })
  })

  const recorder = (capabilities: string[], done: Record<string, unknown> = {}) => {
    const bodies: any[] = []
    const fetch = async (url: string | URL | Request, init?: RequestInit) => {
      if (String(url).endsWith('/api/show')) return Response.json({ capabilities })
      bodies.push(JSON.parse(String(init!.body)))
      return ndjson([{ message: { content: 'ok' } }, { done: true, done_reason: 'stop', eval_count: 5, prompt_eval_count: 40, ...done }])
    }
    return { bodies, fetch }
  }

  it('keeps context off the system prompt and frozen for the turn, so steps share a cache prefix', async () => {
    const { bodies, fetch } = recorder(['completion', 'tools'])
    const model = ollama({ model: 'm', fetch })
    const user = { role: 'user' as const, content: 'how many orders?' }
    const call = { role: 'assistant' as const, content: '', toolCalls: [{ id: 'a', name: 'q', input: {} }] }
    const tool = { role: 'tool' as const, toolCallId: 'a', name: 'q', content: '[42]' }
    await collect(model.stream({ system: 'S', context: 'schema v1', tools: [], messages: [user] }))
    await collect(model.stream({ system: 'S', context: 'schema v1', tools: [], messages: [user, call, tool] }))
    expect(bodies[0].messages[0]).toEqual({ role: 'system', content: 'S' })
    expect(bodies[0].messages[1].content).toBe('<context>\nschema v1\n</context>\n\nhow many orders?')
    // Step 2's prompt starts with step 1's prompt, byte for byte.
    const first = JSON.stringify(bodies[0].messages)
    expect(JSON.stringify(bodies[1].messages.slice(0, 2))).toBe(first)

    // A new turn: the earlier turn's message is sent without its old context.
    const next = { role: 'user' as const, content: 'and last month?' }
    await collect(model.stream({ system: 'S', context: 'schema v2', tools: [], messages: [user, call, tool, { role: 'assistant', content: '42' }, next] }))
    expect(bodies[2].messages[1].content).toBe('how many orders?')
    expect(bodies[2].messages.at(-1).content).toContain('schema v2')
  })

  it("thinks on new requests only, and only when the model can ('auto')", async () => {
    const thinking = recorder(['completion', 'tools', 'thinking'])
    const model = ollama({ model: 'qwen', fetch: thinking.fetch })
    const user = { role: 'user' as const, content: 'q' }
    await collect(model.stream({ system: '', tools: [], messages: [user] }))
    await collect(model.stream({ system: '', tools: [], messages: [user, { role: 'assistant', content: '', toolCalls: [{ id: 'a', name: 't', input: {} }] }, { role: 'tool', toolCallId: 'a', name: 't', content: 'r' }] }))
    expect(thinking.bodies.map((b) => b.think)).toEqual([true, false])

    const plain = recorder(['completion', 'tools'])
    await collect(ollama({ model: 'llama', fetch: plain.fetch }).stream({ system: '', tools: [], messages: [user] }))
    expect(plain.bodies[0].think).toBe(false)
  })

  it('reports prompt read time and decode speed (prompt_eval_count includes cached tokens)', async () => {
    const { fetch } = recorder([], { prompt_eval_count: 1424, prompt_eval_duration: 287e6, eval_count: 30, eval_duration: 1e9 })
    const chunks = await collect(ollama({ model: 'm', fetch }).stream({ system: 'x', tools: [], messages: [{ role: 'user', content: 'q' }] }))
    const finish = chunks.at(-1) as unknown as { metrics: Record<string, unknown> }
    expect(finish.metrics).toMatchObject({ prefillMs: 287, decodeTokensPerSec: 30 })
    expect(finish.metrics).not.toHaveProperty('kvCacheReused')
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

describe('measured Ollama presets', () => {
  it('builds embedders with their measured prompts, dimensions and floors', async () => {
    let body: any
    const { ollamaEmbedder } = await import('../src/models/local.js')
    const e = ollamaEmbedder('qwen3-embedding:4b', {
      fetch: async (_u, init) => {
        body = JSON.parse(String(init!.body))
        return Response.json({ data: [{ index: 0, embedding: Array.from({ length: 2560 }, (_, i) => (i < 1024 ? 1 : 5)) }] })
      },
    })
    const [v] = await e.embed(['vacation days'], 'query')
    expect(body.input[0]).toBe('Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:vacation days')
    expect(v).toHaveLength(1024) // Matryoshka-truncated and re-normalized
    expect(Math.hypot(...v!)).toBeCloseTo(1, 6)
    expect(e.relevanceFloor).toBe(0.43)
    expect(() => ollamaEmbedder('unknown-model')).toThrow(/No preset/)
  })

  it('recommends the best installed model', async () => {
    const { recommendOllamaModel } = await import('../src/models/local.js')
    const m = (id: string) => ({ provider: 'ollama' as const, id, label: id, kind: 'llm' as const })
    expect(recommendOllamaModel([m('qwen3.5:9b'), m('qwen3.8:27b-q4_K_M')])?.tag).toBe('qwen3.8:27b-q4_K_M')
    expect(recommendOllamaModel([m('qwen3.5:9b'), m('qwen3.8:27b-q4_K_M')], 'fast')?.tag).toBe('qwen3.5:9b')
    expect(recommendOllamaModel([m('llama3:8b')])).toBeUndefined()
  })
})

describe('lmstudio context placement', () => {
  it('keeps context off the system prompt so steps share a cache prefix (OpenAI default is unchanged)', async () => {
    const bodies: any[] = []
    const fetch = async (_u: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init!.body)))
      return new Response('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    }
    const user = { role: 'user' as const, content: 'how many orders?' }
    const call = { role: 'assistant' as const, content: '', toolCalls: [{ id: 'a', name: 'q', input: {} }] }
    const tool = { role: 'tool' as const, toolCallId: 'a', name: 'q', content: '[42]' }
    const model = lmstudio({ model: 'qwen/qwen3.6-27b', fetch })
    await collect(model.stream({ system: 'S', context: 'schema v1', tools: [], messages: [user] }))
    await collect(model.stream({ system: 'S', context: 'schema v1', tools: [], messages: [user, call, tool] }))
    expect(bodies[0].messages[0]).toEqual({ role: 'system', content: 'S' })
    expect(bodies[0].messages[1].content).toContain('<context>\nschema v1')
    expect(JSON.stringify(bodies[1].messages.slice(0, 2))).toBe(JSON.stringify(bodies[0].messages))

    const { openaiCompatible } = await import('../src/models/openai.js')
    const remote = openaiCompatible({ baseURL: 'https://api.example/v1', model: 'm', fetch })
    await collect(remote.stream({ system: 'S', context: 'schema v1', tools: [], messages: [user] }))
    expect(bodies[2].messages[0].content).toBe('S\n\nschema v1')
  })

  it("turns reasoning off after tool results ('auto') via reasoning_effort", async () => {
    const bodies: any[] = []
    const fetch = async (_u: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init!.body)))
      return new Response('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    }
    const user = { role: 'user' as const, content: 'q' }
    const model = lmstudio({ model: 'm', fetch })
    await collect(model.stream({ system: '', tools: [], messages: [user] }))
    await collect(model.stream({ system: '', tools: [], messages: [user, { role: 'assistant', content: '', toolCalls: [{ id: 'a', name: 't', input: {} }] }, { role: 'tool', toolCallId: 'a', name: 't', content: 'r' }] }))
    await collect(lmstudio({ model: 'm', fetch, think: false }).stream({ system: '', tools: [], messages: [user] }))
    expect(bodies.map((b) => b.reasoning_effort)).toEqual([undefined, 'none', 'none'])
  })
})
