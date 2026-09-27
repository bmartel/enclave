import { describe, expect, it } from 'vitest'
import { openaiCompatible } from '../src/models/openai.js'
import { collect } from './helpers.js'

function sse(events: unknown[]): Response {
  const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n'
  // Split mid-event to exercise buffering.
  const bytes = new TextEncoder().encode(body)
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      for (let i = 0; i < bytes.length; i += 7) c.enqueue(bytes.slice(i, i + 7))
      c.close()
    },
  })
  return new Response(stream, { status: 200 })
}

describe('openaiCompatible', () => {
  it('streams text and assembles tool call deltas', async () => {
    let body: any
    const model = openaiCompatible({
      baseURL: 'http://local/v1/',
      model: 'm',
      fetch: async (_url, init) => {
        body = JSON.parse(String(init!.body))
        return sse([
          { choices: [{ delta: { reasoning_content: 'hmm' } }] },
          { choices: [{ delta: { content: 'Hi' } }] },
          { choices: [{ delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'look', arguments: '{"a"' } }] } }] },
          { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: ':1}' } }] }, finish_reason: 'tool_calls' }] },
          { choices: [], usage: { prompt_tokens: 3, completion_tokens: 4 } },
        ])
      },
    })
    const chunks = await collect(
      model.stream({
        system: 'S',
        context: 'C',
        tools: [{ name: 'look', description: 'd', inputSchema: { type: 'object' } }],
        messages: [
          { role: 'user', content: 'q' },
          { role: 'assistant', content: '', toolCalls: [{ id: 'p', name: 'look', input: { a: 0 } }] },
          { role: 'tool', toolCallId: 'p', name: 'look', content: 'r' },
        ],
      }),
    )
    expect(chunks).toEqual([
      { type: 'reasoning', delta: 'hmm' },
      { type: 'text', delta: 'Hi' },
      { type: 'tool-call', call: { id: 'c1', name: 'look', input: { a: 1 } } },
      { type: 'finish', reason: 'tool-calls', usage: { inputTokens: 3, outputTokens: 4 } },
    ])
    expect(body.messages[0]).toEqual({ role: 'system', content: 'S\n\nC' })
    expect(body.messages[2].tool_calls[0].function).toEqual({ name: 'look', arguments: '{"a":0}' })
    expect(body.messages[3]).toEqual({ role: 'tool', tool_call_id: 'p', content: 'r' })
    expect(body.tools[0].function.name).toBe('look')
  })

  it('surfaces HTTP errors', async () => {
    const model = openaiCompatible({ baseURL: 'http://x', model: 'm', fetch: async () => new Response('nope', { status: 500 }) })
    await expect(collect(model.stream({ system: '', messages: [], tools: [] }))).rejects.toThrow(/500/)
  })
})
