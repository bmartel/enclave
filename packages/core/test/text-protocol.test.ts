import { describe, expect, it } from 'vitest'
import { fromTextModel, renderMessages, TaggedStreamParser, toTextMessages } from '../src/models/text-protocol.js'
import type { Message, ModelChunk } from '../src/types.js'
import { collect } from './helpers.js'

function parseAll(pieces: string[]): ModelChunk[] {
  const p = new TaggedStreamParser()
  const out: ModelChunk[] = []
  for (const piece of pieces) out.push(...p.push(piece))
  out.push(...p.flush())
  return out
}

const text = (chunks: ModelChunk[]) => chunks.flatMap((c) => (c.type === 'text' ? [c.delta] : [])).join('')
const reasoning = (chunks: ModelChunk[]) => chunks.flatMap((c) => (c.type === 'reasoning' ? [c.delta] : [])).join('')
const calls = (chunks: ModelChunk[]) => chunks.flatMap((c) => (c.type === 'tool-call' ? [c.call] : []))

describe('TaggedStreamParser', () => {
  it('separates think, text and tool calls split across arbitrary chunk boundaries', () => {
    const raw =
      '<think>plan it</think>Looking that up.\n<tool_call>\n{"name": "search", "arguments": {"q": "x < y"}}\n</tool_call>'
    // Feed one character at a time: the worst case for tag detection.
    const chunks = parseAll([...raw])
    expect(reasoning(chunks)).toBe('plan it')
    expect(text(chunks)).toBe('Looking that up.\n')
    expect(calls(chunks)).toEqual([{ id: expect.any(String), name: 'search', input: { q: 'x < y' } }])
  })

  it('trims whitespace left behind by think and tool blocks', () => {
    expect(text(parseAll(['<think>x</think>', '\n\n', 'Answer', ' here']))).toBe('Answer here')
    expect(text(parseAll(['  \n Hi\n\nthere']))).toBe('Hi\n\nthere')
  })

  it('keeps a lone "<" that is not a tag', () => {
    expect(text(parseAll(['a <', ' b <t', 'hing>']))).toBe('a < b <thing>')
  })

  it('handles multiple calls and a missing closing tag at end of stream', () => {
    const chunks = parseAll([
      '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":"{\\"n\\":1}"}',
    ])
    expect(calls(chunks).map((c) => [c.name, c.input])).toEqual([
      ['a', {}],
      ['b', { n: 1 }],
    ])
  })

  it('reports malformed JSON as an invalid tool call so the agent can self-correct', () => {
    const [call] = calls(parseAll(['<tool_call>{not json}</tool_call>']))
    expect(call?.name).toBe('invalid_tool_call')
  })
})

describe('fromTextModel', () => {
  it('renders tools into the system prompt and flattens tool history', async () => {
    let seen: { system: string; messages: { role: string; content: string }[] } | undefined
    const model = fromTextModel({
      id: 't',
      async *streamText(req) {
        seen = req
        yield '<tool_call>{"name":"lookup","arguments":{"id":1}}</tool_call>'
      },
    })
    const chunks = await collect(
      model.stream({
        system: 'SYS',
        context: 'CTX',
        tools: [{ name: 'lookup', description: 'd', inputSchema: { type: 'object' } }],
        messages: [
          { role: 'user', content: 'hi' },
          { role: 'assistant', content: '', toolCalls: [{ id: '1', name: 'lookup', input: { id: 0 } }] },
          { role: 'tool', toolCallId: '1', name: 'lookup', content: '{"ok":true}' },
        ],
      }),
    )
    expect(seen!.system).toContain('SYS')
    expect(seen!.system).toContain('"name":"lookup"')
    // Live context rides on the newest message so the system prompt stays cacheable.
    expect(seen!.system).not.toContain('CTX')
    expect(seen!.messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user'])
    expect(seen!.messages[2]!.content).toBe('<context>\nCTX\n</context>\n\n<tool_response>\n{"ok":true}\n</tool_response>')
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: 'tool-calls' })
  })

  it('keeps the prompt byte-stable across steps so engines can reuse their KV cache', async () => {
    const seen: { system: string; messages: { role: string; content: string }[] }[] = []
    const firstReply = '<think>plan</think><tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    const outputs = [firstReply, 'Done.']
    const model = fromTextModel({
      id: 'kv',
      async *streamText(req) {
        seen.push(structuredClone({ system: req.system, messages: req.messages }))
        yield outputs.shift()!
      },
    })
    const history: Message[] = [{ role: 'user', content: 'go' }]
    const tools = [{ name: 'lookup', description: 'd', inputSchema: { type: 'object' } }]
    const step = async (context: string) => {
      const chunks = await collect(model.stream({ system: 'S', context, tools, messages: history }))
      const finish = chunks.at(-1) as Extract<ModelChunk, { type: 'finish' }>
      const calls = chunks.flatMap((c) => (c.type === 'tool-call' ? [c.call] : []))
      history.push({ role: 'assistant', content: '', ...(calls.length ? { toolCalls: calls } : {}), providerData: finish.providerData! })
      return calls
    }
    const calls = await step('state v1')
    history.push({ role: 'tool', toolCallId: calls[0]!.id, name: 'lookup', content: 'result' })
    await step('state v2')

    // Step 2 = step 1's prompt + the raw generated reply + the new tool result.
    expect(seen[1]!.system).toBe(seen[0]!.system)
    expect(seen[1]!.messages[0]).toEqual(seen[0]!.messages[0])
    expect(seen[0]!.messages[0]!.content).toContain('state v1')
    expect(seen[1]!.messages[1]).toEqual({ role: 'assistant', content: firstReply })
    expect(seen[1]!.messages[2]!.content).toContain('state v2')
  })

  it('drops reasoning from earlier turns but keeps it within the current turn', () => {
    const provider = 'text:m'
    const msgs: Message[] = [
      { role: 'user', content: 'a' },
      { role: 'assistant', content: 'A', providerData: { provider, data: { raw: '<think>old</think>\n\nA' } } },
      { role: 'user', content: 'b' },
      { role: 'assistant', content: '', providerData: { provider, data: { raw: '<think>new</think><tool_call>{}</tool_call>' } } },
      { role: 'tool', toolCallId: 'x', name: 't', content: 'r' },
    ]
    const out = renderMessages(msgs, { provider })
    expect(out[1]!.content).toBe('A')
    expect(out[3]!.content).toBe('<think>new</think><tool_call>{}</tool_call>')
  })

  it('merges consecutive same-role turns', () => {
    const out = toTextMessages([
      { role: 'user', content: 'a' },
      { role: 'user', content: 'b' },
    ])
    expect(out).toEqual([{ role: 'user', content: 'a\nb' }])
  })
})

describe('text protocol robustness', () => {
  it('parses Qwen3.5 / Qwen3-Coder XML function calls', () => {
    const chunks = parseAll([
      '<tool_call>\n<function=execute_sql>\n<parameter=sql>\nselect * from t limit 5\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>',
    ])
    expect(calls(chunks).map((c) => [c.name, c.input])).toEqual([['execute_sql', { sql: 'select * from t limit 5', limit: 5 }]])
  })

  it('drops a hallucinated tool response and everything after it', () => {
    const chunks = parseAll(['<tool_call>{"name":"a","arguments":{}}</tool_call>\n<tool_res', 'ponse>{"fake": true}</tool_response> The answer is 42.'])
    expect(calls(chunks)).toHaveLength(1)
    expect(text(chunks)).toBe('')
  })

  it('passes stop sequences to the backend only when tools are offered', async () => {
    const stops: (string[] | undefined)[] = []
    const model = fromTextModel({
      id: 't',
      contextWindow: 4096,
      async *streamText(req) {
        stops.push(req.stop)
        yield 'ok'
      },
    })
    expect(model.contextWindow).toBe(4096)
    await collect(model.stream({ system: '', messages: [], tools: [] }))
    await collect(model.stream({ system: '', messages: [], tools: [{ name: 'x', description: '', inputSchema: {} }] }))
    expect(stops).toEqual([undefined, ['<tool_response>']])
  })
})

describe('context in history', () => {
  const user = (content: string) => ({ role: 'user' as const, content })
  const call = { role: 'assistant' as const, content: '', toolCalls: [{ id: 'c', name: 'q', input: {} }] }
  const tool = (content: string) => ({ role: 'tool' as const, toolCallId: 'c', name: 'q', content })

  it('attaches context once per turn unless it changes', () => {
    const sent = new WeakMap()
    const u = user('how many orders?')
    renderMessages([u], { sent, context: 'schema v1' })
    const t1 = tool('[1]')
    const step2 = renderMessages([u, call, t1], { sent, context: 'schema v1' })
    expect(step2.map((m) => m.content).join('\n').match(/<context>/g)).toHaveLength(1)
    const step3 = renderMessages([u, call, t1, call, tool('[2]')], { sent, context: 'schema v2' })
    expect(step3.at(-1)!.content).toContain('schema v2') // changed: re-attached
  })

  it("drops earlier turns' context in current-turn mode and keeps it verbatim otherwise", () => {
    const sent = new WeakMap()
    const first = user('what is the wifi password?')
    renderMessages([first], { sent, context: 'passages about wifi' })
    const answer = { role: 'assistant' as const, content: 'maple-harbor-42' }
    const next = user('and parking?')
    const history = [first, answer, next]
    const dropped = renderMessages(history, { sent, context: 'passages about parking', dropOldContext: true })
    expect(dropped[0]!.content).toBe('what is the wifi password?')
    expect(dropped.at(-1)!.content).toContain('passages about parking')
    const kept = renderMessages(history, { sent, context: 'passages about parking' })
    expect(kept[0]!.content).toContain('passages about wifi')
  })
})
