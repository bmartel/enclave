import { describe, expect, it } from 'vitest'
import { fitHistory } from '../src/agent.js'
import { fromTextModel, type TextProtocolOptions } from '../src/models/text-protocol.js'
import type { Message, ModelChunk } from '../src/types.js'
import { collect } from './helpers.js'

/** Drive a text model through several user turns, like the agent loop does. */
async function converse(turns: number, options: TextProtocolOptions, contextWindow?: number) {
  const prompts: { role: string; content: string }[][] = []
  const outputs: string[] = []
  let n = 0
  const model = fromTextModel(
    {
      id: 'stub',
      ...(contextWindow ? { contextWindow } : {}),
      async *streamText(req) {
        prompts.push(structuredClone(req.messages))
        n++
        const out = `<think>${'reasoning '.repeat(40)}#${n}</think>Answer ${n}.`
        outputs.push(out)
        yield out
      },
    },
    options,
  )
  const history: Message[] = []
  const metrics: (number | undefined)[] = []
  for (let t = 1; t <= turns; t++) {
    history.push({ role: 'user', content: `question ${t}` })
    const chunks = await collect(model.stream({ system: 'S', tools: [], messages: history }))
    const finish = chunks.at(-1) as Extract<ModelChunk, { type: 'finish' }>
    metrics.push(finish.metrics?.promptChars)
    history.push({ role: 'assistant', content: `Answer ${t}.`, providerData: finish.providerData! })
  }
  return { prompts, outputs, history, metrics }
}

/** A request can reuse the KV cache when it extends the previous prompt + reply exactly. */
function extendsPrevious(prev: { content: string }[], next: { content: string }[], reply: string) {
  return JSON.stringify([...prev.map((m) => m.content), reply]) === JSON.stringify(next.slice(0, -1).map((m) => m.content))
}

describe('reasoning history', () => {
  it('current-turn drops earlier reasoning (cache rebuilt every turn)', async () => {
    const { prompts, outputs } = await converse(3, { reasoningHistory: 'current-turn' })
    expect(prompts[2]![1]!.content).toBe('Answer 1.')
    expect(extendsPrevious(prompts[1]!, prompts[2]!, outputs[1]!)).toBe(false)
  })

  it('all keeps reasoning so every turn extends the previous prompt byte for byte', async () => {
    const { prompts, outputs } = await converse(4, { reasoningHistory: 'all' })
    for (let t = 1; t < 4; t++) expect(extendsPrevious(prompts[t - 1]!, prompts[t]!, outputs[t - 1]!)).toBe(true)
  })

  it('auto accumulates, compacts past the threshold, and only then rebuilds the cache', async () => {
    // ~450 chars per turn; a 600-token window compacts at ~1150 chars.
    const { prompts, outputs, metrics } = await converse(7, { reasoningHistory: 'auto' }, 600)
    const withThink = prompts.map((p) => p.filter((m) => m.content.includes('<think>')).length)
    const compactions = withThink.flatMap((c, i) => (i > 0 && c < withThink[i - 1]! ? [i] : []))
    const reuse = prompts.slice(1).map((p, i) => extendsPrevious(prompts[i]!, p, outputs[i]!))

    expect(withThink.slice(0, 3)).toEqual([0, 1, 2]) // reasoning accumulates across turns
    expect(compactions.length).toBeGreaterThanOrEqual(1)
    for (const i of compactions) expect(withThink[i]).toBe(0) // compaction drops all older reasoning at once
    // The cache is rebuilt exactly at compactions, and reused on every other turn.
    expect(reuse.flatMap((r, i) => (r ? [] : [i + 1]))).toEqual(compactions)
    expect(reuse.filter(Boolean).length).toBeGreaterThan(compactions.length)
    // The prompt stays bounded by the window.
    expect(Math.max(...(metrics as number[]))).toBeLessThan(600 * 3.2)
  })

  it('auto never compacts mid-turn, so tool steps always extend the cache', async () => {
    // Sweep window sizes so that, for some of them, the threshold is crossed on a
    // tool-result step rather than at the start of a turn.
    for (let contextWindow = 300; contextWindow <= 1500; contextWindow += 50) {
      const prompts: { content: string }[][] = []
      const replies: string[] = []
      const model = fromTextModel(
        {
          id: 'stub',
          contextWindow,
          async *streamText(req) {
            prompts.push(structuredClone(req.messages))
            const out = `<think>${'r'.repeat(300)}</think>` + (req.after === 'user' ? '<tool_call>{"name":"t","arguments":{}}</tool_call>' : 'done')
            replies.push(out)
            yield out
          },
        },
        { reasoningHistory: 'auto' },
      )
      const history: Message[] = []
      for (let turn = 1; turn <= 4; turn++) {
        history.push({ role: 'user', content: `q${turn}` })
        for (let step = 1; step <= 2; step++) {
          const chunks = await collect(model.stream({ system: 'S', tools: [{ name: 't', description: '', inputSchema: {} }], messages: history }))
          const finish = chunks.at(-1) as Extract<ModelChunk, { type: 'finish' }>
          const calls = chunks.flatMap((c) => (c.type === 'tool-call' ? [c.call] : []))
          history.push({ role: 'assistant', content: step === 2 ? 'done' : '', ...(calls.length ? { toolCalls: calls } : {}), providerData: finish.providerData! })
          if (calls.length) history.push({ role: 'tool', toolCallId: calls[0]!.id, name: 't', content: 'ok' })
        }
      }
      // The step after a tool result must always extend the previous prompt exactly.
      for (let i = 1; i < prompts.length; i += 2) {
        expect(extendsPrevious(prompts[i - 1]!, prompts[i]!, replies[i - 1]!), `window ${contextWindow}, step ${i}`).toBe(true)
      }
    }
  })

  it('auto without a known context window behaves like all', async () => {
    const { prompts } = await converse(3, { reasoningHistory: 'auto' })
    expect(prompts[2]!.filter((m) => m.content.includes('<think>'))).toHaveLength(2)
  })

  it('history budgeting counts replayed reasoning, not just the visible answer', () => {
    const big = '<think>' + 'x'.repeat(3000) + '</think>ok'
    const history: Message[] = [
      { role: 'user', content: 'a' },
      { role: 'assistant', content: 'ok', providerData: { provider: 'p', data: { raw: big } } },
      { role: 'user', content: 'b' },
    ]
    // The visible answer is tiny, but the raw turn doesn't fit a 1000-char budget.
    expect(fitHistory(history, 100, 1000)).toEqual([history[2]])
  })
})
