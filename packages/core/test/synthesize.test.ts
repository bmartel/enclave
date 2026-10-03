import { describe, expect, it } from 'vitest'
import { citedNumbers, synthesize, type SynthesisEvent } from '../src/index.js'
import { mockModel } from '../src/testing.js'

const collect = async (gen: AsyncGenerator<SynthesisEvent>) => {
  const events: SynthesisEvent[] = []
  for await (const e of gen) events.push(e)
  return events
}

describe('synthesize', () => {
  it('writes in one pass when the sources fit, citing numbered passages', async () => {
    const model = mockModel(['Herons nest in spring [1]. Frogs breed in ponds [2].'])
    const events = await collect(
      synthesize({
        model,
        instruction: 'Write a briefing.',
        sources: [
          { id: 'a', title: 'Birds', content: 'Herons nest in spring.' },
          { id: 'b', title: 'Ponds', content: 'Frogs breed in ponds.' },
          { id: 'c', title: 'Other', content: 'Unrelated.' },
        ],
      }),
    )
    expect(model.requests).toHaveLength(1)
    expect(model.requests[0]!.system).toMatch(/^Write a briefing\./)
    expect(model.requests[0]!.messages[0]!.content).toContain('[1] (Birds)\nHerons nest in spring.')
    const finish = events.at(-1) as Extract<SynthesisEvent, { type: 'finish' }>
    expect(finish.text).toBe('Herons nest in spring [1]. Frogs breed in ponds [2].')
    expect(finish.citations.map((c) => c.documentId)).toEqual(['a', 'b'])
    expect(finish.passages).toHaveLength(3)
    expect(events.filter((e) => e.type === 'text-delta').map((e) => (e as { delta: string }).delta).join('')).toBe(finish.text)
  })

  it('reads in batches, condenses notes, then writes when sources exceed the context', async () => {
    const sources = Array.from({ length: 12 }, (_, i) => ({ id: `d${i}`, title: `Doc ${i}`, content: `Fact number ${i}. `.repeat(60) }))
    const model = mockModel([
      ...Array.from({ length: 30 }, (_, i) => (req: { system: string; messages: { content: string }[] }) =>
        req.system.startsWith('You read')
          ? `- a fact ${'x'.repeat(1500)} [${(req.messages[0]!.content.match(/\[(\d+)\]/) ?? [])[1]}]`
          : req.system.startsWith('You merge')
            ? `- merged ${'y'.repeat(400)} [1][5]`
            : `Final with [1] and [5] and [99].`,
      ),
    ])
    // A tiny window forces several reading batches and a merge round.
    const events = await collect(synthesize({ model: Object.assign(model, { contextWindow: 2200 }), instruction: 'Write a timeline.', focus: 'dates', sources, outputTokens: 600 }))
    const stages = events.filter((e) => e.type === 'progress').map((e) => (e as { stage: string }).stage)
    expect(stages).toContain('reading')
    expect(stages).toContain('condensing')
    expect(stages.at(-1)).toBe('writing')
    const reading = model.requests.filter((r) => r.system.startsWith('You read'))
    expect(reading.length).toBeGreaterThan(2)
    expect(reading[0]!.messages[0]!.content).toMatch(/^Writer's goal: dates/)
    const write = model.requests.at(-1)!
    expect(write.system).toMatch(/^Write a timeline\./)
    expect(write.messages[0]!.content).toMatch(/^Notes taken from the numbered sources/)
    const finish = events.at(-1) as Extract<SynthesisEvent, { type: 'finish' }>
    // [99] doesn't exist, so only real passages come back.
    expect(finish.citations.map((c) => c.n)).toEqual([1, 5])
  })

  it('finds cited numbers in every style', () => {
    expect(citedNumbers('a [1] b [2, 3] c [[4]] d [5][6] e [link](x)').sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5, 6])
  })
})
