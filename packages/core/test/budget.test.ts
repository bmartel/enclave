import { describe, expect, it } from 'vitest'
import { contextBudget, fitHistory } from '../src/agent.js'
import type { Message } from '../src/types.js'

const turn = (i: number, toolChars = 100): Message[] => [
  { role: 'user', content: `question ${i}` },
  { role: 'assistant', content: '', toolCalls: [{ id: `c${i}`, name: 'search', input: { q: i } }] },
  { role: 'tool', toolCallId: `c${i}`, name: 'search', content: 'r'.repeat(toolChars) },
  { role: 'assistant', content: `answer ${i}` },
]

describe('context budgeting', () => {
  it('is disabled without a context window and reserves room for the reply', () => {
    expect(contextBudget(undefined, 's', undefined, [])).toBeUndefined()
    const b = contextBudget(4096, 'x'.repeat(1000), 'y'.repeat(500), [])!
    expect(b).toBe(Math.floor((4096 - 1024) * 3.2) - 1502)
  })

  it('drops whole older turns first', () => {
    const history = [...turn(1, 2000), ...turn(2, 2000), ...turn(3, 100)]
    const fitted = fitHistory(history, 100, 1500)
    expect(fitted[0]).toEqual({ role: 'user', content: 'question 3' })
    expect(fitted).toHaveLength(4)
  })

  it('shortens earlier tool results inside an oversized current turn without mutating history', () => {
    const history: Message[] = [
      { role: 'user', content: 'q' },
      { role: 'assistant', content: '', toolCalls: [{ id: 'a', name: 's', input: {} }] },
      { role: 'tool', toolCallId: 'a', name: 's', content: 'x'.repeat(5000) },
      { role: 'assistant', content: '', toolCalls: [{ id: 'b', name: 's', input: {} }] },
      { role: 'tool', toolCallId: 'b', name: 's', content: 'y'.repeat(500) },
    ]
    const fitted = fitHistory(history, 100, 2000)
    expect(fitted[2]!.content.length).toBeLessThan(500)
    expect(fitted[4]!.content).toHaveLength(500) // the latest result stays intact
    expect(history[2]!.content).toHaveLength(5000)
  })
})
