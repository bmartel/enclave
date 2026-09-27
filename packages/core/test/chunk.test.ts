import { describe, expect, it } from 'vitest'
import { chunkText } from '../src/rag/chunk.js'

describe('chunkText', () => {
  it('returns short text as one chunk', () => {
    expect(chunkText('  hello  ')).toEqual(['hello'])
    expect(chunkText('')).toEqual([])
  })

  it('respects size, prefers paragraph boundaries and overlaps', () => {
    const paragraphs = Array.from({ length: 12 }, (_, i) => `Paragraph ${i} ` + 'word '.repeat(40)).join('\n\n')
    const chunks = chunkText(paragraphs, { size: 500, overlap: 50 })
    expect(chunks.length).toBeGreaterThan(3)
    for (const c of chunks) expect(c.length).toBeLessThanOrEqual(560)
    // Overlap: each chunk after the first starts with the tail of the previous one.
    expect(chunks[1]!.slice(0, 20)).not.toMatch(/^Paragraph 0 /)
  })

  it('hard-splits text with no separators', () => {
    const chunks = chunkText('x'.repeat(2500), { size: 1000, overlap: 0 })
    expect(chunks.map((c) => c.length)).toEqual([1000, 1000, 500])
  })
})
