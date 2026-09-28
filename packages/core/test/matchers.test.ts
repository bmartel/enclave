import { describe, expect, it } from 'vitest'
import { allOf, anyOf, compareReports, count, declines, noneOf, numberNear, numbersIn, wilson, type AnswerMatcher, type EvalReport } from '../src/eval/index.js'

const t = (m: AnswerMatcher, text: string) => (typeof m === 'function' ? m(text) : m instanceof RegExp ? m.test(text) : text.includes(m))

describe('numbersIn / numberNear', () => {
  it.each([
    ['Total revenue was $12,345.67 in March.', 12345.67],
    ['Total: 12345.67', 12345.67],
    ['about 12.3k dollars', 12300],
    ['1 234,5 EUR', 1234.5],
    ['down -4.2% month over month', -4.2],
    ['The answer is 750.', 750],
    ['**$750.00**', 750],
  ])('finds the number in %j', (text, value) => {
    expect(t(numberNear(value, 0.01), text)).toBe(true)
  })

  it('does not match other numbers or partial digits', () => {
    expect(t(numberNear(75), 'The total is 750.')).toBe(false)
    expect(t(numberNear(500), 'Order #5001 shipped')).toBe(false)
    expect(t(numberNear(12345.67), 'It was 12,345.')).toBe(false)
    // Identifiers like v2 or X200 are not stated quantities.
    expect(numbersIn('v2 of 3 items, 4.5 each, model X200')).toEqual([3, 4.5])
  })
})

describe('count', () => {
  it('accepts digits or words and rejects other numbers', () => {
    expect(t(count(3), 'There are 3 tasks.')).toBe(true)
    expect(t(count(3), 'There are three tasks.')).toBe(true)
    expect(t(count(3), 'There are 13 tasks.')).toBe(false)
    expect(t(count(3), 'Version 3.5 has two tasks')).toBe(false)
  })
})

describe('declines', () => {
  it.each([
    "I don't have that information in the documents.",
    'The handbook does not mention the CEO’s phone number.',
    'There is no Paris office listed in the documents.',
    "I couldn't find any information about that.",
    'That detail is not available in the provided passages.',
    "I'm unable to book flights; that's outside my capabilities.",
    'The passages do not contain the answer.',
    'Personal phone numbers are not published.',
  ])('recognizes %j', (text) => expect(t(declines, text)).toBe(true))

  it.each([
    'The CEO’s phone number is 555-0100.',
    'The Paris office wifi password is bonjour2026.',
    'Visitors park on level B1.',
  ])('rejects confident answers like %j', (text) => expect(t(declines, text)).toBe(false))
})

describe('combinators', () => {
  it('composes', () => {
    expect(t(allOf('a', /b/), 'ab')).toBe(true)
    expect(t(allOf('a', /c/), 'ab')).toBe(false)
    expect(t(anyOf('x', /b/), 'ab')).toBe(true)
    expect(t(noneOf('x', /b/), 'ab')).toBe(false)
  })
})

describe('statistics', () => {
  it('computes Wilson intervals that shrink with more samples', () => {
    const small = wilson(9, 10)
    const large = wilson(90, 100)
    expect(small.rate).toBe(0.9)
    expect(small.low).toBeCloseTo(0.596, 2)
    expect(small.high).toBeCloseTo(0.982, 2)
    expect(large.high - large.low).toBeLessThan(small.high - small.low)
    expect(wilson(10, 10).high).toBe(1)
    expect(wilson(0, 0)).toMatchObject({ rate: 0, low: 0, high: 0 })
  })

  it('flags only regressions whose intervals separate', () => {
    const report = (cases: [string, number, number][], passed: number, total: number) =>
      ({ overall: wilson(passed, total), byCase: cases.map(([name, p, runs]) => ({ name, tags: [], passed: p, runs, passRate: p / runs, meanMs: 0, failures: [] })) }) as unknown as EvalReport
    const before = report([['a', 20, 20], ['b', 3, 3], ['c', 1, 3]], 24, 26)
    const after = report([['a', 5, 20], ['b', 2, 3], ['c', 3, 3]], 10, 26)
    const cmp = compareReports(before, after)
    expect(cmp.regressions).toEqual([
      { name: 'a', before: 1, after: 0.25, significant: true },
      { name: 'b', before: 1, after: 2 / 3, significant: false },
    ])
    expect(cmp.improvements.map((i) => i.name)).toEqual(['c'])
    expect(cmp.overall.significant).toBe(true)
  })
})
