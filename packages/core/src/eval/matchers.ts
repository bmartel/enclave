import type { AnswerMatcher } from './index.js'

/** A matcher that is always a predicate (composable and directly callable). */
export type TextPredicate = ((text: string) => boolean) & { label?: string }

/** Attach a human-readable description, shown in eval failure messages. */
export function labeled(label: string, predicate: (text: string) => boolean): TextPredicate {
  return Object.assign(predicate, { label })
}

const show = (m: AnswerMatcher) => (typeof m === 'function' ? ((m as TextPredicate).label ?? 'predicate') : String(m))

/** Every matcher must match. */
export const allOf = (...matchers: AnswerMatcher[]): TextPredicate =>
  labeled(`all of (${matchers.map(show).join(', ')})`, (text) => matchers.every((m) => test(m, text)))

/** At least one matcher must match. */
export const anyOf = (...matchers: AnswerMatcher[]): TextPredicate =>
  labeled(`any of (${matchers.map(show).join(', ')})`, (text) => matchers.some((m) => test(m, text)))

/** No matcher may match. */
export const noneOf = (...matchers: AnswerMatcher[]): TextPredicate =>
  labeled(`none of (${matchers.map(show).join(', ')})`, (text) => !matchers.some((m) => test(m, text)))

/**
 * The text states a number within `tolerance` of `value`, however it's
 * formatted: 1,234.5 · 1234.50 · $1,234 · 1 234,5 · 12.3k.
 */
export function numberNear(value: number, tolerance = 0.01): TextPredicate {
  return labeled(`the number ${value}`, (text) => numbersIn(text).some((n) => Math.abs(n - value) <= Math.max(tolerance, Math.abs(value) * 1e-9)))
}

/** Numbers mentioned in text, normalizing thousands separators, currency and k/m suffixes. */
export function numbersIn(text: string): number[] {
  const out: number[] = []
  for (const m of text.matchAll(/(?<![\w.])-?\$?\s?(\d{1,3}(?:[,   ]\d{3})+|\d+)(?:[.,](\d+))?\s?([kKmM]\b)?/g)) {
    const whole = m[1]!.replace(/[,   ]/g, '')
    let n = Number(m[2] ? `${whole}.${m[2]}` : whole)
    if (m[3]) n *= /k/i.test(m[3]) ? 1e3 : 1e6
    if (m[0].trimStart().startsWith('-')) n = -n
    if (Number.isFinite(n)) out.push(n)
  }
  return out
}

/** Words spelled as digits (for small counts like "three"). */
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
export function count(value: number): TextPredicate {
  const word = WORDS[value]
  return labeled(`the count ${value}`, anyOf(new RegExp(`(?<![\\d.,])${value}(?![\\d]|[.,]\\d)`), ...(word ? [new RegExp(`\\b${word}\\b`, 'i')] : [])))
}

/**
 * The answer declines or says the information isn't available, rather than
 * inventing it. Covers common phrasings; pair with `notAnswer` for the
 * specific fabrication you're guarding against.
 */
export const declines: TextPredicate = labeled('a decline (not available / not found)', anyOf(
  /\b(don'?t|do not|doesn'?t|does not|didn'?t|did not|can'?t|cannot|could not|couldn'?t|unable|not able)\b[^.]{0,80}\b(know|find|have|contain|mention|include|provide|specify|see|locate|access|help|assist|do|book|perform|access)/i,
  /\b(no|not any)\s+(information|mention|record|details?|data|results?|office|such)\b/i,
  /\bnot (available|mentioned|specified|provided|listed|included|found|documented|covered|stated|published|public|shared|disclosed)\b/i,
  /\b(isn'?t|is not|aren'?t|are not)\s+(available|mentioned|listed|included|specified|provided|documented|covered|published|public|shared|disclosed)\b/i,
  /\bI('?m| am) (not sure|unable|afraid)\b/i,
  /\bthere (is|are) no\b/i,
  /\boutside (of )?(my|the) (scope|capabilities)\b/i,
))

function test(m: AnswerMatcher, text: string): boolean {
  if (typeof m === 'string') return text.toLowerCase().includes(m.toLowerCase())
  if (m instanceof RegExp) return m.test(text)
  return m(text)
}
