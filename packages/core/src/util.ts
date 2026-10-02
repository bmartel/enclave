/** JSON.stringify that survives bigint, Dates, typed arrays and Maps. */
export function safeStringify(value: unknown): string {
  if (typeof value === 'string') return value
  return (
    JSON.stringify(value, (_key, v) => {
      if (typeof v === 'bigint') return v.toString()
      if (v instanceof Map) return Object.fromEntries(v)
      if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return `[binary ${v.byteLength} bytes]`
      return v
    }) ?? 'null'
  )
}

export function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max)}\n…[truncated ${text.length - max} chars]`
}

export async function sha256(text: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}

export function uid(prefix = ''): string {
  return prefix + crypto.randomUUID().replaceAll('-', '').slice(0, 20)
}

/**
 * Run `task` while live-yielding anything it pushes through `emit`,
 * then return the task's result.
 */
export async function* drain<E, T>(task: (emit: (event: E) => void) => Promise<T>): AsyncGenerator<E, T> {
  const buffer: E[] = []
  let wake: (() => void) | undefined
  let settled = false
  let result: T | undefined
  let failure: { error: unknown } | undefined

  task((event) => {
    buffer.push(event)
    wake?.()
  })
    .then(
      (r) => void (result = r),
      (error) => void (failure = { error }),
    )
    .finally(() => {
      settled = true
      wake?.()
    })

  while (true) {
    while (buffer.length) yield buffer.shift()!
    if (settled) break
    await new Promise<void>((resolve) => (wake = resolve))
    wake = undefined
  }
  if (failure) throw failure.error
  return result as T
}

export function toVectorLiteral(values: ArrayLike<number>): string {
  return `[${Array.prototype.join.call(values, ',')}]`
}

const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

/**
 * The date spelled out for a model: today, then the rest of this week and all
 * of next week with full weekday names. Small models get "this Friday" wrong
 * when they must count (one invented September 31st); grouped by week, "this
 * Friday" and "next Friday" read off directly. Put it in a skill's `context`.
 * `today` is a Date or YYYY-MM-DD (read as a calendar date); weeks start Monday.
 */
export function dateContext(today: Date | string = new Date()): string {
  const start =
    typeof today === 'string'
      ? new Date(`${today}T00:00:00Z`)
      : new Date(Date.UTC(today.getFullYear(), today.getMonth(), today.getDate()))
  const day = (offset: number) => {
    const d = new Date(start.getTime() + offset * 86_400_000)
    return { name: WEEKDAYS[d.getUTCDay()]!, iso: d.toISOString().slice(0, 10), weekday: d.getUTCDay() }
  }
  const now = day(0)
  const daysLeftThisWeek = (7 - now.weekday) % 7 // through Sunday
  const list = (from: number, count: number) =>
    Array.from({ length: count }, (_, i) => day(from + i))
      .map((d) => `${d.name} ${d.iso}`)
      .join(', ')
  const lines = [`Today is ${now.name} ${now.iso}. Tomorrow is ${list(1, 1)}.`]
  if (daysLeftThisWeek > 0) lines.push(`Rest of this week: ${list(1, daysLeftThisWeek)}.`)
  lines.push(`Next week: ${list(daysLeftThisWeek + 1, 7)}.`)
  return lines.join('\n')
}

/**
 * True when reasoning is going in circles: some sentence of 40+ characters
 * has appeared `times` times. Thresholds from the evals: three repeats of a
 * shorter sentence ("So, the SQL query would need to…") is normal deliberation,
 * and cutting it there made the model describe SQL instead of running it. Small models can loop ("without the customer's
 * id we can't… therefore… but without the customer's id…") until a budget or
 * timeout ends the turn; cutting the loop and answering directly is faster
 * and, in practice, gives the answer the loop was circling.
 */
export function reasoningLoops(text: string, times = 4): boolean {
  const counts = new Map<string, number>()
  for (const raw of text.split(/(?<=[.?!])\s+|\n+/)) {
    const sentence = raw.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim()
    if (sentence.length < 40) continue
    const n = (counts.get(sentence) ?? 0) + 1
    if (n >= times) return true
    counts.set(sentence, n)
  }
  return false
}

