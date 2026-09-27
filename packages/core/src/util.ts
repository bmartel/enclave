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
