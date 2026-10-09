import { CONTROL_BYTES, F64, F64_OFFSET, FsError, I32, NAME_BYTES, OP, type Op, STATE, type SyncIo } from './protocol.js'

const encoder = new TextEncoder()
const decoder = new TextDecoder()

/**
 * Synchronous OPFS access from the PGlite worker: each call writes a request
 * into shared memory, wakes the I/O worker, and blocks on Atomics.wait until
 * the response lands. Requires cross-origin isolation (SharedArrayBuffer).
 */
export class SharedIoClient implements SyncIo {
  readonly #i32: Int32Array
  readonly #f64: Float64Array
  readonly #bytes: Uint8Array
  readonly #payload: number
  readonly #ping: (() => void) | undefined

  constructor(control: SharedArrayBuffer, data: SharedArrayBuffer, ping?: () => void) {
    if (control.byteLength < CONTROL_BYTES) throw new Error('control block too small')
    this.#i32 = new Int32Array(control, 0, F64_OFFSET / 4)
    this.#f64 = new Float64Array(control, F64_OFFSET, (CONTROL_BYTES - F64_OFFSET) / 8)
    this.#bytes = new Uint8Array(data)
    this.#payload = data.byteLength - NAME_BYTES
    this.#ping = ping
  }

  #call(op: Op, name: string, position = 0, length = 0): number {
    const i32 = this.#i32
    // TextEncoder can't write into shared memory: encode, then copy.
    const encoded = encoder.encode(name)
    if (encoded.byteLength > NAME_BYTES) throw new FsError(28, `file name too long: ${name}`)
    this.#bytes.set(encoded, 0)
    i32[I32.nameLength] = encoded.byteLength
    i32[I32.op] = op
    i32[I32.status] = 0
    this.#f64[F64.position] = position
    this.#f64[F64.length] = length
    Atomics.store(i32, I32.state, STATE.request)
    Atomics.notify(i32, I32.state)
    this.#ping?.()
    while (Atomics.load(i32, I32.state) === STATE.request) Atomics.wait(i32, I32.state, STATE.request)
    const status = i32[I32.status]!
    const result = this.#f64[F64.result]!
    Atomics.store(i32, I32.state, STATE.idle)
    if (status) throw new FsError(status)
    return result
  }

  read(name: string, target: Uint8Array, position: number): number {
    let done = 0
    while (done < target.byteLength) {
      const want = Math.min(this.#payload, target.byteLength - done)
      const got = this.#call(OP.read, name, position + done, want)
      target.set(this.#bytes.subarray(NAME_BYTES, NAME_BYTES + got), done)
      done += got
      if (got < want) break
    }
    return done
  }

  write(name: string, source: Uint8Array, position: number): number {
    let done = 0
    do {
      const n = Math.min(this.#payload, source.byteLength - done)
      this.#bytes.set(source.subarray(done, done + n), NAME_BYTES)
      done += this.#call(OP.write, name, position + done, n)
    } while (done < source.byteLength)
    return done
  }

  truncate(name: string, size: number): void {
    this.#call(OP.truncate, name, 0, size)
  }

  size(name: string): number {
    return this.#call(OP.size, name)
  }

  flush(): void {
    this.#call(OP.flush, '')
  }

  remove(name: string): void {
    this.#call(OP.remove, name)
  }

  list(): string[] {
    const n = this.#call(OP.list, '')
    // TextDecoder rejects shared views: copy out first.
    return JSON.parse(decoder.decode(this.#bytes.slice(NAME_BYTES, NAME_BYTES + n))) as string[]
  }

  sizes(): Record<string, number> {
    const n = this.#call(OP.sizes, '')
    return JSON.parse(decoder.decode(this.#bytes.slice(NAME_BYTES, NAME_BYTES + n))) as Record<string, number>
  }

  sizesOf(names: string[]): Record<string, number> {
    const request = encoder.encode(JSON.stringify(names))
    if (request.byteLength > this.#payload) throw new FsError(28, 'too many names for one call')
    this.#bytes.set(request, NAME_BYTES)
    const n = this.#call(OP.sizesOf, '', 0, request.byteLength)
    return JSON.parse(decoder.decode(this.#bytes.slice(NAME_BYTES, NAME_BYTES + n))) as Record<string, number>
  }

  close(): void {
    this.#call(OP.close, '')
  }
}
