import type { HandleCache } from './handle-cache.js'
import { CONTROL_BYTES, errnoOf, F64, F64_OFFSET, FsError, I32, NAME_BYTES, OP, STATE } from './protocol.js'

const encoder = new TextEncoder()
const decoder = new TextDecoder()

export interface ServerHandle {
  /** True when callers must ping (no Atomics.waitAsync in this runtime). */
  needsPing: boolean
  /** Process a pending request, if any. Wire this to the ping channel. */
  wake(): void
  stop(): void
}

/**
 * Serve SharedIoClient requests against a HandleCache. Uses Atomics.waitAsync
 * where available; otherwise the client pings through a MessagePort and
 * `wake()` must be called for each ping.
 */
export function serveSharedIo(control: SharedArrayBuffer, data: SharedArrayBuffer, cache: HandleCache): ServerHandle {
  const i32 = new Int32Array(control, 0, F64_OFFSET / 4)
  const f64 = new Float64Array(control, F64_OFFSET, (CONTROL_BYTES - F64_OFFSET) / 8)
  const bytes = new Uint8Array(data)
  const payload = data.byteLength - NAME_BYTES
  let stopped = false
  let busy = false

  const handle = async (): Promise<number> => {
    const op = i32[I32.op]!
    // TextDecoder rejects shared views: copy the name out first.
    const name = decoder.decode(bytes.slice(0, i32[I32.nameLength]!))
    const position = f64[F64.position]!
    const length = f64[F64.length]!
    switch (op) {
      case OP.read: {
        const view = bytes.subarray(NAME_BYTES, NAME_BYTES + Math.min(length, payload))
        return cache.use(name, (h) => h.read(view, { at: position }))
      }
      case OP.write: {
        const view = bytes.subarray(NAME_BYTES, NAME_BYTES + Math.min(length, payload))
        return cache.use(name, (h) => h.write(view, { at: position }), { writes: true })
      }
      case OP.truncate:
        await cache.use(name, (h) => h.truncate(length), { writes: true })
        return 0
      case OP.size:
        return cache.size(name)
      case OP.flush:
        cache.flushAll()
        return 0
      case OP.remove:
        await cache.remove(name).catch((err) => {
          if ((err as { name?: string })?.name !== 'NotFoundError') throw err
        })
        return 0
      case OP.list: {
        const json = encoder.encode(JSON.stringify(await cache.list()))
        if (json.byteLength > payload) throw new FsError(28, 'directory listing exceeds the transfer buffer')
        bytes.set(json, NAME_BYTES)
        return json.byteLength
      }
      case OP.sizes: {
        const json = encoder.encode(JSON.stringify(await cache.sizes()))
        if (json.byteLength > payload) throw new FsError(28, 'size listing exceeds the transfer buffer')
        bytes.set(json, NAME_BYTES)
        return json.byteLength
      }
      case OP.close:
        cache.closeAll()
        return 0
      default:
        throw new FsError(28, `unknown op ${op}`)
    }
  }

  const serve = async () => {
    if (busy) return
    busy = true
    try {
      while (!stopped && Atomics.load(i32, I32.state) === STATE.request) {
        try {
          f64[F64.result] = await handle()
          i32[I32.status] = 0
        } catch (err) {
          i32[I32.status] = errnoOf(err)
          f64[F64.result] = 0
          if (!(err instanceof FsError)) console.error('[enclave-ai] OPFS I/O failed', err)
        }
        Atomics.store(i32, I32.state, STATE.response)
        Atomics.notify(i32, I32.state)
      }
    } finally {
      busy = false
    }
  }

  type WaitAsync = (array: Int32Array, index: number, value: number) => { async: boolean; value: Promise<unknown> | string }
  const waitAsync = (Atomics as unknown as { waitAsync?: WaitAsync }).waitAsync
  if (waitAsync) {
    void (async () => {
      while (!stopped) {
        const current = Atomics.load(i32, I32.state)
        if (current === STATE.request) {
          await serve()
          continue
        }
        const waited = waitAsync(i32, I32.state, current)
        if (waited.async) await waited.value
      }
    })()
  }

  return {
    needsPing: !waitAsync,
    wake: () => void serve(),
    stop: () => {
      stopped = true
      cache.closeAll()
    },
  }
}
