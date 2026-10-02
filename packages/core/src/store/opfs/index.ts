import { BoundedOpfsFS } from './fs.js'
import { SharedIoClient } from './client.js'
import { CONTROL_BYTES, DEFAULT_PAYLOAD_BYTES, type InitMessage, NAME_BYTES, type ReadyMessage } from './protocol.js'

export { BoundedOpfsFS } from './fs.js'
export { HandleCache, type AccessHandle, type HandleSource } from './handle-cache.js'
export { SharedIoClient } from './client.js'
export { serveSharedIo } from './server.js'
export { ERRNO, FsError, type SyncIo } from './protocol.js'

export interface BoundedOpfsOptions {
  /** Starts the OPFS I/O worker (a module that calls serveOpfsIo()). */
  ioWorker: () => Worker
  /** Most sync access handles open at once. Default 32. */
  maxOpenHandles?: number
}

/**
 * Open an `opfs://` filesystem from inside a worker: starts the I/O worker,
 * shares memory with it, and returns a PGlite filesystem that talks to it.
 */
export async function openBoundedOpfs(root: string, { ioWorker, maxOpenHandles = 32 }: BoundedOpfsOptions): Promise<BoundedOpfsFS> {
  if (typeof SharedArrayBuffer === 'undefined' || !globalThis.crossOriginIsolated) {
    throw new Error(
      'opfs:// storage needs cross-origin isolation for SharedArrayBuffer. Serve the page with ' +
        '"Cross-Origin-Opener-Policy: same-origin" and "Cross-Origin-Embedder-Policy: require-corp" (or credentialless).',
    )
  }
  const control = new SharedArrayBuffer(CONTROL_BYTES)
  const data = new SharedArrayBuffer(NAME_BYTES + DEFAULT_PAYLOAD_BYTES)
  const io = ioWorker()
  const channel = new MessageChannel()
  const ready = new Promise<ReadyMessage>((resolve, reject) => {
    io.addEventListener('message', (e: MessageEvent) => {
      if (e.data?.type === 'enclave:opfs-ready') resolve(e.data as ReadyMessage)
      else if (e.data?.type === 'enclave:opfs-error') reject(new Error(`OPFS I/O worker: ${e.data.message}`))
    })
    io.addEventListener('error', (e: ErrorEvent) => reject(new Error(e.message || 'OPFS I/O worker failed to start')))
  })
  io.postMessage(
    { type: 'enclave:opfs-init', control, data, root, maxOpenHandles, port: channel.port2 } satisfies InitMessage,
    [channel.port2],
  )
  const { needsPing } = await ready
  const client = new SharedIoClient(control, data, needsPing ? () => channel.port1.postMessage(0) : undefined)
  return new BoundedOpfsFS(root, { io: client })
}
