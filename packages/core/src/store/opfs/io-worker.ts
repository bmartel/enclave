import { HandleCache, type HandleSource } from './handle-cache.js'
import type { FailMessage, InitMessage, ReadyMessage } from './protocol.js'
import { serveSharedIo } from './server.js'

/**
 * Worker entry for `opfs://` databases: owns every OPFS sync access handle
 * and serves synchronous reads and writes to the PGlite worker. Create it
 * from your database worker and pass the factory to servePGlite:
 *
 * ```ts
 * // opfs-io.worker.ts
 * import { serveOpfsIo } from 'enclave-ai/pglite-opfs-io'
 * serveOpfsIo()
 * ```
 */
export function serveOpfsIo(): void {
  self.addEventListener('message', async (event: MessageEvent) => {
    const msg = event.data as InitMessage
    if (msg?.type !== 'enclave:opfs-init') return
    try {
      // The previous page's I/O worker (a reload, a navigation) is torn down after its database worker
      // released the leader lock, so it can still hold this directory's access handles while ours starts:
      // wait until it is gone instead of failing the first open with NoModificationAllowedError.
      await holdDirectoryLock(msg.root)
      const dir = await opfsDirectory(msg.root)
      const cache = new HandleCache(opfsSource(dir), { maxOpen: msg.maxOpenHandles })
      const server = serveSharedIo(msg.control, msg.data, cache)
      if (server.needsPing) msg.port.onmessage = () => server.wake()
      self.postMessage({ type: 'enclave:opfs-ready', needsPing: server.needsPing } satisfies ReadyMessage)
    } catch (err) {
      self.postMessage({ type: 'enclave:opfs-error', message: String((err as Error)?.message ?? err) } satisfies FailMessage)
    }
  })
}

/** Longest wait for another I/O worker to let go of the directory before giving up (the caller may retry). */
export const DIRECTORY_LOCK_TIMEOUT_MS = 15_000

/**
 * Hold a Web Lock named after the directory for this worker's whole life (the
 * browser releases it when the worker ends, however it ends). Rejects when
 * another worker keeps it for longer than DIRECTORY_LOCK_TIMEOUT_MS. Without
 * Web Locks it returns at once: access handles then rely on openWhenFree.
 */
export function holdDirectoryLock(root: string, timeoutMs = DIRECTORY_LOCK_TIMEOUT_MS): Promise<void> {
  const locks = (globalThis.navigator as Navigator | undefined)?.locks
  if (!locks?.request) return Promise.resolve()
  return new Promise<void>((resolve, reject) => {
    const signal = AbortSignal.timeout(timeoutMs)
    locks
      .request(`enclave-opfs:${root}`, { signal }, () => {
        resolve()
        return new Promise<void>(() => {}) // held until the worker ends
      })
      .catch((err: unknown) =>
        reject((err as { name?: string })?.name === 'TimeoutError' ? new Error(`the database files in ${root} are still in use by another worker`) : err),
      )
  })
}

/** Waits before trying a handle again while another worker still holds it (about 3 s in all). */
const BUSY_RETRY_MS = [50, 100, 200, 400, 800, 1500]

/**
 * createSyncAccessHandle, trying again for a few seconds while another worker
 * still holds the file (a closing page's I/O worker, between its lock's
 * release and its handles'): NoModificationAllowedError then passes.
 */
export async function openWhenFree(open: () => Promise<FileSystemSyncAccessHandle>, delays = BUSY_RETRY_MS): Promise<FileSystemSyncAccessHandle> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await open()
    } catch (err) {
      if ((err as { name?: string })?.name !== 'NoModificationAllowedError' || attempt >= delays.length) throw err
      await new Promise((r) => setTimeout(r, delays[attempt]))
    }
  }
}

async function opfsDirectory(path: string): Promise<FileSystemDirectoryHandle> {
  let dir = await navigator.storage.getDirectory()
  for (const part of path.split('/').filter(Boolean)) dir = await dir.getDirectoryHandle(part, { create: true })
  return dir
}

/** HandleSource backed by one OPFS directory. */
export function opfsSource(dir: FileSystemDirectoryHandle): HandleSource {
  return {
    open: async (name) => {
      const file = await dir.getFileHandle(name, { create: true })
      return openWhenFree(() => file.createSyncAccessHandle())
    },
    remove: (name) => dir.removeEntry(name),
    list: async () => {
      const names: string[] = []
      for await (const name of (dir as unknown as { keys(): AsyncIterable<string> }).keys()) names.push(name)
      return names
    },
    // File.size needs no sync access handle (no lock, no quota reservation), and they run together.
    sizes: async (names) => {
      const sizes = await Promise.all(
        names.map(async (name) => {
          try {
            return [name, (await (await dir.getFileHandle(name)).getFile()).size] as const
          } catch {
            return null // a directory, or gone: the filesystem asks again if it needs it
          }
        }),
      )
      return Object.fromEntries(sizes.filter((e): e is readonly [string, number] => e !== null))
    },
  }
}
