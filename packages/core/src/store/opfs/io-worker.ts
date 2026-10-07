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

async function opfsDirectory(path: string): Promise<FileSystemDirectoryHandle> {
  let dir = await navigator.storage.getDirectory()
  for (const part of path.split('/').filter(Boolean)) dir = await dir.getDirectoryHandle(part, { create: true })
  return dir
}

/** HandleSource backed by one OPFS directory. */
export function opfsSource(dir: FileSystemDirectoryHandle): HandleSource {
  return {
    open: async (name) => (await dir.getFileHandle(name, { create: true })).createSyncAccessHandle(),
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
