/**
 * Storage self-check: can this origin actually write to OPFS and Cache
 * Storage right now?
 *
 * Browsers can refuse writes with QuotaExceededError while
 * `navigator.storage.estimate()` still reports gigabytes free. Chrome does this
 * after updating itself underneath a running browser, and some profiles get
 * stuck with a per-origin cap of a few hundred MB. "Free up disk space" is the
 * wrong advice then. This probe writes a small real file to each store, so an
 * app can tell real storage pressure from a browser that is refusing writes.
 */

export type StorageProbeResult = 'ok' | 'refused' | 'unavailable' | 'error'

export interface StorageCheck {
  /** Writing a small file to the origin private file system (PGlite's `opfs://`). */
  opfs: StorageProbeResult
  /** Writing a small response to Cache Storage (where model weights are cached). */
  cache: StorageProbeResult
  /** Bytes used and the quota, from `navigator.storage.estimate()`. */
  usage: number
  quota: number
  /** Whether the browser has agreed not to evict this origin's data. */
  persisted: boolean
  /**
   * Writes were refused although the browser reports plenty of free quota:
   * the browser, not the disk, is the problem.
   */
  refusing: boolean
  /** Something to tell the user when `refusing`, else null. */
  message: string | null
}

export interface StorageCheckOptions {
  /** Bytes to write to each store (default 1 MiB). */
  bytes?: number
  /** Free quota (quota - usage) above which a refusal is the browser's fault (default 1 GiB). */
  minFree?: number
}

const PROBE = 'enclave-storage-probe'

/** Probe OPFS and Cache Storage with small real writes, cleaning up after itself. */
export async function checkStorage({ bytes = 1 << 20, minFree = 1 << 30 }: StorageCheckOptions = {}): Promise<StorageCheck> {
  const storage = typeof navigator !== 'undefined' ? navigator.storage : undefined
  const [opfs, cache] = await Promise.all([probeOpfs(storage, bytes), probeCache(bytes)])
  const estimate = (await storage?.estimate?.().catch(() => null)) ?? {}
  const usage = estimate.usage ?? 0
  const quota = estimate.quota ?? 0
  const persisted = (await storage?.persisted?.().catch(() => false)) ?? false
  const refused = opfs === 'refused' || cache === 'refused'
  const refusing = refused && quota - usage >= minFree
  return { opfs, cache, usage, quota, persisted, refusing, message: refusing ? refusingMessage(quota - usage) : null }
}

/**
 * True when a storage failure (from a model download or the database) is the
 * browser refusing writes rather than real pressure. Runs {@link checkStorage}.
 */
export async function isBrowserRefusingStorage(options?: StorageCheckOptions): Promise<StorageCheck | null> {
  const check = await checkStorage(options)
  return check.refusing ? check : null
}

function refusingMessage(free: number): string {
  return (
    `This browser is refusing to save data, although it reports ${formatBytes(free)} free for this site. ` +
    `Restart the browser completely (quit and reopen it). If that doesn't help, this browser profile's storage is stuck: ` +
    `open the app in a new browser profile.`
  )
}

async function probeOpfs(storage: StorageManager | undefined, bytes: number): Promise<StorageProbeResult> {
  if (!storage?.getDirectory) return 'unavailable'
  let root: FileSystemDirectoryHandle
  try {
    root = await storage.getDirectory()
  } catch (err) {
    return classify(err)
  }
  try {
    const handle = await root.getFileHandle(PROBE, { create: true })
    // createWritable works on the main thread and in workers and goes through
    // the same quota checks as the sync access handles PGlite uses.
    const writable = await (handle as FileSystemFileHandle & { createWritable(): Promise<FileSystemWritableFileStream> }).createWritable()
    try {
      await writable.write(new Uint8Array(bytes))
      await writable.close()
    } catch (err) {
      await writable.abort().catch(() => {})
      throw err
    }
    return 'ok'
  } catch (err) {
    return classify(err)
  } finally {
    await root.removeEntry(PROBE).catch(() => {})
  }
}

async function probeCache(bytes: number): Promise<StorageProbeResult> {
  if (typeof caches === 'undefined') return 'unavailable'
  try {
    const cache = await caches.open(PROBE)
    await cache.put(`/${PROBE}`, new Response(new Uint8Array(bytes)))
    return 'ok'
  } catch (err) {
    return classify(err)
  } finally {
    await caches.delete(PROBE).catch(() => {})
  }
}

function classify(err: unknown): StorageProbeResult {
  const name = String((err as { name?: unknown } | null)?.name ?? '')
  if (name === 'QuotaExceededError') return 'refused'
  if (name === 'SecurityError' || name === 'NotSupportedError') return 'unavailable'
  return 'error'
}

function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`
  return `${Math.max(1, Math.round(n / 1e6))} MB`
}
