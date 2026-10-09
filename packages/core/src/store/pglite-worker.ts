import { PGlite, type PGliteOptions } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'
import { PGliteWorker, worker } from '@electric-sql/pglite/worker'
import { openBoundedOpfs } from './opfs/index.js'

/** Posted by a database worker whose PGlite failed to start, just before it exits. */
export const DB_FATAL = 'enclave:db-fatal'

export interface ServePGliteOptions {
  /**
   * Starts the OPFS I/O worker, required for `opfs://` data directories:
   * `() => new Worker(new URL('./opfs-io.worker.ts', import.meta.url), { type: 'module' })`.
   */
  opfsIo?: () => Worker
  /** Most OPFS sync access handles open at once for `opfs://`. Default 32. */
  maxOpenHandles?: number
  /**
   * Most write-ahead log kept between checkpoints, in MB, for persistent data
   * directories. Default 64. A browser ends the worker without shutting the
   * database down, so every start replays the log since the last checkpoint,
   * and in single-user mode (no checkpointer) Postgres's default only
   * checkpointed after 1 GB of it: the start after a big import replayed it all.
   */
  maxWalSizeMb?: number
}

/**
 * Worker entry. PGlite runs off the main thread; with multiple tabs open, one
 * worker is elected leader and the others proxy to it.
 *
 * ```ts
 * // db.worker.ts
 * import { servePGlite } from 'enclave-ai/pglite-worker'
 * servePGlite({}, { opfsIo: () => new Worker(new URL('./opfs-io.worker.ts', import.meta.url), { type: 'module' }) })
 * ```
 *
 * If PGlite fails to start, the worker reports {@link DB_FATAL} and exits.
 * Staying alive would keep the leader lock (and any storage handles) and wedge
 * every tab; exiting lets createWorkerDb reject and another tab take over.
 */
export function servePGlite(extensions: PGliteOptions['extensions'] = {}, options: ServePGliteOptions = {}): void {
  void worker({
    // Extensions must be registered inside the worker, not on the client.
    init: async ({ dataDir, ...rest }) => {
      try {
        const opfs = dataDir?.startsWith('opfs://') ? dataDir.slice('opfs://'.length) : null
        if (opfs !== null && !options.opfsIo) {
          throw new Error("dataDir 'opfs://…' needs servePGlite(extensions, { opfsIo: () => new Worker(…) })")
        }
        const storage = opfs !== null ? { fs: await openBoundedOpfs(opfs, { ioWorker: options.opfsIo!, maxOpenHandles: options.maxOpenHandles }) } : { dataDir }
        const persistent = opfs !== null || !!dataDir?.startsWith('idb://')
        const own = (rest as { startParams?: string[] }).startParams
        const startParams = own ?? (persistent ? boundedWal(options.maxWalSizeMb ?? 64) : undefined)
        return await PGlite.create({ ...rest, ...storage, ...(startParams ? { startParams } : {}), extensions: { vector, ...extensions } })
      } catch (err) {
        console.error('[enclave-ai] database failed to start', err)
        self.postMessage({ type: DB_FATAL, message: describe(err) })
        self.close()
        throw err
      }
    },
  })
}

/** PGlite's start parameters with the write-ahead log capped at `mb` (at least 64; the minimum kept at half). */
export function boundedWal(mb: number): string[] {
  const max = Math.max(64, Math.round(mb))
  return [...PGlite.defaultStartParams, '-c', `max_wal_size=${max}MB`, '-c', `min_wal_size=${Math.max(32, Math.floor(max / 2))}MB`]
}

function describe(err: unknown): string {
  const e = err as { name?: string; errno?: number; message?: string } | null
  if (e?.name === 'ErrnoError') return `Filesystem error (errno ${e.errno})`
  return String(e?.message ?? err)
}

export interface WorkerDbOptions {
  /**
   * Where data lives:
   * - `opfs://name`: OPFS through a bounded set of sync access handles. The fastest durable storage; needs
   *   cross-origin isolation and `opfsIo` in servePGlite.
   * - `idb://name`: IndexedDB (the whole database is held in memory and flushed).
   * - `memory://`: ephemeral.
   * Default `idb://enclave`.
   */
  dataDir?: string
  /** Distinguishes multiple databases for leader election. */
  id?: string
  relaxedDurability?: boolean
  /** Reject if the database hasn't answered within this many ms. Default 30000. */
  openTimeoutMs?: number
}

/**
 * Connect to the database worker and wait until it answers a query. Rejects,
 * instead of hanging, when the worker reports a startup failure or nothing
 * answers in time (for example a frozen leader in another tab).
 */
export async function createWorkerDb(workerInstance: Worker, options: WorkerDbOptions = {}): Promise<PGliteWorker> {
  const { dataDir = 'idb://enclave', openTimeoutMs = 30_000, ...rest } = options
  let timer: ReturnType<typeof setTimeout> | undefined
  let onMessage: ((e: MessageEvent) => void) | undefined
  let onError: ((e: ErrorEvent) => void) | undefined
  const failure = new Promise<never>((_, reject) => {
    onMessage = (e) => {
      if (e.data?.type === DB_FATAL) reject(new Error(`Database failed to start: ${e.data.message}`))
    }
    onError = (e) => reject(new Error(e.message || 'Database worker failed to start'))
    workerInstance.addEventListener('message', onMessage)
    workerInstance.addEventListener('error', onError)
    timer = setTimeout(
      () => reject(new Error(`Timed out after ${openTimeoutMs}ms opening ${dataDir}. Another tab's database worker may be unresponsive.`)),
      openTimeoutMs,
    )
  })
  try {
    const db = await Promise.race([PGliteWorker.create(workerInstance, { dataDir, ...rest }), failure])
    await Promise.race([db.query('select 1'), failure])
    return db
  } catch (err) {
    workerInstance.terminate()
    throw err
  } finally {
    clearTimeout(timer)
    if (onMessage) workerInstance.removeEventListener('message', onMessage)
    if (onError) workerInstance.removeEventListener('error', onError)
  }
}
