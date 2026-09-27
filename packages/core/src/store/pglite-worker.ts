import { PGlite, type PGliteOptions } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'
import { PGliteWorker, worker } from '@electric-sql/pglite/worker'

/**
 * Worker entry. PGlite runs off the main thread; with multiple tabs open, one
 * worker is elected leader and the others proxy to it.
 *
 * ```ts
 * // db.worker.ts
 * import { servePGlite } from '@enclave/core/pglite-worker'
 * servePGlite()
 * ```
 */
export function servePGlite(extensions: PGliteOptions['extensions'] = {}): void {
  void worker({
    // Extensions must be registered inside the worker, not on the client.
    init: (options) => PGlite.create({ ...options, extensions: { vector, ...extensions } }),
  })
}

export interface WorkerDbOptions {
  /** Default `idb://enclave`. `opfs-ahp://name` is fastest inside a worker. */
  dataDir?: string
  /** Distinguishes multiple databases for leader election. */
  id?: string
  relaxedDurability?: boolean
}

export async function createWorkerDb(workerInstance: Worker, options: WorkerDbOptions = {}): Promise<PGliteWorker> {
  const { dataDir = 'idb://enclave', ...rest } = options
  return PGliteWorker.create(workerInstance, { dataDir, ...rest })
}
