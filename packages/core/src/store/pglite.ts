import { PGlite, type PGliteOptions } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'

export interface CreateDbOptions extends Omit<PGliteOptions, 'extensions'> {
  /**
   * Where data lives. `idb://name` (IndexedDB), `opfs-ahp://name` (OPFS, worker
   * only, fastest), or `memory://` for ephemeral. Default `idb://enclave`.
   */
  dataDir?: string
  extensions?: PGliteOptions['extensions']
}

/**
 * In-thread PGlite with pgvector loaded. For production UIs prefer
 * `createWorkerDb` so queries never block rendering and tabs share one database.
 */
export async function createDb(options: CreateDbOptions = {}): Promise<PGlite> {
  const { extensions, dataDir = 'idb://enclave', ...rest } = options
  return PGlite.create({
    ...rest,
    dataDir,
    extensions: { vector, ...extensions },
  })
}
