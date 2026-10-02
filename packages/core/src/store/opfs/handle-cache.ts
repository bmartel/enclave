import { ERRNO, FsError } from './protocol.js'

/** The subset of FileSystemSyncAccessHandle the cache uses. */
export interface AccessHandle {
  read(buffer: Uint8Array, options: { at: number }): number
  write(buffer: Uint8Array, options: { at: number }): number
  truncate(size: number): void
  getSize(): number
  flush(): void
  close(): void
}

/** Where handles come from: OPFS in browsers, a fake in tests. */
export interface HandleSource {
  open(name: string): Promise<AccessHandle>
  remove(name: string): Promise<void>
  list(): Promise<string[]>
}

interface Entry {
  handle: AccessHandle
  dirty: boolean
}

export interface HandleCacheOptions {
  /** Most handles open at once. Default 32. */
  maxOpen?: number
  /** Never shrink below this many handles when the browser pushes back. Default 4. */
  minOpen?: number
}

const isQuota = (err: unknown) => (err as { name?: string } | null)?.name === 'QuotaExceededError'

/**
 * A bounded, least-recently-used set of open access handles.
 *
 * Closing a handle releases the capacity Chromium reserved for it, so keeping
 * the set small keeps total reservations under the browser's ceiling. If a
 * write still hits QuotaExceededError, every other handle is flushed and
 * closed, the limit is halved, and the operation is retried once; a second
 * failure is a genuine out-of-space condition (ENOSPC).
 */
export class HandleCache {
  readonly #source: HandleSource
  readonly #open = new Map<string, Entry>() // insertion order = LRU order
  readonly #minOpen: number
  #maxOpen: number

  constructor(source: HandleSource, options: HandleCacheOptions = {}) {
    this.#source = source
    this.#maxOpen = Math.max(1, options.maxOpen ?? 32)
    this.#minOpen = Math.max(1, Math.min(options.minOpen ?? 4, this.#maxOpen))
  }

  get openCount(): number {
    return this.#open.size
  }

  get maxOpen(): number {
    return this.#maxOpen
  }

  async #get(name: string): Promise<Entry> {
    const hit = this.#open.get(name)
    if (hit) {
      this.#open.delete(name)
      this.#open.set(name, hit)
      return hit
    }
    while (this.#open.size >= this.#maxOpen) this.#closeOldest()
    let handle: AccessHandle
    try {
      handle = await this.#source.open(name)
    } catch (err) {
      if (!isQuota(err) || this.#open.size === 0) throw err
      this.#shrink(name)
      handle = await this.#source.open(name)
    }
    const entry = { handle, dirty: false }
    this.#open.set(name, entry)
    return entry
  }

  #closeOldest(): void {
    const [name, entry] = this.#open.entries().next().value as [string, Entry]
    this.#closeEntry(name, entry)
  }

  #closeEntry(name: string, entry: Entry): void {
    this.#open.delete(name)
    try {
      if (entry.dirty) entry.handle.flush()
    } finally {
      entry.handle.close()
    }
  }

  /** Close everything except `keep` and lower the limit. */
  #shrink(keep: string): void {
    for (const [name, entry] of [...this.#open]) if (name !== keep) this.#closeEntry(name, entry)
    this.#maxOpen = Math.max(this.#minOpen, Math.floor(this.#maxOpen / 2))
  }

  /** Run `fn` with the file's handle, retrying once after shedding handles on quota pressure. */
  async use<T>(name: string, fn: (handle: AccessHandle) => T, { writes = false } = {}): Promise<T> {
    const entry = await this.#get(name)
    try {
      const out = fn(entry.handle)
      if (writes) entry.dirty = true
      return out
    } catch (err) {
      if (!isQuota(err)) throw err
      this.#shrink(name)
      try {
        const out = fn(entry.handle)
        if (writes) entry.dirty = true
        return out
      } catch (again) {
        if (isQuota(again)) throw new FsError(ERRNO.ENOSPC, 'No space left on the device for this origin')
        throw again
      }
    }
  }

  /** Size without keeping a handle around longer than needed for an unknown file. */
  async size(name: string): Promise<number> {
    return this.use(name, (h) => h.getSize())
  }

  flushAll(): void {
    for (const entry of this.#open.values()) {
      if (!entry.dirty) continue
      entry.handle.flush()
      entry.dirty = false
    }
  }

  async remove(name: string): Promise<void> {
    const entry = this.#open.get(name)
    if (entry) {
      this.#open.delete(name)
      entry.handle.close()
    }
    await this.#source.remove(name)
  }

  list(): Promise<string[]> {
    return this.#source.list()
  }

  closeAll(): void {
    for (const [name, entry] of [...this.#open]) this.#closeEntry(name, entry)
  }
}
