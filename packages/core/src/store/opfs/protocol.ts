/**
 * Wire format between the PGlite worker (synchronous caller) and the OPFS I/O
 * worker (asynchronous owner of every FileSystemSyncAccessHandle).
 *
 * Why a second worker: Chromium reserves quota capacity for every open sync
 * access handle that has been written to, and refuses further writes once the
 * reservations reach a ceiling (~100–150 handles on a typical quota). PGlite's
 * opfs-ahp backend keeps one handle per Postgres file open forever, so initdb
 * alone exceeds it. Opening a handle is async, but PGlite's filesystem calls
 * are sync, so the handles live in a helper worker that opens and closes them
 * on demand (bounded LRU) while the caller blocks on Atomics.wait.
 */

/** Control block: Int32 header followed by Float64 arguments. */
export const CONTROL_BYTES = 64
export const I32 = { state: 0, op: 1, nameLength: 2, status: 3 } as const
export const F64_OFFSET = 32
export const F64 = { position: 0, length: 1, result: 2 } as const

export const STATE = { idle: 0, request: 1, response: 2 } as const

/** Data block: file name area, then payload. */
export const NAME_BYTES = 1024
export const DEFAULT_PAYLOAD_BYTES = 1 << 20

export const OP = {
  read: 1,
  write: 2,
  truncate: 3,
  size: 4,
  flush: 5,
  remove: 6,
  list: 7,
  close: 8,
  sizes: 9,
  sizesOf: 10,
} as const
export type Op = (typeof OP)[keyof typeof OP]

/** Emscripten errno values (PGlite's filesystem layer expects these). */
export const ERRNO = {
  EBADF: 8,
  EBUSY: 10,
  EEXIST: 20,
  EINVAL: 28,
  EIO: 29,
  EISDIR: 31,
  ENOENT: 44,
  ENOSPC: 51,
  ENOTDIR: 54,
  ENOTEMPTY: 55,
} as const

/** Error shape PGlite's BaseFilesystem turns into an Emscripten ErrnoError. */
export class FsError extends Error {
  constructor(
    readonly code: number,
    message?: string,
  ) {
    super(message ?? `errno ${code}`)
    this.name = 'FsError'
  }
}

/** Map an OPFS DOMException to an errno. */
export function errnoOf(err: unknown): number {
  if (err instanceof FsError) return err.code
  switch ((err as { name?: string } | null)?.name) {
    case 'NotFoundError':
      return ERRNO.ENOENT
    case 'QuotaExceededError':
      return ERRNO.ENOSPC
    case 'NoModificationAllowedError':
    case 'InvalidStateError':
      return ERRNO.EBUSY
    case 'TypeMismatchError':
      return ERRNO.EISDIR
    default:
      return ERRNO.EIO
  }
}

/** Messages exchanged over postMessage during setup. */
export interface InitMessage {
  type: 'enclave:opfs-init'
  control: SharedArrayBuffer
  data: SharedArrayBuffer
  /** OPFS directory (relative to the origin root) holding this database. */
  root: string
  maxOpenHandles: number
  /** Wake-up channel for runtimes without Atomics.waitAsync. */
  port: MessagePort
}
export interface ReadyMessage {
  type: 'enclave:opfs-ready'
  /** True when the server cannot use Atomics.waitAsync and needs a ping per request. */
  needsPing: boolean
}
export interface FailMessage {
  type: 'enclave:opfs-error'
  message: string
}

/** The synchronous file I/O the filesystem needs. */
export interface SyncIo {
  /** Read into `target` at `position`; returns bytes read (short at EOF). */
  read(name: string, target: Uint8Array, position: number): number
  /** Write `source` at `position`; creates the file if needed. Returns bytes written. */
  write(name: string, source: Uint8Array, position: number): number
  truncate(name: string, size: number): void
  /** Current size in bytes; 0 for a file that doesn't exist yet. */
  size(name: string): number
  /** Flush every handle written since the last flush. */
  flush(): void
  /** Delete a file (closing its handle first). Missing files are ignored. */
  remove(name: string): void
  /** Names of every file in the database directory. */
  list(): string[]
  /**
   * Every file's size in one call (name → bytes). Optional: without it the
   * filesystem asks for each file's size when it first needs it, which at
   * startup means a round trip, and an OPFS handle opened, per file.
   */
  sizes?(): Record<string, number>
  /** The sizes of some files in one call (name → bytes; 0 for a missing file). Optional. */
  sizesOf?(names: string[]): Record<string, number>
  /** Flush and close all handles. */
  close(): void
}
