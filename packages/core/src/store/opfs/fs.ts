import { BaseFilesystem } from '@electric-sql/pglite/basefs'
import { ERRNO, FsError, type SyncIo } from './protocol.js'

/**
 * A PGlite filesystem that keeps the directory tree in memory and stores file
 * contents through a {@link SyncIo} (one backing file per Postgres file).
 *
 * Metadata durability: two snapshot files, `meta-0` and `meta-1`. Each holds a
 * JSON snapshot on its first line followed by journal lines, one per mutation.
 * A checkpoint writes a fresh snapshot (generation + 1) into the *other* file,
 * so a crash mid-checkpoint leaves the previous snapshot and its journal
 * intact. On load the valid snapshot with the highest generation wins and its
 * journal is replayed; a torn final line is ignored.
 */

const S_IFDIR = 0o040000
const S_IFREG = 0o100000
const BLOCK_SIZE = 4096
const CHECKPOINT_EVERY = 2000
const META = ['meta-0', 'meta-1'] as const
const meta = (index: number) => (index % 2 ? META[1] : META[0])
const dataName = (id: number) => `d${id}`

interface DirNode {
  t: 'd'
  mode: number
  mtime: number
  children: Record<string, Node>
}
interface FileNode {
  t: 'f'
  mode: number
  mtime: number
  id: number
}
type Node = DirNode | FileNode

interface Snapshot {
  v: 1
  gen: number
  nextId: number
  root: DirNode
}

type JournalOp =
  | ['mkdir', string, number, number]
  | ['create', string, number, number, number]
  | ['rename', string, string]
  | ['unlink', string]
  | ['rmdir', string]
  | ['chmod', string, number]
  | ['utimes', string, number]

export interface FsStats {
  dev: number
  ino: number
  mode: number
  nlink: number
  uid: number
  gid: number
  rdev: number
  size: number
  blksize: number
  blocks: number
  atime: number
  mtime: number
  ctime: number
}

const split = (path: string) => path.split('/').filter(Boolean)
const fail = (code: number, message?: string): never => {
  throw new FsError(code, message)
}

export class BoundedOpfsFS extends BaseFilesystem {
  readonly #io: SyncIo
  #state!: Snapshot
  #active = 0 // index into META of the file receiving journal lines
  #journalEnd = 0 // byte offset where the next journal line goes
  #journalLines = 0
  readonly #sizes = new Map<number, number>()
  readonly #fds = new Map<number, FileNode>()
  #nextFd = 1
  readonly #encoder = new TextEncoder()

  constructor(dataDir: string, { io, debug = false }: { io: SyncIo; debug?: boolean }) {
    super(dataDir, { debug })
    this.#io = io
  }

  override async init(
    pg: Parameters<BaseFilesystem['init']>[0],
    options: Parameters<BaseFilesystem['init']>[1],
  ): ReturnType<BaseFilesystem['init']> {
    this.#load()
    this.#sweepOrphans()
    return super.init(pg, options)
  }

  override async syncToFs(relaxedDurability = false): Promise<void> {
    if (this.#journalLines >= CHECKPOINT_EVERY) this.#checkpoint()
    if (!relaxedDurability) this.#io.flush()
  }

  override async closeFs(): Promise<void> {
    this.#checkpoint()
    this.#io.close()
    ;(this.pg as unknown as { Module: { FS: { quit(): void } } } | undefined)?.Module.FS.quit()
  }

  // --- Metadata persistence ------------------------------------------------

  #readAll(name: string): string {
    const size = this.#io.size(name)
    if (!size) return ''
    const buf = new Uint8Array(size)
    const n = this.#io.read(name, buf, 0)
    return new TextDecoder().decode(buf.subarray(0, n))
  }

  #load(): void {
    let best: { index: number; snapshot: Snapshot; lines: string[] } | null = null
    META.forEach((name, index) => {
      const [first, ...lines] = this.#readAll(name).split('\n')
      if (!first) return
      try {
        const snapshot = JSON.parse(first) as Snapshot
        if (snapshot.v !== 1 || typeof snapshot.gen !== 'number') return
        if (!best || snapshot.gen > best.snapshot.gen) best = { index, snapshot, lines }
      } catch {
        // A torn snapshot from an interrupted checkpoint: the other file is valid.
      }
    })
    if (!best) {
      this.#state = { v: 1, gen: 0, nextId: 1, root: { t: 'd', mode: S_IFDIR | 0o777, mtime: Date.now(), children: {} } }
      this.#checkpoint()
      return
    }
    const { index, snapshot, lines } = best as { index: number; snapshot: Snapshot; lines: string[] }
    this.#state = snapshot
    this.#active = index
    let replayed = 0
    for (const line of lines) {
      if (!line) continue
      let op: JournalOp
      try {
        op = JSON.parse(line) as JournalOp
      } catch {
        break // torn tail: everything before it is applied
      }
      try {
        this.#apply(op)
      } catch (err) {
        console.warn('[enclave-ai] skipped journal entry', op, err)
      }
      replayed++
    }
    this.#journalLines = replayed
    this.#journalEnd = this.#io.size(meta(index))
    // Start each session from a compact snapshot.
    if (replayed) this.#checkpoint()
  }

  #checkpoint(): void {
    this.#state.gen++
    const target = this.#state.gen % 2
    const bytes = this.#encoder.encode(JSON.stringify(this.#state))
    this.#io.truncate(meta(target), 0)
    this.#io.write(meta(target), bytes, 0)
    this.#io.flush()
    this.#active = target
    this.#journalEnd = bytes.byteLength
    this.#journalLines = 0
  }

  #journal(op: JournalOp): void {
    const bytes = this.#encoder.encode(`\n${JSON.stringify(op)}`)
    this.#io.write(meta(this.#active), bytes, this.#journalEnd)
    this.#journalEnd += bytes.byteLength
    this.#journalLines++
  }

  /** Apply then journal, so a rejected operation is never recorded. */
  #commit(op: JournalOp): void {
    this.#apply(op)
    this.#journal(op)
  }

  /** Delete backing files that no node references (left by a crash between unlink and delete). */
  #sweepOrphans(): void {
    const live = new Set<string>(META)
    const walk = (dir: DirNode) => {
      for (const node of Object.values(dir.children)) node.t === 'f' ? live.add(dataName(node.id)) : walk(node)
    }
    walk(this.#state.root)
    // Postgres stats every file at startup: learn all sizes in one call
    // (a thousand round trips, each opening an OPFS handle, took seconds on phones).
    const sizes = this.#io.sizes?.()
    for (const name of sizes ? Object.keys(sizes) : this.#io.list()) {
      // Only ever touch names this filesystem creates.
      if (!/^d\d+$/.test(name)) continue
      if (!live.has(name)) this.#io.remove(name)
      else if (sizes) this.#sizes.set(Number(name.slice(1)), sizes[name]!)
    }
  }

  // --- Tree ----------------------------------------------------------------

  #lookup(path: string): Node {
    let node: Node = this.#state.root
    for (const part of split(path)) {
      if (node.t !== 'd') fail(ERRNO.ENOTDIR)
      const next: Node | undefined = (node as DirNode).children[part]
      if (!next) fail(ERRNO.ENOENT, `No such file or directory: ${path}`)
      node = next as Node
    }
    return node
  }

  #parent(path: string): { dir: DirNode; name: string } {
    const parts = split(path)
    const name = parts.pop()
    if (!name) fail(ERRNO.EINVAL, 'root has no parent')
    const dir = this.#lookup(parts.join('/'))
    if (dir.t !== 'd') fail(ERRNO.ENOTDIR)
    return { dir: dir as DirNode, name: name as string }
  }

  #file(node: Node): FileNode {
    if (node.t !== 'f') fail(ERRNO.EISDIR)
    return node as FileNode
  }

  #apply(op: JournalOp): void {
    switch (op[0]) {
      case 'mkdir': {
        const [, path, mode, mtime] = op
        const { dir, name } = this.#parent(path)
        if (dir.children[name]) fail(ERRNO.EEXIST)
        dir.children[name] = { t: 'd', mode, mtime, children: {} }
        return
      }
      case 'create': {
        const [, path, id, mode, mtime] = op
        const { dir, name } = this.#parent(path)
        if (dir.children[name]) fail(ERRNO.EEXIST)
        dir.children[name] = { t: 'f', mode, mtime, id }
        this.#state.nextId = Math.max(this.#state.nextId, id + 1)
        return
      }
      case 'rename': {
        const [, from, to] = op
        const src = this.#parent(from)
        const node = src.dir.children[src.name]
        if (!node) fail(ERRNO.ENOENT)
        const dst = this.#parent(to)
        const existing = dst.dir.children[dst.name]
        if (existing && existing !== node) {
          if (existing.t === 'd' && Object.keys(existing.children).length) fail(ERRNO.ENOTEMPTY)
          if (existing.t !== node!.t) fail(existing.t === 'd' ? ERRNO.EISDIR : ERRNO.ENOTDIR)
        }
        delete src.dir.children[src.name]
        dst.dir.children[dst.name] = node!
        return
      }
      case 'unlink': {
        const { dir, name } = this.#parent(op[1])
        const node = dir.children[name]
        if (!node) fail(ERRNO.ENOENT)
        if (node!.t !== 'f') fail(ERRNO.EISDIR)
        delete dir.children[name]
        return
      }
      case 'rmdir': {
        const { dir, name } = this.#parent(op[1])
        const node = dir.children[name]
        if (!node) fail(ERRNO.ENOENT)
        if (node!.t !== 'd') fail(ERRNO.ENOTDIR)
        if (Object.keys((node as DirNode).children).length) fail(ERRNO.ENOTEMPTY)
        delete dir.children[name]
        return
      }
      case 'chmod':
        this.#lookup(op[1]).mode = op[2]
        return
      case 'utimes':
        this.#lookup(op[1]).mtime = op[2]
        return
    }
  }

  #size(node: FileNode): number {
    let size = this.#sizes.get(node.id)
    if (size === undefined) {
      size = this.#io.size(dataName(node.id))
      this.#sizes.set(node.id, size)
    }
    return size
  }

  #stats(node: Node): FsStats {
    const size = node.t === 'f' ? this.#size(node) : 0
    return {
      dev: 0,
      ino: 0,
      mode: node.mode,
      nlink: 1,
      uid: 0,
      gid: 0,
      rdev: 0,
      size,
      blksize: BLOCK_SIZE,
      blocks: Math.ceil(size / BLOCK_SIZE),
      atime: node.mtime,
      mtime: node.mtime,
      ctime: node.mtime,
    }
  }

  #fd(fd: number): FileNode {
    const node = this.#fds.get(fd)
    if (!node) fail(ERRNO.EBADF)
    return node as FileNode
  }

  // --- BaseFilesystem ------------------------------------------------------

  chmod(path: string, mode: number): void {
    this.#commit(['chmod', path, mode])
  }

  close(fd: number): void {
    this.#fds.delete(fd)
  }

  fstat(fd: number): FsStats {
    return this.#stats(this.#fd(fd))
  }

  lstat(path: string): FsStats {
    return this.#stats(this.#lookup(path))
  }

  mkdir(path: string, options?: { recursive?: boolean; mode?: number }): void {
    const mode = S_IFDIR | ((options?.mode ?? 0o777) & 0o7777)
    if (options?.recursive) {
      const parts = split(path)
      for (let i = 1; i <= parts.length; i++) {
        const sub = `/${parts.slice(0, i).join('/')}`
        try {
          const node = this.#lookup(sub)
          if (node.t !== 'd') fail(ERRNO.ENOTDIR)
        } catch (err) {
          if ((err as FsError).code !== ERRNO.ENOENT) throw err
          this.#commit(['mkdir', sub, mode, Date.now()])
        }
      }
      return
    }
    this.#commit(['mkdir', path, mode, Date.now()])
  }

  open(path: string, _flags?: string, _mode?: number): number {
    const node = this.#file(this.#lookup(path))
    const fd = this.#nextFd++
    this.#fds.set(fd, node)
    return fd
  }

  readdir(path: string): string[] {
    const node = this.#lookup(path)
    if (node.t !== 'd') fail(ERRNO.ENOTDIR)
    return Object.keys((node as DirNode).children)
  }

  read(fd: number, buffer: Uint8Array, offset: number, length: number, position: number): number {
    const node = this.#fd(fd)
    const size = this.#size(node)
    if (position >= size || length === 0) return 0
    const target = new Uint8Array(buffer.buffer, buffer.byteOffset + offset, Math.min(length, size - position))
    return this.#io.read(dataName(node.id), target, position)
  }

  rename(oldPath: string, newPath: string): void {
    const dst = this.#parent(newPath)
    const replaced = dst.dir.children[dst.name]
    const moving = this.#lookup(oldPath)
    this.#commit(['rename', oldPath, newPath])
    if (replaced && replaced !== moving && replaced.t === 'f') this.#discard(replaced)
  }

  rmdir(path: string): void {
    this.#commit(['rmdir', path])
  }

  truncate(path: string, len = 0): void {
    const node = this.#file(this.#lookup(path))
    this.#io.truncate(dataName(node.id), len)
    this.#sizes.set(node.id, len)
  }

  unlink(path: string): void {
    const node = this.#lookup(path)
    this.#commit(['unlink', path])
    if (node.t === 'f') this.#discard(node)
  }

  /** Drop a file's contents once no path refers to it. Open fds keep reading zeros. */
  #discard(node: FileNode): void {
    this.#io.remove(dataName(node.id))
    this.#sizes.delete(node.id)
  }

  utimes(path: string, _atime: number, mtime: number): void {
    this.#commit(['utimes', path, mtime])
  }

  writeFile(path: string, data: string | Uint8Array, options?: { encoding?: string; mode?: number; flag?: string }): void {
    let node: FileNode
    try {
      node = this.#file(this.#lookup(path))
      this.#io.truncate(dataName(node.id), 0)
    } catch (err) {
      if ((err as FsError).code !== ERRNO.ENOENT) throw err
      const id = this.#state.nextId++
      this.#commit(['create', path, id, S_IFREG | ((options?.mode ?? 0o666) & 0o7777), Date.now()])
      node = this.#lookup(path) as FileNode
    }
    const bytes = typeof data === 'string' ? this.#encoder.encode(data) : data
    if (bytes.byteLength) this.#io.write(dataName(node.id), bytes, 0)
    this.#sizes.set(node.id, bytes.byteLength)
  }

  write(fd: number, buffer: ArrayBuffer | Uint8Array, offset: number, length: number, position: number): number {
    const node = this.#fd(fd)
    const source =
      buffer instanceof Uint8Array
        ? new Uint8Array(buffer.buffer, buffer.byteOffset + offset, length)
        : new Uint8Array(buffer, offset, length)
    const written = this.#io.write(dataName(node.id), source, position)
    this.#sizes.set(node.id, Math.max(this.#size(node), position + written))
    return written
  }
}
