import * as fs from 'node:fs'
import * as path from 'node:path'
import type { AccessHandle, HandleSource } from '../src/store/opfs/handle-cache.js'
import { ERRNO, FsError, type SyncIo } from '../src/store/opfs/protocol.js'

const quotaError = () => Object.assign(new Error('No space available for this operation'), { name: 'QuotaExceededError' })

/**
 * A directory of real files exposed like OPFS sync access handles, including
 * Chromium's behavior that motivated the bounded cache: a handle reserves
 * capacity on its first write and keeps it until closed, and writes fail with
 * QuotaExceededError once `cap` handles hold reservations.
 */
export function fakeOpfs(dir: string, { cap = Infinity } = {}) {
  fs.mkdirSync(dir, { recursive: true })
  const reserved = new Set<object>()
  const locked = new Set<string>()
  const stats = { opened: 0, closed: 0, maxReserved: 0, sized: 0 }
  const source: HandleSource = {
    async open(name) {
      if (locked.has(name)) throw Object.assign(new Error('another handle is open'), { name: 'NoModificationAllowedError' })
      const file = path.join(dir, name)
      const fd = fs.openSync(file, fs.existsSync(file) ? 'r+' : 'w+')
      locked.add(name)
      stats.opened++
      const token = {}
      const handle: AccessHandle = {
        read: (buf, { at }) => fs.readSync(fd, buf, 0, buf.byteLength, at),
        write: (buf, { at }) => {
          if (!reserved.has(token)) {
            if (reserved.size >= cap) throw quotaError()
            reserved.add(token)
            stats.maxReserved = Math.max(stats.maxReserved, reserved.size)
          }
          return fs.writeSync(fd, buf, 0, buf.byteLength, at)
        },
        truncate: (size) => fs.ftruncateSync(fd, size),
        getSize: () => fs.fstatSync(fd).size,
        flush: () => fs.fsyncSync(fd),
        close: () => {
          reserved.delete(token)
          locked.delete(name)
          stats.closed++
          fs.closeSync(fd)
        },
      }
      return handle
    },
    async remove(name) {
      const file = path.join(dir, name)
      if (!fs.existsSync(file)) throw Object.assign(new Error('not found'), { name: 'NotFoundError' })
      fs.rmSync(file)
    },
    async list() {
      return fs.readdirSync(dir)
    },
    async sizes(names) {
      stats.sized += names.length
      // Like OPFS: a name that isn't a file (missing, a directory) is left out.
      return Object.fromEntries(names.filter((n) => fs.existsSync(path.join(dir, n))).map((n) => [n, fs.statSync(path.join(dir, n)).size]))
    },
  }
  return { source, stats }
}

/** SyncIo straight over node:fs, for exercising BoundedOpfsFS with a real PGlite. */
export function nodeSyncIo(dir: string): SyncIo & { calls: Record<string, number> } {
  fs.mkdirSync(dir, { recursive: true })
  const calls: Record<string, number> = {}
  const count = (op: string) => (calls[op] = (calls[op] ?? 0) + 1)
  const file = (name: string) => path.join(dir, name)
  const withFd = <T>(name: string, fn: (fd: number) => T): T => {
    const fd = fs.openSync(file(name), fs.existsSync(file(name)) ? 'r+' : 'w+')
    try {
      return fn(fd)
    } finally {
      fs.closeSync(fd)
    }
  }
  return {
    calls,
    read: (name, target, position) => (count('read'), fs.existsSync(file(name)) ? withFd(name, (fd) => fs.readSync(fd, target, 0, target.byteLength, position)) : 0),
    write: (name, source, position) => (count('write'), withFd(name, (fd) => fs.writeSync(fd, source, 0, source.byteLength, position))),
    truncate: (name, size) => (count('truncate'), withFd(name, (fd) => fs.ftruncateSync(fd, size))),
    size: (name) => (count('size'), fs.existsSync(file(name)) ? fs.statSync(file(name)).size : 0),
    sizes: () => (count('sizes'), Object.fromEntries(fs.readdirSync(dir).map((n) => [n, fs.statSync(file(n)).size]))),
    sizesOf: (names) => (count('sizesOf'), Object.fromEntries(names.map((n) => [n, fs.existsSync(file(n)) ? fs.statSync(file(n)).size : 0]))),
    flush: () => void count('flush'),
    remove: (name) => (count('remove'), fs.rmSync(file(name), { force: true })),
    list: () => fs.readdirSync(dir),
    close: () => void count('close'),
  }
}

export { ERRNO, FsError }
