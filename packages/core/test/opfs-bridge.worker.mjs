// Worker thread for opfs-bridge.test.ts: serves the compiled SharedArrayBuffer
// protocol against real files with a simulated browser capacity cap.
import * as fs from 'node:fs'
import * as path from 'node:path'
import { parentPort, workerData } from 'node:worker_threads'
import { HandleCache } from '../dist/store/opfs/handle-cache.js'
import { serveSharedIo } from '../dist/store/opfs/server.js'

const { control, data, dir, cap, maxOpen } = workerData
fs.mkdirSync(dir, { recursive: true })
const reserved = new Set()
let maxReserved = 0
const source = {
  async open(name) {
    const file = path.join(dir, name)
    const fd = fs.openSync(file, fs.existsSync(file) ? 'r+' : 'w+')
    const token = {}
    return {
      read: (buf, { at }) => fs.readSync(fd, buf, 0, buf.byteLength, at),
      write: (buf, { at }) => {
        if (!reserved.has(token)) {
          if (reserved.size >= cap) throw Object.assign(new Error('No space available'), { name: 'QuotaExceededError' })
          reserved.add(token)
          maxReserved = Math.max(maxReserved, reserved.size)
        }
        return fs.writeSync(fd, buf, 0, buf.byteLength, at)
      },
      truncate: (size) => fs.ftruncateSync(fd, size),
      getSize: () => fs.fstatSync(fd).size,
      flush: () => fs.fsyncSync(fd),
      close: () => {
        reserved.delete(token)
        fs.closeSync(fd)
      },
    }
  },
  async remove(name) {
    fs.rmSync(path.join(dir, name))
  },
  async list() {
    return fs.readdirSync(dir)
  },
}
const server = serveSharedIo(control, data, new HandleCache(source, { maxOpen }))
parentPort.on('message', (msg) => {
  if (msg === 'stats') parentPort.postMessage({ maxReserved })
  if (msg === 'ping') server.wake()
})
parentPort.postMessage({ ready: true, needsPing: server.needsPing })
