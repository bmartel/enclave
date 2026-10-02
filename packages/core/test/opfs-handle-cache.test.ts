import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { HandleCache } from '../src/store/opfs/handle-cache.js'
import { fakeOpfs } from './opfs-fixtures.js'

const dirs: string[] = []
const tmp = () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'enclave-opfs-'))
  dirs.push(d)
  return d
}
afterEach(() => {
  for (const d of dirs.splice(0)) fs.rmSync(d, { recursive: true, force: true })
})

const page = (n: number) => new Uint8Array(8192).fill(n % 251)

describe('HandleCache', () => {
  it('reproduces the browser failure when every handle stays open (the opfs-ahp pattern)', async () => {
    const { source } = fakeOpfs(tmp(), { cap: 100 })
    const handles = []
    let failedAt = -1
    for (let i = 0; i < 300; i++) {
      const h = await source.open(`f${i}`)
      handles.push(h)
      try {
        h.write(page(i), { at: 0 })
      } catch (err) {
        failedAt = i
        expect((err as Error).name).toBe('QuotaExceededError')
        break
      }
    }
    expect(failedAt).toBe(100)
    for (const h of handles) h.close()
  })

  it('writes thousands of files under the cap by bounding open handles', async () => {
    const dir = tmp()
    const { source, stats } = fakeOpfs(dir, { cap: 100 })
    const cache = new HandleCache(source, { maxOpen: 32 })
    for (let i = 0; i < 2000; i++) await cache.use(`f${i}`, (h) => h.write(page(i), { at: 0 }), { writes: true })
    cache.flushAll()
    expect(cache.openCount).toBeLessThanOrEqual(32)
    expect(stats.maxReserved).toBeLessThanOrEqual(32)
    // Re-read through evicted-and-reopened handles.
    const out = new Uint8Array(8192)
    for (const i of [0, 777, 1999]) {
      await cache.use(`f${i}`, (h) => h.read(out, { at: 0 }))
      expect(out[0]).toBe(i % 251)
    }
    cache.closeAll()
    expect(fs.readdirSync(dir)).toHaveLength(2000)
  })

  it('sheds handles and halves its limit when the browser cap is lower than configured', async () => {
    const { source, stats } = fakeOpfs(tmp(), { cap: 10 })
    const cache = new HandleCache(source, { maxOpen: 64 })
    for (let i = 0; i < 500; i++) await cache.use(`f${i}`, (h) => h.write(page(i), { at: 0 }), { writes: true })
    expect(cache.maxOpen).toBeLessThanOrEqual(10)
    expect(stats.maxReserved).toBeLessThanOrEqual(10)
    cache.closeAll()
  })

  it('reports ENOSPC when even a single handle cannot write', async () => {
    const { source } = fakeOpfs(tmp(), { cap: 0 })
    const cache = new HandleCache(source, { maxOpen: 4 })
    await expect(cache.use('f', (h) => h.write(page(1), { at: 0 }), { writes: true })).rejects.toMatchObject({ code: 51 })
  })

  it('flushes dirty handles on eviction and removes files', async () => {
    const dir = tmp()
    const { source } = fakeOpfs(dir)
    const cache = new HandleCache(source, { maxOpen: 2 })
    await cache.use('a', (h) => h.write(page(1), { at: 0 }), { writes: true })
    await cache.use('b', (h) => h.write(page(2), { at: 0 }), { writes: true })
    await cache.use('c', (h) => h.write(page(3), { at: 0 }), { writes: true }) // evicts a
    expect(fs.statSync(path.join(dir, 'a')).size).toBe(8192)
    await cache.remove('b')
    expect(fs.existsSync(path.join(dir, 'b'))).toBe(false)
    expect((await cache.list()).sort()).toEqual(['a', 'c'])
    cache.closeAll()
  })
})
