import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'
import { PGlite } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'
import { afterEach, describe, expect, it } from 'vitest'
import { BoundedOpfsFS } from '../src/store/opfs/fs.js'
import { boundedWal } from '../src/store/pglite-worker.js'
import { nodeSyncIo } from './opfs-fixtures.js'

const dirs: string[] = []
const tmp = () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'enclave-opfs-fs-'))
  dirs.push(d)
  return d
}
afterEach(() => {
  for (const d of dirs.splice(0)) fs.rmSync(d, { recursive: true, force: true })
})

const open = (dir: string) => PGlite.create({ fs: new BoundedOpfsFS('test', { io: nodeSyncIo(dir) }), extensions: { vector } })

describe('BoundedOpfsFS with PGlite', () => {
  it('initialises a cluster, persists across close and reopen', async () => {
    const dir = tmp()
    const a = await open(dir)
    await a.exec(`create extension vector; create table notes (id serial primary key, body text, embedding vector(3));
      insert into notes (body, embedding) values ('first', '[1,0,0]'), ('second', '[0,1,0]')`)
    await a.close()

    const b = await open(dir)
    const { rows } = await b.query<{ body: string }>(`select body from notes order by embedding <=> '[0.9,0.1,0]' limit 1`)
    expect(rows[0]?.body).toBe('first')
    await b.close()
    // One backing file per Postgres file plus the two metadata files.
    const names = fs.readdirSync(dir)
    expect(names).toContain('meta-0')
    expect(names).toContain('meta-1')
    expect(names.filter((n) => n.startsWith('d')).length).toBeGreaterThan(300)
  })

  it('recovers committed data after a crash without close (journal replay)', async () => {
    const dir = tmp()
    const a = await open(dir)
    await a.exec(`create table t (n int); insert into t select generate_series(1, 1000)`)
    // No close(): simulate the tab being killed after the transaction synced.
    const b = await open(dir)
    const { rows } = await b.query<{ c: number }>('select count(*)::int as c from t')
    expect(rows[0]?.c).toBe(1000)
    await b.close()
  })

  it('ignores a torn journal tail and sweeps orphaned files', async () => {
    const dir = tmp()
    const a = await open(dir)
    await a.exec('create table kept (x int); insert into kept values (42)')
    await a.close()
    // A file unlinked just before the session ended (its contents still there), then a torn line from an interrupted append.
    const active = ['meta-0', 'meta-1'].map((n) => ({ n, size: fs.statSync(path.join(dir, n)).size, mtime: fs.statSync(path.join(dir, n)).mtimeMs }))
    const newest = active.sort((x, y) => y.mtime - x.mtime)[0]!.n
    fs.appendFileSync(path.join(dir, newest), '\n["create","/orphan",999999,33206,0]\n["unlink","/orphan"]\n["mkdir","/half')
    fs.writeFileSync(path.join(dir, 'd999999'), 'orphan')
    fs.writeFileSync(path.join(dir, 'not-ours.txt'), 'leave me')

    const b = await open(dir)
    const { rows } = await b.query<{ x: number }>('select x from kept')
    expect(rows[0]?.x).toBe(42)
    await b.close()
    expect(fs.existsSync(path.join(dir, 'd999999'))).toBe(false)
    expect(fs.existsSync(path.join(dir, 'not-ours.txt'))).toBe(true)
  })

  it('falls back to the older snapshot when the newer one is torn', async () => {
    const dir = tmp()
    const a = await open(dir)
    await a.exec('create table s (x int); insert into s values (7)')
    await a.close()
    // close() checkpoints into the newer file; corrupt it as if the checkpoint was cut short.
    const files = ['meta-0', 'meta-1'].map((n) => ({ n, gen: JSON.parse(fs.readFileSync(path.join(dir, n), 'utf8').split('\n')[0]!).gen as number }))
    const [newer, older] = files.sort((x, y) => y.gen - x.gen)
    const olderText = fs.readFileSync(path.join(dir, older!.n), 'utf8')
    fs.writeFileSync(path.join(dir, newer!.n), '{"v":1,"gen":')
    expect(olderText.length).toBeGreaterThan(0)

    const b = await open(dir)
    const { rows } = await b.query<{ x: number }>('select x from s')
    expect(rows[0]?.x).toBe(7)
    await b.close()
  })

  it('learns every file size in one call when it opens, instead of one per file', async () => {
    const dir = tmp()
    const a = await open(dir)
    await a.exec('create table t (x int); insert into t values (1)')
    await a.close()
    const io = nodeSyncIo(dir)
    const b = await PGlite.create({ fs: new BoundedOpfsFS('test', { io }), extensions: { vector } })
    expect((await b.query<{ x: number }>('select x from t')).rows[0]?.x).toBe(1)
    await b.close()
    // The snapshot knows every size; at most the files resized since are asked for, in one call.
    expect(io.calls.sizes ?? 0).toBe(0)
    expect(io.calls.sizesOf ?? 0).toBeLessThanOrEqual(1)
    expect(io.calls.size ?? 0).toBeLessThan(20)
  })

  it('knows the sizes of files written after the snapshot, after a crash', async () => {
    const dir = tmp()
    const a = await open(dir)
    await a.exec('create table g (x text)')
    await a.close()
    const b = await open(dir)
    // Grows the table's file after the snapshot; no close().
    await b.exec(`insert into g select repeat('x', 500) from generate_series(1, 2000)`)
    const io = nodeSyncIo(dir)
    const c = await PGlite.create({ fs: new BoundedOpfsFS('test', { io }), extensions: { vector } })
    expect((await c.query<{ n: number }>('select count(*)::int as n from g')).rows[0]?.n).toBe(2000)
    expect(io.calls.sizesOf).toBe(1)
    await c.close()
  })

  it('distrusts sizes an older version carried into its next snapshot', async () => {
    const dir = tmp()
    const a = await open(dir)
    await a.exec('create table carried (x text)')
    await a.close()
    // An older version: next generation, the same (now stale) sizes, the table grown without a journaled resize.
    const [newer] = ['meta-0', 'meta-1']
      .map((n) => ({ n, text: fs.readFileSync(path.join(dir, n), 'utf8') }))
      .filter((m) => m.text)
      .sort((x, y) => JSON.parse(y.text.split('\n')[0]!).gen - JSON.parse(x.text.split('\n')[0]!).gen)
    const snapshot = JSON.parse(newer!.text.split('\n')[0]!)
    const other = newer!.n === 'meta-0' ? 'meta-1' : 'meta-0'
    fs.writeFileSync(path.join(dir, other), JSON.stringify({ ...snapshot, gen: snapshot.gen + 1 }))
    const io = nodeSyncIo(dir)
    const b = await PGlite.create({ fs: new BoundedOpfsFS('test', { io }), extensions: { vector } })
    expect(io.calls.sizes).toBe(1)
    await b.exec(`insert into carried values ('ok')`)
    expect((await b.query<{ x: string }>('select x from carried')).rows[0]?.x).toBe('ok')
    await b.close()
  })

  it('opens a database whose snapshot has no sizes (an older version) by listing every file once', async () => {
    const dir = tmp()
    const a = await open(dir)
    await a.exec('create table old (x int); insert into old values (3)')
    await a.close()
    for (const n of ['meta-0', 'meta-1']) {
      const [first, ...rest] = fs.readFileSync(path.join(dir, n), 'utf8').split('\n')
      if (!first) continue
      const { sizes, ...snapshot } = JSON.parse(first)
      expect(sizes).toBeDefined()
      fs.writeFileSync(path.join(dir, n), [JSON.stringify(snapshot), ...rest].join('\n'))
    }
    const io = nodeSyncIo(dir)
    const b = await PGlite.create({ fs: new BoundedOpfsFS('test', { io }), extensions: { vector } })
    expect((await b.query<{ x: number }>('select x from old')).rows[0]?.x).toBe(3)
    expect(io.calls.sizes).toBe(1)
    await b.close()
  })

  it('reads ahead: a table read in order takes a few round trips, not one per 8 KB page', async () => {
    const dir = tmp()
    const a = await open(dir)
    await a.exec(`create table big (x text); insert into big select repeat(md5(i::text), 20) from generate_series(1, 4000) i`)
    const { rows } = await a.query<{ pages: number }>(`select pg_relation_size('big')::int / 8192 as pages`)
    await a.close()
    const io = nodeSyncIo(dir)
    const b = await PGlite.create({ fs: new BoundedOpfsFS('test', { io }), extensions: { vector } })
    const before = io.calls.read ?? 0
    expect((await b.query<{ n: number }>('select count(*)::int as n from big')).rows[0]?.n).toBe(4000)
    expect((io.calls.read ?? 0) - before).toBeLessThan(rows[0]!.pages / 4)
    await b.close()
  })

  it('reads what was written since a chunk was read ahead', async () => {
    const dir = tmp()
    const fsys = new BoundedOpfsFS('test', { io: nodeSyncIo(dir) })
    await fsys.init({} as never, {} as never)
    fsys.writeFile('/f', new Uint8Array(100_000).fill(1))
    const fd = fsys.open('/f')
    const buf = new Uint8Array(8)
    fsys.read(fd, buf, 0, 8, 50_000)
    expect([...buf]).toEqual(new Array(8).fill(1))
    fsys.write(fd, new Uint8Array(8).fill(2), 0, 8, 50_004)
    fsys.read(fd, buf, 0, 8, 50_000)
    expect([...buf]).toEqual([1, 1, 1, 1, 2, 2, 2, 2])
    // Past the old end: the read-ahead chunk mustn't hide what was appended.
    fsys.write(fd, new Uint8Array(4).fill(3), 0, 4, 100_000)
    const tail = new Uint8Array(6)
    expect(fsys.read(fd, tail, 0, 6, 99_998)).toBe(6)
    expect([...tail]).toEqual([1, 1, 3, 3, 3, 3])
  })
})

describe('worker databases', () => {
  it('cap the write-ahead log, so a start never replays more than that', () => {
    const params = boundedWal(64)
    expect(params[0]).toBe('--single')
    expect(params).toContain('max_wal_size=64MB')
    expect(params).toContain('min_wal_size=32MB')
  })
})
