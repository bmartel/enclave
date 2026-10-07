import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'
import { PGlite } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'
import { afterEach, describe, expect, it } from 'vitest'
import { BoundedOpfsFS } from '../src/store/opfs/fs.js'
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
    // Torn line from an interrupted append, and a file no node references.
    const active = ['meta-0', 'meta-1'].map((n) => ({ n, size: fs.statSync(path.join(dir, n)).size, mtime: fs.statSync(path.join(dir, n)).mtimeMs }))
    const newest = active.sort((x, y) => y.mtime - x.mtime)[0]!.n
    fs.appendFileSync(path.join(dir, newest), '\n["mkdir","/half')
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
    expect(io.calls.sizes).toBe(1)
    // Postgres stats hundreds of files at startup; only files it creates or grows need asking again.
    expect(io.calls.size ?? 0).toBeLessThan(20)
  })
})
