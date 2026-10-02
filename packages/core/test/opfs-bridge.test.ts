import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'
import { Worker } from 'node:worker_threads'
import { PGlite } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { SharedIoClient } from '../src/store/opfs/client.js'
import { BoundedOpfsFS } from '../src/store/opfs/fs.js'
import { CONTROL_BYTES, DEFAULT_PAYLOAD_BYTES, NAME_BYTES } from '../src/store/opfs/protocol.js'

const root = path.resolve(import.meta.dirname, '..')
let dir: string
let worker: Worker
let client: SharedIoClient
const CAP = 60

beforeAll(async () => {
  // The worker thread runs compiled JS.
  execFileSync('npx', ['tsc', '-p', 'tsconfig.build.json'], { cwd: root, stdio: 'inherit' })
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'enclave-opfs-bridge-'))
  const control = new SharedArrayBuffer(CONTROL_BYTES)
  const data = new SharedArrayBuffer(NAME_BYTES + DEFAULT_PAYLOAD_BYTES)
  worker = new Worker(new URL('./opfs-bridge.worker.mjs', import.meta.url), { workerData: { control, data, dir, cap: CAP, maxOpen: 32 } })
  const ready = await new Promise<{ needsPing: boolean }>((resolve) => worker.once('message', resolve))
  client = new SharedIoClient(control, data, ready.needsPing ? () => worker.postMessage('ping') : undefined)
}, 60_000)

afterAll(async () => {
  await worker?.terminate()
  fs.rmSync(dir, { recursive: true, force: true })
})

const stats = () =>
  new Promise<{ maxReserved: number }>((resolve) => {
    worker.once('message', resolve)
    worker.postMessage('stats')
  })

describe('SharedArrayBuffer bridge', () => {
  it('round-trips reads and writes larger than the transfer buffer', () => {
    const big = new Uint8Array(DEFAULT_PAYLOAD_BYTES * 2 + 123).map((_, i) => i % 256)
    expect(client.write('blob', big, 10)).toBe(big.byteLength)
    expect(client.size('blob')).toBe(big.byteLength + 10)
    const back = new Uint8Array(big.byteLength)
    expect(client.read('blob', back, 10)).toBe(big.byteLength)
    expect(back).toEqual(big)
    client.truncate('blob', 5)
    expect(client.size('blob')).toBe(5)
    expect(client.list()).toContain('blob')
    client.remove('blob')
    expect(client.list()).not.toContain('blob')
    expect(() => client.read('missing-is-empty', new Uint8Array(4), 0)).not.toThrow()
  })

  it('runs Postgres (hundreds of relation files) under a 60-handle browser cap', async () => {
    const pg = await PGlite.create({ fs: new BoundedOpfsFS('bridge', { io: client }), extensions: { vector } })
    await pg.exec(`create extension vector;
      create table docs (id serial primary key, body text, embedding vector(3));
      create index on docs using hnsw (embedding vector_cosine_ops);
      insert into docs (body, embedding) select 'doc ' || g, array[random(), random(), random()]::vector from generate_series(1, 2000) g`)
    for (let i = 0; i < 40; i++) await pg.exec(`create table extra_${i} (x int primary key); insert into extra_${i} values (${i})`)
    const { rows } = await pg.query<{ c: number }>('select count(*)::int as c from docs')
    expect(rows[0]?.c).toBe(2000)
    await pg.close()
    expect((await stats()).maxReserved).toBeLessThanOrEqual(CAP)

    const files = fs.readdirSync(dir).filter((n) => n.startsWith('d')).length
    expect(files).toBeGreaterThan(CAP * 5) // far more files than the cap would allow open at once

    const again = await PGlite.create({ fs: new BoundedOpfsFS('bridge', { io: client }), extensions: { vector } })
    const near = await again.query<{ body: string }>(`select body from docs order by embedding <=> '[1,1,1]' limit 1`)
    expect(near.rows).toHaveLength(1)
    expect((await again.query<{ x: number }>('select x from extra_39')).rows[0]?.x).toBe(39)
    await again.close()
  }, 120_000)
})
