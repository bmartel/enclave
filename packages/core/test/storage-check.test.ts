import { afterEach, describe, expect, it, vi } from 'vitest'
import { checkStorage } from '../src/store/storage-check.js'

const quotaError = () => Object.assign(new Error('Quota exceeded'), { name: 'QuotaExceededError' })

/** Fake navigator.storage and caches; `refuse` makes writes fail like a stuck profile. */
function fakeStorage({ refuseOpfs = false, refuseCache = false, usage = 300e6, quota = 11e9, persisted = false } = {}) {
  const removed: string[] = []
  const root = {
    getFileHandle: async () => ({
      createWritable: async () => ({
        write: async () => {
          if (refuseOpfs) throw quotaError()
        },
        close: async () => {},
        abort: async () => {},
      }),
    }),
    removeEntry: async (name: string) => void removed.push(name),
  }
  vi.stubGlobal('navigator', {
    storage: {
      getDirectory: async () => root,
      estimate: async () => ({ usage, quota }),
      persisted: async () => persisted,
    },
  })
  const deleted: string[] = []
  vi.stubGlobal('caches', {
    open: async () => ({
      put: async () => {
        if (refuseCache) throw quotaError()
      },
    }),
    delete: async (name: string) => (deleted.push(name), true),
  })
  return { removed, deleted }
}

afterEach(() => vi.unstubAllGlobals())

describe('checkStorage', () => {
  it('reports healthy storage and cleans up its probes', async () => {
    const { removed, deleted } = fakeStorage({ persisted: true })
    const check = await checkStorage()
    expect(check).toMatchObject({ opfs: 'ok', cache: 'ok', refusing: false, message: null, persisted: true, quota: 11e9 })
    expect(removed).toEqual(['enclave-storage-probe'])
    expect(deleted).toEqual(['enclave-storage-probe'])
  })

  it('flags a browser refusing writes while reporting plenty of free quota', async () => {
    fakeStorage({ refuseCache: true })
    const check = await checkStorage()
    expect(check).toMatchObject({ opfs: 'ok', cache: 'refused', refusing: true })
    expect(check.message).toMatch(/refusing to save data, although it reports 10\.7 GB free/)
    expect(check.message).toMatch(/new browser profile/)
  })

  it('treats a refusal near the quota as real storage pressure', async () => {
    fakeStorage({ refuseOpfs: true, usage: 10.9e9, quota: 11e9 })
    const check = await checkStorage()
    expect(check).toMatchObject({ opfs: 'refused', refusing: false, message: null })
  })

  it('reports stores the environment lacks as unavailable', async () => {
    vi.stubGlobal('navigator', {})
    vi.stubGlobal('caches', undefined)
    const check = await checkStorage()
    expect(check).toMatchObject({ opfs: 'unavailable', cache: 'unavailable', refusing: false, usage: 0, quota: 0 })
  })
})
