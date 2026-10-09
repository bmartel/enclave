import { describe, expect, it } from 'vitest'
import { holdDirectoryLock, openWhenFree } from '../src/store/opfs/io-worker.js'

const busy = () => Object.assign(new Error('Access Handles cannot be created if there is another open Access Handle'), { name: 'NoModificationAllowedError' })

describe('openWhenFree', () => {
  it('tries again while another worker still holds the file', async () => {
    let calls = 0
    const handle = {} as FileSystemSyncAccessHandle
    const got = await openWhenFree(async () => {
      if (++calls < 3) throw busy()
      return handle
    }, [1, 1, 1])
    expect(got).toBe(handle)
    expect(calls).toBe(3)
  })

  it('gives up after its last wait', async () => {
    let calls = 0
    await expect(
      openWhenFree(async () => {
        calls++
        throw busy()
      }, [1, 1]),
    ).rejects.toMatchObject({ name: 'NoModificationAllowedError' })
    expect(calls).toBe(3)
  })

  it('does not retry other failures', async () => {
    let calls = 0
    const err = Object.assign(new Error('gone'), { name: 'NotFoundError' })
    await expect(
      openWhenFree(async () => {
        calls++
        throw err
      }, [1, 1]),
    ).rejects.toBe(err)
    expect(calls).toBe(1)
  })
})

describe('holdDirectoryLock', () => {
  it('waits for the worker holding the directory, and gives up after the timeout', async () => {
    const held = new Map<string, Promise<void>>()
    const locks = {
      request(name: string, options: { signal?: AbortSignal }, fn: () => Promise<void>) {
        const before = held.get(name)
        if (before) {
          return new Promise((resolve, reject) => {
            options.signal?.addEventListener('abort', () => reject(Object.assign(new Error('timed out'), { name: 'TimeoutError' })))
            before.then(() => resolve(fn()))
          })
        }
        const p = fn()
        held.set(name, p)
        return p
      },
    }
    const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
    Object.defineProperty(globalThis, 'navigator', { value: { locks }, configurable: true })
    try {
      await holdDirectoryLock('db', 50)
      await expect(holdDirectoryLock('db', 20)).rejects.toThrow(/still in use by another worker/)
      await holdDirectoryLock('other', 20)
    } finally {
      if (original) Object.defineProperty(globalThis, 'navigator', original)
      else delete (globalThis as { navigator?: unknown }).navigator
    }
  })
})
