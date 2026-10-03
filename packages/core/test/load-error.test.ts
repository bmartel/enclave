import { describe, expect, it } from 'vitest'
import { classifyLoadError, ModelLoadError } from '../src/index.js'
import { toModelLoadError } from '../src/models/load-error.js'

const domError = (name: string, message: string) => Object.assign(new Error(message), { name })

describe('model load errors', () => {
  it('classifies the failures browsers and runtimes actually throw', () => {
    expect(classifyLoadError(domError('QuotaExceededError', 'Quota exceeded.'))).toBe('storage')
    expect(classifyLoadError(new Error('Failed to execute write: No space available for this operation'))).toBe('storage')
    expect(classifyLoadError(new Error('WebGPU is not supported in your current environment'))).toBe('webgpu')
    expect(classifyLoadError(new Error('Unable to find a compatible GPU adapter (requestAdapter returned null)'))).toBe('webgpu')
    expect(classifyLoadError(new Error('Device was lost. This can happen due to insufficient memory'))).toBe('gpu-memory')
    expect(classifyLoadError(new TypeError('Failed to fetch'))).toBe('network')
    expect(classifyLoadError(new Error('Could not load model: status code 404'))).toBe('network')
    expect(classifyLoadError(new Error('Something odd'))).toBe('unknown')
  })

  it('wraps load failures with a user-facing message and keeps the cause', () => {
    const cause = domError('QuotaExceededError', 'Quota exceeded.')
    const err = toModelLoadError(cause, 'qwen3-4b') as ModelLoadError
    expect(err).toBeInstanceOf(ModelLoadError)
    expect(err.name).toBe('ModelLoadError')
    expect(err.reason).toBe('storage')
    expect(err.model).toBe('qwen3-4b')
    expect(err.cause).toBe(cause)
    expect(err.message).toMatch(/isn't enough browser storage to download qwen3-4b/)
    // Unknown failures keep the original detail.
    expect((toModelLoadError(new Error('weird'), 'm') as Error).message).toContain('weird')
  })

  it('passes aborts and existing load errors through unchanged', () => {
    const abort = domError('AbortError', 'aborted')
    expect(toModelLoadError(abort, 'm')).toBe(abort)
    const existing = new ModelLoadError('network', 'm')
    expect(toModelLoadError(existing, 'other')).toBe(existing)
  })
})

import { reviveError } from '../src/transformers/protocol.js'

describe('worker error transport', () => {
  it('rebuilds load errors and keeps DOMException names across the worker boundary', () => {
    const load = reviveError({ error: 'msg', name: 'ModelLoadError', reason: 'storage', model: 'embeddinggemma' })
    expect(load).toBeInstanceOf(ModelLoadError)
    expect((load as ModelLoadError).reason).toBe('storage')
    expect(load.message).toMatch(/embeddinggemma/)
    const quota = reviveError({ error: 'Quota exceeded.', name: 'QuotaExceededError' })
    expect(quota.name).toBe('QuotaExceededError')
    expect(classifyLoadError(quota)).toBe('storage')
  })
})
