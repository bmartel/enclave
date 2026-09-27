import { afterEach, describe, expect, it } from 'vitest'
import { createEnclave } from '../src/index.js'
import { ollama } from '../src/models/local.js'
import { openaiCompatible } from '../src/models/openai.js'
import { anthropic } from '../src/models/anthropic.js'
import { fallback } from '../src/models/fallback.js'
import { selfHostedAppConfig } from '../src/models/webllm.js'
import {
  contentSecurityPolicy,
  guardNetwork,
  localityOfUrl,
  PrivacyError,
  selfHostedTransformers,
  widest,
} from '../src/privacy/index.js'
import { hashEmbedder, mockModel } from '../src/testing.js'
import { memoryDb } from './helpers.js'

describe('locality', () => {
  it('classifies endpoints', () => {
    expect(localityOfUrl('http://localhost:11434')).toBe('device')
    expect(localityOfUrl('http://127.0.0.1:1234/v1')).toBe('device')
    expect(localityOfUrl('http://[::1]:8080')).toBe('device')
    expect(localityOfUrl('http://192.168.1.20:11434')).toBe('local-network')
    expect(localityOfUrl('http://10.0.0.5')).toBe('local-network')
    expect(localityOfUrl('http://gpu-box.local:1234')).toBe('local-network')
    expect(localityOfUrl('https://api.openai.com/v1')).toBe('remote')
    expect(widest('device', 'local-network')).toBe('local-network')
    expect(widest('device', undefined)).toBe('remote')
  })

  it('declares locality on adapters', () => {
    expect(ollama({ model: 'm' }).locality).toBe('device')
    expect(ollama({ model: 'm', baseURL: 'http://192.168.0.9:11434' }).locality).toBe('local-network')
    expect(openaiCompatible({ baseURL: 'https://openrouter.ai/api/v1', model: 'x' }).locality).toBe('remote')
    expect(anthropic({ apiKey: 'test' }).locality).toBe('remote')
    expect(fallback(mockModel([]), anthropic({ apiKey: 'test' })).locality).toBe('remote')
  })
})

describe('privacy policy', () => {
  it('refuses remote models by default and allows explicit opt-in', async () => {
    const db = await memoryDb()
    const remote = anthropic({ apiKey: 'test' })
    await expect(createEnclave({ db, model: remote })).rejects.toThrow(PrivacyError)
    await expect(createEnclave({ db, model: mockModel([]), embedder: { ...hashEmbedder(), locality: undefined } })).rejects.toThrow(/unknown location/)

    const ai = await createEnclave({ db, model: mockModel([]) })
    expect(ai.privacy.allow).toBe('local-network')
    expect(() => ai.setModel(remote)).toThrow(/only allows "local-network"/)
    ai.setModel(ollama({ model: 'qwen3:8b' }))

    const strict = await createEnclave({ db, model: mockModel([]), privacy: { allow: 'device' } })
    expect(() => strict.setModel(ollama({ model: 'm', baseURL: 'http://10.1.1.1:11434' }))).toThrow(PrivacyError)

    const open = await createEnclave({ db, model: remote, privacy: { allow: 'remote' } })
    expect(open.model.id).toMatch(/anthropic/)
    await db.close()
  })
})

describe('guardNetwork', () => {
  let restore: (() => void) | undefined
  afterEach(() => restore?.())

  it('blocks unlisted origins and reports violations', async () => {
    const violations: string[] = []
    const original = globalThis.fetch
    globalThis.fetch = (async () => new Response('ok')) as typeof fetch
    try {
      restore = guardNetwork({ allow: ['https://models.example.com', 'https://*.hf.co'], onViolation: (u) => violations.push(u) })
      await expect(fetch('https://models.example.com/x')).resolves.toBeInstanceOf(Response)
      await expect(fetch('https://cas-bridge.xethub.hf.co/y')).resolves.toBeInstanceOf(Response)
      await expect(fetch('https://tracker.example.net/collect')).rejects.toThrow(/blocked/)
      expect(violations).toEqual(['https://tracker.example.net/collect'])
      restore()
      restore = guardNetwork({ allow: [], mode: 'report', onViolation: (u) => violations.push(u) })
      await expect(fetch('https://elsewhere.example.org')).resolves.toBeInstanceOf(Response)
      expect(violations).toHaveLength(2)
    } finally {
      restore?.()
      restore = undefined
      globalThis.fetch = original
    }
  })
})

describe('deployment helpers', () => {
  it('builds a CSP that pins connect-src', () => {
    const selfHosted = contentSecurityPolicy({ modelHosts: [] })
    expect(selfHosted).toContain("connect-src 'self' blob: data:;")
    expect(selfHosted).toContain("'wasm-unsafe-eval'")
    expect(selfHosted).toContain("form-action 'none'")
    expect(contentSecurityPolicy({ localServers: true })).toMatch(/connect-src[^;]*https:\/\/huggingface\.co[^;]*http:\/\/localhost:11434/)
  })

  it('rewrites WebLLM and Transformers.js sources to a self-hosted base', async () => {
    const config = await selfHostedAppConfig({ baseUrl: 'https://intranet.example/models/', models: ['qwen3-4b'] })
    expect(config.model_list).toHaveLength(1)
    expect(config.model_list[0]).toMatchObject({
      model_id: 'Qwen3-4B-q4f16_1-MLC',
      model: 'https://intranet.example/models/mlc/Qwen3-4B-q4f16_1-MLC/',
      model_lib: expect.stringMatching(/^https:\/\/intranet\.example\/models\/mlc\/libs\/Qwen3-4B-q4f16_1.*\.wasm$/),
    })
    expect(selfHostedTransformers({ baseUrl: 'https://intranet.example/models' })).toEqual({
      remoteHost: 'https://intranet.example/models/hf/',
      remotePathTemplate: '{model}/resolve/{revision}/',
      wasmPaths: {
        mjs: 'https://intranet.example/models/ort/ort-wasm-simd-threaded.asyncify.mjs',
        wasm: 'https://intranet.example/models/ort/ort-wasm-simd-threaded.asyncify.wasm',
      },
      allowRemoteModels: true,
    })
  })
})
