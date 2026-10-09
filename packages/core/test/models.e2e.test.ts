/**
 * Real-model checks on CPU (onnxruntime-node). Downloads ~1.5 GB on first run.
 *   pnpm --filter enclave-ai test:e2e
 */
import { describe, expect, it } from 'vitest'
import { z } from 'zod'
import { createEnclave, defineSkill, tool, type AgentEvent } from '../src/index.js'
import { knowledgeSkill } from '../src/skills/index.js'
import { transformersEmbedder, transformersLLM, transformersReranker } from '../src/transformers/index.js'
import { collect, memoryDb } from './helpers.js'

const run = !!process.env.ENCLAVE_E2E
const cos = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i]!, 0)

const QUERY = 'Which planet is known as the Red Planet?'
const DOCS = [
  "Venus is often called Earth's twin because of its similar size and proximity.",
  'Mars, known for its reddish appearance, is often referred to as the Red Planet.',
  'Jupiter, the largest planet in our solar system, has a prominent red spot.',
  'Saturn, famous for its rings, is sometimes mistaken for the Red Planet.',
]

describe.skipIf(!run)('embedding presets (real weights)', () => {
  for (const preset of ['granite-small-r2', 'granite-multilingual-r2', 'embeddinggemma', 'embeddinggemma-2']) {
    it(`${preset} ranks the relevant passage first`, async () => {
      const e = transformersEmbedder({ preset })
      const [q] = await e.embed([QUERY], 'query')
      const docs = await e.embed(DOCS.map((d) => e.formatDocument!(d)), 'document')
      expect(q).toHaveLength(e.dimensions)
      expect(Math.hypot(...q!)).toBeCloseTo(1, 3)
      const scores = docs.map((d) => cos(q!, d))
      expect(scores.indexOf(Math.max(...scores))).toBe(1)
    }, 300_000)
  }

  it('truncates EmbeddingGemma to a Matryoshka size and renormalizes', async () => {
    const e = transformersEmbedder({ preset: 'embeddinggemma', dimensions: 256 })
    const [v] = await e.embed(['hello world'], 'query')
    expect(v).toHaveLength(256)
    expect(Math.hypot(...v!)).toBeCloseTo(1, 3)
    expect(e.id).toBe('transformers:onnx-community/embeddinggemma-300m-ONNX@256')
    expect(() => transformersEmbedder({ preset: 'embeddinggemma', dimensions: 300 })).toThrow(/supports/)
  }, 300_000)
})

describe.skipIf(!run)('reranker (real weights)', () => {
  it('mxbai-rerank-xsmall scores the answer highest', async () => {
    const r = transformersReranker({ preset: 'mxbai-rerank-xsmall' })
    const scores = await r.rerank(QUERY, DOCS)
    expect(scores).toHaveLength(4)
    expect(scores.indexOf(Math.max(...scores))).toBe(1)
    for (const s of scores) expect(s).toBeGreaterThanOrEqual(0)
  }, 300_000)
})

describe.skipIf(!run)('agent with a real small LLM (Qwen3 0.6B, CPU)', () => {
  const llm = () => transformersLLM({ model: 'tjs-qwen3-0.6b', dtype: 'q4', maxNewTokens: 256 })

  it('calls a tool and uses its result', async () => {
    const weather = defineSkill({
      name: 'weather',
      description: 'Weather lookups',
      instructions: 'Use get_weather to answer weather questions.',
      tools: {
        get_weather: tool({
          description: 'Get the current weather for a city.',
          input: z.object({ city: z.string() }),
          execute: ({ city }) => ({ city, temperatureC: 17, conditions: 'light rain' }),
        }),
      },
    })
    const db = await memoryDb()
    const ai = await createEnclave({ db, model: llm(), skills: [weather], maxSteps: 4 })
    const events: AgentEvent[] = await collect(ai.run("What's the weather in Paris right now?"))
    const call = events.find((e) => e.type === 'tool-call')
    expect(call).toBeDefined()
    expect(call!.type === 'tool-call' && call!.call.name).toBe('get_weather')
    expect(JSON.stringify(call!.type === 'tool-call' && call!.call.input)).toMatch(/paris/i)
    const text = events.flatMap((e) => (e.type === 'text-delta' ? [e.delta] : [])).join('')
    console.log('[qwen3-0.6b answer]', text.trim())
    expect(text).toMatch(/17|rain/i)
    await db.close()
  }, 600_000)

  it('answers from private documents via hybrid search + rerank', async () => {
    const db = await memoryDb()
    const ai = await createEnclave({
      db,
      model: llm(),
      embedder: transformersEmbedder({ preset: 'granite-small-r2' }),
      reranker: transformersReranker({ preset: 'mxbai-rerank-xsmall' }),
      skills: [knowledgeSkill()],
      maxSteps: 4,
    })
    await ai.knowledge!.ingest([
      { title: 'Office wifi', content: 'The guest wifi network is called Visitors and the password is sunflower42.' },
      { title: 'Parking', content: 'Staff parking is on level B2. Visitors park on level B1.' },
      { title: 'Lunch', content: 'The cafeteria serves lunch from 11:30 to 14:00.' },
    ])
    const hits = await ai.knowledge!.search('guest wifi password')
    expect(hits[0]!.title).toBe('Office wifi')
    expect(hits[0]!.rerankScore).toBeGreaterThan(hits[1]!.rerankScore!)

    // Auto-retrieval puts the reranked passage in context, so even a 0.6B model answers correctly.
    const events = await collect(ai.run('What is the guest wifi password?'))
    const text = events.flatMap((e) => (e.type === 'text-delta' ? [e.delta] : [])).join('')
    console.log('[qwen3-0.6b answer]', text.trim())
    expect(text).toMatch(/sunflower42/i)
    await db.close()
  }, 600_000)
})
