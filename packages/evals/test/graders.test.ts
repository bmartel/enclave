/**
 * Proves the graders, not the model. For every production case:
 *  - the reference solution (ideal tool calls + answer, executed for real
 *    against PGlite and the skills) must PASS, so graders accept correct work;
 *  - a do-nothing model ("I'm not sure.") must FAIL, so graders reject
 *    non-answers, unless declining is the correct behaviour (nullPasses).
 * Runs on CPU in seconds; no GPU or model download needed.
 */
import { beforeAll, describe, expect, it } from 'vitest'
import { createEnclave, type Enclave, type Model, type ModelChunk } from '@enclave/core'
import { runEval } from '@enclave/core/eval'
import { createDb } from '@enclave/core/pglite'
import { hashEmbedder } from '@enclave/core/testing'
import { ALL_CASES } from '../src/suites/index.js'
import { prepareWorld, suiteSkills, type ProductionCase, type RefCall } from '../src/world.js'

type Step = { calls?: RefCall[]; text?: string }

/** A model that plays back a script of steps, resolving async references lazily. */
function scriptedModel(getAi: () => Enclave, steps: (() => Promise<Step>)[]): Model {
  let i = 0
  return {
    id: 'reference',
    locality: 'device',
    async *stream(): AsyncGenerator<ModelChunk> {
      const next = steps[i++]
      const step = next ? await next() : { text: '(script exhausted)' }
      void getAi
      if (step.text) yield { type: 'text', delta: step.text }
      for (const [n, call] of (step.calls ?? []).entries()) {
        yield { type: 'tool-call', call: { id: `ref_${i}_${n}`, name: call.name, input: call.input } }
      }
      yield { type: 'finish', reason: step.calls?.length ? 'tool-calls' : 'stop' }
    },
  }
}

function referenceSteps(testCase: ProductionCase, getAi: () => Enclave): (() => Promise<Step>)[] {
  return testCase.reference.flatMap((turn) => {
    const answer = async () => ({ text: typeof turn.answer === 'function' ? await turn.answer(getAi()) : turn.answer })
    if (!turn.calls) return [answer]
    const calls = async () => ({ calls: typeof turn.calls === 'function' ? await turn.calls(getAi()) : turn.calls })
    return [calls, answer]
  })
}

function turnCount(c: ProductionCase): number {
  return c.turns?.length ?? (Array.isArray(c.input) ? c.input.length : 1)
}

let ai: Enclave
let current: Model | undefined
const proxy: Model = {
  id: 'proxy',
  locality: 'device',
  stream: (req) => current!.stream(req),
}

beforeAll(async () => {
  const db = await createDb({ dataDir: 'memory://' })
  ai = await createEnclave({ db, model: proxy, embedder: hashEmbedder(64), skills: suiteSkills() })
  await prepareWorld(ai)
})

describe('suite structure', () => {
  it('has unique names, tags and one reference turn per turn', () => {
    const names = ALL_CASES.map((c) => c.name)
    expect(new Set(names).size).toBe(names.length)
    for (const c of ALL_CASES) {
      expect(c.tags.length, c.name).toBeGreaterThan(0)
      expect(c.reference.length, c.name).toBe(turnCount(c))
    }
  })

  it('covers every production area with enough cases', () => {
    const tags = ALL_CASES.flatMap((c) => c.tags)
    for (const [tag, min] of [['rag', 15], ['sql', 12], ['skill', 10], ['memory', 4], ['multi-turn', 8], ['safety', 8], ['restraint', 3]] as const) {
      expect(tags.filter((t) => t === tag).length, tag).toBeGreaterThanOrEqual(min)
    }
  })
})

describe('reference solutions pass (graders accept correct work)', () => {
  it.each(ALL_CASES.map((c) => [c.name, c] as const))('%s', async (_name, testCase) => {
    current = scriptedModel(() => ai, referenceSteps(testCase, () => ai))
    const report = await runEval(ai, [testCase])
    const result = report.results[0]!
    expect(result.failures, result.turns.map((t) => `turn ${t.index + 1}: ${t.text}`).join('\n')).toEqual([])
    expect(result.passed).toBe(true)
  })
})

/** The reference steps, with the final answer replaced by `text`. */
function withFinalAnswer(testCase: ProductionCase, text: string): (() => Promise<Step>)[] {
  const steps = referenceSteps(testCase, () => ai)
  steps[steps.length - 1] = async () => ({ text })
  return steps
}

const variantRows = ALL_CASES.flatMap((c) => [
  ...(c.variants?.pass ?? []).map((text) => [c.name, 'pass', text, c] as const),
  ...(c.variants?.fail ?? []).map((text) => [c.name, 'fail', text, c] as const),
])

describe('answer variants (graders are robust to phrasing and catch plausible mistakes)', () => {
  it('has variants for the graders most likely to misjudge', () => {
    expect(variantRows.length).toBeGreaterThanOrEqual(50)
  })
  it.each(variantRows)('%s must %s: %s', async (_name, outcome, text, testCase) => {
    current = scriptedModel(() => ai, withFinalAnswer(testCase, text))
    const report = await runEval(ai, [testCase])
    const result = report.results[0]!
    expect(result.passed, result.failures.join('; ')).toBe(outcome === 'pass')
  })
})

describe('a do-nothing model fails (graders reject non-answers)', () => {
  it.each(ALL_CASES.map((c) => [c.name, c] as const))('%s', async (_name, testCase) => {
    current = scriptedModel(() => ai, Array.from({ length: 20 }, () => async () => ({ text: "I'm not sure." })))
    const report = await runEval(ai, [testCase])
    expect(report.results[0]!.passed).toBe(!!testCase.nullPasses)
  })
})
