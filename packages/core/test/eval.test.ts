import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import type { PGlite } from '@electric-sql/pglite'
import { createEnclave } from '../src/index.js'
import { evalRetrieval, formatReport, runEval } from '../src/eval/index.js'
import { Knowledge } from '../src/rag/knowledge.js'
import { sqlSkill } from '../src/skills/index.js'
import { CORE_MIGRATIONS, migrate } from '../src/store/migrate.js'
import { hashEmbedder, mockModel } from '../src/testing.js'
import { memoryDb } from './helpers.js'

let db: PGlite
beforeEach(async () => {
  db = await memoryDb()
})
afterEach(() => db.close())

describe('runEval', () => {
  it('grades tool use, answers and custom checks per case', async () => {
    const model = mockModel([
      // case 1: correct
      { toolCalls: [{ name: 'execute_sql', input: { sql: 'create table t (x int); insert into t values (1), (2)' } }] },
      'Created t with 2 rows.',
      // case 2: answers without the required tool, and hallucinates
      'The password is hunter2.',
      // case 3: multi-turn; only the last turn is graded
      'Hi!',
      'Bonjour !',
    ])
    const ai = await createEnclave({ db, model, skills: [sqlSkill()] })
    const report = await runEval(ai, [
      {
        name: 'create table',
        input: 'make table t with two rows',
        setup: async (e) => void (await e.db.exec('drop table if exists t')),
        expect: {
          tools: ['execute_sql'],
          answer: /2 rows/,
          check: async ({ ai: e }) => {
            const { rows } = await e.db.query<{ n: number }>('select count(*)::int as n from t')
            return rows[0]!.n === 2 || `expected 2 rows, got ${rows[0]!.n}`
          },
        },
      },
      { name: 'must search', input: 'wifi password?', expect: { tools: ['execute_sql'], notAnswer: 'hunter2' } },
      { name: 'french', input: ['hello', 'say hello in French'], expect: { noTools: true, answer: 'bonjour', maxSteps: 1 } },
    ], { label: 'mock' })

    expect(report.byCase.map((c) => [c.name, c.passRate])).toEqual([
      ['create table', 1],
      ['must search', 0],
      ['french', 1],
    ])
    expect(report.results[1]!.failures).toEqual(['did not call execute_sql', 'answer matches forbidden hunter2'])
    expect(report.passRate).toBeCloseTo(2 / 3)
    expect(report.results[0]!.steps).toBe(2)
    expect(report.tokens.meanOutput).toBeGreaterThan(0)
    expect(formatReport(report)).toMatch(/mock: 67% \[\d+%–\d+%\] of 3 runs passed[\s\S]*✗ must search/)
    // Eval threads are cleaned up.
    expect(await ai.threads()).toEqual([])
  })
})

describe('evalRetrieval', () => {
  it('computes recall, MRR and nDCG at k', async () => {
    await migrate(db, 'core', CORE_MIGRATIONS)
    const kb = new Knowledge(db, hashEmbedder(64))
    await kb.ingest([
      { id: 'wifi', content: 'guest wifi password sunflower' },
      { id: 'parking', content: 'visitors park on level B1' },
      { id: 'lunch', content: 'lunch is served at noon' },
    ])
    const report = await evalRetrieval(kb, [
      { query: 'wifi password', relevant: ['wifi'] },
      { query: 'where to park', relevant: ['parking'] },
      { query: 'unrelated banana', relevant: ['lunch'] },
    ], { k: 2 })
    expect(report.perQuery[0]).toMatchObject({ recall: 1, reciprocalRank: 1 })
    expect(report.perQuery[1]!.retrieved[0]).toBe('parking')
    expect(report.mrr).toBeGreaterThan(0.6)
    expect(report.ndcg).toBeGreaterThan(0.6)
  })
})


describe('multi-turn evals', () => {
  it('grades every turn and reports later-turn metrics', async () => {
    const model = mockModel([
      { toolCalls: [{ name: 'execute_sql', input: { sql: 'create table notes (x text)' } }] },
      'Created.',
      'Your first note was about milk.', // turn 2 passes
      'I do not remember.', // turn 3 fails
    ])
    const ai = await createEnclave({ db, model, skills: [sqlSkill()] })
    const report = await runEval(ai, [
      {
        name: 'conversation',
        setup: async (e) => void (await e.db.exec('drop table if exists notes')),
        turns: [
          { input: 'make a notes table', expect: { tools: ['execute_sql'] } },
          { input: 'what was my first note?', expect: { answer: 'milk', noTools: true } },
          { input: 'and the second?', expect: { answer: 'eggs' } },
        ],
      },
    ])
    const [result] = report.results
    expect(result!.turns.map((t) => [t.index, t.graded, t.passed])).toEqual([
      [0, true, true],
      [1, true, true],
      [2, true, false],
    ])
    expect(result!.passed).toBe(false)
    expect(result!.failures).toEqual(['turn 3: answer does not match eggs'])
    expect(report.laterTurns).toMatchObject({ count: 2, passRate: 0.5 })
    expect(formatReport(report)).toContain('later turns (2): 50% passed')
  })
})

describe('fast-fail runs', () => {
  // A model that answers "pass" or "fail" from a script, one entry per run.
  const scripted = (answers: string[]) => mockModel(answers)
  const cases = ['a', 'b', 'c', 'd'].map((name) => ({ name, input: name, expect: { answer: 'pass' } }))

  it('failFast stops at the first failing run and marks the report', async () => {
    const ai = await createEnclave({ db, model: scripted(['pass', 'fail', 'pass', 'pass']) })
    const report = await runEval(ai, cases, { failFast: true })
    expect(report.results.map((r) => r.name)).toEqual(['a', 'b'])
    expect(report.stopped).toMatchObject({ reason: 'fail-fast: b #0', ranRuns: 2, plannedRuns: 4 })
    expect(formatReport(report)).toMatch(/^STOPPED EARLY/)
  })

  it('maxFailures stops once failures exceed the budget', async () => {
    const ai = await createEnclave({ db, model: scripted(['fail', 'pass', 'fail', 'pass']) })
    const report = await runEval(ai, cases, { maxFailures: 1 })
    expect(report.results.map((r) => r.name)).toEqual(['a', 'b', 'c'])
    expect(report.stopped?.reason).toBe('more than 1 failing runs')
  })

  it('runs listed cases first, then the rest in order', async () => {
    const ai = await createEnclave({ db, model: scripted(['pass', 'pass', 'pass', 'pass']) })
    const report = await runEval(ai, cases, { first: ['c', 'a'] })
    expect(report.results.map((r) => r.name)).toEqual(['c', 'a', 'b', 'd'])
    expect(report.stopped).toBeUndefined()
  })
})
