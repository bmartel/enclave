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
    expect(formatReport(report)).toMatch(/mock: 67% passed[\s\S]*✗ must search/)
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

