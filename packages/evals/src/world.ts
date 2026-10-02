import type { Enclave, Skill } from 'enclave-ai'
import type { EvalCase } from 'enclave-ai/eval'
import { knowledgeSkill, memorySkill, sqlSkill } from 'enclave-ai/skills'
import { CORPUS_COLLECTION, FULL_CORPUS } from './fixtures/corpus.js'
import { seedBusinessDb } from './fixtures/business-db.js'
import { crmSkill, seedCrm } from './fixtures/crm-skill.js'

export interface RefCall {
  name: string
  input: unknown
}

/** The ideal behaviour for one turn. Used to prove the graders accept correct work. */
export interface RefTurn {
  calls?: RefCall[] | ((ai: Enclave) => Promise<RefCall[]>)
  answer: string | ((ai: Enclave) => Promise<string>)
}

export interface ProductionCase extends EvalCase {
  tags: string[]
  /** One entry per turn. */
  reference: RefTurn[]
  /**
   * A do-nothing model ("I'm not sure.") legitimately passes this case, e.g.
   * questions the corpus can't answer. Every other case must fail it.
   */
  nullPasses?: boolean
  /**
   * Alternative final answers for grader robustness: `pass` are other correct
   * phrasings that must be accepted; `fail` are plausible wrong answers that
   * must be rejected. Each is graded with the reference tool calls.
   */
  variants?: { pass?: string[]; fail?: string[] }
}

/**
 * The skills every production case runs with, as a real app would configure
 * them. Memory is eager: a lazy skill's context isn't shown until activated,
 * so remembered facts would be invisible in new threads.
 */
export function suiteSkills(): Skill[] {
  return [knowledgeSkill(), sqlSkill(), crmSkill, memorySkill()]
}

/** Load the corpus and seed both databases. Call once per enclave. */
export async function prepareWorld(ai: Enclave): Promise<void> {
  await ai.knowledge!.clear(CORPUS_COLLECTION)
  await ai.knowledge!.ingest(FULL_CORPUS, { collection: CORPUS_COLLECTION })
  await resetData(ai)
}

/** Restore the business and CRM data, for cases that change it. */
export async function resetData(ai: Enclave): Promise<void> {
  await seedBusinessDb(ai.db)
  await seedCrm(ai.db)
}

export const resetMemory = async (ai: Enclave) => {
  await ai.knowledge!.clear('memory')
}
