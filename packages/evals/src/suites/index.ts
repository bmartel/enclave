import { resetMemory, type ProductionCase } from '../world.js'
import { conversationCases } from './conversations.js'
import { crmCases } from './crm.js'
import { memoryCases } from './memory.js'
import { ragCases } from './rag.js'
import { safetyCases } from './safety.js'
import { sqlCases } from './sql.js'

export const SUITES = {
  rag: ragCases,
  sql: sqlCases,
  crm: crmCases,
  memory: memoryCases,
  conversations: conversationCases,
  safety: safetyCases,
} satisfies Record<string, ProductionCase[]>

/**
 * Every case starts with empty memory: a fact the model saved in one case
 * must not leak into the next. (It did: a CRM case read "the user's memory
 * includes a ticket for Wei Chen" saved by an earlier case.)
 */
const isolated = (c: ProductionCase): ProductionCase => ({
  ...c,
  setup: async (ai) => {
    await resetMemory(ai)
    await c.setup?.(ai)
  },
})

export const ALL_CASES: ProductionCase[] = Object.values(SUITES).flat().map(isolated)
