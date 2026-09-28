import type { ProductionCase } from '../world.js'
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

export const ALL_CASES: ProductionCase[] = Object.values(SUITES).flat()
