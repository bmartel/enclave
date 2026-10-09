export { createEnclave, Enclave, Thread, AgentStream } from './enclave.js'
export type { EnclaveOptions, RunOptions, RunResult, ThreadInfo } from './enclave.js'
export { defineSkill } from './skill.js'
export type { Skill, SkillContext } from './skill.js'
export { tool, toJSONSchema } from './tool.js'
export type { ToolDef, ToolContext } from './tool.js'
export { Knowledge, toOrQuery } from './rag/knowledge.js'
export type {
  KnowledgeOptions,
  IngestDocument,
  IngestOptions,
  IngestResult,
  SearchOptions,
  SearchHit,
  CollectionInfo,
  EmbedderRecord,
  ReindexStatus,
} from './rag/knowledge.js'
export { chunkText, type ChunkOptions } from './rag/chunk.js'
export { synthesize, citedNumbers, type SynthesisEvent, type SynthesisSource, type SynthesizeOptions } from './rag/synthesize.js'
export { migrate } from './store/migrate.js'
export { fromTextModel, renderMessages, TaggedStreamParser, type TextModel, type TextRequest } from './models/text-protocol.js'
export { fallback } from './models/fallback.js'
export { runAgent, trimHistory, fitHistory, DEFAULT_SYSTEM, ACTIVATE_SKILL } from './agent.js'
export { PrivacyError } from './privacy/index.js'
export { dateContext, reasoningLoops } from './util.js'
export type { PrivacyPolicy } from './enclave.js'
export type * from './types.js'
export { ModelLoadError, classifyLoadError, type ModelLoadReason } from './models/load-error.js'
export { checkStorage, isBrowserRefusingStorage, type StorageCheck, type StorageCheckOptions, type StorageProbeResult } from './store/storage-check.js'
