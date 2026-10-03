import { DEFAULT_SYSTEM, runAgent, SkillRegistry, type AgentRuntime } from './agent.js'
import { Knowledge, type KnowledgeOptions } from './rag/knowledge.js'
import type { Skill } from './skill.js'
import { CORE_MIGRATIONS, migrate } from './store/migrate.js'
import { assertLocality } from './privacy/index.js'
import type { AgentEvent, ApprovalHandler, Citation, Db, Embedder, FinishReason, KnowledgeScope, Locality, Message, Model, Reranker, Usage } from './types.js'
import { uid } from './util.js'

export interface EnclaveOptions {
  db: Db
  model: Model
  /** Enables `knowledge` (private RAG). Without it, knowledge-backed skills are unavailable. */
  embedder?: Embedder
  /** Cross-encoder for knowledge search. Improves precision of the top results. */
  reranker?: Reranker
  skills?: Skill[]
  /** Replaces the default base system prompt. Skill instructions are appended after it. */
  system?: string
  /** Max model calls per `send`/`run`. Default 12. */
  maxSteps?: number
  /** Max messages sent to the model per step. Default 40. */
  maxHistory?: number
  /** Tool output sent to the model is truncated to this. Default 12,000 chars. */
  maxToolOutputChars?: number
  /** Default handler for tools with `needsApproval`. Without one, such calls are denied. */
  onApproval?: ApprovalHandler
  knowledge?: KnowledgeOptions
  privacy?: PrivacyPolicy
}

export interface PrivacyPolicy {
  /**
   * The furthest a model, embedder or reranker may process data.
   * - `device`: in the browser, or a server on this machine (Ollama on localhost)
   * - `local-network` (default): also servers on private addresses you control
   * - `remote`: internet services; opt in explicitly
   * Components that don't declare a locality count as `remote`.
   */
  allow?: Locality
}

export interface RunOptions {
  signal?: AbortSignal
  onApproval?: ApprovalHandler
  /** Limit knowledge retrieval for this run to some collections, metadata or documents. */
  knowledge?: KnowledgeScope
}

export interface RunResult {
  /** Text of the final assistant message. */
  text: string
  /** Messages produced by this run (assistant + tool). */
  messages: Message[]
  finishReason: FinishReason | 'max-steps' | 'aborted'
  steps: number
  usage: Usage
  /** Passages the answer may cite as [n]. Empty when nothing was retrieved. */
  citations: Citation[]
}

/**
 * A single-consumer stream of agent events. Iterate it for live UI updates,
 * or `await stream.result()` when you only need the outcome.
 */
export class AgentStream implements AsyncIterable<AgentEvent> {
  private consumed = false
  private settled = false
  private readonly produced: Message[] = []
  private citations: Citation[] = []
  private resolveResult!: (r: RunResult) => void
  private rejectResult!: (e: unknown) => void
  private readonly resultPromise = new Promise<RunResult>((resolve, reject) => {
    this.resolveResult = resolve
    this.rejectResult = reject
  })

  constructor(private readonly source: AsyncGenerator<AgentEvent, void>) {
    this.resultPromise.catch(() => undefined)
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<AgentEvent, void> {
    if (this.consumed) throw new Error('AgentStream can only be consumed once')
    this.consumed = true
    try {
      for await (const event of this.source) {
        if (event.type === 'message') this.produced.push(event.message)
        if (event.type === 'citations') this.citations = event.citations
        if (event.type === 'finish') {
          const last = this.produced.findLast((m) => m.role === 'assistant')
          this.settled = true
          this.resolveResult({
            text: last?.content ?? '',
            messages: this.produced,
            finishReason: event.reason,
            steps: event.steps,
            usage: event.usage,
            citations: this.citations,
          })
        }
        yield event
      }
      if (!this.settled) this.rejectResult(new Error('Run ended without a finish event'))
    } catch (error) {
      this.settled = true
      this.rejectResult(error)
      throw error
    }
  }

  /** Resolves when the run finishes. Drains the stream if nobody is iterating it. */
  async result(): Promise<RunResult> {
    if (!this.consumed) for await (const _ of this) void _
    return this.resultPromise
  }

  async text(): Promise<string> {
    return (await this.result()).text
  }
}

export interface ThreadInfo {
  id: string
  title: string | null
  metadata: Record<string, unknown>
  createdAt: Date
  updatedAt: Date
}

/** A persisted conversation. History lives in PGlite and survives reloads. */
export class Thread {
  private history: Message[] | undefined
  private running = false

  constructor(
    readonly id: string,
    private readonly enclave: Enclave,
  ) {}

  send(input: string, options: RunOptions = {}): AgentStream {
    return new AgentStream(this.sendImpl(input, options))
  }

  private async *sendImpl(input: string, options: RunOptions): AsyncGenerator<AgentEvent, void> {
    if (this.running) throw new Error(`Thread ${this.id} already has a run in progress`)
    this.running = true
    const { db } = this.enclave
    try {
      await db.query(
        `insert into enclave.threads (id) values ($1) on conflict (id) do update set updated_at = now()`,
        [this.id],
      )
      const history = (this.history ??= await this.messages())
      const meta = await db.query<{ metadata: { activeSkills?: string[] } }>(
        'select metadata from enclave.threads where id = $1',
        [this.id],
      )
      const activeSkills = new Set(meta.rows[0]?.metadata.activeSkills ?? [])

      const user: Message = { role: 'user', content: input }
      history.push(user)
      await this.persist(user)
      yield { type: 'message', message: user }

      const skillsBefore = activeSkills.size
      for await (const event of runAgent(this.enclave.runtime, {
        history,
        threadId: this.id,
        activeSkills,
        signal: options.signal ?? new AbortController().signal,
        ...(options.onApproval ? { onApproval: options.onApproval } : {}),
        ...(options.knowledge ? { scope: options.knowledge } : {}),
      })) {
        if (event.type === 'message') await this.persist(event.message)
        yield event
      }
      if (activeSkills.size !== skillsBefore) {
        await db.query(
          `update enclave.threads set metadata = metadata || jsonb_build_object('activeSkills', $2::jsonb) where id = $1`,
          [this.id, JSON.stringify([...activeSkills])],
        )
      }
    } finally {
      this.running = false
    }
  }

  private async persist(message: Message): Promise<void> {
    await this.enclave.db.query('insert into enclave.messages (thread_id, message) values ($1, $2)', [
      this.id,
      JSON.stringify(message),
    ])
  }

  async messages(): Promise<Message[]> {
    const { rows } = await this.enclave.db.query<{ message: Message }>(
      'select message from enclave.messages where thread_id = $1 order by id',
      [this.id],
    )
    return rows.map((r) => r.message)
  }

  async rename(title: string): Promise<void> {
    await this.enclave.db.query(
      `insert into enclave.threads (id, title) values ($1, $2)
       on conflict (id) do update set title = excluded.title, updated_at = now()`,
      [this.id, title],
    )
  }

  async delete(): Promise<void> {
    await this.enclave.db.query('delete from enclave.threads where id = $1', [this.id])
    this.history = undefined
  }
}

export class Enclave {
  readonly runtime: AgentRuntime
  private readonly threadCache = new Map<string, Thread>()

  constructor(
    readonly db: Db,
    readonly knowledge: Knowledge | undefined,
    runtime: AgentRuntime,
    /** The locality ceiling enforced for every model, embedder and reranker. */
    readonly privacy: Required<PrivacyPolicy> = { allow: 'local-network' },
  ) {
    this.runtime = runtime
  }

  get model(): Model {
    return this.runtime.model
  }

  /** Swap the model at runtime (e.g. local → remote when online). */
  setModel(model: Model): void {
    assertLocality(`Model ${model.id}`, model.locality, this.privacy.allow)
    this.runtime.model = model
  }

  get skills(): Skill[] {
    return [...this.runtime.registry.skills.values()]
  }

  /** Register a skill after creation. Runs its migrations and setup. */
  async use(skill: Skill): Promise<void> {
    this.runtime.registry.add(skill)
    try {
      await installSkill(this, skill)
    } catch (error) {
      this.runtime.registry.skills.delete(skill.name)
      for (const name of Object.keys(skill.tools ?? {})) this.runtime.registry.tools.delete(name)
      throw error
    }
  }

  /** Open (or lazily create) a persisted thread. */
  thread(id: string = uid('t_')): Thread {
    let thread = this.threadCache.get(id)
    if (!thread) this.threadCache.set(id, (thread = new Thread(id, this)))
    return thread
  }

  async threads(): Promise<ThreadInfo[]> {
    const { rows } = await this.db.query<{
      id: string
      title: string | null
      metadata: Record<string, unknown>
      created_at: Date
      updated_at: Date
    }>('select * from enclave.threads order by updated_at desc')
    return rows.map((r) => ({
      id: r.id,
      title: r.title,
      metadata: r.metadata,
      createdAt: r.created_at,
      updatedAt: r.updated_at,
    }))
  }

  /** Stateless run over the given history; nothing is persisted. */
  run(input: string | Message[], options: RunOptions = {}): AgentStream {
    const history: Message[] = typeof input === 'string' ? [{ role: 'user', content: input }] : [...input]
    return new AgentStream(
      runAgent(this.runtime, {
        history,
        threadId: undefined,
        activeSkills: new Set(),
        signal: options.signal ?? new AbortController().signal,
        ...(options.onApproval ? { onApproval: options.onApproval } : {}),
        ...(options.knowledge ? { scope: options.knowledge } : {}),
      }),
    )
  }
}

async function installSkill(enclave: Enclave, skill: Skill): Promise<void> {
  if (skill.migrations?.length) await migrate(enclave.db, `skill:${skill.name}`, skill.migrations)
  await skill.setup?.({
    db: enclave.db,
    knowledge: enclave.knowledge,
    embedder: enclave.runtime.embedder,
    threadId: undefined,
    messages: [],
  })
}

export async function createEnclave(options: EnclaveOptions): Promise<Enclave> {
  const { db, embedder } = options
  const privacy = { allow: options.privacy?.allow ?? 'local-network' }
  assertLocality(`Model ${options.model.id}`, options.model.locality, privacy.allow)
  if (embedder) assertLocality(`Embedder ${embedder.id}`, embedder.locality, privacy.allow)
  if (options.reranker) assertLocality(`Reranker ${options.reranker.id}`, options.reranker.locality, privacy.allow)
  await migrate(db, 'core', CORE_MIGRATIONS)

  const knowledge = embedder
    ? new Knowledge(db, embedder, { ...options.knowledge, ...(options.reranker ? { reranker: options.reranker } : {}) })
    : undefined
  await knowledge?.init()

  const enclave = new Enclave(db, knowledge, {
    model: options.model,
    db,
    knowledge,
    embedder,
    registry: new SkillRegistry(),
    system: options.system ?? DEFAULT_SYSTEM,
    maxSteps: options.maxSteps ?? 12,
    maxHistory: options.maxHistory ?? 40,
    maxToolOutputChars: options.maxToolOutputChars ?? 12_000,
    onApproval: options.onApproval,
  }, privacy)
  for (const skill of options.skills ?? []) await enclave.use(skill)
  return enclave
}
