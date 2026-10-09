import type { Db, Embedder, EmbedTask, Reranker } from '../types.js'
import { kvGet, kvSet } from '../store/migrate.js'
import { sha256, toVectorLiteral } from '../util.js'
import { chunkText, type ChunkOptions } from './chunk.js'

export interface KnowledgeOptions {
  /** Postgres text search config used for keyword search. Default `english`. */
  language?: string
  /** Texts per embedding call. Default 32. */
  batchSize?: number
  chunk?: ChunkOptions
  /** Search mode used when a search doesn't specify one. Default `hybrid`. */
  defaultMode?: 'hybrid' | 'vector' | 'keyword'
  /** Cross-encoder applied to the top candidates of hybrid/vector search. */
  reranker?: Reranker
  /** Candidates fetched for reranking. Default max(limit × 3, 24), capped at 60. */
  rerankCandidates?: number
  /**
   * When the configured embedder differs from the one that built the index,
   * re-embed everything instead of throwing. Useful when users can switch models.
   */
  autoReindex?: boolean
  onReindexProgress?(done: number, total: number): void
  /**
   * With `autoReindex`: don't re-embed during `init()`. The new vectors are
   * written next to the old ones, a batch per `reindexStep()` (the host calls
   * it while the device is idle); until every chunk has one, search runs in
   * keyword mode and new documents get only the new vectors. The last step
   * swaps the columns and rebuilds the vector index. It survives restarts:
   * the embedder being moved to is kept in `knowledge.embedder.next`.
   */
  reindexInBackground?: boolean
}

/** Which embedder produced the index's vectors. */
export interface EmbedderRecord {
  id: string
  dimensions: number
}

/** A background re-embedding in progress (see `KnowledgeOptions.reindexInBackground`). */
export interface ReindexStatus {
  from: EmbedderRecord
  to: EmbedderRecord
  /** Chunks that have a vector from `to`. */
  done: number
  total: number
}

export interface IngestDocument {
  /** Stable id. Defaults to a hash of `source` (or of the content when no source). */
  id?: string
  content: string
  title?: string
  source?: string
  collection?: string
  metadata?: Record<string, unknown>
}

export interface IngestOptions {
  collection?: string
  chunk?: ChunkOptions
  signal?: AbortSignal
  onProgress?(progress: { done: number; total: number; document: string }): void
}

export interface IngestResult {
  documents: number
  chunks: number
  /** Documents whose metadata changed but text did not (updated without re-embedding). */
  updated: number
  /** Unchanged documents skipped via content hash. */
  skipped: number
}

export interface SearchOptions {
  collection?: string | string[]
  limit?: number
  /** Match documents whose metadata contains this object (jsonb `@>`). */
  filter?: Record<string, unknown>
  mode?: 'hybrid' | 'vector' | 'keyword'
  /** Drop hits whose cosine similarity is below this. Hybrid and vector modes only. */
  minSimilarity?: number
  /** Use the configured reranker. Default true when one is configured. */
  rerank?: boolean
  /** Only search these documents. */
  documentIds?: string[]
  /** What the query is for: task-prompted embedders (EmbeddingGemma) embed it for that task. Default `search`. */
  task?: EmbedTask
}

export interface SearchHit {
  chunkId: number
  documentId: string
  collection: string
  title: string | null
  source: string | null
  content: string
  ordinal: number
  metadata: Record<string, unknown>
  /** Fused rank score (higher is better). */
  score: number
  /** Cosine similarity, when a vector was involved. */
  similarity: number | null
  /** Cross-encoder relevance in [0, 1], when reranked. */
  rerankScore?: number
}

export interface CollectionInfo {
  collection: string
  documents: number
  chunks: number
}

const RRF_K = 60
const EMBEDDER_KEY = 'knowledge.embedder'
const NEXT_KEY = 'knowledge.embedder.next'
const sameEmbedder = (a: EmbedderRecord, b: EmbedderRecord) => a.id === b.id && a.dimensions === b.dimensions
/** A JSON value as a SQL string literal (for the one multi-statement swap). */
const sqlJson = (value: unknown) => `'${JSON.stringify(value).replace(/'/g, "''")}'::jsonb`
const IDENT = /^[a-z_][a-z0-9_]*$/

/**
 * Private, on-device retrieval: documents are chunked, embedded locally and
 * stored in PGlite with an HNSW vector index plus a full-text index. Search
 * fuses both rankings with Reciprocal Rank Fusion.
 */
export class Knowledge {
  readonly language: string
  private ready: Promise<void> | undefined
  /** Set while vectors are being rebuilt in the background for a new embedder. */
  private migrating: { from: EmbedderRecord; to: EmbedderRecord } | null = null
  /** Chunk writes, re-embedding steps and the final swap run one at a time. */
  private writes: Promise<unknown> = Promise.resolve()

  constructor(
    readonly db: Db,
    readonly embedder: Embedder,
    private readonly options: KnowledgeOptions = {},
  ) {
    this.language = options.language ?? 'english'
    if (!IDENT.test(this.language)) throw new Error(`Invalid text search language "${this.language}"`)
  }

  /** Create tables on first use and verify the embedder matches the stored index. */
  init(): Promise<void> {
    return (this.ready ??= this.setup())
  }

  private async setup(): Promise<void> {
    const dims = this.embedder.dimensions
    if (!Number.isInteger(dims) || dims <= 0) throw new Error(`Embedder reports invalid dimensions: ${dims}`)
    await this.db.exec(`
      create extension if not exists vector;
      create table if not exists enclave.documents (
        id text primary key,
        collection text not null,
        title text,
        source text,
        metadata jsonb not null default '{}',
        content_hash text not null,
        chunk_count int not null default 0,
        created_at timestamptz not null default now(),
        updated_at timestamptz not null default now()
      );
      create index if not exists documents_collection_idx on enclave.documents (collection);
      create table if not exists enclave.chunks (
        id bigint generated always as identity primary key,
        document_id text not null references enclave.documents(id) on delete cascade,
        collection text not null,
        ordinal int not null,
        content text not null,
        embedding vector(${dims}) not null,
        fts tsvector generated always as (to_tsvector('${this.language}', content)) stored
      );
      create index if not exists chunks_document_idx on enclave.chunks (document_id);
      create index if not exists chunks_collection_idx on enclave.chunks (collection);
      create index if not exists chunks_embedding_idx on enclave.chunks using hnsw (embedding vector_cosine_ops);
      create index if not exists chunks_fts_idx on enclave.chunks using gin (fts);
    `)
    const want: EmbedderRecord = { id: this.embedder.id, dimensions: dims }
    const stored = await kvGet<EmbedderRecord>(this.db, EMBEDDER_KEY)
    const next = (await kvGet<EmbedderRecord | null>(this.db, NEXT_KEY)) ?? null
    if (!stored) {
      await kvSet(this.db, EMBEDDER_KEY, want)
    } else if (sameEmbedder(stored, want)) {
      // Back to the index's own embedder in the middle of a move to another one.
      if (next) await this.abandonBackground()
    } else if (this.options.autoReindex && this.options.reindexInBackground) {
      await this.beginBackground(stored, want, next)
    } else if (this.options.autoReindex) {
      await this.rebuild(this.options.onReindexProgress ? { onProgress: this.options.onReindexProgress } : {})
    } else {
      throw new Error(
        `Knowledge index was built with embedder "${stored.id}" (${stored.dimensions}d) but "${this.embedder.id}" ` +
          `(${dims}d) is configured. Call knowledge.reindex() to rebuild it.`,
      )
    }
  }

  /** Start (or resume) moving the index to `to`: a second vector column, filled by `reindexStep()`. */
  private async beginBackground(from: EmbedderRecord, to: EmbedderRecord, next: EmbedderRecord | null): Promise<void> {
    // A move to a third embedder: its half-made column goes.
    if (next && !sameEmbedder(next, to)) await this.db.exec('alter table enclave.chunks drop column if exists embedding_next')
    await this.db.exec(`
      alter table enclave.chunks alter column embedding drop not null;
      alter table enclave.chunks add column if not exists embedding_next vector(${to.dimensions});
    `)
    await kvSet(this.db, NEXT_KEY, to)
    this.migrating = { from, to }
    // Nothing to re-embed (an empty index): done at once.
    await this.swapWhenDone()
  }

  /** The configured embedder is the index's own again: drop the half-made column, fill vectors that only it had. */
  private async abandonBackground(): Promise<void> {
    for (;;) {
      if (!(await this.fillBatch('embedding', this.options.batchSize ?? 32))) break
    }
    await this.db.exec(`
      alter table enclave.chunks drop column if exists embedding_next;
      alter table enclave.chunks alter column embedding set not null;
    `)
    await kvSet(this.db, NEXT_KEY, null)
  }

  /** Embed up to `limit` chunks that have no vector in `column`; returns how many it did. */
  private async fillBatch(column: 'embedding' | 'embedding_next', limit: number, signal?: AbortSignal): Promise<number> {
    const { rows } = await this.db.query<{ id: number; content: string; title: string | null }>(
      `select c.id, c.content, d.title from enclave.chunks c join enclave.documents d on d.id = c.document_id
       where c.${column} is null order by c.id limit $1`,
      [limit],
    )
    if (!rows.length) return 0
    signal?.throwIfAborted()
    const vectors = await this.embedBatched(rows.map((r) => this.format(r.content, r.title)), 'document', signal)
    await this.db.query(
      `update enclave.chunks c set ${column} = v.embedding::vector
       from unnest($1::bigint[], $2::text[]) as v(id, embedding) where c.id = v.id`,
      [rows.map((r) => r.id), vectors.map(toVectorLiteral)],
    )
    return rows.length
  }

  /** Every chunk has its new vector: swap the columns, rebuild the vector index, record the embedder (one statement batch). */
  private async swapWhenDone(): Promise<boolean> {
    if (!this.migrating) return true
    const { rows } = await this.db.query<{ left: number }>('select count(*)::int as left from enclave.chunks where embedding_next is null')
    if ((rows[0]?.left ?? 0) > 0) return false
    const { to } = this.migrating
    await this.db.exec(`
      begin;
      drop index if exists enclave.chunks_embedding_idx;
      alter table enclave.chunks drop column embedding;
      alter table enclave.chunks rename column embedding_next to embedding;
      alter table enclave.chunks alter column embedding set not null;
      create index chunks_embedding_idx on enclave.chunks using hnsw (embedding vector_cosine_ops);
      insert into enclave.kv (key, value) values ('${EMBEDDER_KEY}', ${sqlJson(to)})
        on conflict (key) do update set value = excluded.value;
      insert into enclave.kv (key, value) values ('${NEXT_KEY}', 'null'::jsonb)
        on conflict (key) do update set value = excluded.value;
      commit;
    `)
    this.migrating = null
    return true
  }

  private serial<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.writes.then(fn, fn)
    this.writes = run.catch(() => undefined)
    return run
  }

  /** The background re-embedding in progress, or null (none, or finished). */
  async reindexStatus(): Promise<ReindexStatus | null> {
    await this.init()
    if (!this.migrating) return null
    const { rows } = await this.db.query<{ total: number; done: number }>(
      'select count(*)::int as total, count(embedding_next)::int as done from enclave.chunks',
    )
    return { ...this.migrating, done: rows[0]?.done ?? 0, total: rows[0]?.total ?? 0 }
  }

  /** False while a background re-embedding runs: search is keyword-only until it ends. */
  get vectorsReady(): boolean {
    return !this.migrating
  }

  /**
   * Re-embed the next `limit` chunks (default `batchSize`) for the new
   * embedder; the last step swaps the index over. Returns the status after
   * the step, null once there is nothing (left) to do.
   */
  async reindexStep(options: { limit?: number; signal?: AbortSignal } = {}): Promise<ReindexStatus | null> {
    await this.init()
    if (!this.migrating) return null
    const finished = await this.serial(async () => {
      if (!this.migrating) return true
      await this.fillBatch('embedding_next', options.limit ?? this.options.batchSize ?? 32, options.signal)
      return this.swapWhenDone()
    })
    if (finished) return null
    const status = await this.reindexStatus()
    if (status) this.options.onReindexProgress?.(status.done, status.total)
    return status
  }

  async ingest(input: IngestDocument | IngestDocument[], options: IngestOptions = {}): Promise<IngestResult> {
    await this.init()
    const docs = Array.isArray(input) ? input : [input]
    const result: IngestResult = { documents: 0, chunks: 0, updated: 0, skipped: 0 }
    const batchSize = this.options.batchSize ?? 32

    // Documents to embed are pooled so the embedder gets full batches of
    // chunks across documents (a note is often 1-3 chunks; embedding each on
    // its own left the GPU mostly idle), then written in one transaction.
    type Pending = { index: number; id: string; collection: string; doc: IngestDocument; hash: string; pieces: string[] }
    let pending: Pending[] = []
    let pooled = 0
    let reported = 0
    const ids: string[] = []
    const report = (upTo: number) => {
      for (; reported < upTo; reported++) options.onProgress?.({ done: reported + 1, total: docs.length, document: ids[reported] ?? '' })
    }

    const flush = async () => {
      if (!pending.length) return
      const inputs = pending.flatMap((p) => p.pieces.map((piece) => this.format(piece, p.doc.title)))
      const vectors = await this.embedBatched(inputs, 'document', options.signal)
      let at = 0
      // While re-embedding in the background, new chunks get only the new embedder's vectors.
      await this.serial(() => this.db.transaction(async (tx) => {
        const column = this.migrating ? 'embedding_next' : 'embedding'
        for (const p of pending) {
          const own = vectors.slice(at, at + p.pieces.length)
          at += p.pieces.length
          await tx.query(
            `insert into enclave.documents (id, collection, title, source, metadata, content_hash, chunk_count)
             values ($1, $2, $3, $4, $5, $6, $7)
             on conflict (id) do update set
               collection = excluded.collection, title = excluded.title, source = excluded.source,
               metadata = excluded.metadata, content_hash = excluded.content_hash,
               chunk_count = excluded.chunk_count, updated_at = now()`,
            [p.id, p.collection, p.doc.title ?? null, p.doc.source ?? null, JSON.stringify(p.doc.metadata ?? {}), p.hash, p.pieces.length],
          )
          await tx.query('delete from enclave.chunks where document_id = $1', [p.id])
          if (p.pieces.length) {
            await tx.query(
              `insert into enclave.chunks (document_id, collection, ordinal, content, ${column})
               select $1, $2, (t.ord - 1)::int, t.content, t.embedding::vector
               from unnest($3::text[], $4::text[]) with ordinality as t(content, embedding, ord)`,
              [p.id, p.collection, p.pieces, own.map(toVectorLiteral)],
            )
          }
          result.documents++
          result.chunks += p.pieces.length
        }
      }))
      report(pending.at(-1)!.index + 1)
      pending = []
      pooled = 0
    }

    for (const [index, doc] of docs.entries()) {
      options.signal?.throwIfAborted()
      const collection = doc.collection ?? options.collection ?? 'default'
      const id = doc.id ?? `doc_${(await sha256(doc.source ?? doc.content)).slice(0, 24)}`
      ids[index] = id
      // "<text hash>:<metadata hash>": a metadata-only change (a note moved to
      // another folder, retagged) updates the row without re-embedding.
      const textHash = await sha256(JSON.stringify([collection, doc.title ?? '', doc.content]))
      const hash = `${textHash}:${(await sha256(JSON.stringify(doc.metadata ?? {}))).slice(0, 16)}`
      // A document repeated in one call: write what's pooled first, in order.
      if (pending.some((p) => p.id === id)) await flush()

      const existing = await this.db.query<{ content_hash: string }>(
        'select content_hash from enclave.documents where id = $1',
        [id],
      )
      const previous = existing.rows[0]?.content_hash
      if (previous === hash) {
        result.skipped++
        if (!pending.length) report(index + 1)
      } else if (previous?.split(':')[0] === textHash) {
        await this.db.query(
          `update enclave.documents set source = $2, metadata = $3, content_hash = $4, updated_at = now() where id = $1`,
          [id, doc.source ?? null, JSON.stringify(doc.metadata ?? {}), hash],
        )
        result.updated++
        if (!pending.length) report(index + 1)
      } else {
        const pieces = chunkText(doc.content, { ...this.options.chunk, ...options.chunk })
        pending.push({ index, id, collection, doc, hash, pieces })
        pooled += pieces.length
        if (pooled >= batchSize) await flush()
      }
    }
    await flush()
    report(docs.length)
    return result
  }

  async search(query: string, options: SearchOptions = {}): Promise<SearchHit[]> {
    await this.init()
    // Until a background re-embedding ends, query and index vectors come from different models.
    const mode = this.migrating ? 'keyword' : (options.mode ?? this.options.defaultMode ?? 'hybrid')
    const finalLimit = Math.max(1, Math.min(options.limit ?? 8, 100))
    const reranker = options.rerank === false || mode === 'keyword' ? undefined : this.options.reranker
    const limit = reranker ? Math.min(this.options.rerankCandidates ?? Math.max(finalLimit * 3, 24), 60) : finalLimit
    const candidates = Math.max(limit * 4, 20)
    const collections =
      options.collection === undefined ? null : Array.isArray(options.collection) ? options.collection : [options.collection]
    const filter = options.filter ? JSON.stringify(options.filter) : null
    const tsquery = toOrQuery(query)

    const useVector = mode !== 'keyword'
    const useKeyword = mode !== 'vector' && tsquery !== ''
    if (!useVector && !useKeyword) return []

    const vector = useVector
      ? toVectorLiteral((await this.embedder.embed([query], 'query', options.task ? { task: options.task } : undefined))[0]!)
      : null
    const scope =
      `($2::text[] is null or c.collection = any($2::text[])) and ($3::jsonb is null or d.metadata @> $3::jsonb)` +
      ` and ($7::text[] is null or c.document_id = any($7::text[]))`

    const { rows } = await this.db.query<{
      id: number
      score: number
      similarity: number | null
      content: string
      ordinal: number
      document_id: string
      collection: string
      title: string | null
      source: string | null
      metadata: Record<string, unknown>
    }>(
      `with semantic as (
         select c.id, row_number() over (order by c.embedding <=> $1::vector) as rank
         from enclave.chunks c join enclave.documents d on d.id = c.document_id
         where $1::vector is not null and ${scope}
         order by c.embedding <=> $1::vector
         limit $4
       ),
       keyword as (
         select c.id, row_number() over (order by ts_rank_cd(c.fts, q) desc) as rank
         from enclave.chunks c
           join enclave.documents d on d.id = c.document_id,
           to_tsquery('${this.language}', $5) q
         where $5 <> '' and c.fts @@ q and ${scope}
         order by ts_rank_cd(c.fts, q) desc
         limit $4
       ),
       fused as (
         select coalesce(s.id, k.id) as id,
                coalesce(1.0 / (${RRF_K} + s.rank), 0) + coalesce(1.0 / (${RRF_K} + k.rank), 0) as score
         from semantic s full outer join keyword k on s.id = k.id
       )
       select f.id, f.score::float8 as score,
              case when $1::vector is null then null else (1 - (c.embedding <=> $1::vector))::float8 end as similarity,
              c.content, c.ordinal, c.document_id, c.collection, d.title, d.source, d.metadata
       from fused f
         join enclave.chunks c on c.id = f.id
         join enclave.documents d on d.id = c.document_id
       order by f.score desc
       limit $6`,
      [vector, collections, filter, candidates, useKeyword ? tsquery : '', limit, options.documentIds ?? null],
    )

    const hits: SearchHit[] = rows
      .filter((r) => options.minSimilarity === undefined || (r.similarity ?? 1) >= options.minSimilarity)
      .map((r) => ({
        chunkId: Number(r.id),
        documentId: r.document_id,
        collection: r.collection,
        title: r.title,
        source: r.source,
        content: r.content,
        ordinal: r.ordinal,
        metadata: r.metadata,
        score: r.score,
        similarity: r.similarity,
      }))
    if (!reranker || hits.length < 2) return hits.slice(0, finalLimit)

    const scores = await reranker.rerank(
      query,
      hits.map((h) => (h.title ? `${h.title}\n${h.content}` : h.content)),
    )
    return hits
      .map((h, i) => ({ ...h, rerankScore: scores[i]! }))
      .sort((a, b) => b.rerankScore - a.rerankScore)
      .slice(0, finalLimit)
  }

  private format(text: string, title: string | null | undefined): string {
    if (this.embedder.formatDocument) return this.embedder.formatDocument(text, title ?? undefined)
    return title ? `${title}\n\n${text}` : text
  }

  async remove(documentId: string): Promise<boolean> {
    await this.init()
    const { affectedRows } = await this.db.query('delete from enclave.documents where id = $1', [documentId])
    return (affectedRows ?? 0) > 0
  }

  async clear(collection: string): Promise<number> {
    await this.init()
    const { affectedRows } = await this.db.query('delete from enclave.documents where collection = $1', [collection])
    return affectedRows ?? 0
  }

  async collections(): Promise<CollectionInfo[]> {
    await this.init()
    const { rows } = await this.db.query<{ collection: string; documents: number; chunks: number }>(
      `select collection, count(*)::int as documents, coalesce(sum(chunk_count), 0)::int as chunks
       from enclave.documents group by collection order by collection`,
    )
    return rows
  }

  /**
   * Rebuild every embedding with the configured embedder (e.g. after switching
   * models). Chunk text is kept; only vectors and the index change.
   */
  async reindex(options: { signal?: AbortSignal; onProgress?(done: number, total: number): void } = {}): Promise<void> {
    this.ready = undefined
    await this.serial(() => this.rebuild(options))
    await this.init()
  }

  private async rebuild(options: { signal?: AbortSignal; onProgress?(done: number, total: number): void }): Promise<void> {
    const dims = this.embedder.dimensions
    const { rows: present } = await this.db.query<{ ok: boolean }>(`select to_regclass('enclave.chunks') is not null as ok`)
    if (!present[0]?.ok) {
      await kvSet(this.db, EMBEDDER_KEY, { id: this.embedder.id, dimensions: dims })
      return
    }
    await this.db.exec(`
      drop index if exists enclave.chunks_embedding_idx;
      alter table if exists enclave.chunks drop column if exists embedding_next;
      alter table if exists enclave.chunks alter column embedding drop not null;
      alter table if exists enclave.chunks alter column embedding type vector(${dims}) using null;
    `)
    const { rows } = await this.db.query<{ id: number; content: string; title: string | null }>(
      `select c.id, c.content, d.title from enclave.chunks c join enclave.documents d on d.id = c.document_id order by c.id`,
    )
    const batch = this.options.batchSize ?? 32
    for (let i = 0; i < rows.length; i += batch) {
      options.signal?.throwIfAborted()
      const slice = rows.slice(i, i + batch)
      const vectors = await this.embedder.embed(
        slice.map((r) => this.format(r.content, r.title)),
        'document',
      )
      await this.db.query(
        `update enclave.chunks c set embedding = v.embedding::vector
         from unnest($1::bigint[], $2::text[]) as v(id, embedding) where c.id = v.id`,
        [slice.map((r) => r.id), vectors.map(toVectorLiteral)],
      )
      options.onProgress?.(Math.min(i + batch, rows.length), rows.length)
    }
    await this.db.exec(`
      alter table enclave.chunks alter column embedding set not null;
      create index if not exists chunks_embedding_idx on enclave.chunks using hnsw (embedding vector_cosine_ops);
    `)
    // Record the new embedder only once every vector has been rebuilt.
    await kvSet(this.db, EMBEDDER_KEY, { id: this.embedder.id, dimensions: dims })
    await kvSet(this.db, NEXT_KEY, null)
    this.migrating = null
  }

  private async embedBatched(texts: string[], kind: 'document', signal?: AbortSignal): Promise<number[][]> {
    const batch = this.options.batchSize ?? 32
    const out: number[][] = []
    for (let i = 0; i < texts.length; i += batch) {
      signal?.throwIfAborted()
      const vectors = await this.embedder.embed(texts.slice(i, i + batch), kind)
      for (const v of vectors) {
        if (v.length !== this.embedder.dimensions) {
          throw new Error(`Embedder returned ${v.length} dimensions, expected ${this.embedder.dimensions}`)
        }
        out.push(v)
      }
    }
    return out
  }
}

/** Turn free text into an OR-ed tsquery so partial keyword matches still rank. */
export function toOrQuery(text: string): string {
  const terms = new Set(
    (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []).filter((t) => t.length > 1 || /\d/.test(t)),
  )
  return [...terms].map((t) => `'${t}'`).join(' | ')
}
