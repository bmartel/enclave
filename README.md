# enclave

**Private, offline AI agents for web apps.** The language model, embeddings, vector search, a Postgres database and conversation history all run inside the user's browser. Nothing has to leave the device. Add an agent that answers questions from your documents, works with your data and calls your own app's functions, without a backend AI service.

```ts
const ai = await createWebEnclave({ workers, skills: [knowledgeSkill(), sqlSkill()] })
await ai.knowledge!.ingest({ title: 'Handbook', content: handbookText })

for await (const event of ai.thread('support').send('What is the guest wifi password?')) {
  if (event.type === 'text-delta') render(event.delta)
}
```

- **Runs in the browser.** WebLLM (WebGPU) or Transformers.js (WASM) for the model, PGlite (Postgres 17 + pgvector) for storage and search, all in workers.
- **Private RAG.** Ingest documents; the agent retrieves, cites and answers. Search is hybrid (vector plus keyword), and retrieved text is treated as untrusted.
- **Skills.** Typed tools (Zod) with approvals, their own tables and live context. They call straight into your app.
- **Built-in skills.** Documents (`knowledgeSkill`), SQL over the in-browser database (`sqlSkill`) and long-term memory (`memorySkill`).
- **Local servers too.** Ollama and LM Studio, with measured model presets. OpenAI-compatible endpoints and Claude are available when you opt in.
- **Privacy you can enforce.** A locality policy, self-hosted model files and a Content-Security-Policy generator, so the browser itself blocks third-party connections.
- **Measured.** A 65-case production eval suite runs against the real models. The in-browser Qwen3 4B scores 95%; local 27B models score 98.5–99.5%. See [Benchmarks](#benchmarks).

It distils the ideas of [database.build](https://github.com/supabase-community/database-build) into a library: Postgres in the browser, in-browser embeddings, and an LLM that acts through tools.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Choosing where the model runs](#choosing-where-the-model-runs)
- [Documents and private RAG](#documents-and-private-rag)
- [The database and SQL](#the-database-and-sql)
- [Memory](#memory)
- [Writing skills](#writing-skills)
- [Building the UI](#building-the-ui)
- [Privacy and security](#privacy-and-security)
- [Downloads and offline use](#downloads-and-offline-use)
- [Tuning WebLLM](#tuning-webllm)
- [Evaluating your agent](#evaluating-your-agent)
- [Benchmarks](#benchmarks)
- [API reference](#api-reference)
- [Development](#development)
- [Status and limitations](#status-and-limitations)

---

## Install

The package is `@enclave/core` (ESM, TypeScript types included). It isn't published to npm yet. Use it from this repository:

```sh
git clone https://github.com/bmartel/enclave && cd enclave
pnpm install
pnpm --filter @enclave/core build
# then, in your app (pnpm/npm/yarn all support local paths):
pnpm add /path/to/enclave/packages/core zod
```

Requirements:
- **Browser:** Chrome or Edge 121+ with WebGPU for in-browser models. Without WebGPU, small Transformers.js models run on WASM.
- **Bundler:** any that supports module workers via `new URL('./x.worker.ts', import.meta.url)`. Vite works out of the box.
- **Headers (recommended):** serve with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, to enable multi-threaded WASM.

## Quick start

### 1. Create the three workers

Heavy work runs off the main thread. Each worker file is one line:

```ts
// src/db.worker.ts: PGlite (Postgres + pgvector)
import { servePGlite } from '@enclave/core/pglite-worker'; servePGlite()

// src/ml.worker.ts: embeddings, rerankers, Transformers.js models
import { serveTransformers } from '@enclave/core/transformers/worker'; serveTransformers()

// src/llm.worker.ts: WebLLM
import { serveWebLLM } from '@enclave/core/models/webllm-worker'; serveWebLLM()
```

### 2. Create the enclave

```ts
import { createWebEnclave } from '@enclave/core/web'
import { knowledgeSkill, sqlSkill, memorySkill } from '@enclave/core/skills'

const ai = await createWebEnclave({
  workers: {
    db: new Worker(new URL('./db.worker.ts', import.meta.url), { type: 'module' }),
    ml: new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' }),
    llm: new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' }),
  },
  skills: [knowledgeSkill(), sqlSkill(), memorySkill()],
  onProgress: (p) => showProgress(p.stage, p.text, p.progress),   // model downloads, indexing
})

console.log(ai.device, ai.plan)   // what was detected, which models were chosen
```

`createWebEnclave` profiles the device (WebGPU, `shader-f16`, GPU memory) and picks the best chat model, embedder and storage that fit. Everything can be overridden.

The chat model (2.3 GB for Qwen3 4B) downloads on the first message, or during setup with `preloadLLM: true`. After that it loads from the browser cache, offline.

### 3. Add documents and chat

```ts
await ai.knowledge!.ingest([
  { title: 'Wifi', content: 'Guest network: NorthGuest. Password: maple-harbor-42.', source: 'it/wifi.md' },
  { title: 'Expenses', content: expensesPolicyText, source: 'finance/expenses.md' },
])

const thread = ai.thread('support')                       // persisted in the browser database
for await (const e of thread.send('What is the guest wifi password?')) {
  if (e.type === 'text-delta') append(e.delta)
}

// Or just the final answer:
const answer = await ai.thread().send('Create a table of my books and add three classics').text()
```

That's a working private agent: document Q&A with citations, SQL in the browser, and memory across conversations.

A complete example app lives in [`examples/playground`](examples/playground). It has a model picker, downloads, document upload, streaming chat with inline approvals, and evals.

## Concepts

| Concept | What it is |
|---|---|
| **Enclave** (`ai`) | The agent runtime: model, database, knowledge base, skills, privacy policy. |
| **Thread** | A persisted conversation. `ai.thread(id)` resumes one; `ai.threads()` lists them. |
| **Skill** | A unit of capability: instructions, typed tools, optional tables and live context. |
| **Tool** | A typed function the model can call. Inputs are validated with Zod, and the tool can require user approval. |
| **Knowledge** | The document store: chunking, embeddings, hybrid search, optional reranking. |
| **Model / Embedder / Reranker** | Swappable components. Each declares its **locality** (`device`, `local-network`, `remote`), and the privacy policy enforces it. |

Each `send` runs the agent loop: the model sees the system prompt, the skills' instructions, live context (database schema, retrieved passages, memories) and the conversation. It either answers or calls tools; tool results go back to it, and it continues until it answers. Everything streams as typed events.

## Choosing where the model runs

### In the browser (default)

`llm: 'auto'` picks from a curated catalog by GPU memory and capability:

| Preset | Runtime | Download | Use |
|---|---|---|---|
| `qwen3-4b` | WebLLM | 2.3 GB | Default on 8 GB+ desktops. 95% on the production evals. |
| `qwen3-8b` | WebLLM | 4.6 GB | Best in-browser quality; ~6 GB GPU memory. Opt in with `maxDownloadMB`. |
| `qwen3-1.7b` | WebLLM | 1.1 GB | Laptops and integrated GPUs |
| `qwen3-0.6b` | WebLLM | 0.5 GB | Phones and demos |
| `hermes-3-3b`, `llama-3.2-3b`, `phi-4-mini` | WebLLM | 1.8–2.2 GB | Alternatives |
| `tjs-qwen3-1.7b`, `tjs-granite-4-1b`, `tjs-qwen3-0.6b`, … | Transformers.js | 0.5–1.4 GB | Also run without WebGPU (WASM) |

```ts
createWebEnclave({ llm: 'qwen3-1.7b', ... })      // pick a preset
await ai.useModel('qwen3-8b')                       // switch at runtime (the old model is unloaded)

import { detectDevice, rankLLMs } from '@enclave/core/web'
rankLLMs(await detectDevice())                      // every preset that fits, best first
```

Qwen3 is ranked first because it is trained on the tool-call format enclave uses. **Thinking** defaults to `'auto'`: the model reasons on new requests and answers directly after tool results. With thinking off, Qwen3 4B called the right tool 0/4 times; `'auto'` matched always-on quality at lower latency. Set `thinking: false` for plain chat.

### On a local model server (Ollama, LM Studio)

On machines with a capable GPU, a local 27B model is markedly better and about 2.5× faster than the in-browser 4B. Data still stays on the machine: `localhost` counts as `device` locality.

```ts
import { discoverLocalModels, recommendOllamaModel, ollama, ollamaEmbedder, lmstudio } from '@enclave/core/models/local'

const found = await discoverLocalModels()          // Ollama (:11434) and LM Studio (:1234); missing servers are skipped
const pick = recommendOllamaModel(found)            // best measured preset that is installed

const ai = await createWebEnclave({
  workers,
  llm: pick ? ollama({ model: pick.tag, contextWindow: 32768 }) : 'auto',   // fall back to the browser
  embedding: ollamaEmbedder(),                      // embeddinggemma via Ollama (optional)
  skills: [knowledgeSkill(), sqlSkill()],
})
```

Measured presets (M2 Max, 32 GB, default settings):

| Model | Eval pass rate | Median time per case | Download |
|---|---|---|---|
| `qwen3.6:27b-q4_K_M` (Ollama; recommended) | 98.5% (99% with Ollama embeddings) | 15 s | 17 GB |
| `qwen3.6-27b` GGUF (LM Studio) | 99.5% | 14 s | 17.5 GB |
| `qwen3.8:27b-q4_K_M` | 99.0% | 17 s | 17 GB |
| `qwen3.5:9b` (fast) | 93% | 6 s | 6.6 GB |

`qwen3.5:9b` occasionally claimed an action it hadn't taken, so prefer a 27B model where actions matter.

**Ollama notes:**
- enclave uses the native `/api/chat` endpoint, which supports `num_ctx`, thinking and tools.
- Prompts are laid out so Ollama reuses its prompt cache across agent steps: a follow-up step reads its prompt in about 0.3–0.6 s, against 2–8 s cold.
- Ollama allows `localhost` origins by default, so no setup is needed.

**LM Studio notes:**
- **CORS:** browser apps need the server started with `lms server start --cors`. This lets *any* website you visit call the server, so enable it only while needed.
- **Context length:** it's fixed when the model loads (`lms load <model> --context-length 32768`). Pass the same value as `contextWindow`.
- **GGUF vs MLX:** prefer GGUF builds. On Apple silicon, MLX was not faster and scored slightly lower.

```ts
lmstudio({ model: 'qwen/qwen3.6-27b', contextWindow: 32768 })
```

### Remote APIs (opt-in)

```ts
import { openaiCompatible } from '@enclave/core/models/openai'
import { anthropic } from '@enclave/core/models/anthropic'

createWebEnclave({ privacy: { allow: 'remote' }, llm: anthropic({ apiKey }) })
```

Remote models are refused (`PrivacyError`) unless the privacy policy allows `remote`.

## Documents and private RAG

### Ingesting

```ts
const { documents, chunks, skipped } = await ai.knowledge!.ingest(
  [
    { id: 'hr-vacation', title: 'Vacation policy', content: text, source: 'hr/vacation.md', metadata: { dept: 'hr' } },
    // ...
  ],
  { collection: 'handbook', onProgress: ({ done, total }) => bar(done / total) },
)
```

- **Chunking:** markdown-aware (sections, then paragraphs, lines and sentences), about 1,200 characters with overlap. Override with `chunk: { size, overlap }`.
- **Idempotent:** unchanged documents are skipped by content hash. Re-ingesting the same `id` replaces it.
- **Collections:** documents can be grouped into collections (`handbook`, `tickets`…), and searches can target one or several.
- **Other operations:** `remove(id)`, `clear(collection)`, `collections()`, `reindex()`.

### How the agent uses documents

`knowledgeSkill()` gives the agent two paths to your documents:

1. **Auto-retrieval** (default on). Before the model runs, the latest message is searched and relevant passages go into its context. Small models often skip search tools, so this matters.
2. **The `search_knowledge` tool**, for follow-up searches the model decides to make.

Auto-retrieval is deliberately selective, because irrelevant passages derail small models:
- **Relevance floor:** a passage is used only when the best match clears the embedder's calibrated floor (0.35 for EmbeddingGemma). Chit-chat and commands get no passages.
- **Relative cutoff:** weaker matches are dropped relative to the best one.
- **Follow-ups:** short follow-ups ("how much does it cost?") are searched again together with the previous question.
- **Role follow-up:** "who approves this?" follows a role title in the best passage ("Director of Finance") to the page that names the person.
- **No retrieval for reworking:** requests that rework earlier answers ("summarize both of those") skip retrieval.

Answers cite passages as `[1]`, `[2]`. Options:

```ts
knowledgeSkill({
  collections: ['handbook'],          // restrict scope (default: everything except memories)
  limit: 6,                           // results per search_knowledge call
  autoRetrieve: { limit: 3, maxChars: 3600, minSimilarity: 0.35, relativeCutoff: 0.5, followRoles: true },
  // autoRetrieve: false              // tool-only RAG
})
```

### Searching directly

```ts
const hits = await ai.knowledge!.search('parental leave notice period', {
  limit: 5, mode: 'vector',            // 'hybrid' | 'vector' | 'keyword'
  collection: 'handbook', filter: { dept: 'hr' }, minSimilarity: 0.3,
})
// [{ documentId, title, source, content, similarity, score, rerankScore?, metadata }]
```

### Embedding models

| Preset | Dims | Download | Notes |
|---|---|---|---|
| `embeddinggemma` (default with WebGPU) | 768 (Matryoshka 512/256/128) | 197 MB | Best measured: recall@3 0.995 on the 91-query benchmark; 100+ languages |
| `granite-multilingual-r2` (default without WebGPU, and on mobile) | 384 | 98–195 MB | Fast, 200+ languages |
| `granite-small-r2` | 384 | 52–97 MB | Fast, English |
| `qwen3-embedding-0.6b` | 1024 | 567 MB | Heavy |
| `gte-small` | 384 | 34 MB | database.build compatibility |

```ts
createWebEnclave({ embedding: 'granite-small-r2' })
createWebEnclave({ embedding: ollamaEmbedder('embeddinggemma') })   // via Ollama
```

Switching embedders is safe: the index records which embedder built it, and existing documents are re-embedded automatically.

A **reranker** (cross-encoder) is available (`reranker: 'mxbai-rerank-xsmall'`), but it's off by default. On the benchmark, EmbeddingGemma vector search alone was more accurate and far faster.

### Untrusted documents

Retrieved text is treated as data, not instructions:
- Paragraphs that address AI assistants ("ignore previous instructions…") are redacted before the model sees them. They're also kept out of auto-retrieval unless they're the best match.
- In the evals, a planted "system note" in a document no longer changes answers (safety 30/30).
- The helpers are exported as `looksLikeInjection` and `redactInjections`.

## The database and SQL

Every enclave has a PGlite database: Postgres 17 with pgvector, persisted to IndexedDB (or OPFS). Use it directly:

```ts
await ai.db.exec(`create table if not exists books (id bigint primary key generated always as identity, title text, author text)`)
const { rows } = await ai.db.query('select * from books where author = $1', ['Le Guin'])
```

`sqlSkill()` lets the agent design schemas, query and change data:

```ts
sqlSkill({
  schemas: ['public'],        // what the model may see
  readOnly: false,            // true: SELECT only, enforced by a read-only transaction
  approveWrites: true,        // default: ask the user before INSERT/UPDATE/DELETE/DDL
  maxRows: 100,
  sampleValues: 12,           // list values of small lookup columns (0 disables)
})
```

The model sees a live schema and gets `describe_schema` and `execute_sql`. To make it accurate on your data:
- **Document business rules in the schema.** Comments are shown to the model:
  ```sql
  comment on table orders is 'Revenue counts only orders whose status is not cancelled.';
  ```
  Without this rule, the model wrote correct SQL but computed the wrong revenue.
- **Sample values** let it map a user's words to data ("the Fleet Console" is a product).
  - Only lookup-style columns are sampled: tables that other tables reference, shown with their ids, and columns whose values repeat (status, country).
  - Free-text and sensitive-looking columns (email, phone, password, token, key) are never sampled.
- **Internal tables are hidden.** The internal `enclave` schema (documents, threads) is never visible to the model.

## Memory

```ts
memorySkill({ recent: 10 })   // the 10 most recent memories go into context
```

- **Tools:** the agent gets `remember`, `recall` and `forget`. `remember({ fact, replaces })` updates a fact in one call.
- **Context:** recent memories are injected each turn, so the agent knows the user across conversations.
- **Guards:**
  - **Secrets:** `remember` refuses passwords, API keys and card numbers, including bare password-like tokens.
  - **Deletion scope:** `forget` can only delete memories, never documents.
  - **Separation:** memories never show up as document search results.

## Writing skills

A skill packages what the agent needs for one area of your app:

```ts
import { z } from 'zod'
import { defineSkill, tool, dateContext } from '@enclave/core'

export const invoices = defineSkill({
  name: 'invoices',
  description: 'Look up and annotate customer invoices.',
  instructions: 'Use find_invoices before answering billing questions. Amounts are in cents.',
  migrations: [`create table invoice_notes (invoice_id text primary key, note text not null)`],
  tools: {
    find_invoices: tool({
      description: 'Find invoices for a customer.',
      input: z.object({ customer: z.string(), status: z.enum(['open', 'paid']).optional() }),
      execute: async ({ customer, status }, ctx) => {
        ctx.emit({ searching: customer })                  // custom event for your UI
        return myApp.invoices.search({ customer, status }) // call into your app
      },
      toModelOutput: (rows) => rows.slice(0, 20),          // UI gets everything, the model gets 20
    }),
    annotate: tool({
      description: 'Attach a note to an invoice.',
      input: z.object({ invoiceId: z.string(), note: z.string() }),
      needsApproval: true,                                 // the user confirms first
      execute: ({ invoiceId, note }, { db }) =>
        db.query('insert into invoice_notes values ($1, $2) on conflict (invoice_id) do update set note = $2', [invoiceId, note]),
    }),
  },
  context: async ({ db }) => `${dateContext()}\nOpen invoices: ${await myApp.invoices.countOpen()}`,
})

const ai = await createWebEnclave({ workers, skills: [invoices, knowledgeSkill()] })
await ai.use(anotherSkill)   // or add later
```

| Field | Purpose |
|---|---|
| `name`, `description` | Identity. For a `lazy` skill, the description is all the model sees until it activates the skill. |
| `instructions` | Added to the system prompt. |
| `tools` | `tool({ description, input, execute, needsApproval?, toModelOutput? })`. Invalid inputs and thrown errors go back to the model so it can correct itself. |
| `migrations` | SQL applied once per database, in order, tracked per skill. |
| `setup(ctx)` | Runs once when the skill is registered. |
| `context(ctx)` | Fresh text before every model step: current state, today's date, counts. |
| `lazy` | Hidden until the model calls `activate_skill`. Fewer visible tools make small models faster and more accurate. |

Tools receive `ctx` with `db`, `knowledge`, `embedder`, `threadId`, `skill`, `signal` and `emit(data)`. `needsApproval` can be a function: `(input, ctx) => input.amount > 100`.

**Tips for small models** (each learned from the evals):
- **Guide from tool results.** The model reads them at the moment it decides what to do next. When a lookup is ambiguous, return `{ matches, note: 'Several contacts share this name. Ask which one before acting.' }`. The same rule written in the instructions never worked.
- **Give dates.** `dateContext()` lists today, the rest of this week and next week, so "this Friday" is read off directly instead of counted.
- **Return compact data.** Use `toModelOutput` to send the model only what it needs.
- **Validate on the server side.** Throw a clear error, and the model will retry with corrected input.

## Building the UI

### Events

`thread.send()` returns a stream of typed events:

| Event | Use |
|---|---|
| `text-delta` | Stream the answer |
| `reasoning-delta` | Optional "thinking" display |
| `tool-call` / `tool-result` | Show tool activity (`durationMs`, `isError`) |
| `approval-request` | Ask the user (see below) |
| `custom` | Data a tool sent with `ctx.emit` |
| `message` | A complete message was added to the thread |
| `step-start` / `step-finish` | Per model call; `step-finish` carries timing and metrics |
| `finish` | `reason`: `stop`, `max-steps` or `aborted`, with token usage |

```ts
const controller = new AbortController()
const stream = thread.send(input, {
  signal: controller.signal,                                   // stop button
  onApproval: async (call) => confirmDialog(`${call.name}: ${JSON.stringify(call.input)}`),
})
for await (const e of stream) {
  switch (e.type) {
    case 'text-delta': appendText(e.delta); break
    case 'tool-call': showTool(e.call.name, e.call.input); break
    case 'tool-result': finishTool(e.call.id, e.isError); break
  }
}
const { text, steps, usage } = await stream.result()
```

### Approvals

Tools with `needsApproval` pause until your `onApproval` handler resolves. It can be set per enclave (`createWebEnclave({ onApproval })`) or per send. With no handler, approval-gated calls are denied, and the agent reports that honestly instead of claiming success.

### Threads

```ts
const threads = await ai.threads()                 // [{ id, title, updatedAt, ... }]
const messages = await ai.thread(id).messages()    // full history
await ai.thread(id).rename('Q3 revenue')
await ai.thread(id).delete()
await ai.run('One-off question, no history')       // stateless
```

Messages with `synthetic: true` were added by the agent loop. For example, it adds a reminder when the model announces a tool call but doesn't make it. Hide them in your UI.

## Privacy and security

Three layers make "no data leaves the device" enforceable:

**1. Locality policy.** Every model, embedder and reranker declares where it processes data. The enclave refuses components beyond the policy, including on `setModel`.

```ts
createWebEnclave({ privacy: { allow: 'device' } })   // browser + localhost only
// 'local-network' (default): also private-network servers you control
// 'remote': internet APIs, opt-in
```

**2. Self-hosted model files.** By default, weights come from Hugging Face, WebLLM libraries from GitHub, and ONNX Runtime from jsDelivr. These downloads carry no user data, but they reveal the user's IP address and which models you use. Mirror everything to your own origin:

```sh
npx enclave-mirror --out public/models --webllm qwen3-4b --embedding embeddinggemma --ort
```
```ts
createWebEnclave({ selfHost: { baseUrl: '/models' }, ... })
```

**3. Browser-enforced lockdown.** Generate a Content-Security-Policy so the browser itself refuses any other connection, from the page and its workers:

```ts
import { contentSecurityPolicy, guardNetwork } from '@enclave/core/privacy'
contentSecurityPolicy({ modelHosts: [] })        // self-hosted: connect-src 'self'
contentSecurityPolicy({ localServers: true })    // also allow Ollama / LM Studio on localhost
guardNetwork({ allow: [], onViolation: report }) // runtime defense in depth for fetch/XHR/WebSocket
```

Verified: in strict mode (self-hosted files plus CSP), a first-visit session downloaded every model, indexed a document and answered, contacting only the app's own origin.

**Data at rest.** The database, threads, vectors and model caches live in the origin's IndexedDB/OPFS. `createWebEnclave` asks the browser to keep them persistent (`persist: true`).

**Agent safety.**
- Approval-gated writes.
- Redaction of injected instructions in documents.
- Secrets refused from memory.
- Internal tables hidden from SQL.
- Repeated identical tool calls short-circuited.

## Downloads and offline use

```ts
const status = await ai.modelCache.status()
// [{ kind: 'llm', id: 'qwen3-4b', cached: true, active: true, downloadMB: 2300 }, ...]
await ai.modelCache.clear('llm', 'qwen3-8b')
```

- **`cached` means usable offline.** It's true only when every file this device needs is in the browser cache.
- **Download size cap.** First-visit automatic choice is capped at 2.5 GB (`maxDownloadMB`).
- **Test offline on a real hostname.** WebLLM does not cache model libraries served from `localhost`, so test offline behaviour on `127.0.0.1` or a real hostname.

## Tuning WebLLM

```ts
createWebEnclave({
  thinking: 'auto',                    // true | false | 'auto'
  webllm: {
    reasoningHistory: 'current-turn',  // 'current-turn' (default, most accurate) | 'auto' (fastest later turns) | 'all'
    thinkingBudget: 2048,              // reasoning tokens per step before answering directly
    constrainToolCalls: true,          // grammar-constrained tool calls (xgrammar)
  },
})
```

What enclave does for you, and why:
- **KV-cache reuse.** The prompt is laid out so WebLLM continues from its cache. After a tool result, the next step re-reads ~250 tokens instead of the whole prompt.
- **Lean history.** Earlier turns are replayed without their stale context blocks. Later-turn time to first token halved (14 s → 7 s) and accuracy rose.
- **Grammar-constrained tool calls.** Once the model writes `<tool_call>`, it can only produce a valid call to a real tool.
- **Thinking budget and loop cut-off.** Runaway reasoning (over budget, or the same sentence repeating) ends early, and the step is answered directly.
- **Follow-through.** If the model announces a tool call but doesn't make it, the loop reminds it once.

`reasoningHistory: 'auto'` keeps reasoning across turns, so the cache survives. Later turns then start in ~0.4 s instead of ~3 s, at a small accuracy cost (31/33 vs 33/33 later turns). Choose it for latency-sensitive chat.

Every `step-finish` event reports prefill tokens, cache reuse, prompt size, time to first token and decode speed.

## Evaluating your agent

`@enclave/core/eval` runs cases against your enclave, in the browser, on your data:

```ts
import { runEval, formatReport, evalRetrieval, anyOf, numberNear, declines } from '@enclave/core/eval'

const report = await runEval(ai, [
  { name: 'wifi', input: 'Guest wifi password?', expect: { answer: 'maple-harbor-42' } },
  {
    name: 'create books',
    input: 'Create a books table and add two novels',
    setup: (ai) => ai.db.exec('drop table if exists books'),
    expect: {
      tools: ['execute_sql'],
      check: async ({ ai }) => (await ai.db.query('select * from books')).rows.length === 2 || 'expected 2 rows',
    },
  },
  { name: 'restraint', input: 'Say good morning in Spanish', expect: { noTools: true, answer: /buenos/i } },
  {
    name: 'follow-up',
    turns: [
      { input: 'What was revenue in June?', expect: { answer: numberNear(34799) } },
      { input: 'And in May?', expect: { answer: numberNear(30020) } },
    ],
  },
], { repeats: 3 })

console.log(formatReport(report))   // pass rate with 95% CI, consistency, per-tag results, latency, failures
await evalRetrieval(ai.knowledge!, [{ query: 'guest wifi', relevant: ['it-wifi'] }], { k: 3 })   // recall, MRR, nDCG
```

**Iterate fast.** A full suite with 3 repeats can take hours on in-browser models. While fixing failures:

```ts
runEval(ai, cases, { failFast: true })                  // stop at the first failing run
runEval(ai, cases, { maxFailures: 5, first: lastFailures })   // regression guard: likely failures first
```

The production suite in [`packages/evals`](packages/evals) has 65 cases (84 graded turns) covering RAG, SQL, a CRM skill, memory, multi-turn conversations and safety. Its graders are themselves tested against reference solutions, a do-nothing model and real model answers. It runs the same cases against WebLLM, Ollama or LM Studio. See its [README](packages/evals/README.md).

## Benchmarks

Production evals, 65 cases × 3 repeats, Apple M2 Max (32 GB):

| Setup | Runs passed | Cases passing all 3 repeats | Median time per case |
|---|---|---|---|
| In-browser WebLLM Qwen3 4B (start of the eval work) | 75% [69–81%] | 72% | 38 s |
| **In-browser WebLLM Qwen3 4B (current)** | **95% [91–98%]** | **92%** | 38 s |
| Ollama `qwen3.6:27b` | 98.5% | 95% | 15 s |
| Ollama `qwen3.6:27b` + Ollama embeddings | 99% | 97% | 18 s |
| LM Studio `qwen3.6-27b` GGUF + LM Studio embeddings | 99.5% | 98% | 14 s |

The in-browser model passes all safety (30/30), memory (15/15) and approval cases. Its remaining failure is a 4-turn analysis that ends in a dependent multi-row insert, where it guesses ids instead of looking them up. The 27B models pass it.

Retrieval, 91 labeled queries:
- **EmbeddingGemma vector search:** recall@3 0.995, MRR 0.94, 10/10 multilingual queries, about 50 ms per query on CPU.

Full methodology, per-case history and every report: [`packages/evals`](packages/evals/README.md).

## API reference

| Import | Main exports |
|---|---|
| `@enclave/core` | `createEnclave`, `defineSkill`, `tool`, `Knowledge`, `dateContext`, `fallback`, `fromTextModel`, `PrivacyError`, types |
| `@enclave/core/web` | `createWebEnclave`, `browserLLM`, `detectDevice`, `rankLLMs`, `recommendLLM`, `BROWSER_LLMS`, `EMBEDDING_PRESETS`, `RERANKER_PRESETS` |
| `@enclave/core/skills` | `knowledgeSkill`, `sqlSkill`, `memorySkill`, `looksLikeInjection`, `looksLikeSecret` |
| `@enclave/core/models/local` | `ollama`, `lmstudio`, `discoverLocalModels`, `recommendOllamaModel`, `ollamaEmbedder`, `localEmbedder`, `localModel`, `OLLAMA_LLM_PRESETS`, `OLLAMA_EMBEDDING_PRESETS` |
| `@enclave/core/models/webllm` | `webllm`, `selfHostedAppConfig`, `isWebLLMCached`, `deleteWebLLMCache` |
| `@enclave/core/models/webllm-worker` | `serveWebLLM` |
| `@enclave/core/transformers` | `transformersEmbedder`, `transformersReranker`, `transformersLLM`, `configureTransformers` |
| `@enclave/core/transformers/worker` | `serveTransformers` |
| `@enclave/core/models/openai` | `openaiCompatible` |
| `@enclave/core/models/anthropic` | `anthropic` |
| `@enclave/core/models/chrome` | `chromeAI`, `chromeAIAvailable` (Gemini Nano) |
| `@enclave/core/pglite`, `/pglite-worker` | `createDb`, `createWorkerDb`, `servePGlite` |
| `@enclave/core/privacy` | `contentSecurityPolicy`, `guardNetwork`, `selfHostedTransformers`, `localityOfUrl` |
| `@enclave/core/eval` | `runEval`, `formatReport`, `compareReports`, `evalRetrieval`, `wilson`, matchers (`anyOf`, `allOf`, `noneOf`, `numberNear`, `count`, `declines`, `labeled`) |
| `@enclave/core/testing` | `mockModel`, `hashEmbedder` (fast, deterministic tests) |

**Lower level.** Without the web helpers, you can assemble everything yourself:

```ts
import { createEnclave } from '@enclave/core'
import { createDb } from '@enclave/core/pglite'
const ai = await createEnclave({ db: await createDb({ dataDir: 'memory://' }), model, embedder, skills, privacy: { allow: 'device' } })
```

`createEnclave` also accepts `system` (replaces the base prompt), `maxSteps` (default 12), `maxHistory`, `maxToolOutputChars`, `onApproval` and `knowledge` options. It works in Node too, which is how the unit tests run.

## Development

```sh
pnpm install
pnpm test                                     # unit tests: real PGlite + pgvector, scripted model
pnpm --filter @enclave/core test:e2e          # real weights on CPU (embeddings, reranker, Qwen3 0.6B)
pnpm dev                                      # playground at http://localhost:5173
pnpm --filter playground dev:strict           # self-hosted models + CSP: zero third-party requests
pnpm --filter @enclave/evals test             # grader validation (seconds)
pnpm --filter @enclave/evals eval             # production evals on WebGPU (hours; see packages/evals)
```

The playground picks a measured local model (Ollama or LM Studio) when one is installed, and otherwise the best in-browser model. It shows device detection, a model picker with cache state, downloads, document upload, threads and streaming chat with inline approvals.

## Status and limitations

- **Pre-1.0.** APIs may change. Not yet published to npm.
- **Where it's verified:** Chrome with WebGPU on Apple silicon, plus Node for the unit and CPU end-to-end tests. Windows/Linux GPUs, mobile browsers and Chrome built-in AI are untested.
- **In-browser model limits:** Qwen3 4B is strong for its size but still slips occasionally on SQL details and long multi-step writes. Use a local 27B model for heavier workloads.
- **First-visit download:** 2–3 GB for the default in-browser model. Show progress (`onProgress`) and cache state (`modelCache`).
