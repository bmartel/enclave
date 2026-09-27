# enclave

A web-first, offline agent framework. The LLM, embeddings, reranker, vector index, database and conversation history all run in the user's browser. Drop it into an existing web app to add an agent with private RAG and your own skills; no data has to leave the device.

It builds on [database.build](https://github.com/supabase-community/database-build) (Postgres in the browser via PGlite, pgvector, in-browser embeddings, an LLM that drives SQL through tools) and turns those pieces into a library built around WebLLM and Transformers.js. Local servers (Ollama, LM Studio) are supported as an option.

```ts
import { createWebEnclave } from '@enclave/core/web'
import { knowledgeSkill, sqlSkill } from '@enclave/core/skills'

const ai = await createWebEnclave({
  workers: {
    db: new Worker(new URL('./db.worker.ts', import.meta.url), { type: 'module' }),
    ml: new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' }),
    llm: new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' }),
  },
  skills: [knowledgeSkill(), sqlSkill()],
  onProgress: (p) => console.log(p.stage, p.text),
})

await ai.knowledge!.ingest({ title: 'Handbook', content: handbookText })

for await (const event of ai.thread('support').send('What is the guest wifi password?')) {
  if (event.type === 'text-delta') render(event.delta)
}
```

Each worker file is one line:

```ts
// db.worker.ts
import { servePGlite } from '@enclave/core/pglite-worker'; servePGlite()
// ml.worker.ts   (embeddings, reranker, Transformers.js LLMs)
import { serveTransformers } from '@enclave/core/transformers/worker'; serveTransformers()
// llm.worker.ts  (WebLLM)
import { serveWebLLM } from '@enclave/core/models/webllm-worker'; serveWebLLM()
```

## What runs where

| Layer | Default | Runs on |
|---|---|---|
| LLM | [WebLLM](https://github.com/mlc-ai/web-llm) (Qwen3 4B on typical desktops) | WebGPU, in a worker |
| LLM fallback | [Transformers.js](https://huggingface.co/docs/transformers.js) v4 (Qwen3 0.6B) | WASM, when there is no WebGPU |
| Embeddings | EmbeddingGemma 300M (GPU) / Granite Embedding 97M multilingual R2 (CPU) | WebGPU → WASM, in a worker |
| Reranker | mxbai-rerank xsmall (cross-encoder) | Same worker |
| Database + vectors | [PGlite](https://pglite.dev) (Postgres 17) + pgvector HNSW + full-text | Worker, IndexedDB or OPFS |
| Agent loop, skills, threads | This library | Main thread |

`createWebEnclave` profiles the device (WebGPU, `shader-f16`, buffer limits, memory) and picks the best models that fit it. You can override every choice.

## Privacy by construction

Three layers make "no data leaves the device" something the browser enforces, not just a promise:

**1. Locality policy.** Every model, embedder and reranker declares where it processes data: `device` (browser, or a loopback server), `local-network` (private addresses) or `remote`. An enclave refuses components beyond its policy, including on `setModel`. The default is `local-network`, so internet APIs require an explicit opt-in:

```ts
createWebEnclave({ privacy: { allow: 'device' } })     // strictest: browser + localhost only
ai.setModel(anthropic({ apiKey }))                     // throws PrivacyError unless allow: 'remote'
```

**2. Self-hosted model files.** By default the browser downloads weights from Hugging Face, WebLLM's compiled libraries from GitHub, and ONNX Runtime from jsDelivr. These are GET requests with no user data, but they reveal the user's IP address and which models the app uses, and they create a runtime dependency. Mirror everything to your own host once:

```sh
npx enclave-mirror --out public/models \
  --webllm qwen3-4b --embedding embeddinggemma --reranker mxbai-rerank-xsmall --ort
```

```ts
createWebEnclave({ selfHost: { baseUrl: '/models' }, ... })
```

**3. Browser-enforced lockdown.** Serve a Content-Security-Policy so the browser itself refuses any other connection (it applies to workers too), and optionally wrap `fetch`/XHR/WebSocket/beacons at runtime:

```ts
import { contentSecurityPolicy, guardNetwork } from '@enclave/core/privacy'

contentSecurityPolicy({ modelHosts: [] })   // self-hosted: connect-src 'self' only
contentSecurityPolicy({ localServers: true }) // also allow Ollama / LM Studio on localhost
guardNetwork({ allow: [], onViolation: (url) => report(url) })  // defense in depth, per page/worker
```

Data at rest (the PGlite database, threads, the vector index, model caches) lives in the origin's IndexedDB/OPFS on the user's device. `createWebEnclave` requests persistent storage so the browser doesn't evict it.

## Choosing models

### Browser LLMs

`BROWSER_LLMS` in `@enclave/core/web` is a curated catalog. `recommendLLM(device)` picks from it:

- **Fit.** It uses WebLLM's measured VRAM figures plus an estimate of KV-cache growth, and chooses `q4f16` builds when the GPU supports `shader-f16`, `q4f32` otherwise.
- **Context window.** WebLLM builds default to 4K tokens, which is too small for agents. The catalog requests 8K–16K when memory allows. Models squeezed below 8K are ranked down.
- **Tool-use quality.** Qwen3 is ranked first: it is trained on the `<tool_call>` format enclave uses. Qwen3.5 is available but marked experimental, because its WebLLM build caps conversation history (`max_history_size: 1`).
- **First-visit download.** Automatic choice is capped at 2.5 GB. Pass `maxDownloadMB` to allow Qwen3 8B.

| Preset | Runtime | Download | Notes |
|---|---|---|---|
| `qwen3-8b` | WebLLM | 4.6 GB | Best quality; needs ~6 GB of GPU memory |
| `qwen3-4b` | WebLLM | 2.3 GB | Default on 8 GB+ desktops |
| `qwen3-1.7b` | WebLLM | 1.1 GB | Laptops and integrated GPUs |
| `hermes-3-3b`, `llama-3.2-3b`, `phi-4-mini` | WebLLM | 1.8–2.2 GB | Alternatives |
| `qwen3-0.6b` | WebLLM | 0.5 GB | Phones, quick demos |
| `tjs-qwen3-1.7b`, `tjs-granite-4-1b`, `tjs-lfm2-1.2b`, `tjs-qwen3-0.6b`, `tjs-qwen3.5-0.8b` | Transformers.js | 0.5–1.4 GB | Also run on WASM without WebGPU |

```ts
import { browserLLM, detectDevice, rankLLMs } from '@enclave/core/web'
const device = await detectDevice()
rankLLMs(device)                             // everything that fits, best first
await ai.useModel('qwen3-1.7b')              // switch at runtime (same WebLLM worker, old model unloaded)
```

**Thinking defaults to `'auto'` for Qwen3.** With thinking off, Qwen3 4B called the right tool 0/4 times; with thinking on, 4/4. `'auto'` reasons when a new request arrives and skips it after tool results, where it only costs time. It matched always-on quality and cut p90 latency from 32.5 s to 19.4 s. Use `thinking: true` for hard multi-step tasks, `false` for plain chat.

### Embeddings and reranking (Transformers.js)

Presets encode each model's correct pooling, prefixes and per-backend dtype. Getting these wrong silently degrades retrieval.

| Preset | Dims | Download | Notes |
|---|---|---|---|
| `embeddinggemma` | 768 (Matryoshka 512/256/128) | 197 MB | Best quality under 500M params; 100+ languages; `title: … \| text: …` document format |
| `granite-multilingual-r2` | 384 | 98–195 MB | Fast; 200+ languages; 32K tokens |
| `granite-small-r2` | 384 | 52–97 MB | Fast; English; 8K tokens |
| `qwen3-embedding-0.6b` | 1024 (Matryoshka) | 567 MB | Highest quality; heavy |
| `gte-small` | 384 | 34 MB | database.build's model, for compatibility |

| Reranker | Download | Notes |
|---|---|---|
| `mxbai-rerank-xsmall` | 87 MB | Default |
| `ms-marco-minilm` | 23 MB | Fastest |
| `bge-reranker-v2-m3` | 571–700 MB | Multilingual |

```ts
import { transformersEmbedder, transformersReranker } from '@enclave/core/transformers'
transformersEmbedder({ preset: 'embeddinggemma', dimensions: 256, worker })
transformersReranker({ preset: 'mxbai-rerank-xsmall', worker })
```

Switching embedding models is safe. The index records which embedder built it, and `createWebEnclave` re-embeds existing documents automatically (`knowledge.autoReindex`).

### Local servers (optional)

For machines that already run a model server:

```ts
import { discoverLocalModels, localModel, ollama, lmstudio, localEmbedder } from '@enclave/core/models/local'

const found = await discoverLocalModels()           // Ollama (:11434) + LM Studio (:1234); unreachable servers are skipped
await ai.useModel(localModel(found[0]!))
await ai.useModel(ollama({ model: 'qwen3:8b', contextWindow: 16384, think: true }))
await ai.useModel(lmstudio({ model: 'qwen/qwen3-4b' }))
```

- **Ollama** uses the native `/api/chat` endpoint. That endpoint supports `num_ctx`, thinking and tools; the OpenAI-compatible one ignores `num_ctx`, and Ollama's default window is too small for agents.
- **LM Studio** uses its OpenAI-compatible server. Enable CORS in its server settings.
- **Discovery** reports tool support, context length and loaded state.
- **`localEmbedder`** computes embeddings on either server.

Any OpenAI-compatible endpoint also works via `openaiCompatible()`, and Claude via `anthropic()`.

## Skills

A skill bundles instructions, typed tools, its own tables, and live context:

```ts
import { z } from 'zod'
import { defineSkill, tool } from '@enclave/core'

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
        ctx.emit({ searching: customer })                 // streams to the UI
        return myApp.invoices.search({ customer, status }) // call into the host app
      },
      toModelOutput: (rows) => rows.slice(0, 20),         // UI gets everything, the model gets 20
    }),
    annotate: tool({
      description: 'Attach a note to an invoice.',
      input: z.object({ invoiceId: z.string(), note: z.string() }),
      needsApproval: true,
      execute: ({ invoiceId, note }, { db }) =>
        db.query('insert into invoice_notes values ($1, $2) on conflict (invoice_id) do update set note = $2', [invoiceId, note]),
    }),
  },
  context: async ({ db, messages }) => `${(await db.query('select count(*) from invoice_notes')).rows[0].count} annotated invoices`,
})
```

| Field | Purpose |
|---|---|
| `name`, `description` | Identity. The description is all the model sees of a `lazy` skill. |
| `instructions` | Added to the system prompt. |
| `tools` | `tool({ description, input: zodSchema, execute, needsApproval?, toModelOutput? })`. Inputs are validated. Validation and runtime errors go back to the model so it can correct itself. |
| `migrations` | SQL applied once per database, in order, tracked per skill. |
| `setup(ctx)` | Runs once at registration. |
| `context(ctx)` | Fresh state before each model step. `ctx.messages` exposes the conversation. |
| `lazy` | Only the description is listed until the model calls `activate_skill`. Fewer visible tools makes small models faster and more accurate. |

Tool context: `db`, `knowledge`, `embedder`, `threadId`, `skill`, `signal`, `emit(data)`.

### Built-in skills

- **`knowledgeSkill()`**
  - Auto-retrieval (default on): before the model runs, the latest user message is searched and the best reranked passages go into context. This is classic RAG alongside the `search_knowledge` tool. Small models often skip the tool; with auto-retrieval, Qwen3 0.6B answered document questions correctly where it previously invented an answer.
  - Passages are cited as `[n]`.
- **`sqlSkill({ readOnly, approveWrites, maxRows })`**
  - The database.build capability: live schema in context, plus `describe_schema` and `execute_sql`.
  - Writes need approval by default.
  - The internal `enclave` schema is hidden from the model.
- **`memorySkill()`**: `remember`, `recall` and `forget`, backed by the knowledge base. Recent facts are added to context.

## WebLLM performance and quality

Enclave's prompt layout, decoding and agent loop are built around how WebLLM actually executes:

- **KV-cache reuse.** WebLLM skips re-reading the prompt when a request extends the conversation it already holds, byte for byte. To make that happen:
  - The system prompt holds only instructions and tools.
  - Live context (schema, retrieved passages) rides on the newest message.
  - Every earlier message is replayed exactly as it was first sent, including the model's raw reply.

  Measured on Qwen3 4B: after a tool result, the next step prefilled 250 tokens instead of the whole prompt, and time to first token fell from 2.2 s to 0.7 s.
- **Grammar-constrained tool calls.** WebLLM's xgrammar structural tags leave text free. Once the model writes `<tool_call>`, it can only complete a call to a real tool, with arguments that validate against that tool's schema. Schemas xgrammar can't compile fall back to unconstrained decoding.
- **Adaptive thinking.** Qwen3 reasons before acting and answers directly after tool results.
- **Per-step metrics.** Every `step-finish` event carries prefill tokens, KV reuse, time to first token, prefill/decode throughput and grammar compile time.
- **Small-model safeguards.**
  - Context budgeting fits history and tool output into the model's window.
  - Generation stops at `<tool_response>`, so the model can't invent tool results.
  - Malformed calls go back to the model as errors.
  - Both `<tool_call>` JSON and Qwen XML function calls are parsed.
- **Engine management.**
  - One WebLLM engine per worker; switching models unloads the old one.
  - The Transformers.js worker serializes model loads, because concurrent ONNX Runtime WebGPU session creation stalled in testing.

## Tuning with evals

`@enclave/core/eval` grades agent behaviour and retrieval quality in the browser, against your own data, without sending it anywhere:

```ts
import { runEval, formatReport, evalRetrieval } from '@enclave/core/eval'

const report = await runEval(ai, [
  { name: 'wifi', input: 'What is the guest wifi password?', expect: { answer: 'sunflower42' } },
  {
    name: 'create table',
    input: 'Create a books table and add two novels',
    setup: (ai) => ai.db.exec('drop table if exists books'),
    expect: {
      tools: ['execute_sql'],
      check: async ({ ai }) => (await ai.db.query('select * from books')).rows.length === 2 || 'expected 2 rows',
    },
  },
  { name: 'restraint', input: 'Say good morning in Spanish', expect: { noTools: true, answer: /buenos/i } },
], { repeats: 3 })
console.log(formatReport(report))   // pass rate, p50/p90 latency, TTFT, tokens, KV reuse, per-case failures

await evalRetrieval(ai.knowledge!, [{ query: 'guest wifi', relevant: ['wifi-doc'] }], { k: 5 })  // recall, MRR, nDCG
```

The playground's `eval.html` runs a 9-case suite covering private RAG, SQL actions, chained steps, multi-turn and restraint. Configure it by query string: `?model=qwen3-4b&thinking=auto&constrain=1&repeats=3`.

How the current defaults were chosen (Qwen3 4B, WebGPU, Apple silicon):

| Configuration | Pass | p50 | p90 |
|---|---|---|---|
| Previous prompt, thinking always on | 83% (15/18) | 14.1 s | 32.5 s |
| Previous prompt, thinking `auto` | 89% (16/18) | 12.2 s | 19.4 s |
| **Tuned prompt, thinking `auto`, constrained (default)** | **100% (27/27)** | **9.9 s** | **23.4 s** |

The tuned prompt fixed two failure modes the evals exposed:
- The model refused general questions ("I don't have access to translation tools").
- The model printed SQL instead of running it.

On the "create table" case alone it went from 1/6 to 6/6.

## Agent API

```ts
const thread = ai.thread(id?)                           // persisted in PGlite
for await (const e of thread.send(text, { signal, onApproval })) { … }
await thread.send(text).text()                          // or just the final answer
ai.run(textOrMessages)                                  // stateless
await ai.use(skill)                                     // add a skill later
ai.setModel(model) / await ai.useModel('qwen3-1.7b')
ai.device, ai.plan                                      // what was detected and chosen
```

Events: `step-start`, `text-delta`, `reasoning-delta`, `tool-call`, `approval-request`, `tool-result`, `custom`, `message`, `step-finish` (timing, prefill tokens, KV reuse), `finish`.

Lower-level building blocks (`createEnclave`, `createDb`, `createWorkerDb`, `Knowledge`, `webllm`, `transformersLLM`, `fromTextModel`, `fallback`) are all exported for custom setups.

## Entry points

| Import | Provides |
|---|---|
| `@enclave/core` | `createEnclave`, `defineSkill`, `tool`, `Knowledge`, `fallback`, `fromTextModel`, types |
| `@enclave/core/web` | `createWebEnclave`, `browserLLM`, `detectDevice`, catalogs, `recommend*` |
| `@enclave/core/transformers` | `transformersEmbedder`, `transformersReranker`, `transformersLLM` |
| `@enclave/core/transformers/worker` | `serveTransformers()` |
| `@enclave/core/models/webllm` | `webllm`, `isWebLLMCached`, `deleteWebLLMCache` |
| `@enclave/core/models/webllm-worker` | `serveWebLLM()` |
| `@enclave/core/models/local` | `ollama`, `lmstudio`, `discoverLocalModels`, `localModel`, `localEmbedder` |
| `@enclave/core/models/chrome` | `chromeAI` (Gemini Nano) |
| `@enclave/core/models/openai`, `/models/anthropic` | Remote/other endpoints |
| `@enclave/core/pglite`, `/pglite-worker` | `createDb`, `createWorkerDb`, `servePGlite` |
| `@enclave/core/skills` | `knowledgeSkill`, `sqlSkill`, `memorySkill` |
| `@enclave/core/eval` | `runEval`, `formatReport`, `evalRetrieval` |
| `@enclave/core/privacy` | `contentSecurityPolicy`, `guardNetwork`, `selfHostedTransformers`, `localityOfUrl` |
| `@enclave/core/testing` | `mockModel`, `hashEmbedder` |

## Develop

```sh
pnpm install
pnpm test                                   # unit tests: real PGlite + pgvector, scripted model
pnpm --filter @enclave/core test:e2e        # real weights on CPU: embeddings, reranker, Qwen3 0.6B tool use (~1.5 GB download)
pnpm dev                                    # playground at http://localhost:5173 (eval suite at /eval.html)
pnpm --filter playground mirror             # download models into public/models
pnpm --filter playground dev:strict         # self-hosted models + CSP: no third-party requests
```

The playground shows the detected device and a model picker. The picker groups WebLLM models (with fit, cached and too-large markers), Transformers.js models, Chrome built-in AI, and discovered Ollama / LM Studio models. It also has embedding and reranker pickers, download progress, cache deletion, file ingestion, and streaming chat with inline approvals.

## Verified

On Chrome with WebGPU (Apple silicon):

- **Evals.** The 9-case suite passed 27/27 with the default configuration (Qwen3 4B, EmbeddingGemma, mxbai rerank).
- **Strict privacy mode** (`pnpm dev:strict`: self-hosted models plus CSP `connect-src 'self'`). A first-visit session in a fresh profile downloaded every model, indexed a document and answered correctly. The only host it contacted was the app's own origin, with zero CSP violations. The same session in normal mode contacts huggingface.co, its CDN, cdn.jsdelivr.net and raw.githubusercontent.com.
- **KV-cache reuse.** Steps after a tool result prefill ~250 new tokens instead of the full prompt.
- **Node e2e suite.** Checks every embedding preset and the reranker against real weights, plus Qwen3 0.6B completing tool calls and RAG answers on CPU.

Not yet verified: Windows/Linux GPUs, mobile browsers, Chrome built-in AI, and live Ollama / LM Studio servers (their adapters are tested against recorded protocol responses). The eval suite is small (9 cases), so treat its pass rates as directional and extend it with cases from your own domain.
