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

**Thinking mode is on by default for Qwen3.** On the same agent prompt (9 tools), measured in Chrome on WebGPU, Qwen3 4B called the right tool 4/4 times with thinking and 0/4 without. Without thinking it described the SQL it would run instead of running it. Thinking costs latency (about 18 s per step with lazy skills vs 7 s). Pass `thinking: false` for plain chat.

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

## Small-model engineering

These behaviours make 0.6B–8B browser models dependable:

- **Token budgeting.** With a model's `contextWindow`, each step fits history into the window (dropping whole old turns first, then shortening old tool results) and caps tool output.
- **Text tool protocol.**
  - Hermes/Qwen `<tool_call>` JSON, plus Qwen3.5/Qwen3-Coder XML function calls.
  - `<think>` is split into reasoning events.
  - Generation stops at `<tool_response>` and discards anything after it, so the model can't invent tool results.
  - Malformed calls go back to the model as errors.
- **Hybrid retrieval.** Vector and keyword search are fused with Reciprocal Rank Fusion in one SQL statement, then reranked by a cross-encoder. Title-aware document formatting is used where the embedding model supports it.
- **One engine per worker.** Switching WebLLM models reuses the engine and unloads the previous model, so two models are never resident on the GPU at once.
- **Serialized ML worker.** Model loads and inference in the Transformers.js worker run one at a time. Concurrent ONNX Runtime WebGPU session creation stalled in testing.

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

Events: `step-start`, `text-delta`, `reasoning-delta`, `tool-call`, `approval-request`, `tool-result`, `custom`, `message`, `finish`.

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
| `@enclave/core/testing` | `mockModel`, `hashEmbedder` |

## Develop

```sh
pnpm install
pnpm test                                   # unit tests: real PGlite + pgvector, scripted model
pnpm --filter @enclave/core test:e2e        # real weights on CPU: embeddings, reranker, Qwen3 0.6B tool use (~1.5 GB download)
pnpm dev                                    # playground at http://localhost:5173
```

The playground shows the detected device and a model picker. The picker groups WebLLM models (with fit, cached and too-large markers), Transformers.js models, Chrome built-in AI, and discovered Ollama / LM Studio models. It also has embedding and reranker pickers, download progress, cache deletion, file ingestion, and streaming chat with inline approvals.

## Verified

In Chrome with WebGPU on Apple silicon, running the playground with Qwen3 4B (WebLLM), EmbeddingGemma and mxbai rerank:

| Check | Result |
|---|---|
| Ingest a document | ~0.7 s |
| Document question ("guest wifi password?") | Correct, with citation, in every run |
| "Create a table and insert two rows" as the first request in a thread | `execute_sql` called 5/5 (31–49 s each, thinking on) |
| Same request after an earlier question in the thread | 3/4. The miss reasoned about the task, then answered in text |
| Same prompt with thinking off | 0/4 |

The Node e2e suite checks every embedding preset and the reranker against real weights, and a real Qwen3 0.6B completing a tool call and a RAG answer on CPU.

Not yet verified: Windows/Linux GPUs, mobile browsers, Chrome built-in AI, and live Ollama / LM Studio servers (their adapters are tested against recorded protocol responses). A 4B model is not perfectly reliable: keep `needsApproval` on writes, and use `maxDownloadMB` to allow Qwen3 8B where devices can take it.
