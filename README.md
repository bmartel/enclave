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

### Download and cache management

Every downloadable component implements `isCached()`, `load()` and `clearCache()`. `isCached()` is true only when every file needed on this device is in the browser cache, for the dtype and backend the device will actually use: WebLLM weights plus the compiled model library, or ONNX weights, tokenizer and config. `createWebEnclave` exposes the whole catalog:

```ts
const status = await ai.modelCache.status()
// [{ kind: 'llm', id: 'qwen3-4b', cached: true, active: true, downloadMB: 2300 }, { kind: 'embedding', ... }, ...]
await ai.modelCache.clear('llm', 'qwen3-8b')
```

Status is computed against the same hosting configuration the app downloads with, so self-hosted models are reported correctly. WebLLM never caches model libraries served from a `localhost` URL (a development convenience), so test offline behaviour on a real hostname or `127.0.0.1`.

### Local servers (optional)

For machines that already run a model server:

```ts
import { discoverLocalModels, recommendOllamaModel, ollama, ollamaEmbedder, lmstudio } from '@enclave/core/models/local'

const found = await discoverLocalModels()                  // Ollama (:11434) + LM Studio (:1234); unreachable servers are skipped
const pick = recommendOllamaModel(found)                   // best measured preset that is installed
await ai.useModel(ollama({ model: pick?.tag ?? 'qwen3.6:27b-q4_K_M', contextWindow: 32768 }))
const embedder = ollamaEmbedder()                          // embeddinggemma, with its measured prompts and relevance floor
await ai.useModel(lmstudio({ model: 'qwen/qwen3-4b' }))
```

Measured with the production evals, on an M2 Max with 32 GB, Ollama 0.35 and the default GPU memory limit:

| Model | Eval pass rate | Median per case | Size |
|---|---|---|---|
| `qwen3.6:27b-q4_K_M` (quality default) | 98.5% | 15 s | 17 GB |
| `qwen3.8:27b-q4_K_M` | 99.0% | 17 s | 17 GB |
| `qwen3.5:9b` (fast) | 92.8% | 6 s | 6.6 GB |
| In-browser WebLLM Qwen3 4B, for comparison | 91.8% | 38 s | 2.3 GB |

- The 27B models tie on quality. `qwen3.6` is 25% faster at p90.
- `qwen3.5:9b` twice claimed an action that had not happened. Use a 27B where actions matter.

Embeddings on the 91-query retrieval benchmark:

| Embedder | Recall@3 | MRR | Per query |
|---|---|---|---|
| `embeddinggemma` (default) | 0.995 | 0.941 | 21 ms |
| `qwen3-embedding:4b` (1024-d) | 0.995 | 0.908 | 68 ms |
| `qwen3-embedding:0.6b` | 0.973 | 0.886 | 20 ms |
| `bge-m3` | 0.956 | 0.879 | 21 ms |

Qwen3-Embedding needs its exact instruction format. A paraphrased instruction dropped the 0.6b model to 0.912.

- **Ollama** uses the native `/api/chat` endpoint, which supports `num_ctx`, thinking and tools. The OpenAI-compatible endpoint ignores `num_ctx`, and Ollama's default window is too small for agents.
  - Prompts are laid out for Ollama's prompt cache: live context rides on the newest message. A follow-up step reads its prompt in about 0.3–0.6 s, against 2–8 s cold.
  - `think: 'auto'` (the default) reasons on new requests, answers directly after tool results, and applies only to models that support thinking.
- **LM Studio** uses its OpenAI-compatible server. Enable CORS in its server settings.
- **Discovery** reports tool support, context length and loaded state.
- **`localEmbedder`** computes embeddings on either server. Setting `dimensions` below the model's output truncates Matryoshka models; pgvector's HNSW index takes at most 2000 dimensions.

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

For small models, put guidance in tool results, where the model is looking when it decides what to do next. For example, when a lookup is ambiguous, `toModelOutput` can return `{ contacts, note: 'Several contacts share this name. Ask which one before acting.' }`. In the evals, the same rule in the skill's instructions never made Qwen3 4B ask (0/3).

### Built-in skills

- **`knowledgeSkill()`**
  - Auto-retrieval (default on): before the model runs, the latest user message is searched and the most relevant passages go into context. This is classic RAG alongside the `search_knowledge` tool. Small models often skip the tool; with auto-retrieval, Qwen3 0.6B answered document questions correctly where it previously invented an answer.
  - Auto-retrieval stays silent unless something is relevant. Irrelevant passages hurt small models badly: in the production evals, passages attached to "remember that I'm on the Payments team" made the model summarise support SLAs instead of saving the fact.
    - A passage is used only when the best match clears the embedder's `relevanceFloor`. Each preset has a floor, calibrated because cosine scales differ by model: 0.35 for EmbeddingGemma, 0.8 for GTE.
    - Passages must also score at least half the best match's margin above that floor.
    - A short follow-up ("how much does it cost?") is searched again together with the previous question.
  - Documents are treated as untrusted. Paragraphs that address AI assistants ("ignore previous instructions…") are replaced with a removal notice before the model sees them, in auto-retrieval and in `search_knowledge` results. Flagging them wasn't enough: asked to summarise a poisoned document, Qwen3 4B still repeated the planted password in 2 of 3 runs. Such passages are also dropped from auto-retrieval unless they are the best match. `looksLikeInjection` and `redactInjections` are exported.
  - When a passage answers only part of a question (it names a role but not the person), the model is told to search for the rest.
  - Tune with `autoRetrieve: { minSimilarity, relativeCutoff, limit, maxChars, minRerankScore }`.
  - Passages are cited as `[n]`.
- **`sqlSkill({ readOnly, approveWrites, maxRows })`**
  - The database.build capability: live schema in context, plus `describe_schema` and `execute_sql`.
  - Writes need approval by default.
  - Table and column comments (`COMMENT ON`) are shown as SQL comments. They are the place to document business rules the model can't guess, such as "revenue excludes cancelled orders". Without that rule, Qwen3 4B wrote correct SQL but computed the wrong revenue.
  - The internal `enclave` schema is hidden from the model.
- **`memorySkill()`**: `remember`, `recall` and `forget`, backed by the knowledge base. Recent facts are added to context. When the latest message asks to remember or forget something, a one-turn hint tells the model to call the tool. Otherwise small models reply "I'll remember that" and store nothing. `remember({ fact, replaces })` updates a fact in one call. `forget` and `replaces` can only delete memories, never documents. Memories are kept out of `knowledgeSkill` searches (`exclude`, default `['memory']`), so they can't pass as documents. `remember` refuses anything that looks like a credential or card number (`looksLikeSecret`), because prompting alone didn't stop Qwen3 4B from saving a password.

The agent loop answers a repeated identical tool call from the earlier result instead of running it again. This breaks the search loops small models fall into.

Small models sometimes announce a call ("I will now call execute_sql with the corrected query") and then end the turn. The loop catches this when tools already ran in the turn: it adds one reminder, a user message marked `synthetic: true` that UIs can hide, and continues.

`dateContext(today)` spells out today and the next two weeks with weekdays, for a skill's `context`. Qwen3 4B got "this Friday" wrong 2 of 3 times when it had to count days itself.

## WebLLM performance and quality

Enclave's prompt layout, decoding and agent loop are built around how WebLLM actually executes:

- **KV-cache reuse.** WebLLM skips re-reading the prompt when a request extends the conversation it already holds, byte for byte. To make that happen:
  - The system prompt holds only instructions and tools.
  - Live context (schema, retrieved passages) rides on the newest message.
  - Every earlier message is replayed exactly as it was first sent, including the model's raw reply.

  Measured on Qwen3 4B: after a tool result, the next step prefilled 250 tokens instead of the whole prompt, and time to first token fell from 2.2 s to 0.7 s.
- **Grammar-constrained tool calls.** WebLLM's xgrammar structural tags leave text free. Once the model writes `<tool_call>`, it can only complete a call to a real tool, with arguments that validate against that tool's schema. Schemas xgrammar can't compile fall back to unconstrained decoding.
- **Adaptive thinking.** Qwen3 reasons before acting and answers directly after tool results.
- **Thinking budget** (`webllm: { thinkingBudget }`, default 2048 tokens). Past the budget, reasoning is closed and the step is re-asked with thinking off. It exists to stop runaway deliberation, so keep it well above normal reasoning.
  - Qwen3 4B without thinking describes SQL instead of running it. At 1024, the budget cut 17% of SQL turns, and their accuracy dropped.
  - Only 2 of 246 eval turns exceeded 2048 tokens. Cut-off steps report `metrics.thinkingCutOff`.
- **Reasoning history** (`webllm: { reasoningHistory }`), measured on 3 conversations × 3 repeats (33 graded later turns):

  | Mode | Later turns passed | Time to first token (later turns) | Behaviour |
  |---|---|---|---|
  | `current-turn` (default) | **33/33** | 3.2 s | Drops earlier reasoning, as Qwen3's template does. The prompt is re-read at every new user turn. |
  | `auto` | 31/33 | **0.40 s** | Keeps reasoning, so the cache survives across turns. Past 60% of the window it compacts once, only at a turn boundary, then accumulates again. |
  | `all` | similar to `auto` | 0.35 s | Keeps everything until the history budget trims it. |

  With a 4K window, where `auto` compaction fires on the real model, it passed 22/22 later turns. The quality risk of keeping reasoning is carry-over: in the failures we saw, errors or focus from an earlier turn leaked into later ones. Choose `auto` for chat UIs where first-token latency matters; keep the default for accuracy-critical flows.
- **Per-step metrics.** Every `step-finish` event carries prefill tokens, KV reuse, prompt size, compaction, time to first token, prefill/decode throughput and grammar compile time.
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

Multi-turn cases grade every turn (`turns: [{ input, expect }, …]`). The report adds later-turn metrics, where conversation history matters: pass rate, time to first token, prefill tokens, prompt size and KV reuse.

The playground's `eval.html` runs two suites:
- `suite=single`: 9 cases covering private RAG, SQL actions, chained steps and restraint.
- `suite=multi`: 3 conversations of 4–5 turns.

Configure by query string, e.g. `?suite=multi&history=auto&ctx=4096&repeats=3`.

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
- **KV-cache reuse.** Steps after a tool result prefill ~250 new tokens instead of the full prompt. With `reasoningHistory: 'auto'`, later turns prefill ~200 tokens and start answering in 0.4 s (vs 3.2 s).
- **Multi-turn suite.** 3 conversations × 3 repeats: 33/33 later turns with the defaults.
- **Offline from cache** (strict mode, `127.0.0.1`). With the model host blocked after first load, the app indexed, searched, reranked and answered correctly entirely from the browser cache. `modelCache.status()` reported correct cached state before download, after download, across reload and after deletion, with zero off-origin requests.
- **Node e2e suite.** Checks every embedding preset and the reranker against real weights, plus Qwen3 0.6B completing tool calls and RAG answers on CPU.

Not yet verified: Windows/Linux GPUs, mobile browsers, Chrome built-in AI, and live Ollama / LM Studio servers (their adapters are tested against recorded protocol responses). The eval suites are small (9 single-turn cases, 3 conversations), so treat pass rates as directional and extend them with cases from your own domain.
