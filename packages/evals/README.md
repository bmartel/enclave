# @enclave/evals

Production evals for enclave. The suites run the real agent on real WebGPU against a realistic world: an internal knowledge base, a business database, a custom CRM skill and long-term memory. They are graded with checks that are themselves tested.

## What's covered

65 cases (84 graded turns) in six suites:

| Suite | Cases | What it exercises |
|---|---|---|
| `rag` | 18 | Facts, per-office near-duplicates, current vs superseded policies, a retired system, a table, multi-hop across documents, negation, arithmetic over policy, eligibility, a fact buried in a long manual, paraphrase, a German document, two-part questions, and unanswerable / false-premise questions |
| `sql` | 16 | Counts, revenue by month, top customer, AOV, anti-joins, date ranges, NULLs, grouping, month-over-month change, vocabulary mismatch, recent records, schema discovery, writes with approval, a denied destructive change, an ambiguous destructive request |
| `crm` (custom skill) | 12 | Name → id lookup, relative dates, informal priority mapping, disambiguation by company, asking when ambiguous, updating the right ticket, counts, bulk actions, recovering from a validation error, denied and approved deletions, hostile text in arguments |
| `memory` | 4 | Recall in a new conversation, never storing secrets, updating a fact, forgetting on request |
| `conversations` | 6 | Corrections, pronouns across turns, analysis that evolves with writes, support workflows switching between skills and RAG |
| `safety` | 9 | Prompt injection in documents (fact tampering, destructive SQL, memory exfiltration), denied destructive SQL, restraint (no tools for general questions), out-of-scope actions, no data egress |

**The world** (`src/fixtures`):
- **Corpus:** 24 answer documents plus 20 near-miss neighbours (sick leave vs parental leave, holidays vs vacation, travel booking vs meal limits, firmware notes vs troubleshooting), with two planted prompt injections.
- **Business database:** deterministic seed of 40 customers, 12 products, 150 orders and their items.
- **CRM skill:** typed tools, one approval-gated tool, server-side validation, and a fixed "today" (Monday 2026-09-28), so relative dates are testable.

## How the graders are proven

A pass rate is only as trustworthy as its graders. `test/graders.test.ts` runs on CPU in seconds and checks every case three ways:

1. **Reference solutions must pass.** Each case carries its ideal tool calls and answer. They are executed for real against PGlite and the skills, then graded. This proves the grader accepts correct work.
2. **A do-nothing model must fail.** "I'm not sure." must fail every case, except the few where declining is the correct behaviour (`nullPasses`). This proves the grader rejects non-answers.
3. **Answer variants.** 79 alternative final answers: correct phrasings must pass, and plausible wrong answers must fail (superseded values, injected facts, and claiming an action happened when it was denied). This proves the grader is robust to wording.

Expected values for data questions come from reference SQL at grading time, never hard-coded strings. Numbers are matched formatting-insensitively ($12,345.67, 12345.67, 12.3k).

While this suite was being built:
- **The automated checks caught 9 bugs:**
  - a plural mismatch ("accessory" vs "Accessories")
  - four cases a non-answer could pass (two denial cases, an injection case and the exfiltration case)
  - two missing phrasings ("not published", "nothing was dropped")
  - a fixture where every customer name appeared twice
  - one case that needed a `nullPasses` exception, because refusing outright is also correct there
- **Writing variants exposed 5 over-strict graders.** They would have failed correct answers that mention a superseded value as superseded, or that flag an injected value as suspicious.

## Statistics

Reports give pass rates with 95% Wilson intervals, per-tag breakdowns, and consistency across repeats (a case counts only if every repeat passed). They also record latency (p50/p90, time to first token) and KV-cache reuse. `compareReports` flags a regression as significant only when the intervals separate.

## Retrieval benchmark

`src/retrieval.ts` has 91 labeled queries covering keyword, paraphrase, versioning, near-miss, long-document, table, false-premise and multilingual lookups (German, French and Spanish queries, and a German document). `test/retrieval.e2e.test.ts` runs them with real embedding models and rerankers on CPU, and writes `reports/retrieval.json`.

## Relevance calibration

`test/relevance.e2e.test.ts` measures each embedder's top similarity for 95 on-topic questions and 18 conversational messages: commands, follow-ups, chit-chat, and requests meant for other tools. The results set each preset's `relevanceFloor`, the threshold below which auto-retrieval stays silent.

```sh
SIMS_OUT=reports/relevance.txt SIMS_PRESET=gte-small ENCLAVE_E2E=1 npx vitest run test/relevance.e2e.test.ts
```

## Running

```sh
pnpm --filter @enclave/evals test               # grader validation (CPU, seconds)
pnpm --filter @enclave/evals test:retrieval     # retrieval benchmark (CPU, downloads models)
pnpm --filter @enclave/evals eval               # full suite on WebGPU: qwen3-4b, 3 repeats
pnpm --filter @enclave/evals eval "repeats=1&tags=rag,safety"
pnpm --filter @enclave/evals eval "model=qwen3-8b&label=8b" "model=qwen3-1.7b&label=1.7b"
pnpm --filter @enclave/evals compare reports/a.json reports/b.json   # exits 1 on significant regression
```

Runner parameters:

| Parameter | Values |
|---|---|
| `model` | catalog preset |
| `thinking` | `true`, `false` or `auto` |
| `history` | `current-turn`, `all` or `auto` |
| `constrain` | `0` or `1` |
| `embedding` | embedding preset |
| `reranker` | reranker preset, `auto` or `none` |
| `searchMode` | `vector` or `hybrid` |
| `repeats` | number of runs per case |
| `tags` | comma-separated tags to run |
| `only` | substring of case names to run |
| `label` | name for the report |

While a run is going, the log prints every completed run (`✓`/`✗`, with the model's answer on failures). For a summary (progress bar, ETA, pass rate per suite, recent failures) run this from another terminal:

```sh
pnpm --filter @enclave/evals status   # reads reports/progress.json
```

Each finished config writes `reports/<date>-<label>.json` containing every answer, tool call and timing.

Set `CHROME_PATH` and `EVAL_PROFILE` (a browser profile directory, which keeps model caches) as needed.

## Adding a case

```ts
{
  name: 'crm: reopen a ticket',
  tags: ['skill'],
  input: 'Reopen the invoice address ticket for Linus Berg.',
  setup: resetData,
  expect: { tools: ['update_ticket_status'], check: dbState(`select status from crm_tickets where id = 4`, 'open', 'not reopened') },
  reference: [{ calls: [{ name: 'update_ticket_status', input: { ticket_id: 4, status: 'open' } }], answer: 'Reopened ticket #4.' }],
  variants: { pass: ['Ticket #4 is open again.'], fail: [] },
}
```

Run `pnpm --filter @enclave/evals test` before trusting the new case on a model.
