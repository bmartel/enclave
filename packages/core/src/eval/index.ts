import type { Enclave } from '../enclave.js'
import type { Knowledge, SearchOptions } from '../rag/knowledge.js'
import type { AgentEvent, ApprovalHandler, ToolCall } from '../types.js'

// ---------------------------------------------------------------------------
// Agent evals
// ---------------------------------------------------------------------------

export * from './matchers.js'


export type AnswerMatcher = RegExp | string | ((text: string) => boolean)

export interface EvalExpect {
  /** Tools that must be called (any order) during the turn. */
  tools?: string[]
  /** Tools that must not be called. */
  forbidTools?: string[]
  /** The turn must be answered without any tool call. */
  noTools?: boolean
  /** Answer must match. Strings match case-insensitively as substrings. */
  answer?: AnswerMatcher | AnswerMatcher[]
  /** Answer must not match (e.g. hallucinated facts). */
  notAnswer?: AnswerMatcher | AnswerMatcher[]
  /** Custom check: return true, or a failure message. */
  check?(ctx: CheckContext): boolean | string | Promise<boolean | string>
  /** Fail if more model steps than this were needed. */
  maxSteps?: number
}

export interface CheckContext {
  ai: Enclave
  text: string
  calls: ToolCall[]
  /** Tool outputs in call order (full outputs, as the UI sees them). */
  toolResults: { call: ToolCall; output: unknown; isError: boolean }[]
  events: AgentEvent[]
}

export interface EvalTurn {
  input: string
  /** Omit to send the turn without grading it. */
  expect?: EvalExpect
  /** Conversation to send this turn in (default `main`). Use several to test cross-thread memory. */
  thread?: string
  /** Approval decision for this turn; overrides the case's. */
  approve?: boolean | ApprovalHandler
}

export interface EvalCase {
  name: string
  /** User turns sent in order in a fresh thread; `expect` grades the last one. */
  input?: string | string[]
  expect?: EvalExpect
  /** Multi-turn conversations with per-turn expectations. Takes precedence over `input`. */
  turns?: EvalTurn[]
  /** Prepare state (ingest documents, create tables…). Runs before every repeat. */
  setup?(ai: Enclave): void | Promise<void>
  /** Clean up after every repeat. */
  teardown?(ai: Enclave): void | Promise<void>
  /** Categories for per-tag reporting, e.g. `['rag', 'multi-hop']`. */
  tags?: string[]
  /** Approval decision for tools that need it. Default: approve. */
  approve?: boolean | ApprovalHandler
  /** Fail a turn that runs longer than this. Default: EvalOptions.turnTimeoutMs. */
  timeoutMs?: number
}

export interface TurnResult {
  index: number
  input: string
  graded: boolean
  passed: boolean
  failures: string[]
  text: string
  calls: { name: string; input: unknown }[]
  toolErrors: number
  steps: number
  durationMs: number
  timeToFirstTokenMs: number | undefined
  /** Tokens actually prefilled across the turn's steps (small when the KV cache was reused). */
  prefillTokens: number
  outputTokens: number
  /** Largest prompt sent during the turn, in characters. */
  promptChars: number | undefined
  /** Older reasoning was compacted during this turn. */
  compacted: boolean
  kvReuseRate: number | undefined
  /** Characters of reasoning streamed during the turn. */
  reasoningChars: number
  /** End of the turn's reasoning, kept for failed turns (why did it time out?). */
  reasoningTail?: string
}

export interface CaseResult {
  name: string
  tags: string[]
  repeat: number
  passed: boolean
  failures: string[]
  /** Final turn's answer and tool calls. */
  text: string
  calls: { name: string; input: unknown }[]
  toolErrors: number
  steps: number
  /** Total time of graded turns. */
  durationMs: number
  timeToFirstTokenMs: number | undefined
  prefillTokens: number
  outputTokens: number
  kvReuseRate: number | undefined
  turns: TurnResult[]
}

export interface RateWithInterval {
  passed: number
  total: number
  rate: number
  /** 95% Wilson score interval. */
  low: number
  high: number
}

export interface EvalReport {
  label: string | undefined
  passRate: number
  /** Pass rate over all runs, with a 95% confidence interval. */
  overall: RateWithInterval
  /** Cases that passed on every repeat (consistency), and on at least one. */
  consistency: { allRepeats: RateWithInterval; anyRepeat: RateWithInterval; repeats: number }
  byTag: ({ tag: string } & RateWithInterval)[]
  results: CaseResult[]
  /** Set when failFast/maxFailures ended the run early: rates cover only the runs made. */
  stopped?: { reason: string; ranRuns: number; plannedRuns: number }
  byCase: { name: string; tags: string[]; passRate: number; passed: number; runs: number; meanMs: number; failures: string[] }[]
  latency: { meanMs: number; p50Ms: number; p90Ms: number; ttftP50Ms: number | undefined }
  tokens: { meanOutput: number; meanPrefill: number }
  toolErrorRate: number
  kvReuseRate: number | undefined
  /** Graded turns after the first in multi-turn cases: where conversation history matters. */
  laterTurns:
    | {
        count: number
        passRate: number
        ttftP50Ms: number | undefined
        meanPrefillTokens: number
        meanPromptChars: number | undefined
        kvReuseRate: number | undefined
      }
    | undefined
}

export interface EvalOptions {
  /** Shown in reports; useful when comparing configurations. */
  label?: string
  /** Runs per case. Small models are stochastic: use 3+ before trusting a number. */
  repeats?: number
  /** Approval decisions during evals. Default: approve everything. */
  onApproval?: ApprovalHandler
  onResult?(result: CaseResult): void
  signal?: AbortSignal
  /** Time limit for each turn (one user message and the agent's full response). Default 4 minutes. */
  turnTimeoutMs?: number
  /** Only run cases carrying at least one of these tags. */
  tags?: string[]
  /**
   * Stop at the first failing run. For checking a fix on cases that must
   * pass: a wrong fix costs one run, not a whole suite.
   */
  failFast?: boolean
  /**
   * Stop once more runs than this have failed. For regression guards: set it
   * to a baseline's failure count plus a small allowance.
   */
  maxFailures?: number
  /** Case names to run first (e.g. last run's failures), so a regression shows up early. */
  first?: string[]
}

/**
 * Run cases against an enclave and grade them. Everything executes locally,
 * so apps can tune prompts, skills, models and retrieval on real data without
 * that data leaving the device.
 */
export async function runEval(ai: Enclave, cases: EvalCase[], options: EvalOptions = {}): Promise<EvalReport> {
  const repeats = options.repeats ?? 1
  const results: CaseResult[] = []
  const tagged = options.tags?.length ? cases.filter((c) => c.tags?.some((t) => options.tags!.includes(t))) : cases
  const priority = new Map(options.first?.map((name, i) => [name, i]))
  // Stable sort: listed cases first (in the given order), the rest as defined.
  const selected = [...tagged].sort((a, b) => (priority.get(a.name) ?? Infinity) - (priority.get(b.name) ?? Infinity))
  let failures = 0
  let stopped: string | undefined
  run: for (const testCase of selected) {
    for (let repeat = 0; repeat < repeats; repeat++) {
      options.signal?.throwIfAborted()
      const result = await runCase(ai, testCase, repeat, options)
      results.push(result)
      options.onResult?.(result)
      if (!result.passed) failures++
      if (!result.passed && options.failFast) {
        stopped = `fail-fast: ${result.name} #${result.repeat}`
        break run
      }
      if (options.maxFailures !== undefined && failures > options.maxFailures) {
        stopped = `more than ${options.maxFailures} failing runs`
        break run
      }
    }
  }
  const report = summarize(options.label, results, repeats)
  if (stopped) report.stopped = { reason: stopped, ranRuns: results.length, plannedRuns: selected.length * repeats }
  return report
}

function turnsOf(testCase: EvalCase): EvalTurn[] {
  if (testCase.turns?.length) return testCase.turns
  const inputs = Array.isArray(testCase.input) ? testCase.input : testCase.input === undefined ? [] : [testCase.input]
  if (!inputs.length) throw new Error(`Eval case "${testCase.name}" has no input or turns`)
  return inputs.map((input, i) => ({ input, ...(i === inputs.length - 1 && testCase.expect ? { expect: testCase.expect } : {}) }))
}

async function runCase(ai: Enclave, testCase: EvalCase, repeat: number, options: EvalOptions): Promise<CaseResult> {
  const turns: TurnResult[] = []
  const threads = new Map<string, ReturnType<Enclave['thread']>>()
  const turnTimeout = testCase.timeoutMs ?? options.turnTimeoutMs ?? 240_000
  let aborted = false

  try {
    await testCase.setup?.(ai)
    for (const [index, turn] of turnsOf(testCase).entries()) {
      const key = turn.thread ?? 'main'
      let thread = threads.get(key)
      if (!thread) threads.set(key, (thread = ai.thread()))
      const decision = turn.approve ?? testCase.approve ?? options.onApproval ?? true
      const onApproval: ApprovalHandler = typeof decision === 'function' ? decision : () => decision
      // Timed per turn: a later turn in a long conversation gets the same budget.
      const timeout = AbortSignal.timeout(turnTimeout)
      const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout
      const started = performance.now()
      const events: AgentEvent[] = []
      const failures: string[] = []
      try {
        for await (const event of thread.send(turn.input, { onApproval, signal })) events.push(event)
        if (signal.aborted) throw new Error(timeout.aborted ? 'timed out' : 'aborted')
      } catch (error) {
        failures.push(`error: ${timeout.aborted ? 'timed out' : error instanceof Error ? error.message : String(error)}`)
        aborted = true
      }
      turns.push(await gradeTurn(ai, index, turn, events, failures, performance.now() - started))
      if (aborted) break
    }
  } catch (error) {
    turns.push({
      index: turns.length, input: '(setup)', graded: true, passed: false,
      failures: [`setup failed: ${error instanceof Error ? error.message : String(error)}`],
      text: '', calls: [], toolErrors: 0, steps: 0, durationMs: 0, timeToFirstTokenMs: undefined,
      prefillTokens: 0, outputTokens: 0, promptChars: undefined, compacted: false, kvReuseRate: undefined, reasoningChars: 0,
    })
    aborted = true
  } finally {
    for (const thread of threads.values()) await thread.delete().catch(() => undefined)
    await Promise.resolve(testCase.teardown?.(ai)).catch(() => undefined)
  }

  const graded = turns.filter((t) => t.graded)
  const last = turns.at(-1)
  const reuse = graded.flatMap((t) => (t.kvReuseRate === undefined ? [] : [t.kvReuseRate]))
  const ttfts = graded.flatMap((t) => (t.timeToFirstTokenMs === undefined ? [] : [t.timeToFirstTokenMs]))
  const multi = turns.length > 1
  return {
    name: testCase.name,
    tags: testCase.tags ?? [],
    repeat,
    passed: graded.length > 0 && graded.every((t) => t.passed) && !aborted,
    failures: graded.flatMap((t) => t.failures.map((f) => (multi ? `turn ${t.index + 1}: ${f}` : f))),
    text: last?.text ?? '',
    calls: last?.calls ?? [],
    toolErrors: graded.reduce((n, t) => n + t.toolErrors, 0),
    steps: graded.reduce((n, t) => n + t.steps, 0),
    durationMs: Math.round(graded.reduce((n, t) => n + t.durationMs, 0)),
    timeToFirstTokenMs: ttfts.length ? Math.round(mean(ttfts)) : undefined,
    prefillTokens: graded.reduce((n, t) => n + t.prefillTokens, 0),
    outputTokens: graded.reduce((n, t) => n + t.outputTokens, 0),
    kvReuseRate: reuse.length ? mean(reuse) : undefined,
    turns,
  }
}

async function gradeTurn(
  ai: Enclave,
  index: number,
  turn: EvalTurn,
  events: AgentEvent[],
  failures: string[],
  durationMs: number,
): Promise<TurnResult> {
  const calls = events.flatMap((e) => (e.type === 'tool-call' ? [e.call] : []))
  const steps = events.flatMap((e) => (e.type === 'step-finish' ? [e] : []))
  const finish = events.findLast((e) => e.type === 'finish')
  const last = events.findLast((e) => e.type === 'message' && e.message.role === 'assistant')
  const text = last?.type === 'message' ? last.message.content : ''
  const reasoning = events.map((e) => (e.type === 'reasoning-delta' ? e.delta : '')).join('')
  const expect = turn.expect

  if (expect) {
    if (finish?.type === 'finish' && finish.reason === 'max-steps') failures.push('hit maxSteps')
    const called = new Set(calls.map((c) => c.name))
    for (const name of expect.tools ?? []) if (!called.has(name)) failures.push(`did not call ${name}`)
    for (const name of expect.forbidTools ?? []) if (called.has(name)) failures.push(`called forbidden ${name}`)
    if (expect.noTools && calls.length) failures.push(`expected no tools, called ${[...called].join(', ')}`)
    for (const m of list(expect.answer)) if (!matches(m, text)) failures.push(`answer does not match ${describe(m)}`)
    for (const m of list(expect.notAnswer)) if (matches(m, text)) failures.push(`answer matches forbidden ${describe(m)}`)
    if (expect.maxSteps !== undefined && steps.length > expect.maxSteps) failures.push(`${steps.length} steps > ${expect.maxSteps}`)
    if (expect.check && !failures.some((f) => f.startsWith('error'))) {
      try {
        const toolResults = events.flatMap((e) => (e.type === 'tool-result' ? [{ call: e.call, output: e.output, isError: e.isError }] : []))
      const verdict = await expect.check({ ai, text, calls, toolResults, events })
        if (verdict !== true) failures.push(typeof verdict === 'string' ? verdict : 'check failed')
      } catch (error) {
        failures.push(`check threw: ${error instanceof Error ? error.message : String(error)}`)
      }
    }
  }

  const metrics = steps.flatMap((s) => (s.metrics ? [s.metrics] : []))
  const withReuse = metrics.filter((m) => m.kvCacheReused !== undefined)
  const promptSizes = metrics.flatMap((m) => (m.promptChars === undefined ? [] : [m.promptChars]))
  return {
    index,
    input: turn.input,
    graded: !!expect,
    passed: failures.length === 0,
    failures,
    text,
    calls: calls.map((c) => ({ name: c.name, input: c.input })),
    toolErrors: events.filter((e) => e.type === 'tool-result' && e.isError).length,
    steps: steps.length,
    durationMs: Math.round(durationMs),
    timeToFirstTokenMs: steps[0]?.metrics?.timeToFirstTokenMs,
    prefillTokens: steps.reduce((n, s) => n + (s.usage?.inputTokens ?? 0), 0),
    outputTokens: steps.reduce((n, s) => n + (s.usage?.outputTokens ?? 0), 0),
    promptChars: promptSizes.length ? Math.max(...promptSizes) : undefined,
    compacted: metrics.some((m) => m.compacted),
    kvReuseRate: withReuse.length ? withReuse.filter((m) => m.kvCacheReused).length / withReuse.length : undefined,
    reasoningChars: reasoning.length,
    ...(failures.length && reasoning ? { reasoningTail: reasoning.slice(-1500) } : {}),
  }
}

function summarize(label: string | undefined, results: CaseResult[], repeats: number): EvalReport {
  const names = [...new Set(results.map((r) => r.name))]
  const passedCount = results.filter((r) => r.passed).length
  const perCase = names.map((name) => results.filter((r) => r.name === name))
  const tags = [...new Set(results.flatMap((r) => r.tags))].sort()
  const durations = results.map((r) => r.durationMs).sort((a, b) => a - b)
  const ttfts = results.flatMap((r) => (r.timeToFirstTokenMs === undefined ? [] : [r.timeToFirstTokenMs])).sort((a, b) => a - b)
  const reuse = results.flatMap((r) => (r.kvReuseRate === undefined ? [] : [r.kvReuseRate]))
  const steps = results.reduce((n, r) => n + r.steps, 0)

  const later = results.flatMap((r) => r.turns.filter((t) => t.graded && t.index > 0))
  const laterTtft = later.flatMap((t) => (t.timeToFirstTokenMs === undefined ? [] : [t.timeToFirstTokenMs])).sort((a, b) => a - b)
  const laterPrompt = later.flatMap((t) => (t.promptChars === undefined ? [] : [t.promptChars]))
  const laterReuse = later.flatMap((t) => (t.kvReuseRate === undefined ? [] : [t.kvReuseRate]))

  return {
    label,
    passRate: rate(passedCount, results.length),
    overall: wilson(passedCount, results.length),
    consistency: {
      allRepeats: wilson(perCase.filter((rs) => rs.every((r) => r.passed)).length, perCase.length),
      anyRepeat: wilson(perCase.filter((rs) => rs.some((r) => r.passed)).length, perCase.length),
      repeats,
    },
    byTag: tags.map((tag) => {
      const rs = results.filter((r) => r.tags.includes(tag))
      return { tag, ...wilson(rs.filter((r) => r.passed).length, rs.length) }
    }),
    results,
    byCase: perCase.map((rs) => ({
      name: rs[0]!.name,
      tags: rs[0]!.tags,
      passRate: rate(rs.filter((r) => r.passed).length, rs.length),
      passed: rs.filter((r) => r.passed).length,
      runs: rs.length,
      meanMs: Math.round(mean(rs.map((r) => r.durationMs))),
      failures: [...new Set(rs.flatMap((r) => r.failures))],
    })),
    latency: {
      meanMs: Math.round(mean(durations)),
      p50Ms: percentile(durations, 0.5),
      p90Ms: percentile(durations, 0.9),
      ttftP50Ms: ttfts.length ? percentile(ttfts, 0.5) : undefined,
    },
    tokens: {
      meanOutput: Math.round(mean(results.map((r) => r.outputTokens))),
      meanPrefill: Math.round(mean(results.map((r) => r.prefillTokens))),
    },
    toolErrorRate: rate(results.reduce((n, r) => n + r.toolErrors, 0), Math.max(steps, 1)),
    kvReuseRate: reuse.length ? mean(reuse) : undefined,
    laterTurns: later.length
      ? {
          count: later.length,
          passRate: rate(later.filter((t) => t.passed).length, later.length),
          ttftP50Ms: laterTtft.length ? percentile(laterTtft, 0.5) : undefined,
          meanPrefillTokens: Math.round(mean(later.map((t) => t.prefillTokens))),
          meanPromptChars: laterPrompt.length ? Math.round(mean(laterPrompt)) : undefined,
          kvReuseRate: laterReuse.length ? mean(laterReuse) : undefined,
        }
      : undefined,
  }
}

/** Plain-text table for consoles and logs. */
export function formatReport(report: EvalReport): string {
  const pct = (x: number) => `${Math.round(x * 100)}%`
  const ci = (r: RateWithInterval) => `${pct(r.rate)} [${pct(r.low)}–${pct(r.high)}]`
  const lines = [
    ...(report.stopped ? [`STOPPED EARLY (${report.stopped.reason}) after ${report.stopped.ranRuns} of ${report.stopped.plannedRuns} runs`] : []),
    `${report.label ?? 'eval'}: ${ci(report.overall)} of ${report.overall.total} runs passed` +
      (report.consistency.repeats > 1 ? ` · ${ci(report.consistency.allRepeats)} of cases passed all ${report.consistency.repeats} repeats` : ''),
    `  latency p50 ${(report.latency.p50Ms / 1000).toFixed(1)}s · p90 ${(report.latency.p90Ms / 1000).toFixed(1)}s` +
      (report.latency.ttftP50Ms !== undefined ? ` · TTFT p50 ${report.latency.ttftP50Ms}ms` : '') +
      ` · ${report.tokens.meanOutput} out tok/case` +
      (report.kvReuseRate !== undefined ? ` · KV reuse ${pct(report.kvReuseRate)}` : ''),
  ]
  if (report.byTag.length) {
    lines.push(`  by tag: ${report.byTag.map((t) => `${t.tag} ${t.passed}/${t.total}`).join(' · ')}`)
  }
  const later = report.laterTurns
  if (later) {
    lines.push(
      `  later turns (${later.count}): ${pct(later.passRate)} passed` +
        (later.ttftP50Ms !== undefined ? ` · TTFT p50 ${later.ttftP50Ms}ms` : '') +
        ` · ${later.meanPrefillTokens} prefill tok/turn` +
        (later.meanPromptChars !== undefined ? ` · prompt ${later.meanPromptChars} chars` : '') +
        (later.kvReuseRate !== undefined ? ` · KV reuse ${pct(later.kvReuseRate)}` : ''),
    )
  }
  lines.push(
    ...report.byCase.map(
      (c) =>
        `  ${c.passRate === 1 ? '✓' : c.passRate === 0 ? '✗' : '~'} ${c.name.padEnd(32)} ${pct(c.passRate).padStart(4)}  ${(c.meanMs / 1000).toFixed(1)}s${c.failures.length ? `  ${c.failures.join('; ')}` : ''}`,
    ),
  )
  return lines.join('\n')
}

// ---------------------------------------------------------------------------
// Retrieval evals
// ---------------------------------------------------------------------------

export interface RetrievalCase {
  query: string
  /** Document ids that should be retrieved. */
  relevant: string[]
}

export interface RetrievalReport {
  k: number
  /** Share of relevant documents found in the top k, averaged over queries. */
  recall: number
  /** Mean reciprocal rank of the first relevant document. */
  mrr: number
  ndcg: number
  perQuery: { query: string; recall: number; reciprocalRank: number; retrieved: string[] }[]
}

/**
 * Measure retrieval quality: use it to choose an embedding model, reranker,
 * chunk size or hybrid/vector/keyword mode for your own documents.
 */
export async function evalRetrieval(
  knowledge: Knowledge,
  cases: RetrievalCase[],
  options: { k?: number } & Omit<SearchOptions, 'limit'> = {},
): Promise<RetrievalReport> {
  const k = options.k ?? 5
  const perQuery: RetrievalReport['perQuery'] = []
  let ndcgSum = 0
  for (const c of cases) {
    const hits = await knowledge.search(c.query, { ...options, limit: k * 3 })
    // Several chunks can come from one document: rank by first appearance.
    const retrieved = [...new Set(hits.map((h) => h.documentId))].slice(0, k)
    const relevant = new Set(c.relevant)
    const found = retrieved.filter((id) => relevant.has(id)).length
    const firstRank = retrieved.findIndex((id) => relevant.has(id))
    const dcg = retrieved.reduce((sum, id, i) => sum + (relevant.has(id) ? 1 / Math.log2(i + 2) : 0), 0)
    const ideal = Array.from({ length: Math.min(relevant.size, k) }, (_, i) => 1 / Math.log2(i + 2)).reduce((a, b) => a + b, 0)
    ndcgSum += ideal ? dcg / ideal : 0
    perQuery.push({
      query: c.query,
      recall: relevant.size ? found / Math.min(relevant.size, k) : 0,
      reciprocalRank: firstRank === -1 ? 0 : 1 / (firstRank + 1),
      retrieved,
    })
  }
  return {
    k,
    recall: mean(perQuery.map((q) => q.recall)),
    mrr: mean(perQuery.map((q) => q.reciprocalRank)),
    ndcg: cases.length ? ndcgSum / cases.length : 0,
    perQuery,
  }
}

// ---------------------------------------------------------------------------

function list<T>(value: T | T[] | undefined): T[] {
  return value === undefined ? [] : Array.isArray(value) ? value : [value]
}

function matches(m: AnswerMatcher, text: string): boolean {
  if (typeof m === 'string') return text.toLowerCase().includes(m.toLowerCase())
  if (m instanceof RegExp) return m.test(text)
  return m(text)
}

function describe(m: AnswerMatcher): string {
  return typeof m === 'function' ? ((m as { label?: string }).label ?? 'predicate') : String(m)
}

/** 95% Wilson score interval: honest bounds for pass rates on small samples. */
export function wilson(passed: number, total: number, z = 1.96): RateWithInterval {
  if (!total) return { passed, total, rate: 0, low: 0, high: 0 }
  const p = passed / total
  const denom = 1 + (z * z) / total
  const centre = (p + (z * z) / (2 * total)) / denom
  const half = (z * Math.sqrt((p * (1 - p)) / total + (z * z) / (4 * total * total))) / denom
  return { passed, total, rate: p, low: Math.max(0, centre - half), high: Math.min(1, centre + half) }
}

export interface ReportComparison {
  overall: { before: RateWithInterval; after: RateWithInterval; delta: number; significant: boolean }
  /** Cases whose pass rate dropped. `significant` when the intervals don't overlap. */
  regressions: { name: string; before: number; after: number; significant: boolean }[]
  improvements: { name: string; before: number; after: number; significant: boolean }[]
}

/**
 * Compare two reports (e.g. before/after a prompt change). A difference is
 * flagged significant only when the 95% intervals don't overlap, which is
 * conservative: it avoids chasing noise from a handful of runs.
 */
export function compareReports(before: EvalReport, after: EvalReport): ReportComparison {
  const disjoint = (a: RateWithInterval, b: RateWithInterval) => a.high < b.low || b.high < a.low
  const regressions: ReportComparison['regressions'] = []
  const improvements: ReportComparison['improvements'] = []
  for (const next of after.byCase) {
    const prev = before.byCase.find((c) => c.name === next.name)
    if (!prev || prev.passRate === next.passRate) continue
    const entry = {
      name: next.name,
      before: prev.passRate,
      after: next.passRate,
      significant: disjoint(wilson(prev.passed, prev.runs), wilson(next.passed, next.runs)),
    }
    ;(next.passRate < prev.passRate ? regressions : improvements).push(entry)
  }
  return {
    overall: {
      before: before.overall,
      after: after.overall,
      delta: after.overall.rate - before.overall.rate,
      significant: disjoint(before.overall, after.overall),
    },
    regressions,
    improvements,
  }
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0)
const rate = (n: number, d: number) => (d ? n / d : 0)
function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))]!
}
