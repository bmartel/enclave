import { formatReport, runEval } from '@enclave/core/eval'
import { knowledgeSkill, memorySkill, sqlSkill } from '@enclave/core/skills'
import { createWebEnclave, type ThinkingMode } from '@enclave/core/web'
import { HANDBOOK, suite } from './eval-suite'

const params = new URLSearchParams(location.search)
const $ = (id: string) => document.getElementById(id)!
const thinkingParam = params.get('thinking') ?? 'true'
const thinking: ThinkingMode = thinkingParam === 'auto' ? 'auto' : thinkingParam !== 'false'
const config = {
  model: params.get('model') ?? 'qwen3-4b',
  thinking,
  reasoningHistory: (params.get('history') ?? 'current-turn') as 'current-turn' | 'all',
  constrainToolCalls: params.get('constrain') !== '0',
  repeats: Number(params.get('repeats') ?? 1),
  only: params.get('only'),
}
const label = `${config.model} thinking=${config.thinking} history=${config.reasoningHistory} constrain=${config.constrainToolCalls}`

const ai = await createWebEnclave({
  workers: {
    db: new Worker(new URL('./db.worker.ts', import.meta.url), { type: 'module' }),
    ml: new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' }),
    llm: new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' }),
  },
  dataDir: 'memory://',
  llm: config.model,
  thinking: config.thinking,
  webllm: { reasoningHistory: config.reasoningHistory, constrainToolCalls: config.constrainToolCalls },
  skills: [knowledgeSkill(), sqlSkill(), { ...memorySkill(), lazy: true }],
  onProgress: (p) => ($('status').textContent = p.text),
})
await ai.knowledge!.ingest(HANDBOOK, { collection: 'handbook' })
await (ai.model as { load?(): Promise<void> }).load?.()

const cases = config.only ? suite.filter((c) => c.name.includes(config.only!)) : suite
const report = await runEval(ai, cases, {
  label,
  repeats: config.repeats,
  onResult: (r) => ($('status').textContent += `\n${r.passed ? '✓' : '✗'} ${r.name} (${(r.durationMs / 1000).toFixed(1)}s)${r.failures.length ? ` — ${r.failures.join('; ')}` : ''}`),
})
$('report').textContent = formatReport(report)
Object.assign(globalThis, { __evalReport: report })
