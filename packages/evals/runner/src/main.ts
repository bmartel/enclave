import { formatReport, runEval } from '@enclave/core/eval'
import { lmstudio, localEmbedder, ollama, ollamaEmbedder, OLLAMA_EMBEDDING_PRESETS } from '@enclave/core/models/local'
import { createWebEnclave, type ThinkingMode } from '@enclave/core/web'
import { ALL_CASES } from '../../src/suites/index.js'
import { prepareWorld, suiteSkills } from '../../src/world.js'

const params = new URLSearchParams(location.search)
const $ = (id: string) => document.getElementById(id)!
const log = (line: string) => ($('status').textContent += `\n${line}`)

const thinkingParam = params.get('thinking') ?? 'auto'
const config = {
  model: params.get('model') ?? 'qwen3-4b',
  thinking: (thinkingParam === 'auto' ? 'auto' : thinkingParam !== 'false') as ThinkingMode,
  reasoningHistory: (params.get('history') ?? 'current-turn') as 'current-turn' | 'all' | 'auto',
  constrainToolCalls: params.get('constrain') !== '0',
  embedding: params.get('embedding') ?? 'auto',
  reranker: params.get('reranker') ?? 'auto',
  searchMode: params.get('searchMode') as 'vector' | 'hybrid' | null,
  repeats: Number(params.get('repeats') ?? 3),
  tags: params.get('tags')?.split(',').filter(Boolean),
  only: params.get('only'),
  /** Ollama context window (`num_ctx`), for `model=ollama:<tag>`. */
  ctx: Number(params.get('ctx') ?? 32768),
}
// `model=ollama:qwen3.8:27b-q4_K_M` runs the suite on a local Ollama model
// instead of WebLLM; everything else (world, graders, statistics) is shared.
const ollamaTag = config.model.startsWith('ollama:') ? config.model.slice('ollama:'.length) : undefined
// `model=lmstudio:<identifier>`: a model already loaded in LM Studio (`lms load … --context-length`).
const lmstudioId = config.model.startsWith('lmstudio:') ? config.model.slice('lmstudio:'.length) : undefined

/** `lmstudio:<id>` embeddings, with the measured prompts when the model matches a preset (e.g. embeddinggemma). */
function lmstudioEmbedder(id: string) {
  const preset = Object.entries(OLLAMA_EMBEDDING_PRESETS).find(([name]) => id.toLowerCase().includes(name.split(':')[0]!))?.[1]
  if (!preset) throw new Error(`No embedding preset matches ${id}`)
  return localEmbedder({ ...preset, provider: 'lmstudio', model: id })
}
const label = params.get('label') ?? `${config.model} thinking=${config.thinking} history=${config.reasoningHistory} embedding=${config.embedding} reranker=${config.reranker}${config.searchMode ? ` search=${config.searchMode}` : ''}`

try {
  const ai = await createWebEnclave({
    workers: {
      db: new Worker(new URL('./db.worker.ts', import.meta.url), { type: 'module' }),
      ml: new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' }),
      llm: new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' }),
    },
    dataDir: 'memory://',
    llm: ollamaTag
      ? ollama({ model: ollamaTag, contextWindow: config.ctx, think: config.thinking })
      : lmstudioId
        ? lmstudio({ model: lmstudioId, contextWindow: config.ctx, think: config.thinking })
        : config.model,
    thinking: config.thinking,
    // `embedding=ollama:embeddinggemma` embeds through Ollama with the measured preset.
    embedding: config.embedding.startsWith('ollama:')
      ? ollamaEmbedder(config.embedding.slice('ollama:'.length))
      : config.embedding.startsWith('lmstudio:')
        ? lmstudioEmbedder(config.embedding.slice('lmstudio:'.length))
        : config.embedding,
    reranker: config.reranker === 'none' ? false : config.reranker,
    webllm: { reasoningHistory: config.reasoningHistory, constrainToolCalls: config.constrainToolCalls },
    skills: suiteSkills(),
    ...(config.searchMode ? { knowledge: { defaultMode: config.searchMode } } : {}),
    onProgress: (p) => ($('status').textContent = p.text),
  })
  // Fail fast when the model can't answer at all (server down, CORS blocked),
  // instead of recording a 0% report.
  $('status').textContent = 'Checking the model responds…'
  for await (const chunk of ai.model.stream({ system: 'Reply with OK.', messages: [{ role: 'user', content: 'OK?' }], tools: [] })) {
    if (chunk.type === 'finish') break
  }
  $('status').textContent = 'Preparing world (corpus, databases)…'
  await prepareWorld(ai)
  await (ai.model as { load?(): Promise<void> }).load?.()
  $('status').textContent = `Running ${label}`

  const cases = config.only ? ALL_CASES.filter((c) => c.name.includes(config.only!)) : ALL_CASES
  const planned = config.tags ? cases.filter((c) => c.tags.some((t) => config.tags!.includes(t))) : cases
  // Lets the run script report progress against the total.
  Object.assign(globalThis, { __evalPlan: { label, runs: planned.length * config.repeats, cases: planned.length, repeats: config.repeats } })
  const report = await runEval(ai, cases, {
    label,
    repeats: config.repeats,
    ...(config.tags ? { tags: config.tags } : {}),
    // Failures include what the model actually said, so a live run can be judged
    // (model mistake vs grader mistake) without waiting for the final report.
    onResult: (r) =>
      log(
        `${r.passed ? '✓' : '✗'} ${r.name} #${r.repeat} ${(r.durationMs / 1000).toFixed(1)}s` +
          (r.failures.length ? ` — ${r.failures.join('; ')} → ${JSON.stringify(r.text.slice(0, 160))}` : ''),
      ),
  })
  $('report').textContent = formatReport(report)
  Object.assign(globalThis, {
    __evalReport: { ...report, config, environment: { device: ai.device, plan: ai.plan, model: ai.model.id, userAgent: navigator.userAgent } },
  })
} catch (error) {
  Object.assign(globalThis, { __evalError: error instanceof Error ? `${error.message}\n${error.stack}` : String(error) })
  log(`FATAL: ${String(error)}`)
}
