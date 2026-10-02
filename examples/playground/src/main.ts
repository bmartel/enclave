import type { Model, Thread, ToolCall } from '@enclave/core'
import { chromeAI, chromeAIAvailable } from '@enclave/core/models/chrome'
import {
  discoverLocalModels,
  localModel,
  lmstudio,
  ollama,
  ollamaEmbedder,
  recommendOllamaModel,
  OLLAMA_EMBEDDING_PRESETS,
  OLLAMA_LLM_PRESETS,
  type LocalModelInfo,
} from '@enclave/core/models/local'
import { ACCEPT, importTable, loadFiles, tesseractOcr } from '@enclave/core/loaders'
import { knowledgeSkill, memorySkill, sqlSkill } from '@enclave/core/skills'
// pdf.js's worker, bundled and served from this origin.
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import {
  BROWSER_LLMS,
  EMBEDDING_PRESETS,
  RERANKER_PRESETS,
  browserLLM,
  createWebEnclave,
  rankLLMs,
  recommendLLM,
  resolveLLM,
  type WebProgress,
} from '@enclave/core/web'
import { notesSkill } from './notes-skill'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const setStatus = (text: string) => ($('status').textContent = text)

// ---------------------------------------------------------------------------
// Settings (per browser)
// ---------------------------------------------------------------------------

function stored(key: string, fallback = ''): string {
  try {
    return localStorage.getItem(`enclave:${key}`) ?? fallback
  } catch {
    return fallback
  }
}
function store(key: string, value: string): void {
  try {
    localStorage.setItem(`enclave:${key}`, value)
  } catch {
    /* private mode */
  }
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

function onProgress(p: WebProgress) {
  setStatus(p.text)
  const bar = $('progress')
  if (p.progress !== undefined && p.progress < 1) {
    bar.hidden = false
    $('progress-bar').style.width = `${Math.round(p.progress * 100)}%`
  } else if (p.progress === 1) {
    bar.hidden = true
  }
}

// ---------------------------------------------------------------------------
// Boot: device → database → retrieval models → chat model
// ---------------------------------------------------------------------------

const workers = {
  db: new Worker(new URL('./db.worker.ts', import.meta.url), { type: 'module' }),
  ml: new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' }),
  llm: new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' }),
}

// Model selections are encoded as `browser:<preset>`, `chrome`, `ollama:<model>`, `lmstudio:<model>`.
let selection = stored('model')
const thinking = $<HTMLInputElement>('thinking')
thinking.checked = stored('thinking', 'true') === 'true'

// Measured Ollama models get the context window they were evaluated with.
const measured = new Set(OLLAMA_LLM_PRESETS.map((p) => p.tag))

function modelFromSelection(value: string, local: LocalModelInfo[] = []): Model | undefined {
  const [kind, ...rest] = value.split(':')
  const id = rest.join(':')
  if (kind === 'chrome') return chromeAI()
  if (kind === 'ollama' || kind === 'lmstudio') {
    const think = thinking.checked ? ('auto' as const) : false
    const contextWindow = measured.has(id) ? 32768 : undefined
    const info = local.find((m) => m.provider === kind && m.id === id)
    if (info) return localModel(info, { think, ...(contextWindow ? { contextWindow } : {}) })
    return kind === 'ollama' ? ollama({ model: id, think, ...(contextWindow ? { contextWindow } : {}) }) : lmstudio({ model: id })
  }
  return undefined
}

// `pnpm dev:strict`: every model file from this origin, only on-device components.
const strict = import.meta.env.VITE_STRICT === '1'

// Local servers first: a measured Ollama model beats the in-browser models by
// a wide margin (98.5% vs 92% on the evals). A refused localhost connection
// fails instantly, so this costs nothing when no server runs.
let localModels: LocalModelInfo[] = strict ? [] : await discoverLocalModels({ timeoutMs: 800 }).catch(() => [])
const bare = (id: string) => id.replace(/:latest$/, '')
const installed = (value: string) => {
  const [kind, ...rest] = value.split(':')
  return localModels.some((m) => m.provider === kind && bare(m.id) === bare(rest.join(':')))
}
let bootNote = ''
if (!selection) {
  const pick = recommendOllamaModel(localModels)
  if (pick) {
    selection = `ollama:${pick.tag}`
    bootNote = ` (found ${pick.tag} in Ollama)`
  }
} else if ((selection.startsWith('ollama:') || selection.startsWith('lmstudio:')) && !installed(selection)) {
  // The saved local model isn't reachable (server stopped?): fall back to the browser.
  bootNote = ` (${selection.split(':').slice(1).join(':')} is not reachable; using the in-browser model)`
  selection = ''
}

// `ollama:<preset>` embeddings, offered when that model is installed in Ollama.
const embeddingChoice = stored('embedding', 'auto')
const ollamaEmbedding = embeddingChoice.startsWith('ollama:') && installed(embeddingChoice) ? embeddingChoice.slice('ollama:'.length) : undefined

const ai = await createWebEnclave({
  workers,
  dataDir: 'idb://enclave-playground',
  ...(strict ? { selfHost: { baseUrl: '/models' }, privacy: { allow: 'device' as const } } : {}),
  llm: selection.startsWith('browser:') ? selection.slice(8) : (modelFromSelection(selection) ?? 'auto'),
  embedding: ollamaEmbedding ? ollamaEmbedder(ollamaEmbedding) : embeddingChoice.startsWith('ollama:') ? 'auto' : embeddingChoice,
  ...(ollamaEmbedding === 'embeddinggemma' ? { knowledge: { defaultMode: 'vector' as const } } : {}),
  reranker: stored('reranker', 'auto') === 'none' ? false : stored('reranker', 'auto'),
  thinking: thinking.checked,
  // Secondary skills are lazy: fewer visible tools keeps small models focused and faster.
  skills: [knowledgeSkill(), sqlSkill(), { ...memorySkill(), lazy: true }, { ...notesSkill, lazy: true }],
  onProgress,
})

const { device } = ai
$('device').textContent = device.webgpu
  ? `WebGPU ${device.shaderF16 ? '· f16' : '· f32 only'} · ~${(device.gpuBudgetMB / 1024).toFixed(1)} GB budget${
      device.adapter?.vendor ? ` · ${device.adapter.vendor}` : ''
    }`
  : 'No WebGPU · CPU (WASM) only'
$('device').title = JSON.stringify(device, null, 2)
if (!selection) {
  const rec = recommendLLM(device)
  selection = rec ? `browser:${rec.preset.id}` : ''
}
setStatus(`Ready. Model: ${ai.model.id}${bootNote}`)

// ---------------------------------------------------------------------------
// Model picker
// ---------------------------------------------------------------------------

const modelSelect = $<HTMLSelectElement>('model')
let chromeAvailable = false

async function renderModelOptions() {
  const fits = new Set(rankLLMs(device).map((c) => c.preset.id))
  const recommended = recommendLLM(device)?.preset.id
  const status = await ai.modelCache.status()
  const cached = new Map(status.filter((e) => e.kind === 'llm').map((e) => [e.id, e.cached]))
  markRetrievalCache(status)

  const group = (label: string, options: HTMLOptionElement[]) => {
    if (!options.length) return undefined
    const g = document.createElement('optgroup')
    g.label = label
    g.append(...options)
    return g
  }
  const option = (value: string, text: string, disabled = false) => {
    const o = new Option(text, value)
    o.disabled = disabled
    return o
  }
  const browserOption = (p: (typeof BROWSER_LLMS)[number]) => {
    const choice = resolveLLM(p, device)
    const flags = [
      p.id === recommended ? '★' : '',
      cached.get(p.id) ? '✓ cached' : `${(p.downloadMB / 1024).toFixed(1)} GB`,
      `${choice.contextWindow / 1024}k ctx`,
      fits.has(p.id) ? '' : 'too large for this device',
    ].filter(Boolean)
    return option(`browser:${p.id}`, `${p.label} — ${flags.join(' · ')}`, !fits.has(p.id))
  }

  modelSelect.replaceChildren(
    ...[
      group('In browser · WebLLM (WebGPU)', BROWSER_LLMS.filter((p) => p.runtime === 'webllm').map(browserOption)),
      group('In browser · Transformers.js', BROWSER_LLMS.filter((p) => p.runtime === 'transformers').map(browserOption)),
      chromeAvailable ? group('In browser · Chrome built-in', [option('chrome', 'Gemini Nano (Chrome)')]) : undefined,
      group(
        'This computer · Ollama',
        localModels
          .filter((m) => m.provider === 'ollama' && m.kind === 'llm')
          // Measured presets first, in their ranked order.
          .sort((a, b) => rank(a.id) - rank(b.id))
          .map((m) => {
            const preset = OLLAMA_LLM_PRESETS.find((p) => p.tag === m.id)
            const flags = [
              preset ? `★ ${(preset.passRate * 100).toFixed(1)}% on evals · ~${preset.p50Seconds}s/answer` : '',
              m.parameterSize ?? '',
              m.capabilities && !m.capabilities.includes('tools') ? 'no tool support' : '',
            ].filter(Boolean)
            return option(`ollama:${m.id}`, `${m.id}${flags.length ? ` — ${flags.join(' · ')}` : ''}`)
          }),
      ),
      group(
        'This computer · LM Studio',
        localModels
          .filter((m) => m.provider === 'lmstudio' && m.kind === 'llm')
          .map((m) => option(`lmstudio:${m.id}`, `${m.id}${m.loaded ? ' · loaded' : ''}`)),
      ),
    ].filter((g): g is HTMLOptGroupElement => !!g),
  )
  modelSelect.value = selection
  describeSelection()
}

const rank = (tag: string) => {
  const i = OLLAMA_LLM_PRESETS.findIndex((p) => p.tag === tag)
  return i < 0 ? OLLAMA_LLM_PRESETS.length : i
}

function describeSelection() {
  const info = $('model-info')
  const forget = $<HTMLButtonElement>('forget')
  const load = $<HTMLButtonElement>('load')
  if (selection.startsWith('browser:')) {
    const preset = BROWSER_LLMS.find((p) => p.id === selection.slice(8))
    if (!preset) return
    const choice = resolveLLM(preset, device)
    info.textContent = `${preset.params} params · ~${(choice.estimatedMB / 1024).toFixed(1)} GB GPU memory · ${choice.modelId}${
      preset.notes ? ` · ${preset.notes}` : ''
    }`
    forget.hidden = false
    load.hidden = false
  } else if (selection === 'chrome') {
    info.textContent = 'On-device Gemini Nano managed by Chrome.'
    forget.hidden = true
    load.hidden = true
  } else {
    const [provider, ...rest] = selection.split(':')
    const m = localModels.find((x) => x.provider === provider && x.id === rest.join(':'))
    const preset = OLLAMA_LLM_PRESETS.find((p) => p.tag === rest.join(':'))
    info.textContent = `Served by ${provider === 'ollama' ? 'Ollama' : 'LM Studio'} on this computer${
      m?.contextLength ? ` · ${m.contextLength / 1024}k max ctx` : ''
    }${m?.quantization ? ` · ${m.quantization}` : ''}${preset?.note ? ` · ${preset.note}` : ''}`
    forget.hidden = true
    load.hidden = true
  }
}

async function applySelection() {
  store('model', selection)
  describeSelection()
  const model = selection.startsWith('browser:')
    ? await browserLLM(selection.slice(8), { workers, device, thinking: thinking.checked ? 'auto' : false, onProgress })
    : modelFromSelection(selection, localModels)
  if (model) {
    ai.setModel(model)
    setStatus(`Model: ${model.id}${'load' in model ? ' (downloads on first message)' : ''}`)
  }
}

modelSelect.onchange = () => {
  selection = modelSelect.value
  void applySelection()
}
thinking.onchange = () => {
  store('thinking', String(thinking.checked))
  void applySelection()
}
$('load').onclick = async () => {
  const model = ai.model as Model & { load?(): Promise<void> }
  try {
    await model.load?.()
    setStatus(`${model.id} loaded`)
    $('progress').hidden = true
    await renderModelOptions()
  } catch (error) {
    setStatus(`Load failed: ${error instanceof Error ? error.message : String(error)}`)
  }
}
$('forget').onclick = async () => {
  if (!selection.startsWith('browser:')) return
  const preset = BROWSER_LLMS.find((p) => p.id === selection.slice(8))!
  await ai.modelCache.clear('llm', preset.id)
  setStatus(`Deleted cached weights for ${preset.label}`)
  await renderModelOptions()
}
$('discover').onclick = async () => {
  setStatus('Looking for Ollama and LM Studio…')
  localModels = await discoverLocalModels()
  setStatus(
    localModels.length
      ? `Found ${localModels.filter((m) => m.kind === 'llm').length} local models`
      : 'No local servers found (Ollama on :11434, LM Studio on :1234 with CORS enabled)',
  )
  await renderModelOptions()
}

// ---------------------------------------------------------------------------
// Retrieval settings (applied on reload; documents are re-embedded automatically)
// ---------------------------------------------------------------------------

const embeddingSelect = $<HTMLSelectElement>('embedding')
embeddingSelect.append(
  new Option(`Auto (${'label' in ai.plan.embedding ? ai.plan.embedding.label : ai.plan.embedding.id})`, 'auto'),
  ...EMBEDDING_PRESETS.map((p) => new Option(`${p.label} · ${p.dimensions}d · ${p.downloadMB} MB`, p.id)),
  // Ollama embedders with measured presets, when installed.
  ...Object.keys(OLLAMA_EMBEDDING_PRESETS)
    .filter((model) => installed(`ollama:${model}`))
    .map((model) => new Option(`${model} (Ollama)`, `ollama:${model}`)),
)
embeddingSelect.value = stored('embedding', 'auto')
embeddingSelect.onchange = () => {
  store('embedding', embeddingSelect.value)
  location.reload()
}

/** Append ✓ cached to embedding/reranker options that are fully downloaded. */
function markRetrievalCache(status: Awaited<ReturnType<typeof ai.modelCache.status>>) {
  for (const [select, kind] of [[embeddingSelect, 'embedding'], [rerankerSelect, 'reranker']] as const) {
    for (const option of select.options) {
      const entry = status.find((e) => e.kind === kind && e.id === option.value)
      option.textContent = option.textContent!.replace(/ · ✓ cached$/, '') + (entry?.cached ? ' · ✓ cached' : '')
    }
  }
}

const rerankerSelect = $<HTMLSelectElement>('reranker')
rerankerSelect.append(
  new Option('Auto', 'auto'),
  new Option('None', 'none'),
  ...RERANKER_PRESETS.map((p) => new Option(`${p.label} · ${p.downloadMB} MB`, p.id)),
)
rerankerSelect.value = stored('reranker', 'auto')
rerankerSelect.onchange = () => {
  store('reranker', rerankerSelect.value)
  location.reload()
}

// ---------------------------------------------------------------------------
// Knowledge ingestion
// ---------------------------------------------------------------------------

async function refreshCollections() {
  $('collections').replaceChildren(
    ...(await ai.knowledge!.collections()).map((c) => {
      const li = document.createElement('li')
      li.textContent = `${c.collection}: ${c.documents} docs · ${c.chunks} chunks`
      return li
    }),
  )
}

// OCR for images and scanned PDFs, when `pnpm mirror` has put tesseract.js under /models/ocr.
const ocr = (await fetch('/models/ocr/worker.min.js', { method: 'HEAD' }).then((r) => r.ok && !/html/.test(r.headers.get('content-type') ?? ''), () => false))
  ? tesseractOcr({ lib: () => import('tesseract.js'), baseUrl: '/models/ocr', onProgress: ({ status, progress }) => setStatus(`OCR: ${status} ${Math.round(progress * 100)}%`) })
  : undefined

async function ingestFiles(files: FileList | File[]) {
  if (!files.length) return
  const started = performance.now()
  try {
    // PDF, Word, PowerPoint, Excel, EPUB, HTML, CSV, JSON and text, all read in this tab.
    const loaded = await loadFiles(files, {
      collection: 'files',
      pdf: { lib: () => import('pdfjs-dist'), workerSrc: pdfWorkerSrc },
      ...(ocr ? { ocr } : {}),
      onProgress: ({ done, total, name }) => setStatus(`Reading ${name} (${done}/${total})…`),
    })
    const result = await ai.knowledge!.ingest(loaded.documents, {
      onProgress: ({ done, total }) => setStatus(`Indexing ${done}/${total}…`),
    })
    // Spreadsheets and CSVs also become SQL tables, so the agent can total and filter them.
    const imported: string[] = []
    for (const table of loaded.tables) {
      const { table: name, rows } = await importTable(ai.db, table, { ifExists: 'replace' })
      imported.push(`${name} (${rows} rows)`)
    }
    const notes = [
      ...loaded.warnings,
      ...loaded.errors.map((e) => e.error.message),
      ...(imported.length ? [`Tables: ${imported.join(', ')}`] : []),
    ]
    setStatus(
      `Indexed ${result.documents} docs (${result.chunks} chunks, ${result.skipped} unchanged) in ${Math.round(performance.now() - started)} ms` +
        (notes.length ? `. ${notes.join(' ')}` : ''),
    )
  } catch (error) {
    setStatus(`Indexing failed: ${error instanceof Error ? error.message : String(error)}`)
  }
  await refreshCollections()
}

$<HTMLInputElement>('files').accept = ACCEPT
$<HTMLInputElement>('files').onchange = (e) => ingestFiles((e.target as HTMLInputElement).files!)
const drop = $('drop')
drop.ondragover = (e) => (e.preventDefault(), drop.classList.add('over'))
drop.ondragleave = () => drop.classList.remove('over')
drop.ondrop = (e) => {
  e.preventDefault()
  drop.classList.remove('over')
  if (e.dataTransfer?.files.length) void ingestFiles(e.dataTransfer.files)
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

const log = $('log')
let thread: Thread = ai.thread(stored('thread') || undefined)
store('thread', thread.id)

function bubble(role: string, text = ''): HTMLElement {
  const el = document.createElement('div')
  el.className = `msg ${role}`
  el.textContent = text
  log.append(el)
  log.scrollTop = log.scrollHeight
  return el
}

function toolCard(call: ToolCall): HTMLDetailsElement {
  const el = document.createElement('details')
  el.className = 'tool'
  const summary = document.createElement('summary')
  summary.textContent = `⚙ ${call.name}`
  const pre = document.createElement('pre')
  pre.textContent = JSON.stringify(call.input, null, 2)
  el.append(summary, pre)
  log.append(el)
  return el
}

function askApproval(call: ToolCall): Promise<boolean> {
  return new Promise((resolve) => {
    const box = bubble('approval', `Allow ${call.name}?`)
    const pre = document.createElement('pre')
    pre.textContent = JSON.stringify(call.input, null, 2)
    const yes = document.createElement('button')
    yes.textContent = 'Allow'
    const no = document.createElement('button')
    no.textContent = 'Deny'
    const decide = (ok: boolean) => {
      box.replaceChildren(`${ok ? '✓ Allowed' : '✗ Denied'} ${call.name}`)
      resolve(ok)
    }
    yes.onclick = () => decide(true)
    no.onclick = () => decide(false)
    box.append(pre, yes, no)
  })
}

async function renderHistory() {
  log.replaceChildren()
  for (const m of await thread.messages()) {
    if (m.role === 'user') bubble('user', m.content)
    else if (m.role === 'assistant') {
      if (m.content) bubble('assistant', m.content)
      m.toolCalls?.forEach(toolCard)
    }
  }
}

async function refreshThreads() {
  $('threads').replaceChildren(
    ...(await ai.threads()).map((t) => {
      const li = document.createElement('li')
      li.textContent = t.title ?? t.id
      li.className = t.id === thread.id ? 'active' : ''
      li.onclick = async () => {
        thread = ai.thread(t.id)
        store('thread', t.id)
        await renderHistory()
        await refreshThreads()
      }
      return li
    }),
  )
}

$('new-thread').onclick = async () => {
  thread = ai.thread()
  store('thread', thread.id)
  log.replaceChildren()
  await refreshThreads()
}

let controller: AbortController | undefined
$('stop').onclick = () => controller?.abort()

$<HTMLFormElement>('composer').onsubmit = async (e) => {
  e.preventDefault()
  const input = $<HTMLTextAreaElement>('input')
  const text = input.value.trim()
  if (!text || controller) return
  input.value = ''
  bubble('user', text)
  controller = new AbortController()
  $('stop').hidden = false
  $('send').hidden = true

  let current: HTMLElement | undefined
  let thoughts: HTMLDetailsElement | undefined
  const cards = new Map<string, HTMLDetailsElement>()
  const started = performance.now()
  let firstToken: number | undefined
  let chars = 0
  try {
    for await (const event of thread.send(text, { signal: controller.signal, onApproval: askApproval })) {
      switch (event.type) {
        case 'step-start':
          current = undefined
          thoughts = undefined
          break
        case 'reasoning-delta':
          firstToken ??= performance.now()
          chars += event.delta.length
          if (!thoughts) {
            thoughts = document.createElement('details')
            thoughts.className = 'thinking'
            thoughts.append(
              Object.assign(document.createElement('summary'), { textContent: 'thinking' }),
              document.createElement('pre'),
            )
            log.append(thoughts)
          }
          thoughts.querySelector('pre')!.textContent += event.delta
          break
        case 'text-delta':
          firstToken ??= performance.now()
          chars += event.delta.length
          current ??= bubble('assistant')
          current.textContent += event.delta
          log.scrollTop = log.scrollHeight
          break
        case 'tool-call':
          cards.set(event.call.id, toolCard(event.call))
          break
        case 'tool-result': {
          const card = cards.get(event.call.id)
          const pre = document.createElement('pre')
          pre.className = event.isError ? 'error' : 'result'
          pre.textContent = JSON.stringify(event.output, null, 2).slice(0, 4000)
          card?.append(pre)
          card?.querySelector('summary')?.append(` · ${Math.round(event.durationMs)} ms${event.isError ? ' · error' : ''}`)
          break
        }
        case 'custom':
          bubble('custom', `${event.tool}: ${JSON.stringify(event.data)}`)
          break
        case 'finish': {
          // Counts thinking tokens too; ~4 chars per token.
          const ttft = firstToken ? `${((firstToken - started) / 1000).toFixed(1)}s to first token · ` : ''
          const secs = (performance.now() - (firstToken ?? started)) / 1000
          setStatus(`${ttft}~${Math.round(chars / 4 / Math.max(secs, 0.01))} tok/s · ${event.steps} steps · ${event.reason}`)
        }
      }
    }
    const info = (await ai.threads()).find((t) => t.id === thread.id)
    if (info && !info.title) await thread.rename(text.slice(0, 48))
  } catch (error) {
    bubble('error', error instanceof Error ? error.message : String(error))
  } finally {
    controller = undefined
    $('stop').hidden = true
    $('send').hidden = false
    $('progress').hidden = true
    await refreshThreads()
    await refreshCollections()
  }
}

$<HTMLTextAreaElement>('input').onkeydown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    $<HTMLFormElement>('composer').requestSubmit()
  }
}

chromeAvailable = await chromeAIAvailable().catch(() => false)
await Promise.all([renderModelOptions(), renderHistory(), refreshThreads(), refreshCollections()])
document.body.dataset.ready = 'true'

// Dev-only handle for debugging from the console or automated tests.
if (import.meta.env.DEV) Object.assign(globalThis, { __enclave: ai, __browserLLM: browserLLM, __workers: workers, __device: device })
