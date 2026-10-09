#!/usr/bin/env node
/**
 * Mirror model files to your own static host so the app never contacts a
 * third party at runtime. Produces the layout `selfHost: { baseUrl }` expects:
 *
 *   <out>/mlc/<model-id>/resolve/main/*      WebLLM weights
 *   <out>/mlc/libs/*.wasm                     WebLLM compiled model libraries
 *   <out>/hf/<repo>/resolve/main/*            Transformers.js models
 *   <out>/ort/*                               ONNX Runtime WASM
 *   <out>/ocr/{worker.min.js,core/,lang/}     tesseract.js OCR (tesseractOcr({ baseUrl: '<out URL>/ocr' }))
 *
 * Usage:
 *   enclave-mirror --out public/models \
 *     --webllm Qwen3-4B-q4f16_1-MLC \
 *     --hf onnx-community/embeddinggemma-300m-ONNX:q4,q8 \
 *     --hf mixedbread-ai/mxbai-rerank-xsmall-v1:fp32,q8 \
 *     --ort \
 *     --ocr eng,deu
 *
 * Presets from the catalog work too: --webllm qwen3-4b --embedding embeddinggemma --reranker mxbai-rerank-xsmall
 * --transcriber whisper-base (the Whisper files every backend loads: fp32 encoder, q4 + q8 decoders)
 */
import { createWriteStream } from 'node:fs'
import { copyFile, mkdir, readdir, stat } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { parseArgs } from 'node:util'

const { values } = parseArgs({
  options: {
    out: { type: 'string', default: 'public/models' },
    webllm: { type: 'string', multiple: true, default: [] },
    hf: { type: 'string', multiple: true, default: [] },
    embedding: { type: 'string', multiple: true, default: [] },
    reranker: { type: 'string', multiple: true, default: [] },
    transcriber: { type: 'string', multiple: true, default: [] },
    ort: { type: 'boolean', default: false },
    ocr: { type: 'string', multiple: true, default: [] },
    'f32': { type: 'boolean', default: false, description: 'Also mirror q4f32 WebLLM builds (GPUs without shader-f16)' },
    help: { type: 'boolean', default: false },
  },
})

if (values.help || (!values.webllm.length && !values.hf.length && !values.embedding.length && !values.reranker.length && !values.transcriber.length && !values.ort && !values.ocr.length)) {
  console.log((await import('node:fs')).readFileSync(new URL(import.meta.url), 'utf8').split('\n').slice(2, 22).join('\n').replace(/^ \* ?/gm, ''))
  process.exit(values.help ? 0 : 1)
}

const require = createRequire(import.meta.url)
const out = values.out
const HF = 'https://huggingface.co'

async function exists(path, size) {
  try {
    const s = await stat(path)
    return size === undefined || s.size === size
  } catch {
    return false
  }
}

async function download(url, path, size) {
  if (await exists(path, size)) return console.log(`  = ${path}`)
  await mkdir(dirname(path), { recursive: true })
  const response = await fetch(url, { redirect: 'follow' })
  if (!response.ok || !response.body) throw new Error(`${response.status} ${url}`)
  await pipeline(Readable.fromWeb(response.body), createWriteStream(path))
  console.log(`  ↓ ${path}${size ? ` (${(size / 2 ** 20).toFixed(1)} MB)` : ''}`)
}

async function repoFiles(repo) {
  const response = await fetch(`${HF}/api/models/${repo}?blobs=true`)
  if (!response.ok) throw new Error(`Unknown repo ${repo} (${response.status})`)
  const info = await response.json()
  return info.siblings.map((s) => ({ name: s.rfilename, size: s.size }))
}

// dtype → ONNX file suffix used by Transformers.js
const SUFFIX = { fp32: '', fp16: '_fp16', q8: '_quantized', int8: '_int8', uint8: '_uint8', q4: '_q4', q4f16: '_q4f16', bnb4: '_bnb4' }

function suffixOf(dtype) {
  if (!(dtype in SUFFIX)) throw new Error(`Unknown dtype ${dtype}; use one of ${Object.keys(SUFFIX).join(', ')}`)
  return SUFFIX[dtype]
}

/**
 * `modules` (optional) limits ONNX files to these base names and dtypes, for
 * presets with a dtype per file (Whisper: { encoder_model: ['fp32'], decoder_model_merged: ['q4', 'q8'] }).
 */
async function mirrorTransformers(repo, dtypes, modules) {
  console.log(`hf ${repo} [${modules ? Object.entries(modules).map(([m, d]) => `${m}: ${d.join(', ')}`).join('; ') : dtypes.join(', ')}]`)
  const wanted = dtypes.map(suffixOf)
  const perModule = modules && Object.fromEntries(Object.entries(modules).map(([m, d]) => [m, d.map(suffixOf)]))
  for (const file of await repoFiles(repo)) {
    const onnx = file.name.match(/^onnx\/(.+?)(_fp16|_quantized|_int8|_uint8|_q4f16|_q4|_bnb4)?\.onnx(_data(_\d+)?)?$/)
    if (onnx) {
      // Alternate exports (e.g. model_no_gather_q4) are never loaded by Transformers.js.
      if (onnx[1].includes('no_gather')) continue
      if (perModule ? !perModule[onnx[1]]?.includes(onnx[2] ?? '') : !wanted.includes(onnx[2] ?? '')) continue
    } else if (!/\.(json|txt|model|tiktoken|jinja)$/.test(file.name) || file.name.includes('/')) {
      continue
    }
    await download(`${HF}/${repo}/resolve/main/${file.name}`, join(out, 'hf', repo, 'resolve', 'main', file.name), file.size)
  }
}

async function mirrorWebLLM(ids) {
  const { prebuiltAppConfig } = await import('@mlc-ai/web-llm')
  const catalog = await import('../dist/web/catalog.js').catch(() => undefined)
  for (let id of ids) {
    const preset = catalog?.findLLM(id)
    const builds = preset
      ? [`${preset.model}-q4f16_1-MLC`, ...(values.f32 ? [`${preset.model}-q4f32_1-MLC`] : [])]
      : [id]
    for (id of builds) {
      const record = prebuiltAppConfig.model_list.find((r) => r.model_id === id)
      if (!record) throw new Error(`Unknown WebLLM model ${id}`)
      console.log(`webllm ${id}`)
      const repo = new URL(record.model).pathname.replace(/^\/|\/$/g, '')
      for (const file of await repoFiles(repo)) {
        if (/^(README|\.gitattributes)/.test(file.name)) continue
        await download(`${HF}/${repo}/resolve/main/${file.name}`, join(out, 'mlc', id, 'resolve', 'main', file.name), file.size)
      }
      await download(record.model_lib, join(out, 'mlc', 'libs', record.model_lib.split('/').pop()))
    }
  }
}

function packageRoot(entry, name) {
  let dir = dirname(entry)
  while (dir !== dirname(dir)) {
    try {
      if (require(join(dir, 'package.json')).name === name) return dir
    } catch {
      /* keep walking */
    }
    dir = dirname(dir)
  }
  throw new Error(`Could not locate ${name}`)
}

async function mirrorOrt() {
  console.log('onnxruntime-web wasm')
  // Use the exact onnxruntime-web build that Transformers.js depends on.
  const tjs = packageRoot(require.resolve('@huggingface/transformers'), '@huggingface/transformers')
  const ort = packageRoot(createRequire(join(tjs, 'package.json')).resolve('onnxruntime-web'), 'onnxruntime-web')
  const ortDist = join(ort, 'dist')
  await mkdir(join(out, 'ort'), { recursive: true })
  for (const f of await readdir(ortDist)) {
    if (!/^ort-wasm-simd-threaded.*\.(wasm|mjs)$/.test(f)) continue
    await copyFile(join(ortDist, f), join(out, 'ort', f))
    console.log(`  ⧉ ${join(out, 'ort', f)}`)
  }
}

/**
 * tesseract.js from the app's own install (so the worker matches the version it
 * imports), its matching WASM core (LSTM engine builds), and gzipped language data.
 */
async function mirrorOcr(languages) {
  console.log(`tesseract.js OCR [${languages.join(', ')}]`)
  let entry
  for (const from of [join(process.cwd(), 'package.json'), import.meta.url]) {
    try {
      entry = createRequire(from).resolve('tesseract.js')
      break
    } catch {
      /* try the next location */
    }
  }
  if (!entry) throw new Error('tesseract.js is not installed. Add it to your app: pnpm add tesseract.js')
  const tesseract = packageRoot(entry, 'tesseract.js')
  const core = packageRoot(createRequire(join(tesseract, 'package.json')).resolve('tesseract.js-core'), 'tesseract.js-core')
  await mkdir(join(out, 'ocr', 'core'), { recursive: true })
  await copyFile(join(tesseract, 'dist', 'worker.min.js'), join(out, 'ocr', 'worker.min.js'))
  console.log(`  ⧉ ${join(out, 'ocr', 'worker.min.js')}`)
  for (const f of await readdir(core)) {
    if (!/^tesseract-core(-simd|-relaxedsimd)?-lstm\.wasm\.js$/.test(f)) continue
    await copyFile(join(core, f), join(out, 'ocr', 'core', f))
    console.log(`  ⧉ ${join(out, 'ocr', 'core', f)}`)
  }
  for (const lang of languages) {
    await download(
      `https://cdn.jsdelivr.net/npm/@tesseract.js-data/${lang}/4.0.0_best_int/${lang}.traineddata.gz`,
      join(out, 'ocr', 'lang', `${lang}.traineddata.gz`),
    )
  }
}

const catalog = await import('../dist/web/catalog.js').catch(() => undefined)
// Text-only presets of multimodal repos (EmbeddingGemma 2): only the text model's ONNX files, never the vision/audio encoders.
const textOnly = new Map()
const presetRepos = [
  ...values.embedding.map((id) => {
    const p = catalog?.findEmbedding(id)
    if (!p) throw new Error(`Unknown embedding preset ${id} (build the package first)`)
    if (p.textOnly) textOnly.set(p.model, [...new Set(Object.values(p.dtype))])
    return `${p.model}:${[...new Set(Object.values(p.dtype))].join(',')}`
  }),
  ...values.reranker.map((id) => {
    const p = catalog?.findReranker(id)
    if (!p) throw new Error(`Unknown reranker preset ${id} (build the package first)`)
    return `${p.model}:${[...new Set(Object.values(p.dtype))].join(',')}`
  }),
]

for (const spec of [...values.hf, ...presetRepos]) {
  const [repo, dtypes = 'q8'] = spec.split(':')
  await mirrorTransformers(repo, dtypes.split(','), textOnly.has(repo) ? { model: textOnly.get(repo) } : undefined)
}
for (const id of values.transcriber) {
  const p = catalog?.findTranscriber?.(id)
  if (!p) throw new Error(`Unknown transcriber preset ${id} (build the package first)`)
  // Each backend's dtype is one string or one per ONNX file; mirror every file any backend loads.
  const modules = {}
  for (const dtype of Object.values(p.dtype)) {
    for (const [module, d] of Object.entries(typeof dtype === 'string' ? { encoder_model: dtype, decoder_model_merged: dtype } : dtype)) {
      modules[module] = [...new Set([...(modules[module] ?? []), d])]
    }
  }
  await mirrorTransformers(p.model, [], modules)
}
if (values.webllm.length) await mirrorWebLLM(values.webllm)
if (values.ort) await mirrorOrt()
if (values.ocr.length) await mirrorOcr(values.ocr.flatMap((v) => v.split(',')).filter(Boolean))
console.log(`\nDone. Serve ${out} and pass selfHost: { baseUrl: '<its URL>' } to createWebEnclave().`)
