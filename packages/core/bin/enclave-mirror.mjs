#!/usr/bin/env node
/**
 * Mirror model files to your own static host so the app never contacts a
 * third party at runtime. Produces the layout `selfHost: { baseUrl }` expects:
 *
 *   <out>/mlc/<model-id>/resolve/main/*      WebLLM weights
 *   <out>/mlc/libs/*.wasm                     WebLLM compiled model libraries
 *   <out>/hf/<repo>/resolve/main/*            Transformers.js models
 *   <out>/ort/*                               ONNX Runtime WASM
 *
 * Usage:
 *   enclave-mirror --out public/models \
 *     --webllm Qwen3-4B-q4f16_1-MLC \
 *     --hf onnx-community/embeddinggemma-300m-ONNX:q4,q8 \
 *     --hf mixedbread-ai/mxbai-rerank-xsmall-v1:fp32,q8 \
 *     --ort
 *
 * Presets from the catalog work too: --webllm qwen3-4b --embedding embeddinggemma --reranker mxbai-rerank-xsmall
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
    ort: { type: 'boolean', default: false },
    'f32': { type: 'boolean', default: false, description: 'Also mirror q4f32 WebLLM builds (GPUs without shader-f16)' },
    help: { type: 'boolean', default: false },
  },
})

if (values.help || (!values.webllm.length && !values.hf.length && !values.embedding.length && !values.reranker.length && !values.ort)) {
  console.log((await import('node:fs')).readFileSync(new URL(import.meta.url), 'utf8').split('\n').slice(2, 20).join('\n').replace(/^ \* ?/gm, ''))
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

async function mirrorTransformers(repo, dtypes) {
  console.log(`hf ${repo} [${dtypes.join(', ')}]`)
  const wanted = dtypes.map((d) => {
    if (!(d in SUFFIX)) throw new Error(`Unknown dtype ${d}; use one of ${Object.keys(SUFFIX).join(', ')}`)
    return SUFFIX[d]
  })
  for (const file of await repoFiles(repo)) {
    const onnx = file.name.match(/^onnx\/(.+?)(_fp16|_quantized|_int8|_uint8|_q4f16|_q4|_bnb4)?\.onnx(_data(_\d+)?)?$/)
    if (onnx) {
      // Alternate exports (e.g. model_no_gather_q4) are never loaded by Transformers.js.
      if (onnx[1].includes('no_gather') || !wanted.includes(onnx[2] ?? '')) continue
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

const catalog = await import('../dist/web/catalog.js').catch(() => undefined)
const presetRepos = [
  ...values.embedding.map((id) => {
    const p = catalog?.findEmbedding(id)
    if (!p) throw new Error(`Unknown embedding preset ${id} (build the package first)`)
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
  await mirrorTransformers(repo, dtypes.split(','))
}
if (values.webllm.length) await mirrorWebLLM(values.webllm)
if (values.ort) await mirrorOrt()
console.log(`\nDone. Serve ${out} and pass selfHost: { baseUrl: '<its URL>' } to createWebEnclave().`)
