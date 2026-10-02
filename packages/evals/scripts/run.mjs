#!/usr/bin/env node
/**
 * Run the production suites on real WebGPU in headless Chrome.
 *
 *   pnpm --filter @enclave/evals eval                         # defaults: qwen3-4b, 3 repeats, all suites
 *   pnpm --filter @enclave/evals eval "repeats=1&tags=rag"   # one or more query-string configs
 *
 * Env: CHROME_PATH (default: macOS Chrome), EVAL_PROFILE (browser profile dir, keeps model caches).
 * Writes reports/<date>-<label>.json and prints a summary per config.
 */
import { execSync, spawn } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'
import { formatReport } from 'enclave-ai/eval'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const configs = process.argv.slice(2).length ? process.argv.slice(2) : ['']
const port = 5299
const chrome = process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const profile = process.env.EVAL_PROFILE ?? join(root, '.cache', 'chrome-profile')
const commit = (() => {
  try {
    return execSync('git rev-parse --short HEAD', { cwd: root }).toString().trim()
  } catch {
    return 'unknown'
  }
})()

// Serve a production build: closer to how apps ship, and immune to source
// edits while a long run is in progress (no hot reload).
const outDir = join(root, '.cache', 'runner-dist')
execSync(`npx vite build runner --config runner/vite.config.ts --outDir ${JSON.stringify(outDir)} --emptyOutDir --logLevel error`, { cwd: root, stdio: 'inherit' })
const server = spawn('npx', ['vite', 'preview', 'runner', '--config', 'runner/vite.config.ts', '--outDir', outDir, '--port', String(port), '--strictPort', '--host', '127.0.0.1'], { cwd: root, stdio: 'ignore' })
process.on('exit', () => server.kill())
for (let i = 0; i < 60; i++) {
  try {
    if ((await fetch(`http://127.0.0.1:${port}/`)).ok) break
  } catch {}
  await new Promise((r) => setTimeout(r, 500))
}

mkdirSync(join(root, 'reports'), { recursive: true })
mkdirSync(profile, { recursive: true })
const context = await chromium.launchPersistentContext(profile, { executablePath: chrome, headless: true, args: ['--enable-unsafe-webgpu'] })
let failed = false

// A run that shows no progress for this long is restarted (each turn is
// capped at 4 minutes, and the longest case has 4 turns: 30 minutes of silence means something is stuck).
const STALL_MS = Number(process.env.EVAL_STALL_MIN ?? 30) * 60_000
const ATTEMPTS = 3

/**
 * `failedIn=<report>` → run only the cases that failed in that report;
 * `firstFrom=<report>` → run those cases first. Reports are read from disk
 * here and passed to the page as case names.
 */
function expandReportParams(qs) {
  const params = new URLSearchParams(qs)
  const failedNames = (file) => {
    const report = JSON.parse(readFileSync(file.startsWith('/') ? file : join(root, file), 'utf8'))
    return [...new Set(report.results.filter((r) => !r.passed).map((r) => r.name))]
  }
  if (params.has('failedIn')) {
    params.set('cases', failedNames(params.get('failedIn')).join('|'))
    params.delete('failedIn')
  }
  if (params.has('firstFrom')) {
    params.set('first', failedNames(params.get('firstFrom')).join('|'))
    params.delete('firstFrom')
  }
  return params.toString()
}
configs.splice(0, configs.length, ...configs.map(expandReportParams))

for (const qs of configs) {
  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    const outcome = await runConfig(qs)
    if (outcome === 'done') break
    if (outcome === 'stalled' && attempt < ATTEMPTS) {
      console.error(`  ↻ restarting ${qs || '(defaults)'} (attempt ${attempt + 1} of ${ATTEMPTS})`)
      continue
    }
    failed = true
    break
  }
}

/** Run one config in a fresh page: 'done', 'stalled' or 'failed'. */
async function runConfig(qs) {
  const page = await context.newPage()
  let printed = 0
  page.on('console', (m) => { if (m.type() === 'error') console.error('  [page]', m.text().slice(0, 200)) })
  // A crash loop (e.g. a worker that can't load its WASM) should fail fast, not hang.
  let pageErrors = 0
  page.on('pageerror', (e) => {
    if (++pageErrors === 1 || pageErrors % 50 === 0) console.error('  [pageerror]', e.message.slice(0, 200))
    if (pageErrors > 200) page.evaluate((m) => (globalThis.__evalError = `crash loop: ${m}`), e.message).catch(() => {})
  })
  const started = Date.now()
  await page.goto(`http://127.0.0.1:${port}/?${qs}`)
  const progressFile = join(root, 'reports', 'progress.json')
  let lastSignature = ''
  let lastChange = Date.now()
  let onStall
  const stalled = new Promise((resolve) => (onStall = resolve))
  // Stream progress lines as they appear, and watch for stalls.
  const timer = setInterval(async () => {
    const text = await page.textContent('#status', { timeout: 10_000 }).catch(() => '')
    // Print every completed run once (failures include the model's answer).
    const lines = (text ?? '').split('\n').filter((l) => /^[✓✗]/.test(l))
    for (const line of lines.slice(printed)) console.log(`  ${line}`)
    printed = Math.max(printed, lines.length)
    // Live progress for `pnpm --filter @enclave/evals status`.
    const plan = await page.evaluate(() => globalThis.__evalPlan ?? null).catch(() => null)
    const phase = (text ?? '').split('\n')[0] ?? ''
    writeFileSync(
      progressFile,
      JSON.stringify({ config: qs || '(defaults)', configIndex: configs.indexOf(qs) + 1, configCount: configs.length, startedAt: new Date(started).toISOString(), updatedAt: new Date().toISOString(), plan, phase: plan ? 'running' : phase, completed: lines }, null, 2),
    )
    // Any streamed token, phase change or finished run counts as progress.
    const signature = `${text?.length ?? 0}:${phase}`
    if (signature !== lastSignature) {
      lastSignature = signature
      lastChange = Date.now()
    } else if (Date.now() - lastChange > STALL_MS) {
      onStall(`no progress for ${Math.round(STALL_MS / 60_000)} min (at: ${phase || 'start'})`)
    }
  }, 5000)
  try {
    const finished = page.waitForFunction(() => globalThis.__evalReport || globalThis.__evalError, null, { timeout: 0, polling: 2000 }).then(() => null)
    const stall = await Promise.race([finished, stalled])
    if (stall) {
      console.error(`\n✗ ${qs || '(defaults)'}: ${stall}`)
      return 'stalled'
    }
    const error = await page.evaluate(() => globalThis.__evalError)
    if (error) throw new Error(error)
    const report = await page.evaluate(() => globalThis.__evalReport)
    const file = join(root, 'reports', `${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}-${report.label.replace(/[^\w.=-]+/g, '_')}.json`)
    writeFileSync(file, JSON.stringify({ ...report, commit, wallClockSec: Math.round((Date.now() - started) / 1000) }, null, 2))
    console.log(`\n${formatReport(report)}\n  → ${file}`)
    return 'done'
  } catch (error) {
    console.error(`\n✗ ${qs || '(defaults)'}: ${error instanceof Error ? error.message : String(error)}`)
    return 'failed'
  } finally {
    clearInterval(timer)
    await page.close().catch(() => {})
  }
}
await context.close()
server.kill()
process.exit(failed ? 1 : 0)
