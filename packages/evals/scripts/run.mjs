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
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'
import { formatReport } from '@enclave/core/eval'

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

for (const qs of configs) {
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
  // Stream progress lines as they appear.
  const timer = setInterval(async () => {
    const text = await page.textContent('#status').catch(() => '')
    // Print every completed run once (failures include the model's answer).
    const lines = (text ?? '').split('\n').filter((l) => /^[✓✗]/.test(l))
    for (const line of lines.slice(printed)) console.log(`  ${line}`)
    printed = lines.length
    // Live progress for `pnpm --filter @enclave/evals status`.
    const plan = await page.evaluate(() => globalThis.__evalPlan ?? null).catch(() => null)
    const phase = (text ?? '').split('\n')[0] ?? ''
    writeFileSync(
      progressFile,
      JSON.stringify({ config: qs || '(defaults)', configIndex: configs.indexOf(qs) + 1, configCount: configs.length, startedAt: new Date(started).toISOString(), updatedAt: new Date().toISOString(), plan, phase: plan ? 'running' : phase, completed: lines }, null, 2),
    )
  }, 5000)
  try {
    await page.waitForFunction(() => globalThis.__evalReport || globalThis.__evalError, null, { timeout: 6 * 3600_000, polling: 2000 })
    const error = await page.evaluate(() => globalThis.__evalError)
    if (error) throw new Error(error)
    const report = await page.evaluate(() => globalThis.__evalReport)
    const file = join(root, 'reports', `${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}-${report.label.replace(/[^\w.=-]+/g, '_')}.json`)
    writeFileSync(file, JSON.stringify({ ...report, commit, wallClockSec: Math.round((Date.now() - started) / 1000) }, null, 2))
    console.log(`\n${formatReport(report)}\n  → ${file}`)
  } catch (error) {
    failed = true
    console.error(`\n✗ ${qs || '(defaults)'}: ${error instanceof Error ? error.message : String(error)}`)
  } finally {
    clearInterval(timer)
    await page.close()
  }
}
await context.close()
server.kill()
process.exit(failed ? 1 : 0)
