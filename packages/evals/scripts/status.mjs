#!/usr/bin/env node
/** Summarize a running (or finished) eval: pnpm --filter @enclave/evals status */
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const file = join(dirname(fileURLToPath(import.meta.url)), '..', 'reports', 'progress.json')
if (!existsSync(file)) {
  console.log('No eval has reported progress yet (reports/progress.json is missing).')
  process.exit(0)
}
const p = JSON.parse(readFileSync(file, 'utf8'))
const staleSec = (Date.now() - new Date(p.updatedAt).getTime()) / 1000
const runs = p.completed.map((line) => {
  const m = line.match(/^([✓✗]) (.+?) #(\d+) ([\d.]+)s(?: — (.*))?$/)
  return m && { passed: m[1] === '✓', name: m[2], suite: m[2].split(':')[0], seconds: +m[4], detail: m[5] ?? '' }
}).filter(Boolean)

const elapsedMin = (Date.now() - new Date(p.startedAt).getTime()) / 60000
console.log(`config ${p.configIndex}/${p.configCount}: ${p.config}`)
if (!p.plan) {
  console.log(`  preparing: ${p.phase}`)
} else {
  const total = p.plan.runs
  const done = runs.length
  const passed = runs.filter((r) => r.passed).length
  const perRun = runs.length ? runs.reduce((n, r) => n + r.seconds, 0) / runs.length : 0
  const etaMin = perRun ? ((total - done) * perRun) / 60 : NaN
  const bar = '█'.repeat(Math.round((20 * done) / total)).padEnd(20, '░')
  console.log(`  ${bar} ${done}/${total} runs (${Math.round((100 * done) / total)}%) · ${elapsedMin.toFixed(0)} min elapsed · ETA ~${Number.isNaN(etaMin) ? '?' : Math.round(etaMin)} min`)
  console.log(`  passing so far: ${passed}/${done} (${done ? Math.round((100 * passed) / done) : 0}%)`)
  const suites = [...new Set(runs.map((r) => r.suite))]
  for (const s of suites) {
    const rs = runs.filter((r) => r.suite === s)
    console.log(`    ${s.padEnd(14)} ${rs.filter((r) => r.passed).length}/${rs.length}`)
  }
  const failures = runs.filter((r) => !r.passed).slice(-5)
  if (failures.length) {
    console.log('  recent failures:')
    for (const f of failures) console.log(`    ✗ ${f.name}: ${f.detail.slice(0, 220)}`)
  }
}
if (staleSec > 600) console.log(`\n  ⚠ no update for ${Math.round(staleSec / 60)} min: the run may have stopped or finished.`)
