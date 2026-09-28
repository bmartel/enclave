#!/usr/bin/env node
/** Compare two eval reports: pnpm --filter @enclave/evals compare reports/a.json reports/b.json */
import { readFileSync } from 'node:fs'
import { compareReports } from '@enclave/core/eval'

const [a, b] = process.argv.slice(2)
if (!a || !b) {
  console.error('usage: compare <before.json> <after.json>')
  process.exit(2)
}
const before = JSON.parse(readFileSync(a, 'utf8'))
const after = JSON.parse(readFileSync(b, 'utf8'))
const cmp = compareReports(before, after)
const pct = (x) => `${Math.round(x * 100)}%`
const ci = (r) => `${pct(r.rate)} [${pct(r.low)}–${pct(r.high)}]`
console.log(`before: ${before.label}\nafter:  ${after.label}\n`)
console.log(`overall ${ci(cmp.overall.before)} → ${ci(cmp.overall.after)} (${cmp.overall.delta >= 0 ? '+' : ''}${pct(cmp.overall.delta)})${cmp.overall.significant ? ' SIGNIFICANT' : ' (within noise)'}`)
for (const r of cmp.regressions) console.log(`  ▼ ${r.name}: ${pct(r.before)} → ${pct(r.after)}${r.significant ? ' SIGNIFICANT' : ''}`)
for (const r of cmp.improvements) console.log(`  ▲ ${r.name}: ${pct(r.before)} → ${pct(r.after)}${r.significant ? ' SIGNIFICANT' : ''}`)
process.exit(cmp.regressions.some((r) => r.significant) || (cmp.overall.significant && cmp.overall.delta < 0) ? 1 : 0)
