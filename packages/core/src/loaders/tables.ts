import type { Db } from '../types.js'

/** Rows and columns from a CSV, spreadsheet sheet or JSON array. */
export interface TableData {
  /** Sheet name or file name. */
  name: string
  columns: string[]
  /** Cell values as text; `null` for empty cells. */
  rows: (string | null)[][]
  /** The file it came from. */
  source?: string
  /** Table name `importTable` uses by default, unique per file and sheet (e.g. `budget_q3_forecast`). */
  sqlName?: string
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

/** RFC 4180 CSV with quoted fields, escaped quotes and newlines inside quotes. */
export function parseCsv(text: string, delimiter = sniffDelimiter(text)): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i++
        } else quoted = false
      } else field += ch
    } else if (ch === '"' && field === '') quoted = true
    else if (ch === delimiter) {
      row.push(field)
      field = ''
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else field += ch
  }
  if (field !== '' || row.length) {
    row.push(field)
    rows.push(row)
  }
  return rows.filter((r) => r.some((c) => c.trim() !== ''))
}

/** The delimiter that splits the first lines into the same, largest number of fields. */
export function sniffDelimiter(text: string): string {
  const lines = text.split(/\r?\n/).filter((l) => l.trim()).slice(0, 10)
  let best = ','
  let bestScore = 0
  for (const d of [',', '\t', ';', '|']) {
    const counts = lines.map((l) => countOutsideQuotes(l, d))
    const min = Math.min(...counts)
    if (min === 0) continue
    const consistent = counts.filter((c) => c === counts[0]).length / counts.length
    const score = min * consistent
    if (score > bestScore) {
      best = d
      bestScore = score
    }
  }
  return best
}

function countOutsideQuotes(line: string, d: string): number {
  let n = 0
  let quoted = false
  for (const ch of line) {
    if (ch === '"') quoted = !quoted
    else if (ch === d && !quoted) n++
  }
  return n
}

/** First row as headers; blank or duplicate headers get unique names. */
export function toTable(name: string, grid: (string | null)[][], source?: string): TableData {
  const [header = [], ...body] = grid
  const width = Math.max(header.length, ...body.map((r) => r.length))
  const seen = new Map<string, number>()
  const columns = Array.from({ length: width }, (_, i) => {
    const base = (header[i] ?? '').trim() || `column_${i + 1}`
    const n = (seen.get(base.toLowerCase()) ?? 0) + 1
    seen.set(base.toLowerCase(), n)
    return n > 1 ? `${base} ${n}` : base
  })
  const rows = body.map((r) => columns.map((_, i) => (r[i] == null || r[i]!.trim() === '' ? null : r[i]!.trim())))
  return { name, columns, rows, ...(source ? { source } : {}) }
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

/** An array of flat objects becomes a table; anything else becomes readable text. */
export function jsonToTableOrText(value: unknown, name: string): { table?: TableData; text: string } {
  if (Array.isArray(value) && value.length && value.every(isFlatRecord)) {
    const columns = [...new Set(value.flatMap((r) => Object.keys(r as object)))]
    const rows = (value as Record<string, unknown>[]).map((r) => columns.map((c) => cellText(r[c])))
    const table: TableData = { name, columns, rows }
    return { table, text: tableToText(table) }
  }
  return { text: jsonToText(value).trim() }
}

function isFlatRecord(v: unknown): boolean {
  return !!v && typeof v === 'object' && !Array.isArray(v) && Object.values(v).every((x) => x === null || typeof x !== 'object')
}

function cellText(v: unknown): string | null {
  return v === null || v === undefined || v === '' ? null : String(v)
}

/** YAML-like rendering: easier for a model to read than braces and quotes. */
export function jsonToText(value: unknown, indent = ''): string {
  if (value === null || typeof value !== 'object') return `${indent}${String(value)}\n`
  if (Array.isArray(value)) {
    return value
      .map((v) => (v !== null && typeof v === 'object' ? `${indent}-\n${jsonToText(v, indent + '  ')}` : `${indent}- ${String(v)}\n`))
      .join('')
  }
  return Object.entries(value)
    .map(([k, v]) => (v !== null && typeof v === 'object' ? `${indent}${k}:\n${jsonToText(v, indent + '  ')}` : `${indent}${k}: ${String(v)}\n`))
    .join('')
}

// ---------------------------------------------------------------------------
// Tables as text (for search)
// ---------------------------------------------------------------------------

/**
 * One self-describing line per row ("Region: West; Revenue: 1200"), so every
 * chunk keeps its column names. Rows past `maxRows` are left out.
 */
export function tableToText(table: TableData, maxRows = Infinity): string {
  const shown = table.rows.slice(0, maxRows)
  const header = `Table ${table.name}: ${table.rows.length} rows. Columns: ${table.columns.join(', ')}.`
  const lines = shown.map((r) =>
    table.columns
      .map((c, i) => (r[i] == null ? '' : `${c}: ${r[i]}`))
      .filter(Boolean)
      .join('; '),
  )
  return [header, '', ...lines].join('\n')
}

// ---------------------------------------------------------------------------
// Import into Postgres (for sqlSkill)
// ---------------------------------------------------------------------------

export interface ImportTableOptions {
  /** Table name. Default: derived from the table's name. */
  name?: string
  schema?: string
  /** When the table exists: throw (default), drop and recreate, or append rows. */
  ifExists?: 'error' | 'replace' | 'append'
  /** Stored with `COMMENT ON TABLE`, which `sqlSkill` shows the model. Default: where it came from. */
  comment?: string
}

export interface ImportTableResult {
  /** Qualified name, e.g. `public.sales_2025`. */
  table: string
  rows: number
  columns: { name: string; source: string; type: string }[]
}

/**
 * Create a Postgres table from tabular data, with column types inferred from
 * the values (integer, bigint, numeric, boolean, date, timestamp, text), so the
 * agent can answer questions with SQL instead of reading rows as text.
 */
export async function importTable(db: Db, table: TableData, options: ImportTableOptions = {}): Promise<ImportTableResult> {
  const schema = options.schema ?? 'public'
  const name = options.name ?? sqlIdentifier(table.sqlName ?? table.name.replace(/\.[a-z0-9]+$/i, ''), 'imported')
  const used = new Map<string, number>()
  const columns = table.columns.map((source, i) => {
    let col = sqlIdentifier(source, `column_${i + 1}`)
    const n = (used.get(col) ?? 0) + 1
    used.set(col, n)
    if (n > 1) col = `${col}_${n}`
    return { name: col, source, type: inferType(table.rows.map((r) => r[i] ?? null)) }
  })
  const qualified = `${quoteIdent(schema)}.${quoteIdent(name)}`

  await db.transaction(async (tx) => {
    const { rows: existing } = await tx.query<{ ok: boolean }>(`select to_regclass($1) is not null as ok`, [`${quoteIdent(schema)}.${quoteIdent(name)}`])
    const exists = existing[0]?.ok
    const mode = options.ifExists ?? 'error'
    if (exists && mode === 'error') throw new Error(`Table ${schema}.${name} already exists. Pass ifExists: 'replace' or 'append'.`)
    if (exists && mode === 'replace') await tx.exec(`drop table ${qualified}`)
    if (!exists || mode === 'replace') {
      await tx.exec(`create table ${qualified} (${columns.map((c) => `${quoteIdent(c.name)} ${c.type}`).join(', ')})`)
      const comment = options.comment ?? `Imported from ${table.source ?? table.name}${table.source && table.source !== table.name ? ` (${table.name})` : ''}.`
      await tx.exec(`comment on table ${qualified} is ${quoteLiteral(comment)}`)
      for (const c of columns) {
        if (c.source !== c.name) await tx.exec(`comment on column ${qualified}.${quoteIdent(c.name)} is ${quoteLiteral(`Original column: ${c.source}`)}`)
      }
    }
    // Stay well under Postgres's 65,535 bind parameters per statement.
    const batch = Math.max(1, Math.min(500, Math.floor(30_000 / Math.max(1, columns.length))))
    for (let start = 0; start < table.rows.length; start += batch) {
      const slice = table.rows.slice(start, start + batch)
      const params: (string | null)[] = []
      const values = slice.map((r) => `(${columns.map((c, i) => (params.push(normalize(r[i] ?? null, c.type)), `$${params.length}`)).join(', ')})`)
      await tx.query(`insert into ${qualified} (${columns.map((c) => quoteIdent(c.name)).join(', ')}) values ${values.join(', ')}`, params)
    }
  })
  return { table: `${schema}.${name}`, rows: table.rows.length, columns }
}

const INT = /^-?(0|[1-9]\d*)$/
const NUM = /^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$|^-?\.\d+$/
const BOOL = /^(true|false)$/i
const DATE = /^\d{4}-\d{2}-\d{2}$/
const TIMESTAMP = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/
const TIMESTAMPTZ = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}(:?\d{2})?)$/

/** The narrowest type every non-empty value fits. Leading zeros stay text (zip codes, ids). */
export function inferType(values: (string | null)[]): string {
  const present = values.filter((v): v is string => v !== null && v !== '')
  if (!present.length) return 'text'
  const all = (re: RegExp) => present.every((v) => re.test(v))
  if (all(INT)) {
    const fits = present.every((v) => Math.abs(Number(v)) <= 2_147_483_647)
    if (fits) return 'integer'
    if (present.every((v) => BigInt(v) <= 9_223_372_036_854_775_807n && BigInt(v) >= -9_223_372_036_854_775_808n)) return 'bigint'
    return 'numeric'
  }
  if (all(NUM)) return 'numeric'
  if (all(BOOL)) return 'boolean'
  if (all(DATE) && present.every(validDate)) return 'date'
  if (all(TIMESTAMP) && present.every(validDate)) return 'timestamp'
  if (all(TIMESTAMPTZ) && present.every(validDate)) return 'timestamptz'
  return 'text'
}

function validDate(v: string): boolean {
  const [y, m, d] = v.slice(0, 10).split('-').map(Number) as [number, number, number]
  const date = new Date(Date.UTC(y, m - 1, d))
  return date.getUTCFullYear() === y && date.getUTCMonth() === m - 1 && date.getUTCDate() === d
}

function normalize(v: string | null, type: string): string | null {
  if (v === null || v === '') return null
  return type === 'boolean' ? v.toLowerCase() : v
}

const RESERVED = new Set(
  'all analyse analyze and any array as asc asymmetric both case cast check collate column constraint create current_catalog current_date current_role current_time current_timestamp current_user default deferrable desc distinct do else end except false fetch for foreign from grant group having in initially intersect into lateral leading limit localtime localtimestamp not null offset on only or order placing primary references returning select session_user some symmetric table then to trailing true union unique user using variadic when where window with'.split(' '),
)

/** snake_case, ASCII, ≤ 63 bytes, never a reserved word, so the model can write SQL without quoting. */
export function sqlIdentifier(text: string, fallback: string): string {
  let id = text
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .toLowerCase()
    .replace(/%/g, '_pct')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
  if (!id) id = fallback
  if (/^\d/.test(id)) id = `_${id}`
  if (RESERVED.has(id)) id = `${id}_`
  return id.slice(0, 63)
}

const quoteIdent = (id: string) => `"${id.replace(/"/g, '""')}"`
const quoteLiteral = (s: string) => `'${s.replace(/'/g, "''")}'`
