import { z } from 'zod'
import { defineSkill } from '../skill.js'
import { tool } from '../tool.js'
import type { Db } from '../types.js'

export interface SqlSkillOptions {
  /** Schemas the model may see and query. Default `['public']`. */
  schemas?: string[]
  /** Reject anything but reads, enforced by a read-only transaction. Default false. */
  readOnly?: boolean
  /** Ask the user before running statements that change data or schema. Default true. */
  approveWrites?: boolean
  /** Rows per result returned to the model. Default 100. */
  maxRows?: number
}

const MUTATION = /\b(insert|update|delete|merge|create|alter|drop|truncate|grant|revoke|copy|vacuum|reindex|cluster|refresh|call|do)\b/i
const INTERNAL = /\benclave\s*\./i

/** Strip comments and string literals so keyword checks don't misfire. */
function stripSql(sql: string): string {
  return sql
    .replace(/--[^\n]*/g, ' ')
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/'(?:[^']|'')*'/g, "''")
    .replace(/\$([a-z_]*)\$[\s\S]*?\$\1\$/gi, "''")
}

export function isMutation(sql: string): boolean {
  return MUTATION.test(stripSql(sql))
}

/**
 * The core database.build capability: the model designs schemas, writes and
 * runs Postgres against the user's local PGlite database.
 */
export function sqlSkill(options: SqlSkillOptions = {}) {
  const schemas = options.schemas ?? ['public']
  const maxRows = options.maxRows ?? 100
  const readOnly = options.readOnly ?? false

  return defineSkill({
    name: 'sql',
    description: 'Query and modify the local Postgres database.',
    instructions: `You have a Postgres 17 database (PGlite) running locally in the user's browser. Visible schemas: ${schemas.join(', ')}.
- The pgvector extension is available (use <=> for cosine distance).
- Primary keys: "id bigint primary key generated always as identity". Prefer text over varchar.
- When the user asks to create, change or query data, run the SQL with execute_sql. Never just show SQL for the user to run.
- Don't conclude data is missing from table or column names alone: amounts like revenue or totals are often computed across several columns or tables, so query to check.
- Always add a LIMIT to exploratory queries (default 5, max ${maxRows}).
- Check the current state section or call describe_schema before writing queries against tables you haven't seen.${
      readOnly ? '\n- The database is read-only: only SELECT queries are allowed.' : ''
    }`,
    tools: {
      describe_schema: tool({
        description: 'Describe tables, columns, types and constraints in the database.',
        input: z.object({
          table: z.string().optional().describe('Only describe this table.'),
        }),
        execute: async ({ table }, { db }) => {
          const ddl = await describe(db, schemas, table)
          return ddl || (table ? `No table named "${table}".` : 'The database has no tables yet.')
        },
      }),
      execute_sql: tool({
        description:
          'Run one or more SQL statements and return the results. Returns up to ' + maxRows + ' rows per statement.',
        input: z.object({ sql: z.string().min(1) }),
        needsApproval: ({ sql }) => !readOnly && (options.approveWrites ?? true) && isMutation(sql),
        execute: async ({ sql }, { db }) => {
          if (INTERNAL.test(stripSql(sql))) throw new Error('The "enclave" schema is internal and cannot be queried.')
          if (readOnly && isMutation(sql)) throw new Error('The database is read-only.')
          const results = readOnly
            ? await db.transaction(async (tx) => {
                await tx.exec('set transaction read only')
                return tx.exec(sql)
              })
            : await db.exec(sql)
          return results.map((r) => ({
            columns: r.fields.map((f) => f.name),
            rows: r.rows,
            rowCount: r.rows.length,
            ...(r.affectedRows !== undefined ? { affectedRows: r.affectedRows } : {}),
          }))
        },
        toModelOutput: (results) =>
          results.map((r) => ({
            ...r,
            rows: r.rows.slice(0, maxRows).map(compactRow),
            ...(r.rows.length > maxRows ? { truncated: `showing ${maxRows} of ${r.rows.length} rows` } : {}),
          })),
      }),
    },
    context: async ({ db }) => {
      const ddl = await describe(db, schemas)
      if (!ddl) return 'Database is empty.'
      return ddl.length > 4000 ? `${ddl.slice(0, 4000)}\n… (call describe_schema for the rest)` : ddl
    },
  })
}

function compactRow(row: unknown): unknown {
  if (!row || typeof row !== 'object') return row
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(row)) {
    if (typeof value === 'string' && value.length > 400) {
      out[key] = value.startsWith('[') ? `[vector ${value.split(',').length}d]` : `${value.slice(0, 400)}…`
    } else {
      out[key] = value
    }
  }
  return out
}

async function describe(db: Db, schemas: string[], table?: string): Promise<string> {
  const { rows: columns } = await db.query<{
    table_schema: string
    table_name: string
    column_name: string
    type: string
    not_null: boolean
    default_value: string | null
    column_comment: string | null
    table_comment: string | null
  }>(
    `select n.nspname as table_schema, c.relname as table_name, a.attname as column_name,
            format_type(a.atttypid, a.atttypmod) as type, a.attnotnull as not_null,
            pg_get_expr(d.adbin, d.adrelid) as default_value,
            col_description(c.oid, a.attnum) as column_comment, obj_description(c.oid, 'pg_class') as table_comment
     from pg_attribute a
       join pg_class c on c.oid = a.attrelid
       join pg_namespace n on n.oid = c.relnamespace
       left join pg_attrdef d on d.adrelid = a.attrelid and d.adnum = a.attnum
     where n.nspname = any($1) and c.relkind in ('r', 'p', 'v', 'm')
       and a.attnum > 0 and not a.attisdropped
       and ($2::text is null or c.relname = $2)
     order by n.nspname, c.relname, a.attnum`,
    [schemas, table ?? null],
  )
  const { rows: constraints } = await db.query<{ table_schema: string; table_name: string; def: string }>(
    `select n.nspname as table_schema, c.relname as table_name, pg_get_constraintdef(k.oid) as def
     from pg_constraint k
       join pg_class c on c.oid = k.conrelid
       join pg_namespace n on n.oid = c.relnamespace
     where n.nspname = any($1) and ($2::text is null or c.relname = $2)
     order by k.contype, k.conname`,
    [schemas, table ?? null],
  )

  // COMMENT ON is where apps document business rules ("revenue excludes
  // cancelled orders"); the model sees them as SQL comments.
  const tables = new Map<string, { def: string; comment?: string }[]>()
  const notes = new Map<string, string>()
  for (const c of columns) {
    const key = schemas.length === 1 ? c.table_name : `${c.table_schema}.${c.table_name}`
    const parts = [c.column_name, c.type]
    if (c.not_null) parts.push('not null')
    if (c.default_value) parts.push(`default ${c.default_value}`)
    if (c.table_comment) notes.set(key, oneLine(c.table_comment))
    const line = { def: parts.join(' '), ...(c.column_comment ? { comment: oneLine(c.column_comment) } : {}) }
    tables.set(key, [...(tables.get(key) ?? []), line])
  }
  for (const k of constraints) {
    const key = schemas.length === 1 ? k.table_name : `${k.table_schema}.${k.table_name}`
    tables.get(key)?.push({ def: k.def })
  }
  return [...tables]
    .map(([name, lines]) => {
      const body = lines.map((l, i) => `${l.def}${i < lines.length - 1 ? ',' : ''}${l.comment ? ` -- ${l.comment}` : ''}`)
      const note = notes.get(name)
      return `${note ? `-- ${note}\n` : ''}table ${name} (\n  ${body.join('\n  ')}\n)`
    })
    .join('\n')
}

const oneLine = (text: string) => text.replace(/\s+/g, ' ').trim()
