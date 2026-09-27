import type { Db } from '../types.js'

const BOOTSTRAP = `
create schema if not exists enclave;
create table if not exists enclave.migrations (
  scope text not null,
  version int not null,
  applied_at timestamptz not null default now(),
  primary key (scope, version)
);
`

export const CORE_MIGRATIONS = [
  `
  create table enclave.kv (
    key text primary key,
    value jsonb not null
  );
  create table enclave.threads (
    id text primary key,
    title text,
    metadata jsonb not null default '{}',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
  );
  create table enclave.messages (
    id bigint generated always as identity primary key,
    thread_id text not null references enclave.threads(id) on delete cascade,
    message jsonb not null,
    created_at timestamptz not null default now()
  );
  create index messages_thread_idx on enclave.messages (thread_id, id);
  `,
]

/**
 * Apply `migrations` for `scope` that have not run yet. Each migration runs in
 * its own transaction and is recorded by index, so append-only edits are safe.
 */
export async function migrate(db: Db, scope: string, migrations: readonly string[]): Promise<number> {
  await db.exec(BOOTSTRAP)
  const { rows } = await db.query<{ version: number | null }>(
    'select max(version) as version from enclave.migrations where scope = $1',
    [scope],
  )
  const current = rows[0]?.version ?? -1
  let applied = 0
  for (let version = current + 1; version < migrations.length; version++) {
    await db.transaction(async (tx) => {
      await tx.exec(migrations[version]!)
      await tx.query('insert into enclave.migrations (scope, version) values ($1, $2)', [scope, version])
    })
    applied++
  }
  return applied
}

export async function kvGet<T>(db: Db, key: string): Promise<T | undefined> {
  const { rows } = await db.query<{ value: T }>('select value from enclave.kv where key = $1', [key])
  return rows[0]?.value
}

export async function kvSet(db: Db, key: string, value: unknown): Promise<void> {
  await db.query(
    'insert into enclave.kv (key, value) values ($1, $2) on conflict (key) do update set value = excluded.value',
    [key, JSON.stringify(value)],
  )
}
