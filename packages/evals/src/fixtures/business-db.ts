import type { Db } from '@enclave/core'

/** Deterministic PRNG so every seed produces identical data. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const FIRST = ['Ava', 'Ben', 'Chloe', 'Dev', 'Elena', 'Farid', 'Grace', 'Hiro', 'Ines', 'Jonas', 'Kara', 'Liam', 'Maya', 'Noah', 'Olga', 'Pavel', 'Quinn', 'Rosa', 'Sami', 'Tara']
const LAST = ['Nguyen', 'Schmidt', 'Okafor', 'Silva', 'Kowalski', 'Haddad', 'Tanaka', 'Moreau', 'Rossi', 'Larsen']
const COUNTRIES = ['Canada', 'Canada', 'Canada', 'Germany', 'Germany', 'United States', 'United States', 'France']
const SEGMENTS = ['enterprise', 'smb', 'smb', 'consumer']

export const PRODUCTS: [string, string, number][] = [
  ['X100 Scanner', 'hardware', 399],
  ['X200 Scanner', 'hardware', 649],
  ['X300 Scanner', 'hardware', 899],
  ['Charging Dock', 'accessory', 79],
  ['Carry Case', 'accessory', 39],
  ['Spare Battery', 'accessory', 59],
  ['Screen Protector', 'accessory', 19],
  ['Care Plan 1yr', 'service', 99],
  ['Care Plan 3yr', 'service', 249],
  ['Fleet Console', 'software', 499],
  ['Legacy Cable', 'accessory', 9],
  ['Training Session', 'service', 350],
]

/** Recreate the business schema with the same data every time. */
export async function seedBusinessDb(db: Db): Promise<void> {
  const rand = mulberry32(20260928)
  const pick = <T>(xs: T[]) => xs[Math.floor(rand() * xs.length)]!

  const customers: string[] = []
  for (let i = 1; i <= 40; i++) {
    // Unique names: the surname shifts on the second pass through first names.
    const first = FIRST[(i * 7) % FIRST.length]!
    const last = LAST[(i * 3 + Math.floor((i - 1) / FIRST.length)) % LAST.length]!
    const name = `${first} ${last}`
    // Every 6th customer has no email on file.
    const email = i % 6 === 0 ? 'null' : `'${first.toLowerCase()}.${last.toLowerCase()}${i}@example.com'`
    const created = `2025-${String(1 + (i % 12)).padStart(2, '0')}-${String(1 + (i % 27)).padStart(2, '0')}`
    customers.push(`('${name}', ${email}, '${pick(COUNTRIES)}', '${pick(SEGMENTS)}', '${created}')`)
  }

  const orders: string[] = []
  const items: string[] = []
  for (let o = 1; o <= 150; o++) {
    const customer = 1 + Math.floor(rand() * 40)
    const month = 1 + Math.floor(rand() * 9) // Jan..Sep 2026
    const day = 1 + Math.floor(rand() * 28)
    const r = rand()
    const status = r < 0.1 ? 'cancelled' : r < 0.25 ? 'pending' : r < 0.55 ? 'shipped' : 'delivered'
    orders.push(`(${customer}, '${status}', '2026-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}')`)
    const lines = 1 + Math.floor(rand() * 3)
    const used = new Set<number>()
    for (let l = 0; l < lines; l++) {
      // Legacy Cable (id 11) is never ordered.
      let product = 1 + Math.floor(rand() * 12)
      if (product === 11) product = 7
      if (used.has(product)) continue
      used.add(product)
      const quantity = 1 + Math.floor(rand() * 4)
      items.push(`(${o}, ${product}, ${quantity}, ${PRODUCTS[product - 1]![2]})`)
    }
  }

  await db.exec(`
    drop table if exists order_items;
    drop table if exists orders;
    drop table if exists products;
    drop table if exists customers;
    create table customers (
      id bigint primary key generated always as identity,
      name text not null,
      email text,
      country text not null,
      segment text not null,
      created_at date not null
    );
    create table products (
      id bigint primary key generated always as identity,
      name text not null,
      category text not null,
      price numeric(10,2) not null,
      active boolean not null default true
    );
    create table orders (
      id bigint primary key generated always as identity,
      customer_id bigint not null references customers(id),
      status text not null check (status in ('pending','shipped','delivered','cancelled')),
      ordered_at date not null
    );
    create table order_items (
      order_id bigint not null references orders(id) on delete cascade,
      product_id bigint not null references products(id),
      quantity int not null,
      unit_price numeric(10,2) not null,
      primary key (order_id, product_id)
    );
    insert into customers (name, email, country, segment, created_at) values ${customers.join(',')};
    insert into products (name, category, price, active) values ${PRODUCTS.map(([n, c, p]) => `('${n}', '${c}', ${p}, ${n !== 'Legacy Cable'})`).join(',')};
    insert into orders (customer_id, status, ordered_at) values ${orders.join(',')};
    insert into order_items (order_id, product_id, quantity, unit_price) values ${items.join(',')};
  `)
}

/** Run a reference query and return the first column of the first row. */
export async function scalar<T = number>(db: Db, sql: string): Promise<T> {
  const { rows } = await db.query<Record<string, unknown>>(sql)
  const value = Object.values(rows[0] ?? {})[0]
  return (typeof value === 'string' && /^-?\d+(\.\d+)?$/.test(value) ? Number(value) : value) as T
}

/** Revenue excludes cancelled orders. */
export const REVENUE_SQL = `select coalesce(sum(oi.quantity * oi.unit_price), 0)::numeric(12,2)
  from order_items oi join orders o on o.id = oi.order_id where o.status <> 'cancelled'`
