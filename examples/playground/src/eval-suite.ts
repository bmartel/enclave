import type { Enclave } from '@enclave/core'
import type { EvalCase } from '@enclave/core/eval'

export const HANDBOOK = [
  { id: 'wifi', title: 'Wifi', content: 'The guest wifi network is called Visitors and the password is sunflower42. Staff use single sign-on.' },
  { id: 'parking', title: 'Parking', content: 'Staff park on level B2. Visitors park on level B1 and must register at reception.' },
  { id: 'expenses', title: 'Expenses', content: 'Submit receipts within 30 days. Meals are reimbursed up to 45 dollars per day.' },
  { id: 'leave', title: 'Vacation', content: 'Full-time employees accrue 20 vacation days per year. Requests go to your manager two weeks ahead.' },
]

const table = (sql: string) => async (ai: Enclave) => void (await ai.db.exec(sql))
const count = async (ai: Enclave, sql: string) => (await ai.db.query<{ n: number }>(sql)).rows[0]!.n

/** A small, realistic suite: private RAG, SQL actions, multi-step chaining, restraint. */
export const suite: EvalCase[] = [
  { name: 'rag: wifi password', input: 'What is the guest wifi password?', expect: { answer: 'sunflower42' } },
  { name: 'rag: visitor parking', input: 'Where should visitors park?', expect: { answer: /B1/ } },
  { name: 'rag: meal limit', input: 'How much can I expense for meals per day?', expect: { answer: /45/ } },
  {
    name: 'rag: unknown fact',
    input: "What is the CEO's phone number?",
    expect: { notAnswer: /\d{3}[-.\s]?\d{3}[-.\s]?\d{4}/, answer: /(not|n't|no )[^.]*(know|find|contain|mention|information|available|provided)/i },
  },
  {
    name: 'sql: create + insert',
    input: 'Create a table called books with title and author columns, and add two classic novels to it.',
    setup: table('drop table if exists books'),
    expect: {
      tools: ['execute_sql'],
      check: async ({ ai }) => (await count(ai, 'select count(*)::int as n from books')) === 2 || 'books does not have 2 rows',
    },
  },
  {
    name: 'sql: aggregate existing data',
    input: 'What is the total amount of all orders?',
    setup: table(`drop table if exists orders;
      create table orders (id bigint primary key generated always as identity, customer text, amount numeric);
      insert into orders (customer, amount) values ('acme', 120.50), ('globex', 79.50), ('initech', 300);`),
    expect: { tools: ['execute_sql'], answer: /500/ },
  },
  {
    name: 'sql: chained write then read',
    input: 'Add cherry to the fruits table, then tell me how many fruits are in it.',
    setup: table(`drop table if exists fruits; create table fruits (name text); insert into fruits values ('apple'), ('banana');`),
    expect: {
      tools: ['execute_sql'],
      answer: /\b3\b|three/i,
      check: async ({ ai }) => (await count(ai, "select count(*)::int as n from fruits where name ilike 'cherry'")) === 1 || 'cherry not inserted',
    },
  },
  {
    name: 'follow-up turn uses tools',
    input: ['What is the guest wifi password?', 'Create a table named visits with a visitor column and add one row for Ada.'],
    setup: table('drop table if exists visits'),
    expect: {
      tools: ['execute_sql'],
      check: async ({ ai }) => (await count(ai, 'select count(*)::int as n from visits')) === 1 || 'visits does not have 1 row',
    },
  },
  { name: 'restraint: no tools for chit-chat', input: 'Say good morning in Spanish.', expect: { noTools: true, answer: /buenos d[ií]as/i } },
]
