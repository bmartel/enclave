import type { Enclave } from 'enclave-ai'
import type { EvalCase } from 'enclave-ai/eval'

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

/**
 * Multi-turn conversations graded on every turn. Later turns depend on earlier
 * ones, which is where reasoning history and KV-cache reuse across turns matter.
 */
export const conversations: EvalCase[] = [
  {
    name: 'tasks: build, update, query',
    setup: table('drop table if exists tasks'),
    turns: [
      {
        input: 'Create a table called tasks with a title column and a done boolean column, then add three tasks: buy milk, write report, call mom.',
        expect: { tools: ['execute_sql'], check: async ({ ai }) => (await count(ai, 'select count(*)::int as n from tasks')) === 3 || 'tasks does not have 3 rows' },
      },
      {
        input: 'Mark the report task as done.',
        expect: {
          tools: ['execute_sql'],
          check: async ({ ai }) => (await count(ai, "select count(*)::int as n from tasks where title ilike '%report%' and done")) === 1 || 'report not marked done',
        },
      },
      // Mentioning the report as done is fine; the open ones must be listed.
      { input: 'Which tasks are still open?', expect: { answer: [/milk/i, /mom/i] } },
      {
        input: 'Delete the milk task.',
        expect: { tools: ['execute_sql'], check: async ({ ai }) => (await count(ai, "select count(*)::int as n from tasks where title ilike '%milk%'")) === 0 || 'milk task still present' },
      },
      { input: 'How many rows are in the tasks table now, done or not?', expect: { answer: /\b2\b|two/i } },
    ],
  },
  {
    name: 'handbook: follow-ups',
    turns: [
      { input: 'What is the guest wifi password?', expect: { answer: 'sunflower42' } },
      { input: 'And where should visitors park?', expect: { answer: /B1/ } },
      { input: 'Combine your last two answers into one sentence for a visitor.', expect: { answer: ['sunflower42', /B1/] } },
      { input: 'How many vacation days do full-time employees get?', expect: { answer: /\b20\b|twenty/i } },
    ],
  },
  {
    name: 'orders: analysis that evolves',
    setup: table(`drop table if exists orders;
      create table orders (id bigint primary key generated always as identity, customer text, amount numeric);
      insert into orders (customer, amount) values ('acme', 120.50), ('globex', 79.50), ('initech', 300);`),
    turns: [
      { input: 'What is the total amount of all orders?', expect: { tools: ['execute_sql'], answer: /500/ } },
      { input: 'Which customer has spent the most?', expect: { answer: /initech/i } },
      {
        input: 'Add a new order of 250 for acme.',
        expect: { tools: ['execute_sql'], check: async ({ ai }) => (await count(ai, "select count(*)::int as n from orders where customer = 'acme'")) === 2 || 'acme order not added' },
      },
      { input: 'Now which customer has spent the most?', expect: { answer: /acme/i } },
      { input: 'And what is the new total amount across all orders?', expect: { answer: /750/ } },
    ],
  },
]
