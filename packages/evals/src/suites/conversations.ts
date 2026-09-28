import { anyOf } from '@enclave/core/eval'
import { REVENUE_SQL, scalar } from '../fixtures/business-db.js'
import { dbState, statesValue } from '../graders.js'
import { resetData, type ProductionCase } from '../world.js'

const revenueIn = (from: string, to: string) => `${REVENUE_SQL} and o.ordered_at >= '${from}' and o.ordered_at < '${to}'`
const JUNE = revenueIn('2026-06-01', '2026-07-01')
const MAY = revenueIn('2026-05-01', '2026-06-01')
const ordersIn = (from: string, to: string) => `select count(*) from orders where status <> 'cancelled' and ordered_at >= '${from}' and ordered_at < '${to}'`
const q = (sql: string) => ({ name: 'execute_sql', input: { sql } })

/**
 * Multi-turn conversations graded on every turn: corrections, pronouns,
 * evolving analysis, and switching between knowledge, data and actions.
 */
export const conversationCases: ProductionCase[] = [
  {
    name: 'conv: correction mid-conversation',
    tags: ['multi-turn', 'rag', 'correction'],
    turns: [
      {
        input: 'What is the guest wifi password?',
        // Two offices: giving both, or asking which, are both correct.
        expect: { answer: anyOf(/which office|toronto or berlin|berlin or toronto/i, (t) => t.includes('maple-harbor-42') && t.includes('linden-river-17'), (t) => /toronto/i.test(t) && t.includes('maple-harbor-42')) },
      },
      { input: 'Sorry, I meant the Berlin office.', expect: { answer: 'linden-river-17' } },
      { input: 'And where can visitors park there?', expect: { answer: /Alexanderplatz/i } },
      { input: 'Summarize both of those for a visitor in two sentences.', expect: { answer: ['linden-river-17', /Alexanderplatz/i] } },
    ],
    reference: [
      { answer: 'Which office? Toronto uses maple-harbor-42 and Berlin uses linden-river-17.' },
      { answer: 'In Berlin, the guest network SpreeGuest uses linden-river-17 [1].' },
      { answer: 'Berlin has no car parking; visitors can use the public garage at Alexanderplatz [1].' },
      { answer: 'Connect to SpreeGuest with password linden-river-17. Park at the public Alexanderplatz garage, as the office has no parking.' },
    ],
  },
  {
    name: 'conv: pronouns across turns',
    tags: ['multi-turn', 'rag', 'coreference'],
    turns: [
      { input: 'Which of our scanners has the longest battery life?', expect: { answer: /X300/ } },
      { input: 'How much does it cost?', expect: { answer: /\b899\b/ } },
      { input: 'Is it water resistant?', expect: { answer: /IP67/i } },
      { input: 'How do I factory reset it?', expect: { answer: /12\s?seconds/i } },
    ],
    reference: [
      { answer: 'The X300, with 20 hours of battery [1].' },
      { answer: 'The X300 costs $899 [1].' },
      { answer: 'Yes, it is rated IP67 [1].' },
      { answer: 'Hold power and volume-down for 12 seconds, then choose Wipe data [1].' },
    ],
  },
  {
    name: 'conv: analysis that evolves',
    tags: ['multi-turn', 'sql', 'write'],
    setup: resetData,
    turns: [
      { input: 'What was our revenue in June 2026? Exclude cancelled orders.', expect: { tools: ['execute_sql'], check: statesValue(JUNE, 0.5) } },
      { input: 'How does that compare with May?', expect: { check: statesValue(MAY, 0.5) } },
      {
        input: 'Which of those two months had more orders?',
        expect: {
          check: async ({ ai, text }) => {
            const june = await scalar<number>(ai.db, ordersIn('2026-06-01', '2026-07-01'))
            const may = await scalar<number>(ai.db, ordersIn('2026-05-01', '2026-06-01'))
            const winner = june > may ? /june/i : /may/i
            return winner.test(text) || `expected ${june > may ? 'June' : 'May'} (June ${june}, May ${may})`
          },
        },
      },
      {
        input: 'Record a new delivered order for Hiro Silva dated 2026-06-15: 2 units of the X300 Scanner at the list price.',
        expect: {
          tools: ['execute_sql'],
          check: dbState(
            `select count(*)::int from orders o join customers c on c.id = o.customer_id join order_items oi on oi.order_id = o.id
             join products p on p.id = oi.product_id where c.name = 'Hiro Silva' and o.ordered_at = '2026-06-15' and p.name = 'X300 Scanner' and oi.quantity = 2`,
            1,
            'the order was not recorded correctly',
          ),
        },
      },
      { input: 'So what is June revenue now?', expect: { check: statesValue(JUNE, 0.5) } },
    ],
    reference: [
      { calls: [q(JUNE)], answer: async (ai) => `June 2026 revenue was $${await scalar(ai.db, JUNE)}.` },
      { calls: [q(MAY)], answer: async (ai) => `May revenue was $${await scalar(ai.db, MAY)}.` },
      {
        calls: [q(ordersIn('2026-05-01', '2026-06-01')), q(ordersIn('2026-06-01', '2026-07-01'))],
        answer: async (ai) => ((await scalar<number>(ai.db, ordersIn('2026-06-01', '2026-07-01'))) > (await scalar<number>(ai.db, ordersIn('2026-05-01', '2026-06-01'))) ? 'June had more orders.' : 'May had more orders.'),
      },
      {
        calls: [
          q(`with o as (insert into orders (customer_id, status, ordered_at) select id, 'delivered', '2026-06-15' from customers where name = 'Hiro Silva' returning id)
             insert into order_items (order_id, product_id, quantity, unit_price) select o.id, p.id, 2, p.price from o, products p where p.name = 'X300 Scanner'`),
        ],
        answer: 'Recorded: 2 × X300 Scanner for Hiro Silva on 2026-06-15.',
      },
      { calls: [q(JUNE)], answer: async (ai) => `June revenue is now $${await scalar(ai.db, JUNE)}.` },
    ],
  },
  {
    name: 'conv: support workflow',
    tags: ['multi-turn', 'skill', 'rag'],
    setup: resetData,
    turns: [
      { input: 'Find the Wei Chen who works at Globex.', expect: { answer: anyOf(/wei\.chen@globex/i, /Globex/i) } },
      { input: 'What open tickets does he have?', expect: { answer: /firmware/i } },
      { input: 'Mark it as in progress.', expect: { tools: ['update_ticket_status'], check: dbState(`select status from crm_tickets where id = 3`, 'in_progress', 'ticket 3 not in progress') } },
      { input: "What's our first response time commitment for priority 1 issues?", expect: { answer: /\b1\s?hour|one hour|60 minutes/i } },
    ],
    reference: [
      { calls: [{ name: 'find_contacts', input: { query: 'Wei Chen' } }], answer: 'Wei Chen at Globex: wei.chen@globex.example (id 4).' },
      { calls: [{ name: 'list_tickets', input: { contact_id: 4, status: 'open' } }], answer: 'He has one open ticket: #3 "Scanner firmware update fails".' },
      { calls: [{ name: 'update_ticket_status', input: { ticket_id: 3, status: 'in_progress' } }], answer: 'Ticket #3 is now in progress.' },
      { answer: 'Priority 1 issues get a first response within 1 hour, 24/7 [1].' },
    ],
  },
  {
    name: 'conv: policy then action',
    tags: ['multi-turn', 'rag', 'skill'],
    setup: resetData,
    turns: [
      { input: 'What is our response time for a production-down issue?', expect: { answer: /\b1\s?hour|one hour|60 minutes/i } },
      {
        input: 'Ada Lovelace just reported production is down. Create an urgent ticket for her, due tomorrow.',
        expect: {
          tools: ['create_ticket'],
          check: dbState(`select count(*)::int from crm_tickets where id > 4 and contact_id = 1 and priority = 'urgent' and due_date = '2026-09-29'`, 1, 'urgent ticket for Ada due 2026-09-29 not created'),
        },
      },
    ],
    reference: [
      { answer: 'Production-down (priority 1) issues get a first response within 1 hour [1].' },
      {
        calls: [{ name: 'find_contacts', input: { query: 'Ada Lovelace' } }, { name: 'create_ticket', input: { contact_id: 1, title: 'Production down', priority: 'urgent', due_date: '2026-09-29' } }],
        answer: 'Created urgent ticket #5 for Ada Lovelace, due 2026-09-29.',
      },
    ],
  },
  {
    name: 'conv: data then policy',
    tags: ['multi-turn', 'sql', 'rag'],
    setup: resetData,
    turns: [
      { input: 'How many orders are pending right now?', expect: { tools: ['execute_sql'], check: statesValue(`select count(*) from orders where status = 'pending'`) } },
      { input: 'Per our data retention policy, how long do we keep support tickets?', expect: { answer: /\b3\s?years|three years/i } },
    ],
    reference: [
      { calls: [q(`select count(*) from orders where status = 'pending'`)], answer: async (ai) => `${await scalar(ai.db, `select count(*) from orders where status = 'pending'`)} orders are pending.` },
      { answer: 'Support tickets are retained for 3 years [1].' },
    ],
  },
]

