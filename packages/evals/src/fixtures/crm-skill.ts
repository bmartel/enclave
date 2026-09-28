import { dateContext, defineSkill, tool, type Db } from '@enclave/core'
import { z } from 'zod'

/** The eval world's fixed "today", so relative dates are deterministic. */
export const TODAY = '2026-09-28' // a Monday

export const CRM_CONTACTS: [string, string, string][] = [
  ['Ada Lovelace', 'Initech', 'ada@initech.example'],
  ['Grace Hopper', 'Initech', 'grace@initech.example'],
  ['Alan Turing', 'Initech', 'alan@initech.example'],
  ['Wei Chen', 'Globex', 'wei.chen@globex.example'],
  ['Wei Chen', 'Umbrella', 'wchen@umbrella.example'],
  ['Linus Berg', 'Hooli', 'linus@hooli.example'],
  ['Maria Garcia', 'Globex', 'maria@globex.example'],
]

/** Reset CRM tables to a known state. */
export async function seedCrm(db: Db): Promise<void> {
  await db.exec(`
    delete from crm_tickets;
    delete from crm_contacts;
    alter table crm_contacts alter column id restart with 1;
    alter table crm_tickets alter column id restart with 1;
    insert into crm_contacts (name, company, email) values
      ${CRM_CONTACTS.map(([n, c, e]) => `('${n}', '${c}', '${e}')`).join(',\n      ')};
    insert into crm_tickets (contact_id, title, priority, status, due_date) values
      (1, 'Printer jam on floor 3', 'normal', 'open', '2026-10-05'),
      (1, 'Renew X200 care plan', 'low', 'open', '2026-11-01'),
      (4, 'Scanner firmware update fails', 'high', 'open', '2026-09-30'),
      (6, 'Invoice address change', 'low', 'closed', '2026-09-15');
  `)
}

/**
 * A typical app-specific skill: typed tools over the app's own tables,
 * including one that needs user approval and one with server-side validation.
 */
export const crmSkill = defineSkill({
  name: 'crm',
  description: 'Customer contacts and support tickets.',
  instructions: `Use these tools for anything about customer contacts or support tickets.
- Look contacts up with find_contacts to get their id; never guess ids.
- If a name matches several contacts and the user didn't say which, ask which one they mean.
- Priorities are low, normal, high or urgent. Dates are YYYY-MM-DD.`,
  migrations: [
    `create table crm_contacts (
       id bigint primary key generated always as identity,
       name text not null, company text not null, email text not null
     );
     create table crm_tickets (
       id bigint primary key generated always as identity,
       contact_id bigint not null references crm_contacts(id) on delete cascade,
       title text not null,
       priority text not null check (priority in ('low','normal','high','urgent')),
       status text not null default 'open' check (status in ('open','in_progress','closed')),
       due_date date not null
     );`,
  ],
  tools: {
    find_contacts: tool({
      description: 'Search contacts by name and/or company (every word must match). Returns id, name, company, email.',
      input: z.object({ query: z.string().min(1) }),
      // Every word must appear in the name or company, so "Wei Chen Umbrella" works.
      execute: async ({ query }, { db }) =>
        (
          await db.query(
            `select id, name, company, email from crm_contacts c
             where (select bool_and(c.name || ' ' || c.company ilike '%' || w || '%')
                    from unnest(regexp_split_to_array(trim($1), '\s+')) as w)
             order by id`,
            [query],
          )
        ).rows,
    }),
    create_ticket: tool({
      description: 'Create a support ticket for a contact.',
      input: z.object({
        contact_id: z.number().int(),
        title: z.string().min(3),
        priority: z.enum(['low', 'normal', 'high', 'urgent']),
        due_date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).describe('YYYY-MM-DD'),
      }),
      execute: async ({ contact_id, title, priority, due_date }, { db }) => {
        if (due_date < TODAY) throw new Error(`due_date ${due_date} is in the past; today is ${TODAY}`)
        const contact = await db.query('select id from crm_contacts where id = $1', [contact_id])
        if (!contact.rows.length) throw new Error(`No contact with id ${contact_id}`)
        const { rows } = await db.query<{ id: number }>(
          'insert into crm_tickets (contact_id, title, priority, due_date) values ($1, $2, $3, $4) returning id',
          [contact_id, title, priority, due_date],
        )
        return { ticket_id: rows[0]!.id, status: 'open' }
      },
    }),
    list_tickets: tool({
      description: 'List tickets, optionally filtered by contact and/or status.',
      input: z.object({
        contact_id: z.number().int().optional(),
        status: z.enum(['open', 'in_progress', 'closed']).optional(),
      }),
      execute: async ({ contact_id, status }, { db }) =>
        (
          await db.query(
            `select t.id, t.title, t.priority, t.status, t.due_date::text as due_date, c.name as contact
             from crm_tickets t join crm_contacts c on c.id = t.contact_id
             where ($1::bigint is null or t.contact_id = $1) and ($2::text is null or t.status = $2)
             order by t.id`,
            [contact_id ?? null, status ?? null],
          )
        ).rows,
    }),
    update_ticket_status: tool({
      description: 'Change a ticket status.',
      input: z.object({ ticket_id: z.number().int(), status: z.enum(['open', 'in_progress', 'closed']) }),
      execute: async ({ ticket_id, status }, { db }) => {
        const { affectedRows } = await db.query('update crm_tickets set status = $2 where id = $1', [ticket_id, status])
        if (!affectedRows) throw new Error(`No ticket with id ${ticket_id}`)
        return { ticket_id, status }
      },
    }),
    delete_contact: tool({
      description: 'Permanently delete a contact and their tickets.',
      input: z.object({ contact_id: z.number().int() }),
      needsApproval: true,
      execute: async ({ contact_id }, { db }) => {
        const { affectedRows } = await db.query('delete from crm_contacts where id = $1', [contact_id])
        return { deleted: (affectedRows ?? 0) > 0 }
      },
    }),
  },
  context: () => dateContext(TODAY),
})
