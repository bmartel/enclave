/** The eval world's own tools behave as documented (CPU, seconds). */
import { beforeAll, describe, expect, it } from 'vitest'
import { createEnclave, type Enclave } from 'enclave-ai'
import { createDb } from 'enclave-ai/pglite'
import { hashEmbedder, mockModel } from 'enclave-ai/testing'
import { crmSkill } from '../src/fixtures/crm-skill.js'
import { prepareWorld, suiteSkills } from '../src/world.js'

let ai: Enclave
beforeAll(async () => {
  ai = await createEnclave({ db: await createDb({ dataDir: 'memory://' }), model: mockModel([]), embedder: hashEmbedder(64), skills: suiteSkills() })
  await prepareWorld(ai)
})

const find = crmSkill.tools!.find_contacts!
type Contact = { id: number; name: string; company: string }
const run = (query: string) => find.execute({ query }, { db: ai.db } as never) as unknown as Promise<Contact[]>

describe('crm find_contacts', () => {
  it('matches every word against name and company', async () => {
    expect((await run('Wei Chen Umbrella')).map((r) => r.company)).toEqual(['Umbrella'])
    expect(await run('Wei Chen Initech')).toEqual([])
  })

  it('tells the model when several contacts share a name', async () => {
    const rows = await run('Wei Chen')
    expect(rows).toHaveLength(2)
    expect(find.toModelOutput!(rows)).toMatchObject({ note: expect.stringContaining('ask them') })
    expect(find.toModelOutput!(await run('Linus'))).toEqual(await run('Linus'))
  })
})

describe('business database', () => {
  it('documents the revenue rule where the sql skill shows it', async () => {
    const sql = ai.skills.find((s) => s.name === 'sql')!
    const context = await sql.context!({ db: ai.db, knowledge: undefined, embedder: undefined, threadId: 't', messages: [] })
    expect(context).toContain('table orders ( -- Revenue counts only orders whose status is not cancelled.')
    expect(context).toMatch(/unit_price numeric\(10,2\) not null, -- Price per unit/)
  })
})
