import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { PGlite } from '@electric-sql/pglite'
import * as pdfjs from 'pdfjs-dist/legacy/build/pdf.mjs'
import { detectFormat, htmlToMarkdown, importTable, inferType, loadFile, loadFiles, parseCsv, sqlIdentifier, UnsupportedFileError } from '../src/loaders/index.js'
import { readZip } from '../src/loaders/zip.js'
import { Knowledge } from '../src/rag/knowledge.js'
import { CORE_MIGRATIONS, migrate } from '../src/store/migrate.js'
import { hashEmbedder } from '../src/testing.js'
import { makePdf, makeZip, para } from './fixtures.js'
import { memoryDb } from './helpers.js'

const pdf = { lib: pdfjs as never }
const file = (name: string, data: string | Uint8Array, type?: string) => ({ name, data, ...(type ? { type } : {}) })

describe('zip', () => {
  it('reads stored and deflated entries', async () => {
    const zip = readZip(makeZip({ 'a.txt': 'stored text', 'dir/b.txt': 'deflated '.repeat(50) }, { store: ['a.txt'] }))
    expect(zip.names()).toEqual(['a.txt', 'dir/b.txt'])
    expect(await zip.text('a.txt')).toBe('stored text')
    expect(await zip.text('dir/b.txt')).toBe('deflated '.repeat(50))
    expect(await zip.text('missing')).toBeUndefined()
  })
})

describe('HTML', () => {
  it('keeps the article and drops scripts, navigation and chrome', () => {
    const { title, text } = htmlToMarkdown(`<!doctype html><html><head><title>Expenses &amp; Travel</title>
      <style>p { color: red }</style><script>var x = "<p>not text</p>"</script></head>
      <body><nav><a href="/">Home</a> <a href="/about">About</a></nav>
      <main><h1>Expense policy</h1>
        <p>Submit   receipts within <b>30&nbsp;days</b>.<br>Late claims need approval.</p>
        <!-- internal note -->
        <h2>Limits</h2>
        <ul><li>Meals: $60/day</li><li>Hotels:<ul><li>Standard room</li></ul></li></ul>
        <ol start="3"><li>Third</li><li>Fourth</li></ol>
        <table><tr><th>Item</th><th>Limit</th></tr><tr><td>Taxi</td><td>$40 | $50</td></tr></table>
        <pre><code>total = sum(receipts)
  print(total)</code></pre>
        <p hidden>Hidden text</p><div aria-hidden="true">Decorative</div>
      </main>
      <footer>© 2026 Example Corp</footer></body></html>`)
    expect(title).toBe('Expenses & Travel')
    expect(text).toBe(
      [
        '# Expense policy',
        'Submit receipts within 30 days.\nLate claims need approval.',
        '## Limits',
        '- Meals: $60/day\n- Hotels:\n  - Standard room',
        '3. Third\n4. Fourth',
        '| Item | Limit |\n| --- | --- |\n| Taxi | $40 \\| $50 |',
        '```\ntotal = sum(receipts)\n  print(total)\n```',
      ].join('\n\n'),
    )
  })

  it('keeps the whole body when there is no main element, minus footers and hidden parts', () => {
    const { title, text } = htmlToMarkdown(`<body><header><h1>Wifi</h1></header><div role="navigation">Menu</div>
      <p>Guest network: NorthGuest</p><p style="display: none">secret</p><footer>Footer</footer></body>`)
    expect(title).toBe('Wifi')
    expect(text).toBe('# Wifi\n\nGuest network: NorthGuest')
  })

  it('survives malformed markup', () => {
    const { text } = htmlToMarkdown('<p>One <b>two<p>Three &lt; four &#x2014; &#169; &bogus; <div>Five')
    expect(text).toBe('One two\n\nThree < four — © &bogus;\n\nFive')
  })
})

describe('CSV and JSON', () => {
  it('parses quoted fields and sniffs the delimiter', () => {
    expect(parseCsv('name,notes\n"Smith, J","said ""hi""\nthen left"\n\nLee,ok\r\n')).toEqual([
      ['name', 'notes'],
      ['Smith, J', 'said "hi"\nthen left'],
      ['Lee', 'ok'],
    ])
    expect(parseCsv('a;b;c\n1,5;2;3\n')).toEqual([
      ['a', 'b', 'c'],
      ['1,5', '2', '3'],
    ])
  })

  it('loads a CSV as a table and as self-describing text rows', async () => {
    const loaded = await loadFile(file('sales 2025.csv', 'Region,Revenue,Quarter\nWest,1200,Q1\nEast,,Q1\n'), { collection: 'data' })
    expect(loaded.format).toBe('csv')
    expect(loaded.tables[0]).toMatchObject({ name: 'sales 2025', columns: ['Region', 'Revenue', 'Quarter'], rows: [['West', '1200', 'Q1'], ['East', null, 'Q1']] })
    expect(loaded.documents[0]).toMatchObject({ title: 'sales 2025', source: 'sales 2025.csv', collection: 'data' })
    expect(loaded.documents[0]!.content).toBe('Table sales 2025: 2 rows. Columns: Region, Revenue, Quarter.\n\nRegion: West; Revenue: 1200; Quarter: Q1\nRegion: East; Quarter: Q1')
  })

  it('decodes Windows-1252 exports from Excel', async () => {
    const bytes = new Uint8Array([...new TextEncoder().encode('city\nMontr'), 0xe9, ...new TextEncoder().encode('al\n')])
    const loaded = await loadFile(file('cities.csv', bytes))
    expect(loaded.tables[0]!.rows).toEqual([['Montréal']])
  })

  it('caps text rows and says so', async () => {
    const csv = 'n\n' + Array.from({ length: 30 }, (_, i) => i).join('\n')
    const loaded = await loadFile(file('n.csv', csv), { maxTextRows: 10 })
    expect(loaded.tables[0]!.rows).toHaveLength(30)
    expect(loaded.documents[0]!.content.split('\n')).toHaveLength(12)
    expect(loaded.warnings[0]).toMatch(/first 10 of 30 rows/)
  })

  it('turns JSON arrays into tables and other JSON into readable text', async () => {
    const rows = await loadFile(file('people.json', JSON.stringify([{ name: 'Ada', age: 36 }, { name: 'Alan', team: 'Bletchley' }])))
    expect(rows.tables[0]).toMatchObject({ columns: ['name', 'age', 'team'], rows: [['Ada', '36', null], ['Alan', null, 'Bletchley']] })
    const config = await loadFile(file('config.json', JSON.stringify({ server: { host: 'db.local', ports: [5432, 5433] }, debug: false })))
    expect(config.tables).toEqual([])
    expect(config.documents[0]!.content).toBe('server:\n  host: db.local\n  ports:\n    - 5432\n    - 5433\ndebug: false')
    const lines = await loadFile(file('events.jsonl', '{"event":"login"}\n{"event":"logout"}\n'))
    expect(lines.tables[0]!.rows).toEqual([['login'], ['logout']])
  })
})

describe('importTable', () => {
  let db: PGlite
  beforeAll(async () => {
    db = await memoryDb()
  })
  afterAll(() => db.close())

  it('infers column types', () => {
    expect(inferType(['1', '-20', null])).toBe('integer')
    expect(inferType(['1', '3000000000'])).toBe('bigint')
    expect(inferType(['1.5', '2', '1e3'])).toBe('numeric')
    expect(inferType(['00123', '45'])).toBe('text')
    expect(inferType(['TRUE', 'false'])).toBe('boolean')
    expect(inferType(['2025-01-31', '2024-02-29'])).toBe('date')
    expect(inferType(['2025-02-30'])).toBe('text')
    expect(inferType(['2025-01-31 09:30', '2025-01-31T10:00:00'])).toBe('timestamp')
    expect(inferType(['2025-01-31T10:00:00Z'])).toBe('timestamptz')
    expect(inferType(['$1,200'])).toBe('text')
    expect(inferType([null, ''])).toBe('text')
  })

  it('makes SQL-friendly names', () => {
    expect(sqlIdentifier('Unit Price (€)', 'x')).toBe('unit_price')
    expect(sqlIdentifier('unitPrice', 'x')).toBe('unit_price')
    expect(sqlIdentifier('Growth %', 'x')).toBe('growth_pct')
    expect(sqlIdentifier('2025 Revenue', 'x')).toBe('_2025_revenue')
    expect(sqlIdentifier('Order', 'x')).toBe('order_')
    expect(sqlIdentifier('Café', 'x')).toBe('cafe')
    expect(sqlIdentifier('!!!', 'fallback')).toBe('fallback')
  })

  it('creates a typed, commented table the agent can query', async () => {
    const { tables } = await loadFile(file('Sales 2025.csv', 'Region,Revenue,Closed On,Zip,Won,Order\nWest,1200.50,2025-01-31,02134,true,1\nEast,800,2025-02-15,10001,false,2\n'))
    const result = await importTable(db, tables[0]!)
    expect(result.table).toBe('public.sales_2025')
    expect(result.columns.map((c) => `${c.name} ${c.type}`)).toEqual(['region text', 'revenue numeric', 'closed_on date', 'zip text', 'won boolean', 'order_ integer'])
    const { rows } = await db.query<{ total: string; zips: string }>(`select sum(revenue)::text as total, string_agg(zip, ',' order by zip) as zips from sales_2025 where won or closed_on < '2025-03-01'`)
    expect(rows[0]).toEqual({ total: '2000.50', zips: '02134,10001' })
    const comment = await db.query<{ c: string }>(`select obj_description('public.sales_2025'::regclass, 'pg_class') as c`)
    expect(comment.rows[0]!.c).toBe('Imported from Sales 2025.csv (Sales 2025).')
  })

  it('refuses to overwrite unless asked, and can append', async () => {
    const table = { name: 'pets', columns: ['Name'], rows: [['Rex'], ['Tom']] }
    await importTable(db, table)
    await expect(importTable(db, table)).rejects.toThrow(/already exists/)
    await importTable(db, table, { ifExists: 'append' })
    expect((await db.query('select * from pets')).rows).toHaveLength(4)
    await importTable(db, { ...table, rows: [['Kit']] }, { ifExists: 'replace' })
    expect((await db.query('select name from pets')).rows).toEqual([{ name: 'Kit' }])
  })

  it('inserts large tables in batches', async () => {
    const rows = Array.from({ length: 1234 }, (_, i) => [String(i), `item ${i}`])
    await importTable(db, { name: 'items', columns: ['id', 'label'], rows })
    expect((await db.query<{ n: number }>('select count(*)::int as n from items')).rows[0]!.n).toBe(1234)
  })
})

// ---------------------------------------------------------------------------
// Office
// ---------------------------------------------------------------------------

const W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
const p = (text: string, pPr = '') => `<w:p>${pPr ? `<w:pPr>${pPr}</w:pPr>` : ''}<w:r><w:t xml:space="preserve">${text}</w:t></w:r></w:p>`

function docx(): Uint8Array {
  return makeZip({
    '[Content_Types].xml': '<Types/>',
    'docProps/core.xml': '<cp:coreProperties xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Travel Handbook</dc:title></cp:coreProperties>',
    // A German template: style ids aren't "Heading1", but the names are.
    'word/styles.xml': `<w:styles ${W}>
      <w:style w:type="paragraph" w:styleId="berschrift1"><w:name w:val="heading 1"/></w:style>
      <w:style w:type="paragraph" w:styleId="berschrift2"><w:name w:val="heading 2"/></w:style>
      <w:style w:type="paragraph" w:styleId="MyHeading"><w:name w:val="My Heading"/><w:basedOn w:val="berschrift2"/></w:style>
    </w:styles>`,
    'word/numbering.xml': `<w:numbering ${W}>
      <w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/></w:lvl></w:abstractNum>
      <w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/></w:lvl><w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum>
      <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num><w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
    </w:numbering>`,
    'word/document.xml': `<w:document ${W}><w:body>
      ${p('Booking travel', '<w:pStyle w:val="berschrift1"/>')}
      <w:p><w:r><w:t>Book through the </w:t></w:r><w:hyperlink><w:r><w:t>travel portal</w:t></w:r></w:hyperlink><w:del><w:r><w:delText>old site</w:delText></w:r></w:del><w:r><w:fldChar/><w:instrText>HYPERLINK x</w:instrText><w:t>.</w:t><w:tab/><w:t>Ask</w:t><w:br/><w:t>early.</w:t></w:r></w:p>
      ${p('Approvals', '<w:pStyle w:val="MyHeading"/>')}
      ${p('Get manager sign-off', '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>')}
      ${p('Book within 24 hours', '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>')}
      ${p('Economy class', '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>')}
      ${p('Except flights over 6 hours', '<w:numPr><w:ilvl w:val="1"/><w:numId w:val="2"/></w:numPr>')}
      <w:tbl><w:tr><w:tc>${p('Region')}</w:tc><w:tc>${p('Per diem')}</w:tc></w:tr><w:tr><w:tc>${p('Europe')}</w:tc><w:tc>${p('€80')}${p('incl. meals')}</w:tc></w:tr></w:tbl>
      <w:sdt><w:sdtContent>${p('Inside a content control')}</w:sdtContent></w:sdt>
      ${p('   ')}
      <w:sectPr/>
    </w:body></w:document>`,
  })
}

const A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="r"'
const shape = (type: string | null, paragraphs: string[]) =>
  `<p:sp><p:nvSpPr>${type ? `<p:nvPr><p:ph type="${type}"/></p:nvPr>` : '<p:nvPr/>'}</p:nvSpPr><p:txBody>${paragraphs.join('')}</p:txBody></p:sp>`
const ap = (text: string, lvl = 0) => `<a:p>${lvl ? `<a:pPr lvl="${lvl}"/>` : ''}<a:r><a:t>${text}</a:t></a:r></a:p>`

function pptx(): Uint8Array {
  return makeZip({
    'ppt/presentation.xml': `<p:presentation ${A}><p:sldIdLst><p:sldId r:id="rId3"/><p:sldId r:id="rId2"/></p:sldIdLst></p:presentation>`,
    'ppt/_rels/presentation.xml.rels': '<Relationships><Relationship Id="rId2" Type=".../slide" Target="slides/slide1.xml"/><Relationship Id="rId3" Type=".../slide" Target="slides/slide2.xml"/></Relationships>',
    // slide2.xml is shown first.
    'ppt/slides/slide2.xml': `<p:sld ${A}><p:cSld><p:spTree>${shape('ctrTitle', [ap('Q3 Review')])}${shape('subTitle', [ap('Sales team')])}${shape('sldNum', [ap('1')])}</p:spTree></p:cSld></p:sld>`,
    'ppt/slides/slide1.xml': `<p:sld ${A}><p:cSld><p:spTree>${shape('title', [ap('Results')])}<p:grpSp>${shape('body', [ap('Revenue up 12%'), ap('Driven by EMEA', 1)])}${shape(null, [ap('Source: CRM export')])}</p:grpSp>
      <p:graphicFrame><a:graphic><a:graphicData><a:tbl><a:tr><a:tc><a:txBody>${ap('Region')}</a:txBody></a:tc><a:tc><a:txBody>${ap('Growth')}</a:txBody></a:tc></a:tr><a:tr><a:tc><a:txBody>${ap('EMEA')}</a:txBody></a:tc><a:tc><a:txBody>${ap('18%')}</a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>
    </p:spTree></p:cSld></p:sld>`,
    'ppt/slides/_rels/slide1.xml.rels': '<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/></Relationships>',
    'ppt/notesSlides/notesSlide1.xml': `<p:notes ${A}><p:cSld><p:spTree>${shape('sldImg', [])}${shape('body', [ap('Mention the Berlin deal.')])}</p:spTree></p:cSld></p:notes>`,
  })
}

function xlsx(prefix = ''): Uint8Array {
  const x = (tag: string) => `${prefix}${tag}`
  return makeZip({
    'xl/workbook.xml': `<${x('workbook')} xmlns:r="r"><${x('workbookPr')}/><${x('sheets')}><${x('sheet')} name="Orders" r:id="rId1"/><${x('sheet')} name="Empty" r:id="rId2"/><${x('sheet')} name="Notes" r:id="rId3"/></${x('sheets')}></${x('workbook')}>`,
    'xl/_rels/workbook.xml.rels': '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="ws"/><Relationship Id="rId2" Target="/xl/worksheets/sheet2.xml" Type="ws"/><Relationship Id="rId3" Target="worksheets/sheet3.xml" Type="ws"/></Relationships>',
    'xl/sharedStrings.xml': `<${x('sst')}><${x('si')}><${x('t')}>Customer</${x('t')}></${x('si')}><${x('si')}><${x('t')}>Date</${x('t')}></${x('si')}><${x('si')}><${x('r')}><${x('t')}>Acme </${x('t')}></${x('r')}><${x('r')}><${x('t')}>Corp</${x('t')}></${x('r')}></${x('si')}><${x('si')}><${x('t')}>Total</${x('t')}></${x('si')}></${x('sst')}>`,
    'xl/styles.xml': `<${x('styleSheet')}><${x('numFmts')}><${x('numFmt')} numFmtId="164" formatCode="[$-409]d\\-mmm\\-yy;@"/><${x('numFmt')} numFmtId="165" formatCode="&quot;$&quot;#,##0.00"/></${x('numFmts')}><${x('cellXfs')}><${x('xf')} numFmtId="0"/><${x('xf')} numFmtId="14"/><${x('xf')} numFmtId="164"/><${x('xf')} numFmtId="165"/><${x('xf')} numFmtId="22"/></${x('cellXfs')}></${x('styleSheet')}>`,
    'xl/worksheets/sheet1.xml': `<${x('worksheet')}><${x('sheetData')}>
      <${x('row')} r="1"><${x('c')} r="A1" t="s"><${x('v')}>0</${x('v')}></${x('c')}><${x('c')} r="B1" t="s"><${x('v')}>1</${x('v')}></${x('c')}><${x('c')} r="D1" t="s"><${x('v')}>3</${x('v')}></${x('c')}><${x('c')} r="E1" t="inlineStr"><${x('is')}><${x('t')}>Paid</${x('t')}></${x('is')}></${x('c')}></${x('row')}>
      <${x('row')} r="2"><${x('c')} r="A2" t="s"><${x('v')}>2</${x('v')}></${x('c')}><${x('c')} r="B2" s="1"><${x('v')}>45688</${x('v')}></${x('c')}><${x('c')} r="C2"><${x('v')}>0.30000000000000004</${x('v')}></${x('c')}><${x('c')} r="D2" s="3"><${x('v')}>1250.5</${x('v')}></${x('c')}><${x('c')} r="E2" t="b"><${x('v')}>1</${x('v')}></${x('c')}></${x('row')}>
      <${x('row')} r="4"><${x('c')} r="A4" t="str"><${x('v')}>Globex</${x('v')}></${x('c')}><${x('c')} r="B4" s="2"><${x('v')}>45700</${x('v')}></${x('c')}><${x('c')} r="D4" t="e"><${x('v')}>#DIV/0!</${x('v')}></${x('c')}><${x('c')} r="E4" s="4"><${x('v')}>45700.75</${x('v')}></${x('c')}></${x('row')}>
    </${x('sheetData')}></${x('worksheet')}>`,
    'xl/worksheets/sheet2.xml': `<${x('worksheet')}><${x('sheetData')}/></${x('worksheet')}>`,
    'xl/worksheets/sheet3.xml': `<${x('worksheet')}><${x('sheetData')}><${x('row')}><${x('c')} t="inlineStr"><${x('is')}><${x('t')}>Note</${x('t')}></${x('is')}></${x('c')}></${x('row')}><${x('row')}><${x('c')} t="inlineStr"><${x('is')}><${x('t')}>Call back</${x('t')}></${x('is')}></${x('c')}></${x('row')}></${x('sheetData')}></${x('worksheet')}>`,
  })
}

function epub(): Uint8Array {
  return makeZip(
    {
      mimetype: 'application/epub+zip',
      'META-INF/container.xml': '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
      'OEBPS/content.opf': `<package><metadata><dc:title>Field Manual</dc:title></metadata>
        <manifest><item id="c1" href="text/one.xhtml"/><item id="c2" href="text/two%20b.xhtml"/><item id="toc" href="nav.xhtml"/></manifest>
        <spine><itemref idref="c2"/><itemref idref="toc" linear="no"/><itemref idref="c1"/></spine></package>`,
      'OEBPS/text/one.xhtml': '<html><head><title>One</title></head><body><h1>Pumps</h1><p>Prime the pump first.</p></body></html>',
      'OEBPS/text/two b.xhtml': '<html><body><h1>Safety</h1><p>Wear gloves.</p></body></html>',
      'OEBPS/nav.xhtml': '<html><body><p>Table of contents</p></body></html>',
    },
    { store: ['mimetype'] },
  )
}

describe('Office and EPUB', () => {
  it('reads Word headings, lists, tables and tracked changes', async () => {
    const loaded = await loadFile(file('handbook.docx', docx()))
    expect(loaded.format).toBe('docx')
    expect(loaded.documents[0]!.title).toBe('Travel Handbook')
    expect(loaded.documents[0]!.content).toBe(
      [
        '# Booking travel',
        'Book through the travel portal. Ask\nearly.',
        '## Approvals',
        '1. Get manager sign-off\n2. Book within 24 hours\n- Economy class\n  - Except flights over 6 hours',
        '| Region | Per diem |\n| --- | --- |\n| Europe | €80 incl. meals |',
        'Inside a content control',
      ].join('\n\n'),
    )
  })

  it('finds headings and lists in Word files formatted by hand', async () => {
    const run = (text: string, sz?: number, tabs = false) =>
      `<w:r>${sz ? `<w:rPr><w:b/><w:sz w:val="${sz}"/></w:rPr>` : ''}${tabs ? '<w:tab/>' : ''}<w:t xml:space="preserve">${text}</w:t></w:r>`
    const para = (...runs: string[]) => `<w:p>${runs.join('')}</w:p>`
    const bytes = makeZip({
      'word/document.xml': `<w:document ${W}><w:body>
        ${para(run('Onboarding', 48))}
        ${para(run('Welcome to the team. This guide covers your first week, from equipment to payroll.', 24))}
        ${para(run('First day', 36))}
        ${para(run('1', undefined, true), '<w:r><w:tab/><w:t>Collect your laptop.</w:t></w:r>')}
        ${para(run('2', undefined, true), '<w:r><w:tab/><w:t>Meet your buddy.</w:t></w:r>')}
        ${para(run('• Bring photo ID.'))}
        ${para(run('Payroll runs on the 25th of every month, and the first payslip arrives after your first full month.', 24))}
      </w:body></w:document>`,
    })
    const loaded = await loadFile(file('onboarding.docx', bytes))
    expect(loaded.documents[0]!.content).toBe(
      [
        '# Onboarding',
        'Welcome to the team. This guide covers your first week, from equipment to payroll.',
        '## First day',
        '1. Collect your laptop.\n2. Meet your buddy.\n- Bring photo ID.',
        'Payroll runs on the 25th of every month, and the first payslip arrives after your first full month.',
      ].join('\n\n'),
    )
  })

  it('reads slides in presentation order, with tables and speaker notes', async () => {
    const loaded = await loadFile(file('q3.pptx', pptx()))
    expect(loaded.documents[0]!.title).toBe('Q3 Review')
    expect(loaded.documents[0]!.metadata).toMatchObject({ format: 'pptx', slides: 2 })
    expect(loaded.documents[0]!.content).toBe(
      [
        '## Slide 1: Q3 Review',
        'Sales team',
        '## Slide 2: Results',
        '- Revenue up 12%\n  - Driven by EMEA\nSource: CRM export\n| Region | Growth |\n| --- | --- |\n| EMEA | 18% |',
        'Speaker notes: Mention the Berlin deal.',
      ].join('\n\n'),
    )
    const noNotes = await loadFile(file('q3.pptx', pptx()), { slideNotes: false })
    expect(noNotes.documents[0]!.content).not.toContain('Berlin')
  })

  it.each([['plain', ''], ['prefixed', 'x:']])('reads Excel sheets with types and dates (%s)', async (_label, prefix) => {
    const loaded = await loadFile(file('orders.xlsx', xlsx(prefix)))
    expect(loaded.tables.map((t) => [t.name, t.sqlName])).toEqual([['Orders', 'orders'], ['Notes', 'orders_notes']])
    expect(loaded.tables[0]).toMatchObject({
      columns: ['Customer', 'Date', 'column_3', 'Total', 'Paid'],
      rows: [
        ['Acme Corp', '2025-01-31', '0.3', '1250.5', 'TRUE'],
        ['Globex', '2025-02-12', null, null, '2025-02-12 18:00'],
      ],
      source: 'orders.xlsx',
    })
    expect(loaded.documents[0]!.content).toContain('Customer: Acme Corp; Date: 2025-01-31; column_3: 0.3; Total: 1250.5; Paid: TRUE')
    expect(loaded.documents[0]!.metadata).toMatchObject({ tables: ['Orders', 'Notes'] })
  })

  it('reads EPUB chapters in spine order', async () => {
    const loaded = await loadFile(file('manual.epub', epub()))
    expect(loaded.format).toBe('epub')
    expect(loaded.documents[0]).toMatchObject({ title: 'Field Manual', content: '# Safety\n\nWear gloves.\n\n# Pumps\n\nPrime the pump first.' })
  })
})

// ---------------------------------------------------------------------------
// PDF
// ---------------------------------------------------------------------------

function handbookPdf(): Uint8Array {
  const header = { text: 'ACME Corp — Employee Handbook', size: 8, y: 770 }
  const footer = (n: number) => ({ text: `Page ${n} of 3`, size: 8, y: 30 })
  return makePdf(
    [
      [
        header,
        { text: 'Leave policy', size: 20, y: 700 },
        ...para(['Employees accrue fifteen vacation days per year. Re-', 'quests go to your manager at least two weeks in', 'advance.'], 660),
        ...para(['Unused days carry over once.'], 600),
        { text: 'Sick leave', size: 14, y: 560 },
        ...para(['• Up to ten paid days', '• A doctor\'s note after three days'], 530),
        footer(1),
      ],
      [
        header,
        ...para(['Parental leave is sixteen weeks at full pay.'], 700),
        // A table: cells on one line, separated by wide gaps.
        { text: 'Region', size: 11, y: 660 }, { text: 'Days', size: 11, y: 660, x: 200 },
        { text: 'Europe', size: 11, y: 646 }, { text: '25', size: 11, y: 646, x: 200 },
        footer(2),
      ],
      [header, footer(3)],
    ],
    'Employee Handbook 2026',
  )
}

describe('PDF', () => {
  it('rebuilds headings, paragraphs and bullets, without running headers and page numbers', async () => {
    const loaded = await loadFile(file('handbook.pdf', handbookPdf()), { pdf })
    expect(loaded.format).toBe('pdf')
    expect(loaded.documents).toHaveLength(1)
    expect(loaded.documents[0]!.title).toBe('Employee Handbook 2026')
    expect(loaded.documents[0]!.metadata).toMatchObject({ format: 'pdf', pages: 3 })
    expect(loaded.documents[0]!.content).toBe(
      [
        '# Leave policy',
        'Employees accrue fifteen vacation days per year. Requests go to your manager at least two weeks in advance.',
        'Unused days carry over once.',
        '## Sick leave',
        "- Up to ten paid days\n- A doctor's note after three days",
        'Parental leave is sixteen weeks at full pay.',
        '| Region | Days |\n| --- | --- |\n| Europe | 25 |',
      ].join('\n\n'),
    )
    // Page 3 has no text: it's reported as possibly scanned.
    expect(loaded.warnings[0]).toMatch(/1 page\(s\) have no text layer \(3\)/)
  })

  it('splits into one document per page for page citations', async () => {
    const loaded = await loadFile(file('handbook.pdf', handbookPdf()), { pdf, pdfSplit: 'page' })
    expect(loaded.documents.map((d) => [d.id, d.title, d.metadata?.page])).toEqual([
      ['handbook.pdf#page=1', 'Employee Handbook 2026 (page 1)', 1],
      ['handbook.pdf#page=2', 'Employee Handbook 2026 (page 2)', 2],
    ])
  })

  it('explains how to enable PDFs when pdf.js is missing', async () => {
    await expect(loadFile(file('a.pdf', handbookPdf()))).rejects.toThrow(/pdfjs-dist/)
  })

  it('warns about scanned PDFs', async () => {
    const blank = makePdf([[], []])
    const plain = await loadFile(file('scan.pdf', blank), { pdf })
    expect(plain.documents).toEqual([])
    expect(plain.warnings[0]).toMatch(/scanned PDF with no text layer/)
  })
})

// ---------------------------------------------------------------------------
// Detection, images and batches
// ---------------------------------------------------------------------------

describe('loadFile and loadFiles', () => {
  it('detects formats from content first, then the name', async () => {
    const bytes = (s: string) => new TextEncoder().encode(s)
    expect(await detectFormat('report', handbookPdf())).toBe('pdf')
    expect(await detectFormat('renamed.zip', docx())).toBe('docx')
    expect(await detectFormat('deck.bin', pptx())).toBe('pptx')
    expect(await detectFormat('x', xlsx())).toBe('xlsx')
    expect(await detectFormat('book', epub())).toBe('epub')
    expect(await detectFormat('notes.md', bytes('# Hi'))).toBe('markdown')
    expect(await detectFormat('main.ts', bytes('export const a = 1'))).toBe('text')
    expect(await detectFormat('page', bytes('<p>hi</p>'), 'text/html')).toBe('html')
    expect(await detectFormat('photo', new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 0]))).toBe('image')
    expect(await detectFormat('blob.bin', new Uint8Array([1, 2, 0, 4]))).toBeUndefined()
  })

  it('reads Markdown front matter titles', async () => {
    const loaded = await loadFile(file('wifi.md', '---\ntitle: "Guest wifi"\ntags: [it]\n---\nNetwork: NorthGuest\n'))
    expect(loaded.documents[0]).toMatchObject({ title: 'Guest wifi', content: 'Network: NorthGuest\n' })
  })

  it('reads images only with OCR', async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    const without = await loadFile(file('receipt.png', png))
    expect(without.documents).toEqual([])
    expect(without.warnings[0]).toMatch(/Pass an `ocr` function/)
    const seen: string[] = []
    const withOcr = await loadFile(file('receipt.png', png, 'image/png'), {
      ocr: async (image, info) => (seen.push(`${info.name}:${image.type}`), 'Total: $42.10'),
    })
    expect(seen).toEqual(['receipt.png:image/png'])
    expect(withOcr.documents[0]).toMatchObject({ title: 'receipt', content: 'Total: $42.10', metadata: { ocr: true } })
  })

  it('accepts browser File objects', async () => {
    const f = new File(['Hello from a file'], 'hello.txt', { type: 'text/plain', lastModified: Date.UTC(2026, 0, 2) })
    const loaded = await loadFile(f)
    expect(loaded.documents[0]).toMatchObject({ title: 'hello', content: 'Hello from a file', metadata: { modified: '2026-01-02T00:00:00.000Z' } })
  })

  it('rejects legacy Office and unknown binaries with a clear message', async () => {
    await expect(loadFile(file('old.doc', new Uint8Array([0xd0, 0xcf, 0x11, 0xe0])))).rejects.toThrow(/Save it as \.docx/)
    await expect(loadFile(file('blob.bin', new Uint8Array([1, 2, 0, 4])))).rejects.toBeInstanceOf(UnsupportedFileError)
    await expect(loadFile(file('big.txt', 'x'.repeat(100)), { maxBytes: 10 })).rejects.toThrow(/over the/)
  })

  it('loads a batch, keeping going past bad files', async () => {
    const progress: number[] = []
    const result = await loadFiles(
      [file('a.md', '# A\nalpha'), file('bad.bin', new Uint8Array([0, 1, 2])), file('b.csv', 'x\n1\n'), file('c.docx', docx())],
      { collection: 'uploads', onProgress: ({ done }) => progress.push(done) },
    )
    expect(progress).toEqual([1, 2, 3, 4])
    expect(result.documents.map((d) => d.source)).toEqual(['a.md', 'b.csv', 'c.docx'])
    expect(result.documents.every((d) => d.collection === 'uploads')).toBe(true)
    expect(result.tables.map((t) => t.name)).toEqual(['b'])
    expect(result.errors.map((e) => e.name)).toEqual(['bad.bin'])
  })

  it('produces documents the knowledge base can search', async () => {
    const db = await memoryDb()
    await migrate(db, 'core', CORE_MIGRATIONS)
    const kb = new Knowledge(db, hashEmbedder(128))
    const { documents } = await loadFiles([file('handbook.docx', docx()), file('handbook.pdf', handbookPdf())], { pdf })
    const result = await kb.ingest(documents)
    expect(result.documents).toBe(2)
    const hits = await kb.search('parental leave weeks', { mode: 'keyword' })
    expect(hits[0]?.title).toBe('Employee Handbook 2026')
    expect((await kb.ingest(documents)).skipped).toBe(2)
    await db.close()
  })
})
