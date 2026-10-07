import { htmlToMarkdown } from './html.js'
import { toTable, type TableData } from './tables.js'
import { child, childElements, find, findAll, isElement, parseXml, textOf, type XmlElement } from './xml.js'
import { resolvePath, type ZipArchive } from './zip.js'

const xml = async (zip: ZipArchive, path: string, stripPrefixes = false) => {
  const text = await zip.text(path)
  return text === undefined ? undefined : parseXml(text, { stripPrefixes })
}

/** Relationship id → target path, for a part such as `word/document.xml`. */
async function relationships(zip: ZipArchive, part: string): Promise<Map<string, { target: string; type: string }>> {
  const slash = part.lastIndexOf('/')
  const relsPath = `${part.slice(0, slash + 1)}_rels/${part.slice(slash + 1)}.rels`
  const doc = await xml(zip, relsPath)
  const map = new Map<string, { target: string; type: string }>()
  for (const r of findAll(doc, 'Relationship')) {
    if (r.attrs.TargetMode === 'External') continue
    map.set(r.attrs.Id ?? '', { target: resolvePath(part, r.attrs.Target ?? ''), type: r.attrs.Type ?? '' })
  }
  return map
}

/** `dc:title` from the package properties, when the author set one. */
async function coreTitle(zip: ZipArchive): Promise<string | undefined> {
  const title = textOf(find(await xml(zip, 'docProps/core.xml'), 'dc:title')).trim()
  return title || undefined
}

const markdownRow = (cells: string[]) => `| ${cells.map((c) => c.replace(/\s+/g, ' ').trim().replace(/\|/g, '\\|')).join(' | ')} |`

function markdownTable(rows: string[][]): string {
  const filled = rows.filter((r) => r.some((c) => c.trim()))
  if (!filled.length) return ''
  const width = Math.max(...filled.map((r) => r.length))
  const padded = filled.map((r) => [...r, ...Array(width - r.length).fill('')])
  return [markdownRow(padded[0]!), markdownRow(padded[0]!.map(() => '---')), ...padded.slice(1).map(markdownRow)].join('\n')
}

// ---------------------------------------------------------------------------
// Word (.docx)
// ---------------------------------------------------------------------------

export async function docxToMarkdown(zip: ZipArchive): Promise<{ title?: string; text: string }> {
  const doc = await xml(zip, 'word/document.xml')
  if (!doc) throw new Error('Not a Word document (word/document.xml is missing).')
  const styles = await xml(zip, 'word/styles.xml')
  const headingOf = headingStyles(styles)
  const sizeOf = styleSizes(styles)
  const listFormats = numberingFormats(await xml(zip, 'word/numbering.xml'))
  const counters = new Map<string, number>()
  /** `size` (half-points) is kept for plain paragraphs, to find headings made with direct formatting. */
  const blocks: { text: string; list: boolean; size?: number }[] = []

  const paragraph = (p: XmlElement) => {
    const pPr = child(p, 'w:pPr')
    const raw = runText(p)
    const text = raw.replace(/\t+/g, ' ').replace(/ {2,}/g, ' ').trim()
    if (!text) return
    const styleId = child(pPr, 'w:pStyle')?.attrs['w:val']
    const outline = child(pPr, 'w:outlineLvl')?.attrs['w:val']
    const level = outline !== undefined && Number(outline) < 6 ? Number(outline) + 1 : styleId ? headingOf.get(styleId) : undefined
    if (level) {
      blocks.push({ text: `${'#'.repeat(level)} ${text.replace(/\n+/g, ' ')}`, list: false })
      return
    }
    const numPr = child(pPr, 'w:numPr')
    const numId = child(numPr, 'w:numId')?.attrs['w:val']
    if (numPr && numId && numId !== '0') {
      const ilvl = Number(child(numPr, 'w:ilvl')?.attrs['w:val'] ?? 0)
      const ordered = listFormats.get(`${numId}:${ilvl}`) === 'ordered'
      const key = `${numId}:${ilvl}`
      const n = (counters.get(key) ?? 0) + 1
      counters.set(key, n)
      // A new item at a shallower level restarts deeper counters.
      for (const k of counters.keys()) if (k.startsWith(`${numId}:`) && Number(k.split(':')[1]) > ilvl) counters.delete(k)
      blocks.push({ text: `${'  '.repeat(ilvl)}${ordered ? `${n}.` : '-'} ${text}`, list: true })
      return
    }
    // Lists typed by hand: "1<tab>Item", "•<tab>Item", "- Item".
    const typedNumber = /^\s*(\d{1,3}|[a-zA-Z])[.)]?\t+/.exec(raw)
    if (typedNumber) {
      blocks.push({ text: `${typedNumber[1]}. ${text.slice(typedNumber[1]!.length).replace(/^[.)]?\s*/, '')}`, list: true })
      return
    }
    if (TYPED_BULLET.test(text)) {
      blocks.push({ text: text.replace(TYPED_BULLET, '- '), list: true })
      return
    }
    const defaultSize = sizeOf(styleId)
    const sizes = findAll(p, 'w:r')
      .filter((r) => textOf(child(r, 'w:t')).trim())
      .map((r) => Number(child(child(r, 'w:rPr'), 'w:sz')?.attrs['w:val'] ?? defaultSize))
    blocks.push({ text, list: false, size: sizes.length ? Math.max(...sizes) : defaultSize })
  }

  const table = (tbl: XmlElement) => {
    const rows = childElements(tbl, 'w:tr').map((tr) =>
      childElements(tr, 'w:tc').map((tc) =>
        findAll(tc, 'w:p')
          .map((p) => runText(p).replace(/\s+/g, ' ').trim())
          .filter(Boolean)
          .join(' '),
      ),
    )
    const md = markdownTable(rows)
    if (md) blocks.push({ text: md, list: false })
  }

  const walk = (el: XmlElement) => {
    for (const c of childElements(el)) {
      if (c.name === 'w:p') paragraph(c)
      else if (c.name === 'w:tbl') table(c)
      else if (c.name === 'w:sdt' || c.name === 'w:sdtContent' || c.name === 'w:customXml') walk(c)
    }
  }
  walk(find(doc, 'w:body') ?? doc)
  inferHeadings(blocks)

  const text = blocks.map((b, i) => (i === 0 ? '' : b.list && blocks[i - 1]!.list ? '\n' : '\n\n') + b.text).join('')
  const title = (await coreTitle(zip)) ?? /^# (.+)$/m.exec(text)?.[1]
  return { ...(title ? { title } : {}), text }
}

const TYPED_BULLET = /^[•●▪◦‣■□➢►✓–*-]\s+/

/**
 * Documents written without heading styles (common from converters and
 * hand-formatted files) mark headings with a larger font. Short paragraphs
 * set larger than the body text become headings, ranked by size.
 */
function inferHeadings(blocks: { text: string; size?: number }[]) {
  const weight = new Map<number, number>()
  for (const b of blocks) if (b.size) weight.set(b.size, (weight.get(b.size) ?? 0) + b.text.length)
  const body = [...weight.entries()].sort((a, b) => b[1] - a[1])[0]?.[0]
  if (!body) return
  const isHeading = (b: { text: string; size?: number }) => !!b.size && b.size >= body * 1.15 && b.text.length <= 150 && !b.text.includes('\n')
  const sizes = [...new Set(blocks.filter(isHeading).map((b) => b.size!))].sort((a, b) => b - a)
  for (const b of blocks) if (isHeading(b)) b.text = `${'#'.repeat(Math.min(3, sizes.indexOf(b.size!) + 1))} ${b.text}`
}

/** Paragraph style id → font size in half-points (following `basedOn`), with the document default. */
function styleSizes(styles: XmlElement | undefined): (styleId?: string) => number {
  const fallback = Number(find(find(styles, 'w:rPrDefault'), 'w:sz')?.attrs['w:val'] ?? 22)
  const info = new Map<string, { size?: number; basedOn?: string }>()
  for (const s of findAll(styles, 'w:style')) {
    const size = child(child(s, 'w:rPr'), 'w:sz')?.attrs['w:val']
    info.set(s.attrs['w:styleId'] ?? '', {
      ...(size ? { size: Number(size) } : {}),
      ...(child(s, 'w:basedOn') ? { basedOn: child(s, 'w:basedOn')!.attrs['w:val']! } : {}),
    })
  }
  const sizeOf = (id: string | undefined, depth = 0): number => {
    const s = id ? info.get(id) : undefined
    if (s?.size) return s.size
    return s?.basedOn && depth < 5 ? sizeOf(s.basedOn, depth + 1) : fallback
  }
  return (id) => sizeOf(id ?? 'Normal')
}

/** Text of a paragraph's runs, including hyperlinks and tracked insertions; skips deletions and field codes. */
function runText(el: XmlElement): string {
  let out = ''
  for (const c of el.children) {
    if (!isElement(c)) continue
    switch (c.name) {
      case 'w:t':
        out += textOf(c)
        break
      case 'w:tab':
        out += '\t'
        break
      case 'w:br':
      case 'w:cr':
        out += '\n'
        break
      case 'w:noBreakHyphen':
        out += '-'
        break
      case 'w:del':
      case 'w:instrText':
      case 'w:pPr':
      case 'w:rPr':
      case 'w:footnoteReference':
      case 'w:endnoteReference':
        break
      default:
        out += runText(c)
    }
  }
  return out
}

/** Paragraph style id → heading level, from style names ("heading 2", "Title") or outline levels. */
function headingStyles(styles: XmlElement | undefined): Map<string, number> {
  const info = new Map<string, { name: string; outline?: number; basedOn?: string }>()
  for (const s of findAll(styles, 'w:style')) {
    if (s.attrs['w:type'] !== 'paragraph') continue
    const outline = find(s, 'w:outlineLvl')?.attrs['w:val']
    info.set(s.attrs['w:styleId'] ?? '', {
      name: (child(s, 'w:name')?.attrs['w:val'] ?? '').toLowerCase(),
      ...(outline !== undefined ? { outline: Number(outline) } : {}),
      ...(child(s, 'w:basedOn') ? { basedOn: child(s, 'w:basedOn')!.attrs['w:val']! } : {}),
    })
  }
  const levelOf = (id: string, depth = 0): number | undefined => {
    const s = info.get(id)
    const name = s?.name ?? id.toLowerCase()
    const m = /^heading ?(\d)$/.exec(name)
    if (m) return Math.min(6, Number(m[1]))
    if (name === 'title') return 1
    if (name === 'subtitle') return 2
    if (s?.outline !== undefined && s.outline < 6) return s.outline + 1
    return s?.basedOn && depth < 5 ? levelOf(s.basedOn, depth + 1) : undefined
  }
  const map = new Map<string, number>()
  for (const id of info.keys()) {
    const level = levelOf(id)
    if (level) map.set(id, level)
  }
  // Documents without styles.xml still use the built-in ids.
  for (let i = 1; i <= 6; i++) if (!map.has(`Heading${i}`)) map.set(`Heading${i}`, i)
  if (!map.has('Title')) map.set('Title', 1)
  return map
}

/** `numId:level` → whether that list level is numbered or bulleted. */
function numberingFormats(numbering: XmlElement | undefined): Map<string, 'ordered' | 'bullet'> {
  const abstract = new Map<string, Map<number, string>>()
  for (const a of findAll(numbering, 'w:abstractNum')) {
    const levels = new Map<number, string>()
    for (const lvl of childElements(a, 'w:lvl')) levels.set(Number(lvl.attrs['w:ilvl'] ?? 0), child(lvl, 'w:numFmt')?.attrs['w:val'] ?? 'bullet')
    abstract.set(a.attrs['w:abstractNumId'] ?? '', levels)
  }
  const map = new Map<string, 'ordered' | 'bullet'>()
  for (const num of findAll(numbering, 'w:num')) {
    const levels = abstract.get(child(num, 'w:abstractNumId')?.attrs['w:val'] ?? '')
    for (const [level, fmt] of levels ?? []) map.set(`${num.attrs['w:numId']}:${level}`, fmt === 'bullet' || fmt === 'none' ? 'bullet' : 'ordered')
  }
  return map
}

// ---------------------------------------------------------------------------
// PowerPoint (.pptx)
// ---------------------------------------------------------------------------

/** One slide's text: its 1-based position in the deck and title, for citing it. */
export interface SlideSection {
  slide: number
  title?: string
  text: string
}

export async function pptxToMarkdown(
  zip: ZipArchive,
  options: { notes?: boolean } = {},
): Promise<{ title?: string; text: string; slides: number; sections: SlideSection[] }> {
  const slides = await slideOrder(zip)
  const sections: string[] = []
  const bySlide: SlideSection[] = []
  let deckTitle = await coreTitle(zip)
  for (const [i, path] of slides.entries()) {
    const slide = await xml(zip, path)
    const tree = find(slide, 'p:spTree')
    let title = ''
    const body: string[] = []
    const visit = (el: XmlElement) => {
      for (const c of childElements(el)) {
        if (c.name === 'p:grpSp') visit(c)
        else if (c.name === 'p:sp') {
          const type = find(c, 'p:ph')?.attrs.type
          const paragraphs = shapeParagraphs(c)
          if (!paragraphs.length || type === 'sldNum' || type === 'dt' || type === 'ftr') continue
          if ((type === 'title' || type === 'ctrTitle') && !title) title = paragraphs.map((p) => p.text).join(' ')
          else body.push(...paragraphs.map((p) => `${p.marker ? '  '.repeat(p.level) : ''}${p.marker}${p.text}`))
        } else if (c.name === 'p:graphicFrame') {
          const tbl = find(c, 'a:tbl')
          if (tbl) body.push(markdownTable(childElements(tbl, 'a:tr').map((tr) => childElements(tr, 'a:tc').map((tc) => textOf(tc)))))
        }
      }
    }
    if (tree) visit(tree)
    if (i === 0 && !deckTitle && title) deckTitle = title
    let notes = ''
    if (options.notes !== false) {
      const rels = await relationships(zip, path)
      const notesPath = [...rels.values()].find((r) => r.type.endsWith('/notesSlide'))?.target
      if (notesPath) {
        const notesXml = await xml(zip, notesPath)
        notes = findAll(notesXml, 'p:sp')
          .filter((sp) => find(sp, 'p:ph')?.attrs.type === 'body')
          .flatMap((sp) => shapeParagraphs(sp).map((p) => p.text))
          .join(' ')
      }
    }
    const heading = `## Slide ${i + 1}${title ? `: ${title}` : ''}`
    const parts = [heading, body.join('\n'), notes ? `Speaker notes: ${notes}` : ''].filter(Boolean)
    if (parts.length > 1 || title) {
      sections.push(parts.join('\n\n'))
      bySlide.push({ slide: i + 1, ...(title ? { title } : {}), text: parts.join('\n\n') })
    }
  }
  return { ...(deckTitle ? { title: deckTitle } : {}), text: sections.join('\n\n'), slides: slides.length, sections: bySlide }
}

/**
 * Paragraphs of a shape. Body placeholders are bulleted by default; text
 * boxes aren't. `a:buNone`, `a:buChar` and `a:buAutoNum` override either.
 */
function shapeParagraphs(shape: XmlElement): { text: string; level: number; marker: string }[] {
  const ph = find(shape, 'p:ph')
  const bulletedByDefault = !!ph && !['title', 'ctrTitle', 'subTitle'].includes(ph.attrs.type ?? 'body')
  const numbers = new Map<number, number>()
  return childElements(find(shape, 'p:txBody'), 'a:p')
    .map((p) => {
      const pPr = child(p, 'a:pPr')
      const level = Number(pPr?.attrs.lvl ?? 0)
      const text = p.children
        .filter(isElement)
        .map((r) => (r.name === 'a:br' ? ' ' : r.name === 'a:r' || r.name === 'a:fld' ? textOf(child(r, 'a:t')) : ''))
        .join('')
        .replace(/\s+/g, ' ')
        .trim()
      let marker = ''
      if (child(pPr, 'a:buAutoNum')) {
        const n = (numbers.get(level) ?? 0) + 1
        numbers.set(level, n)
        marker = `${n}. `
      } else if (child(pPr, 'a:buChar') || (bulletedByDefault && !child(pPr, 'a:buNone'))) marker = '- '
      if (!child(pPr, 'a:buAutoNum')) numbers.delete(level)
      return { text, level, marker }
    })
    .filter((p) => p.text)
}

/** Slide parts in presentation order (falls back to file-name order). */
async function slideOrder(zip: ZipArchive): Promise<string[]> {
  const presentation = await xml(zip, 'ppt/presentation.xml')
  const rels = await relationships(zip, 'ppt/presentation.xml')
  const ordered = findAll(presentation, 'p:sldId')
    .map((s) => rels.get(s.attrs['r:id'] ?? '')?.target)
    .filter((p): p is string => !!p && zip.has(p))
  if (ordered.length) return ordered
  return zip
    .names()
    .filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))
    .sort((a, b) => Number(/(\d+)\.xml$/.exec(a)![1]) - Number(/(\d+)\.xml$/.exec(b)![1]))
}

// ---------------------------------------------------------------------------
// Excel (.xlsx)
// ---------------------------------------------------------------------------

export async function xlsxToTables(zip: ZipArchive, source?: string): Promise<TableData[]> {
  const workbook = await xml(zip, 'xl/workbook.xml', true)
  if (!workbook) throw new Error('Not an Excel workbook (xl/workbook.xml is missing).')
  const rels = await relationships(zip, 'xl/workbook.xml')
  const date1904 = /^(1|true)$/.test(find(workbook, 'workbookPr')?.attrs.date1904 ?? '')
  const shared = findAll(await xml(zip, 'xl/sharedStrings.xml', true), 'si').map(stringItem)
  const dateStyles = dateStyleIndexes(await xml(zip, 'xl/styles.xml', true))

  const tables: TableData[] = []
  for (const sheet of findAll(workbook, 'sheet')) {
    const path = rels.get(sheet.attrs['r:id'] ?? '')?.target
    const sheetXml = path ? await xml(zip, path, true) : undefined
    if (!sheetXml) continue
    const grid: (string | null)[][] = []
    for (const row of findAll(sheetXml, 'row')) {
      const cells: (string | null)[] = []
      let next = 0
      for (const c of childElements(row, 'c')) {
        const col = c.attrs.r ? columnIndex(c.attrs.r) : next
        next = col + 1
        cells[col] = cellValue(c, shared, dateStyles, date1904)
      }
      if (cells.some((v) => v != null && v !== '')) grid.push(Array.from(cells, (v) => v ?? null))
    }
    if (grid.length) tables.push(toTable(sheet.attrs.name ?? `Sheet${tables.length + 1}`, grid, source))
  }
  return tables
}

/** Shared or inline string: plain text plus rich-text runs, without phonetic hints. */
function stringItem(si: XmlElement): string {
  return si.children
    .filter(isElement)
    .map((c) => (c.name === 't' ? textOf(c) : c.name === 'r' ? textOf(child(c, 't')) : ''))
    .join('')
}

function cellValue(c: XmlElement, shared: string[], dateStyles: Set<number>, date1904: boolean): string | null {
  const type = c.attrs.t ?? 'n'
  if (type === 'inlineStr') return stringItem(child(c, 'is') ?? c) || null
  const v = textOf(child(c, 'v'))
  if (v === '') return null
  switch (type) {
    case 's':
      return shared[Number(v)] ?? null
    case 'b':
      return v === '1' ? 'TRUE' : 'FALSE'
    case 'e':
      return null
    case 'str':
    case 'd':
      return v
    default: {
      const n = Number(v)
      if (!Number.isFinite(n)) return v
      if (dateStyles.has(Number(c.attrs.s ?? 0))) return excelDate(n, date1904)
      // Excel stores binary floats ("0.30000000000000004"); show what the user typed.
      return Number.isInteger(n) ? String(n) : String(parseFloat(n.toPrecision(15)))
    }
  }
}

function columnIndex(ref: string): number {
  let n = 0
  for (const ch of ref.replace(/\d+$/, '').toUpperCase()) n = n * 26 + (ch.charCodeAt(0) - 64)
  return n - 1
}

/** ISO date (or date-time) from an Excel serial number. */
export function excelDate(serial: number, date1904 = false): string {
  const ms = Math.round(((date1904 ? serial + 1462 : serial) - 25569) * 86_400_000)
  const iso = new Date(ms).toISOString()
  if (serial < 1 && !date1904) return iso.slice(11, 16)
  const time = iso.slice(11, 19)
  return time === '00:00:00' ? iso.slice(0, 10) : `${iso.slice(0, 10)} ${time.endsWith(':00') ? time.slice(0, 5) : time}`
}

/** Indexes into `cellXfs` whose number format displays a date or time. */
function dateStyleIndexes(styles: XmlElement | undefined): Set<number> {
  const custom = new Map<number, string>()
  for (const f of findAll(styles, 'numFmt')) custom.set(Number(f.attrs.numFmtId), f.attrs.formatCode ?? '')
  const isDate = (id: number) => {
    if ((id >= 14 && id <= 22) || (id >= 45 && id <= 47) || (id >= 27 && id <= 36) || (id >= 50 && id <= 58)) return true
    const code = custom.get(id)
    if (!code) return false
    // Drop quoted literals, escapes and [color]/[locale] sections before looking for date tokens.
    const bare = code.replace(/"[^"]*"|\\.|\[[^\]]*\]/g, '')
    return /[dmyhs]/i.test(bare) && !/^general$/i.test(bare.trim())
  }
  const set = new Set<number>()
  childElements(find(styles, 'cellXfs'), 'xf').forEach((xf, i) => {
    if (isDate(Number(xf.attrs.numFmtId ?? 0))) set.add(i)
  })
  return set
}

// ---------------------------------------------------------------------------
// EPUB
// ---------------------------------------------------------------------------

/**
 * One EPUB chapter: its 1-based position among the spine's linear items (what
 * readers page through), its path in the archive and its first heading.
 */
export interface ChapterSection {
  chapter: number
  href: string
  title?: string
  text: string
}

export async function epubToMarkdown(zip: ZipArchive): Promise<{ title?: string; text: string; chapters: ChapterSection[] }> {
  const container = await xml(zip, 'META-INF/container.xml')
  const opfPath = find(container, 'rootfile')?.attrs['full-path']
  const opf = opfPath ? await xml(zip, opfPath) : undefined
  if (!opfPath || !opf) throw new Error('Not a valid EPUB (no package document).')
  const manifest = new Map<string, string>()
  for (const item of findAll(opf, 'item')) manifest.set(item.attrs.id ?? '', resolvePath(opfPath, item.attrs.href ?? ''))
  const chapters: ChapterSection[] = []
  let position = 0
  for (const ref of findAll(opf, 'itemref')) {
    if (ref.attrs.linear === 'no') continue
    const path = manifest.get(ref.attrs.idref ?? '')
    position++
    const html = path ? await zip.text(path) : undefined
    if (!html || !path) continue
    const { text } = htmlToMarkdown(html)
    const heading = /^#{1,3} (.+)$/m.exec(text)?.[1]?.trim()
    if (text) chapters.push({ chapter: position, href: path, ...(heading ? { title: heading } : {}), text })
  }
  const title = textOf(find(opf, 'dc:title')).trim()
  return { ...(title ? { title } : {}), text: chapters.map((c) => c.text).join('\n\n'), chapters }
}
