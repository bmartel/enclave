/**
 * PDF text extraction with pdf.js (`pdfjs-dist`), supplied by the app so its
 * bundler serves the library and its worker from your own origin.
 *
 * Rebuilds reading text from positioned glyph runs: lines, paragraphs,
 * headings (by font size), bullets and de-hyphenation, and drops running
 * headers, footers and page numbers.
 */

/** The parts of the pdf.js module this loader uses. */
export interface PdfJsLike {
  getDocument(params: Record<string, unknown>): { promise: Promise<PdfDocumentLike>; destroy(): Promise<void> }
  GlobalWorkerOptions: { workerSrc: string; workerPort: unknown }
}
interface PdfDocumentLike {
  numPages: number
  getPage(n: number): Promise<PdfPageLike>
  getMetadata(): Promise<{ info?: object }>
}
interface PdfPageLike {
  getTextContent(): Promise<{ items: unknown[] }>
  getViewport(params: { scale: number }): { width: number; height: number }
  render(params: Record<string, unknown>): { promise: Promise<void> }
}
interface TextItemLike {
  str: string
  transform: number[]
  width: number
  height: number
  hasEOL: boolean
}

export interface PdfOptions {
  /** The pdf.js module, or a function that imports it: `() => import('pdfjs-dist')`. */
  lib?: PdfJsLike | (() => Promise<PdfJsLike>)
  /** URL of pdf.js's worker script, e.g. `import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'`. */
  workerSrc?: string
  /** A ready-made pdf.js worker; overrides `workerSrc`. */
  workerPort?: Worker
  /** Base URL of pdf.js's CMaps (needed for some CJK PDFs). Self-host for offline use. */
  cMapUrl?: string
  /** Base URL of pdf.js's standard font data. */
  standardFontDataUrl?: string
  password?: string
}

export interface PdfPage {
  page: number
  text: string
}

export interface PdfResult {
  title?: string
  pages: PdfPage[]
  pageCount: number
  /** Page numbers with no text layer (scanned); filled by OCR when available. */
  scanned: number[]
}

export type OcrFunction = (image: Blob, info: { name: string; page?: number }) => Promise<string>

export async function pdfToText(data: Uint8Array, name: string, options: PdfOptions = {}, ocr?: OcrFunction): Promise<PdfResult> {
  if (!options.lib) {
    throw new Error(
      `Reading PDFs needs pdf.js. Install pdfjs-dist and pass pdf: { lib: () => import('pdfjs-dist'), workerSrc } (see the loaders docs).`,
    )
  }
  const lib = typeof options.lib === 'function' ? await options.lib() : options.lib
  if (options.workerPort) lib.GlobalWorkerOptions.workerPort = options.workerPort
  else if (options.workerSrc) lib.GlobalWorkerOptions.workerSrc = options.workerSrc

  const task = lib.getDocument({
    // pdf.js transfers the buffer to its worker; keep the caller's copy intact.
    data: new Uint8Array(data),
    ...(options.password ? { password: options.password } : {}),
    ...(options.cMapUrl ? { cMapUrl: options.cMapUrl, cMapPacked: true } : {}),
    ...(options.standardFontDataUrl ? { standardFontDataUrl: options.standardFontDataUrl } : {}),
    isEvalSupported: false,
    disableFontFace: true,
    useSystemFonts: false,
    verbosity: 0,
  })
  const doc = await task.promise
  try {
    const pageLines: Line[][] = []
    for (let n = 1; n <= doc.numPages; n++) {
      const page = await doc.getPage(n)
      const content = await page.getTextContent()
      pageLines.push(toLines(content.items.filter(isTextItem)))
    }
    removeRunningLines(pageLines)
    const bodySize = dominantSize(pageLines.flat())
    const levelOf = headingLevels(pageLines.flat(), bodySize)

    const pages: PdfPage[] = []
    const scanned: number[] = []
    for (const [i, lines] of pageLines.entries()) {
      let text = toParagraphs(lines, bodySize, levelOf)
      if (text.replace(/\s/g, '').length < 20) {
        scanned.push(i + 1)
        if (ocr) text = (await ocrPage(await doc.getPage(i + 1), name, i + 1, ocr)).trim() || text
      }
      if (text.trim()) pages.push({ page: i + 1, text })
    }
    const info = ((await doc.getMetadata().catch(() => undefined))?.info ?? {}) as Record<string, unknown>
    // Titles like "Microsoft Word - draft.docx" or "Untitled" are worse than the file name.
    const raw = typeof info.Title === 'string' ? info.Title.trim() : ''
    const title = raw && !/^untitled|\.(docx?|pdf|pptx?)$/i.test(raw) ? raw : undefined
    return { ...(title ? { title } : {}), pages, pageCount: doc.numPages, scanned }
  } finally {
    await task.destroy()
  }
}

function isTextItem(item: unknown): item is TextItemLike {
  return !!item && typeof item === 'object' && 'str' in item && 'transform' in item
}

interface Line {
  text: string
  /** Segments separated by wide gaps: table cells or tab stops. */
  cells: string[]
  size: number
  y: number
  x: number
  endX: number
}

function toLines(items: TextItemLike[]): Line[] {
  const lines: Line[] = []
  let cur: Line | undefined
  const flush = () => {
    const cells = cur?.cells.map((c) => c.replace(/\s+/g, ' ').trim()).filter(Boolean) ?? []
    if (cur && cells.length) lines.push({ ...cur, text: cells.join(' '), cells })
    cur = undefined
  }
  for (const item of items) {
    const [, , c = 0, d = 0, x = 0, y = 0] = item.transform
    const size = Math.hypot(c, d) || item.height || 10
    if (cur && Math.abs(y - cur.y) > Math.max(size, cur.size) * 0.5) flush()
    if (!cur) cur = { text: '', cells: [''], size: 0, y, x, endX: x }
    else if (item.str.trim()) {
      const gap = x - cur.endX
      if (gap > size * 0.8) cur.cells.push('')
      else if (gap > size * 0.15 && !/\s$/.test(cur.cells[cur.cells.length - 1]!) && !/^\s/.test(item.str)) cur.cells[cur.cells.length - 1] += ' '
    }
    cur.cells[cur.cells.length - 1] += item.str
    if (item.str.trim()) {
      cur.endX = x + item.width
      cur.size = Math.max(cur.size, size)
    }
    if (item.hasEOL) flush()
  }
  flush()
  return lines
}

const runningKey = (text: string) => text.toLowerCase().replace(/\d+/g, '#').replace(/\s+/g, ' ').trim()
const PAGE_NUMBER = /^(page\s*)?[#\d]+(\s*(of|\/)\s*[#\d]+)?$|^[-–—]\s*\d+\s*[-–—]$/i

/** Drop page numbers and lines repeated at the top or bottom of most pages. */
function removeRunningLines(pages: Line[][]) {
  const edge = (lines: Line[]) => [...lines.slice(0, 2), ...lines.slice(-2)]
  const counts = new Map<string, number>()
  for (const lines of pages) for (const key of new Set(edge(lines).map((l) => runningKey(l.text)))) counts.set(key, (counts.get(key) ?? 0) + 1)
  const threshold = Math.max(3, Math.ceil(pages.length * 0.5))
  for (const [i, lines] of pages.entries()) {
    const candidates = new Set(edge(lines))
    pages[i] = lines.filter((l) => {
      if (!candidates.has(l)) return true
      const key = runningKey(l.text)
      return !(PAGE_NUMBER.test(l.text.trim()) || (pages.length >= 3 && (counts.get(key) ?? 0) >= threshold))
    })
  }
}

/** The font size most of the text is set in. */
function dominantSize(lines: Line[]): number {
  const weight = new Map<number, number>()
  for (const l of lines) {
    const size = roundSize(l.size)
    weight.set(size, (weight.get(size) ?? 0) + l.text.length)
  }
  return [...weight.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 10
}

const BULLET = /^[•●▪‣◦○■□➢►✓\-*–]\s+/

const isHeadingLike = (l: Line, bodySize: number) => l.size / bodySize >= 1.15 && l.text.length <= 150
const roundSize = (size: number) => Math.round(size * 2) / 2

/** Heading levels by rank: the largest heading size is `#`, the next `##`, then `###`. */
function headingLevels(lines: Line[], bodySize: number): (l: Line) => number {
  const sizes = [...new Set(lines.filter((l) => isHeadingLike(l, bodySize)).map((l) => roundSize(l.size)))].sort((a, b) => b - a)
  return (l) => (isHeadingLike(l, bodySize) ? Math.min(3, sizes.indexOf(roundSize(l.size)) + 1) : 0)
}

const LIST_ITEM = /^(- |\d{1,2}\. )/

function toParagraphs(lines: Line[], bodySize: number, headingLevel: (l: Line) => number): string {
  if (!lines.length) return ''
  const isBody = (l: Line) => Math.abs(l.size - bodySize) < 0.5 && l.cells.length === 1
  // Line spacing within body text; gaps around headings and between paragraphs are larger.
  const gaps = lines
    .slice(1)
    .map((l, i) => ({ gap: lines[i]!.y - l.y, body: isBody(l) && isBody(lines[i]!) }))
    .filter((g) => g.gap > 0 && g.body)
    .map((g) => g.gap)
    .sort((a, b) => a - b)
  const spacing = gaps[Math.floor(gaps.length / 2)] ?? bodySize * 1.2
  // A body line that stops well short of the usual width and ends a sentence ends its paragraph
  // (list items whose bullets are drawn as shapes, short paragraphs at normal spacing).
  const widths = lines.filter(isBody).map((l) => l.endX - l.x).sort((a, b) => a - b)
  const fullWidth = widths.length >= 3 ? widths[Math.floor(widths.length * 0.75)]! : Infinity
  const endsShort = (l: Line) => isBody(l) && l.endX - l.x < fullWidth * 0.75 && /[.!?:;]$/.test(l.text)

  const blocks: { text: string; heading: number; size: number; row: boolean }[] = []
  let prev: Line | undefined
  for (const line of lines) {
    const heading = headingLevel(line)
    const row = line.cells.length > 1
    const last = blocks[blocks.length - 1]
    const gap = prev ? prev.y - line.y : 0
    const newBlock =
      !last ||
      row ||
      last.row ||
      heading !== last.heading ||
      (heading > 0 && Math.abs(line.size - last.size) > 0.5) ||
      gap > spacing * 1.4 ||
      gap < 0 || // a new column or text box
      (!!prev && endsShort(prev)) ||
      BULLET.test(line.text) ||
      /^\(?\d{1,2}[.)]\s/.test(line.text)
    if (row) blocks.push({ text: `| ${line.cells.map((c) => c.replace(/\|/g, '\\|')).join(' | ')} |`, heading: 0, size: line.size, row: true })
    else if (newBlock) blocks.push({ text: line.text.replace(BULLET, '- '), heading, size: line.size, row: false })
    else if (/[A-Za-zÀ-ÿ]-$/.test(last.text) && /^[a-zà-ÿ]/.test(line.text)) last.text = last.text.slice(0, -1) + line.text
    else last.text += ` ${line.text}`
    prev = line
  }
  return blocks
    .map((b, i) => {
      const before = blocks[i - 1]
      let text = b.heading ? `${'#'.repeat(b.heading)} ${b.text}` : b.text
      // The first row of a table gets a separator so the rows read as a Markdown table.
      if (b.row && !before?.row) {
        const columns = b.text.split(/(?<!\\)\|/).length - 2
        text += `\n|${' --- |'.repeat(columns)}`
      }
      const tight = before && ((b.row && before.row) || (LIST_ITEM.test(b.text) && LIST_ITEM.test(before.text)))
      return (i === 0 ? '' : tight ? '\n' : '\n\n') + text
    })
    .join('')
}

async function ocrPage(page: PdfPageLike, name: string, n: number, ocr: OcrFunction): Promise<string> {
  if (typeof OffscreenCanvas === 'undefined') return ''
  const viewport = page.getViewport({ scale: 2 })
  const canvas = new OffscreenCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height))
  const context = canvas.getContext('2d')
  if (!context) return ''
  await page.render({ canvas, canvasContext: context, viewport }).promise
  return ocr(await canvas.convertToBlob({ type: 'image/png' }), { name, page: n })
}
