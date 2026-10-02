/**
 * Turn files into documents for `ai.knowledge.ingest()` and tables for
 * `importTable()`. Everything runs locally: Office, EPUB, HTML, CSV and JSON
 * need no dependencies; PDF uses pdf.js and images use an OCR function, both
 * supplied by the app.
 *
 *   const { documents, tables } = await loadFiles(input.files, { pdf: { lib: () => import('pdfjs-dist'), workerSrc } })
 *   await ai.knowledge!.ingest(documents)
 *   for (const t of tables) await importTable(ai.db, t)
 */
import type { IngestDocument } from '../rag/knowledge.js'
import { htmlToMarkdown } from './html.js'
import { docxToMarkdown, epubToMarkdown, pptxToMarkdown, xlsxToTables } from './office.js'
import { pdfToText, type OcrFunction, type PdfOptions } from './pdf.js'
import { jsonToTableOrText, parseCsv, sqlIdentifier, tableToText, toTable, type TableData } from './tables.js'
import { isZip, readZip } from './zip.js'

export { importTable, inferType, parseCsv, sqlIdentifier, tableToText } from './tables.js'
export type { ImportTableOptions, ImportTableResult, TableData } from './tables.js'
export type { OcrFunction, PdfJsLike, PdfOptions } from './pdf.js'
export { htmlToMarkdown } from './html.js'

export type DocumentFormat = 'text' | 'markdown' | 'html' | 'pdf' | 'docx' | 'pptx' | 'xlsx' | 'epub' | 'csv' | 'json' | 'image'

/** A browser `File`/`Blob`, or raw bytes or text with a name. */
export type FileInput = Blob | { name: string; data: ArrayBuffer | Uint8Array | string; type?: string; lastModified?: number }

export interface LoadOptions {
  /** Collection for the documents. */
  collection?: string
  /** Extra metadata stored with each document. */
  metadata?: Record<string, unknown>
  /** Overrides the source (default: the file name). Useful for paths or URLs. */
  source?: string
  /** pdf.js and its worker. Required for PDFs. */
  pdf?: PdfOptions
  /**
   * PDFs as one document per page (`'page'`, ids `<source>#page=N`) so answers
   * can cite pages, or one document per file (default).
   */
  pdfSplit?: 'file' | 'page'
  /** Reads text from images and scanned PDF pages. Without it, images are skipped. */
  ocr?: OcrFunction
  /** Include PowerPoint speaker notes. Default true. */
  slideNotes?: boolean
  /**
   * Also add spreadsheet, CSV and JSON-array rows to the documents as text, one
   * self-describing line per row, so they can be searched. Default true.
   * Tables are always returned in `tables` for `importTable()`.
   */
  tablesAsText?: boolean
  /** Rows per table included as text. Default 2000; larger tables belong in SQL. */
  maxTextRows?: number
  /** Refuse files larger than this many bytes. Default 100 MB. */
  maxBytes?: number
}

export interface LoadedFile {
  name: string
  format: DocumentFormat
  documents: IngestDocument[]
  tables: TableData[]
  /** Things the user should know: truncated tables, scanned pages without OCR, skipped images. */
  warnings: string[]
}

export class UnsupportedFileError extends Error {
  constructor(
    readonly fileName: string,
    message: string,
  ) {
    super(message)
    this.name = 'UnsupportedFileError'
  }
}

const EXTENSIONS: Record<string, DocumentFormat> = {
  md: 'markdown', markdown: 'markdown', mdx: 'markdown', mdown: 'markdown',
  html: 'html', htm: 'html', xhtml: 'html',
  pdf: 'pdf', docx: 'docx', docm: 'docx', dotx: 'docx', pptx: 'pptx', pptm: 'pptx', xlsx: 'xlsx', xlsm: 'xlsx', epub: 'epub',
  csv: 'csv', tsv: 'csv', tab: 'csv', json: 'json', jsonl: 'json', ndjson: 'json', geojson: 'json',
  png: 'image', jpg: 'image', jpeg: 'image', webp: 'image', gif: 'image', bmp: 'image', tif: 'image', tiff: 'image',
}
const LEGACY_OFFICE = /\.(doc|xls|ppt|dot|xlt|pot)$/i

/** The file input elements' `accept` value for everything the loaders read. */
export const ACCEPT = '.txt,.md,.markdown,.mdx,.html,.htm,.pdf,.docx,.pptx,.xlsx,.epub,.csv,.tsv,.json,.jsonl,.png,.jpg,.jpeg,.webp'

/** Detect a file's format from its leading bytes, then its extension and MIME type. */
export async function detectFormat(name: string, bytes: Uint8Array, mime = ''): Promise<DocumentFormat | undefined> {
  const head = String.fromCharCode(...bytes.subarray(0, 8))
  if (head.startsWith('%PDF-')) return 'pdf'
  if (isZip(bytes)) {
    const zip = readZip(bytes)
    if (zip.has('word/document.xml')) return 'docx'
    if (zip.has('ppt/presentation.xml')) return 'pptx'
    if (zip.has('xl/workbook.xml')) return 'xlsx'
    if ((await zip.text('mimetype'))?.trim() === 'application/epub+zip' || zip.has('META-INF/container.xml')) return 'epub'
    return undefined
  }
  if (head.startsWith('\x89PNG') || head.startsWith('\xff\xd8\xff') || head.startsWith('GIF8') || (head.startsWith('RIFF') && String.fromCharCode(...bytes.subarray(8, 12)) === 'WEBP')) {
    return 'image'
  }
  if (bytes[0] === 0xd0 && bytes[1] === 0xcf && bytes[2] === 0x11 && bytes[3] === 0xe0) return undefined // legacy Office
  const ext = /\.([a-z0-9]+)$/i.exec(name)?.[1]?.toLowerCase()
  const byExt = ext ? EXTENSIONS[ext] : undefined
  if (byExt) return byExt
  if (/html/.test(mime)) return 'html'
  if (/csv|tab-separated/.test(mime)) return 'csv'
  if (/json/.test(mime)) return 'json'
  if (/markdown/.test(mime)) return 'markdown'
  if (/^image\//.test(mime)) return 'image'
  // Anything else that looks like text (source code, logs, notes) is read as text.
  return looksLikeText(bytes) ? 'text' : undefined
}

function looksLikeText(bytes: Uint8Array): boolean {
  const sample = bytes.subarray(0, 8192)
  if (sample[0] === 0xff && sample[1] === 0xfe) return true
  if (sample[0] === 0xfe && sample[1] === 0xff) return true
  let control = 0
  for (const b of sample) {
    if (b === 0) return false
    if (b < 9 || (b > 13 && b < 32)) control++
  }
  return control <= sample.length * 0.02
}

/** UTF-8 (or UTF-16 with a BOM); falls back to Windows-1252, which Excel uses for CSV exports. */
export function decodeText(bytes: Uint8Array): string {
  if (bytes[0] === 0xff && bytes[1] === 0xfe) return new TextDecoder('utf-16le').decode(bytes.subarray(2))
  if (bytes[0] === 0xfe && bytes[1] === 0xff) return new TextDecoder('utf-16be').decode(bytes.subarray(2))
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes).replace(/^﻿/, '')
  } catch {
    return new TextDecoder('windows-1252').decode(bytes)
  }
}

async function readInput(input: FileInput): Promise<{ name: string; bytes: Uint8Array; type: string; lastModified?: number }> {
  if (typeof Blob !== 'undefined' && input instanceof Blob) {
    const file = input as Blob & { name?: string; lastModified?: number }
    return {
      name: file.name ?? 'untitled',
      bytes: new Uint8Array(await file.arrayBuffer()),
      type: file.type,
      ...(file.lastModified ? { lastModified: file.lastModified } : {}),
    }
  }
  const raw = input as Exclude<FileInput, Blob>
  // A plain Uint8Array view, also for Node Buffers (which pdf.js rejects).
  const bytes =
    typeof raw.data === 'string'
      ? new TextEncoder().encode(raw.data)
      : raw.data instanceof Uint8Array
        ? new Uint8Array(raw.data.buffer, raw.data.byteOffset, raw.data.byteLength)
        : new Uint8Array(raw.data)
  return { name: raw.name, bytes, type: raw.type ?? '', ...(raw.lastModified ? { lastModified: raw.lastModified } : {}) }
}

const baseName = (name: string) => name.split(/[\\/]/).pop() ?? name
const stem = (name: string) => baseName(name).replace(/\.[a-z0-9]+$/i, '')

/** Read one file into documents (for search) and tables (for SQL). */
export async function loadFile(input: FileInput, options: LoadOptions = {}): Promise<LoadedFile> {
  const { name, bytes, type, lastModified } = await readInput(input)
  if (bytes.byteLength > (options.maxBytes ?? 100 * 1024 * 1024)) {
    throw new UnsupportedFileError(name, `${baseName(name)} is ${Math.round(bytes.byteLength / 1048576)} MB, over the ${Math.round((options.maxBytes ?? 104857600) / 1048576)} MB limit.`)
  }
  if (LEGACY_OFFICE.test(name)) {
    throw new UnsupportedFileError(name, `${baseName(name)} is a legacy Office file. Save it as .docx, .xlsx or .pptx and try again.`)
  }
  const format = await detectFormat(name, bytes, type)
  if (!format) throw new UnsupportedFileError(name, `Can't read ${baseName(name)}: unsupported file type.`)

  const source = options.source ?? name
  const warnings: string[] = []
  const tables: TableData[] = []
  const documents: IngestDocument[] = []
  const metadata = (extra: Record<string, unknown> = {}) => ({
    format,
    file: baseName(name),
    ...(lastModified ? { modified: new Date(lastModified).toISOString() } : {}),
    ...extra,
    ...options.metadata,
  })
  const add = (content: string, title: string | undefined, extra: Record<string, unknown> = {}, id?: string) => {
    if (!content.trim()) return
    documents.push({
      ...(id ? { id } : {}),
      title: title?.trim() || stem(name),
      content,
      source,
      ...(options.collection ? { collection: options.collection } : {}),
      metadata: metadata(extra),
    })
  }
  const addTables = (found: TableData[]) => {
    tables.push(...found)
    if (options.tablesAsText === false) return
    const max = options.maxTextRows ?? 2000
    const text = found.map((t) => tableToText(t, max)).join('\n\n')
    for (const t of found) if (t.rows.length > max) warnings.push(`${t.name}: only the first ${max} of ${t.rows.length} rows are searchable as text. Import it as a table to query all rows.`)
    add(text, stem(name), { tables: found.map((t) => t.name) })
  }

  switch (format) {
    case 'text':
      add(decodeText(bytes), undefined)
      break
    case 'markdown': {
      const { title, body } = frontMatter(decodeText(bytes))
      add(body, title ?? /^# (.+)$/m.exec(body)?.[1])
      break
    }
    case 'html': {
      const { title, text } = htmlToMarkdown(decodeText(bytes))
      add(text, title)
      break
    }
    case 'csv': {
      const grid = parseCsv(decodeText(bytes), /\.(tsv|tab)$/i.test(name) ? '\t' : undefined)
      if (grid.length) addTables([toTable(stem(name), grid, source)])
      break
    }
    case 'json': {
      const text = decodeText(bytes)
      let value: unknown
      try {
        value = /\.(jsonl|ndjson)$/i.test(name)
          ? text.split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l))
          : JSON.parse(text)
      } catch {
        add(text, undefined)
        warnings.push(`${baseName(name)} isn't valid JSON; indexed as plain text.`)
        break
      }
      const result = jsonToTableOrText(value, stem(name))
      if (result.table) addTables([{ ...result.table, source }])
      else add(result.text, undefined)
      break
    }
    case 'docx': {
      const { title, text } = await docxToMarkdown(readZip(bytes))
      add(text, title)
      break
    }
    case 'pptx': {
      const { title, text, slides } = await pptxToMarkdown(readZip(bytes), { notes: options.slideNotes !== false })
      add(text, title, { slides })
      break
    }
    case 'xlsx': {
      // One sheet, or a sheet named like the file: the file's name. Otherwise file and sheet ("budget_forecast").
      const sheets = await xlsxToTables(readZip(bytes), source)
      const file = sqlIdentifier(stem(name), 'sheet')
      addTables(sheets.map((t) => ({ ...t, sqlName: sheets.length === 1 || sqlIdentifier(t.name, '') === file ? file : sqlIdentifier(`${stem(name)} ${t.name}`, 'sheet') })))
      break
    }
    case 'epub': {
      const { title, text } = await epubToMarkdown(readZip(bytes))
      add(text, title)
      break
    }
    case 'pdf': {
      const result = await pdfToText(bytes, baseName(name), options.pdf, options.ocr)
      if (result.scanned.length && !options.ocr) {
        warnings.push(
          result.pages.length === 0
            ? `${baseName(name)} is a scanned PDF with no text layer. Pass an \`ocr\` function to read it.`
            : `${baseName(name)}: ${result.scanned.length} page(s) have no text layer (${result.scanned.slice(0, 5).join(', ')}${result.scanned.length > 5 ? '…' : ''}). Pass an \`ocr\` function to read them.`,
        )
      }
      if (options.pdfSplit === 'page') {
        for (const p of result.pages) add(p.text, `${result.title ?? stem(name)} (page ${p.page})`, { page: p.page }, `${source}#page=${p.page}`)
      } else {
        add(result.pages.map((p) => p.text).join('\n\n'), result.title, { pages: result.pageCount })
      }
      break
    }
    case 'image': {
      if (!options.ocr) {
        warnings.push(`${baseName(name)} is an image. Pass an \`ocr\` function to read text from images.`)
        break
      }
      const blob = new Blob([bytes as Uint8Array<ArrayBuffer>], { type: type || 'image/png' })
      add(await options.ocr(blob, { name: baseName(name) }), undefined, { ocr: true })
      break
    }
  }
  if (!documents.length && !tables.length && !warnings.length) warnings.push(`${baseName(name)} has no readable text.`)
  return { name, format, documents, tables, warnings }
}

export interface LoadFilesResult {
  documents: IngestDocument[]
  tables: TableData[]
  files: LoadedFile[]
  warnings: string[]
  /** Files that couldn't be read; the rest still load. */
  errors: { name: string; error: Error }[]
}

/** Read many files. One unreadable file doesn't stop the others. */
export async function loadFiles(
  inputs: Iterable<FileInput> | ArrayLike<FileInput>,
  options: LoadOptions & { onProgress?(progress: { done: number; total: number; name: string }): void } = {},
): Promise<LoadFilesResult> {
  const list = Array.from(inputs as ArrayLike<FileInput>)
  // Each file is its own source; a single `source` override only makes sense for loadFile.
  const { source: _source, onProgress, ...fileOptions } = options
  const result: LoadFilesResult = { documents: [], tables: [], files: [], warnings: [], errors: [] }
  for (const [i, input] of list.entries()) {
    const name = 'name' in input && typeof input.name === 'string' ? input.name : `file ${i + 1}`
    try {
      const file = await loadFile(input, fileOptions)
      result.files.push(file)
      result.documents.push(...file.documents)
      result.tables.push(...file.tables)
      result.warnings.push(...file.warnings)
    } catch (error) {
      result.errors.push({ name, error: error instanceof Error ? error : new Error(String(error)) })
    }
    onProgress?.({ done: i + 1, total: list.length, name })
  }
  return result
}

/** Strip YAML front matter, keeping its `title`. */
function frontMatter(text: string): { title?: string; body: string } {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text)
  if (!m) return { body: text }
  const title = /^title:\s*["']?(.+?)["']?\s*$/m.exec(m[1]!)?.[1]
  return { ...(title ? { title } : {}), body: text.slice(m[0].length) }
}
