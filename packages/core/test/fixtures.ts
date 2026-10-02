import { crc32, deflateRawSync } from 'node:zlib'

/** A real ZIP archive (deflated entries unless `store` is set), as Office and EPUB use. */
export function makeZip(files: Record<string, string | Uint8Array>, options: { store?: string[] } = {}): Uint8Array {
  const enc = new TextEncoder()
  const locals: Buffer[] = []
  const centrals: Buffer[] = []
  let offset = 0
  for (const [name, content] of Object.entries(files)) {
    const data = Buffer.from(typeof content === 'string' ? enc.encode(content) : content)
    const store = options.store?.includes(name)
    const body = store ? data : deflateRawSync(data)
    const nameBytes = Buffer.from(enc.encode(name))
    const crc = crc32(data)
    const local = Buffer.alloc(30)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(20, 4)
    local.writeUInt16LE(0x0800, 6)
    local.writeUInt16LE(store ? 0 : 8, 8)
    local.writeUInt32LE(crc, 14)
    local.writeUInt32LE(body.length, 18)
    local.writeUInt32LE(data.length, 22)
    local.writeUInt16LE(nameBytes.length, 26)
    locals.push(local, nameBytes, body)
    const central = Buffer.alloc(46)
    central.writeUInt32LE(0x02014b50, 0)
    central.writeUInt16LE(20, 4)
    central.writeUInt16LE(20, 6)
    central.writeUInt16LE(0x0800, 8)
    central.writeUInt16LE(store ? 0 : 8, 10)
    central.writeUInt32LE(crc, 16)
    central.writeUInt32LE(body.length, 20)
    central.writeUInt32LE(data.length, 24)
    central.writeUInt16LE(nameBytes.length, 28)
    central.writeUInt32LE(offset, 42)
    centrals.push(central, nameBytes)
    offset += 30 + nameBytes.length + body.length
  }
  const cd = Buffer.concat(centrals)
  const eocd = Buffer.alloc(22)
  eocd.writeUInt32LE(0x06054b50, 0)
  eocd.writeUInt16LE(Object.keys(files).length, 8)
  eocd.writeUInt16LE(Object.keys(files).length, 10)
  eocd.writeUInt32LE(cd.length, 12)
  eocd.writeUInt32LE(offset, 16)
  return new Uint8Array(Buffer.concat([...locals, cd, eocd]))
}

export interface PdfLine {
  text: string
  size: number
  y: number
  x?: number
}

/** A minimal, valid PDF: Helvetica text placed line by line. */
export function makePdf(pages: PdfLine[][], title?: string): Uint8Array {
  const esc = (s: string) => s.replace(/[\\()]/g, (c) => `\\${c}`)
  const objects: string[] = []
  const add = (body: string) => objects.push(body)
  const pageIds = pages.map((_, i) => 5 + i * 2)
  add('<< /Type /Catalog /Pages 2 0 R >>')
  add(`<< /Type /Pages /Kids [${pageIds.map((id) => `${id} 0 R`).join(' ')}] /Count ${pages.length} >>`)
  add('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>')
  add(title ? `<< /Title (${esc(title)}) >>` : '<< >>')
  for (const [i, lines] of pages.entries()) {
    const stream = lines.map((l) => `BT /F1 ${l.size} Tf 1 0 0 1 ${l.x ?? 72} ${l.y} Tm (${esc(l.text)}) Tj ET`).join('\n')
    add(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents ${pageIds[i]! + 1} 0 R >>`)
    add(`<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`)
  }
  let out = '%PDF-1.4\n'
  const offsets: number[] = []
  objects.forEach((body, i) => {
    offsets.push(out.length)
    out += `${i + 1} 0 obj\n${body}\nendobj\n`
  })
  const xref = out.length
  out += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${offsets.map((o) => `${String(o).padStart(10, '0')} 00000 n \n`).join('')}`
  out += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R /Info 4 0 R >>\nstartxref\n${xref}\n%%EOF\n`
  // Helvetica with WinAnsiEncoding: a few characters live outside Latin-1.
  const winAnsi: Record<string, number> = { '•': 0x95, '—': 0x97, '–': 0x96, '’': 0x92, '“': 0x93, '”': 0x94, '€': 0x80 }
  return new Uint8Array([...out].map((ch) => winAnsi[ch] ?? ch.charCodeAt(0) & 0xff))
}

/** A body paragraph's worth of lines at a given size and spacing. */
export function para(lines: string[], top: number, size = 11, leading = 14): PdfLine[] {
  return lines.map((text, i) => ({ text, size, y: top - i * leading }))
}
