/**
 * Minimal ZIP reader for Office and EPUB files. Inflates with the platform's
 * `DecompressionStream`, so it needs no dependency and works in workers.
 */
export interface ZipArchive {
  names(): string[]
  has(name: string): boolean
  bytes(name: string): Promise<Uint8Array | undefined>
  text(name: string): Promise<string | undefined>
}

interface Entry {
  method: number
  compressedSize: number
  localOffset: number
}

export function isZip(data: Uint8Array): boolean {
  return data[0] === 0x50 && data[1] === 0x4b && data[2] === 0x03 && data[3] === 0x04
}

export function readZip(data: Uint8Array): ZipArchive {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
  // The end-of-central-directory record sits in the last 22 bytes plus an optional comment.
  let eocd = -1
  for (let i = data.length - 22; i >= Math.max(0, data.length - 22 - 0xffff); i--) {
    if (view.getUint32(i, true) === 0x06054b50) {
      eocd = i
      break
    }
  }
  if (eocd < 0) throw new Error('Not a valid ZIP archive (no central directory).')
  const count = view.getUint16(eocd + 10, true)
  let offset = view.getUint32(eocd + 16, true)
  if (offset === 0xffffffff) throw new Error('ZIP64 archives are not supported.')

  const decoder = new TextDecoder()
  const entries = new Map<string, Entry>()
  for (let i = 0; i < count; i++) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw new Error('Corrupt ZIP central directory.')
    const nameLength = view.getUint16(offset + 28, true)
    const extraLength = view.getUint16(offset + 30, true)
    const commentLength = view.getUint16(offset + 32, true)
    const name = decoder.decode(data.subarray(offset + 46, offset + 46 + nameLength))
    entries.set(name, {
      method: view.getUint16(offset + 10, true),
      compressedSize: view.getUint32(offset + 20, true),
      localOffset: view.getUint32(offset + 42, true),
    })
    offset += 46 + nameLength + extraLength + commentLength
  }

  const bytes = async (name: string) => {
    const entry = entries.get(name) ?? entries.get(name.replace(/^\//, ''))
    if (!entry) return undefined
    const local = entry.localOffset
    if (view.getUint32(local, true) !== 0x04034b50) throw new Error(`Corrupt ZIP entry: ${name}`)
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true)
    const raw = data.subarray(start, start + entry.compressedSize)
    if (entry.method === 0) return raw
    if (entry.method === 8) return inflateRaw(raw)
    throw new Error(`Unsupported ZIP compression method ${entry.method} for ${name}.`)
  }
  return {
    names: () => [...entries.keys()],
    has: (name) => entries.has(name),
    bytes,
    text: async (name) => {
      const b = await bytes(name)
      return b && new TextDecoder().decode(b)
    },
  }
}

async function inflateRaw(raw: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([raw as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('deflate-raw'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

/** Resolve a relationship target against the part that references it. */
export function resolvePath(base: string, target: string): string {
  if (target.startsWith('/')) return target.slice(1)
  const parts = base.split('/').slice(0, -1)
  for (const segment of decodeURIComponent(target).split('/')) {
    if (segment === '..') parts.pop()
    else if (segment && segment !== '.') parts.push(segment)
  }
  return parts.join('/')
}
