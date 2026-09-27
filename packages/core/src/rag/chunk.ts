export interface ChunkOptions {
  /** Target chunk size in characters (~4 chars per token). Default 1200. */
  size?: number
  /** Characters of trailing context carried into the next chunk. Default 150. */
  overlap?: number
}

// Split on the coarsest boundary that works: sections, paragraphs, lines, sentences, words.
const SEPARATORS = [/\n(?=#{1,6} )/, /\n\s*\n/, /\n/, /(?<=[.!?])\s+/, /\s+/]

/**
 * Recursive, structure-aware text splitter. Keeps markdown sections and
 * paragraphs intact where possible, then packs pieces up to `size`.
 */
export function chunkText(text: string, options: ChunkOptions = {}): string[] {
  const size = options.size ?? 1200
  const overlap = Math.min(options.overlap ?? 150, Math.floor(size / 2))
  const clean = text.replace(/\r\n?/g, '\n').trim()
  if (!clean) return []
  if (clean.length <= size) return [clean]

  const pieces = split(clean, size, 0)
  const chunks: string[] = []
  let current = ''
  for (const piece of pieces) {
    if (current && current.length + piece.length + 1 > size) {
      chunks.push(current.trim())
      const tail = overlap ? current.slice(-overlap) : ''
      // Start overlap on a word boundary.
      current = tail.includes(' ') ? tail.slice(tail.indexOf(' ') + 1) : tail
    }
    current = current ? `${current}\n${piece}` : piece
  }
  if (current.trim()) chunks.push(current.trim())
  return chunks
}

function split(text: string, size: number, level: number): string[] {
  if (text.length <= size) return [text]
  const separator = SEPARATORS[level]
  if (!separator) {
    const out: string[] = []
    for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size))
    return out
  }
  return text
    .split(separator)
    .filter((part) => part.trim())
    .flatMap((part) => split(part.trim(), size, level + 1))
}
