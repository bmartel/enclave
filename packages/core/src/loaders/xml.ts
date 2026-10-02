/**
 * A small, forgiving XML parser for machine-written documents (Office, EPUB).
 * No DTDs or namespaces resolution: names keep their prefix (`w:p`).
 * Unlike DOMParser it also works in workers and Node.
 */
export interface XmlElement {
  name: string
  attrs: Record<string, string>
  children: XmlNode[]
}
export type XmlNode = XmlElement | string

const TOKEN = /<!--[\s\S]*?-->|<!\[CDATA\[([\s\S]*?)\]\]>|<\?[\s\S]*?\?>|<![^>]*>|<\/([^\s>]+)\s*>|<([^\s/>]+)((?:[^>"']|"[^"]*"|'[^']*')*?)(\/?)>|([^<]+)/g
const ATTR = /([^\s=]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g

export function parseXml(text: string, options: { stripPrefixes?: boolean } = {}): XmlElement {
  const local = (n: string) => (options.stripPrefixes ? n.slice(n.indexOf(':') + 1) : n)
  const root: XmlElement = { name: '#document', attrs: {}, children: [] }
  const stack: XmlElement[] = [root]
  for (const m of text.matchAll(TOKEN)) {
    const top = stack[stack.length - 1]!
    if (m[1] !== undefined) top.children.push(m[1])
    else if (m[2] !== undefined) {
      // Pop to the matching element; tolerate stray closing tags.
      const name = local(m[2])
      const at = stack.findLastIndex((e) => e.name === name)
      if (at > 0) stack.length = at
    } else if (m[3] !== undefined) {
      const attrs: Record<string, string> = {}
      for (const a of (m[4] ?? '').matchAll(ATTR)) attrs[a[1]!] = decodeEntities(a[2] ?? a[3] ?? '')
      const el: XmlElement = { name: local(m[3]), attrs, children: [] }
      top.children.push(el)
      if (!m[5]) stack.push(el)
    } else if (m[6] !== undefined) top.children.push(decodeEntities(m[6]))
  }
  return root
}

export const isElement = (n: XmlNode): n is XmlElement => typeof n !== 'string'

/** Direct children with this name. */
export function childElements(el: XmlElement | undefined, name?: string): XmlElement[] {
  return el ? el.children.filter((c): c is XmlElement => isElement(c) && (!name || c.name === name)) : []
}

export function child(el: XmlElement | undefined, name: string): XmlElement | undefined {
  return el?.children.find((c): c is XmlElement => isElement(c) && c.name === name)
}

/** First descendant with this name (depth-first, document order). */
export function find(el: XmlElement | undefined, name: string): XmlElement | undefined {
  if (!el) return undefined
  for (const c of el.children) {
    if (!isElement(c)) continue
    if (c.name === name) return c
    const hit = find(c, name)
    if (hit) return hit
  }
  return undefined
}

/** All descendants with this name, in document order. */
export function findAll(el: XmlElement | undefined, name: string, out: XmlElement[] = []): XmlElement[] {
  for (const c of el?.children ?? []) {
    if (!isElement(c)) continue
    if (c.name === name) out.push(c)
    else findAll(c, name, out)
  }
  return out
}

/** Concatenated text of an element and its descendants. */
export function textOf(el: XmlElement | undefined): string {
  if (!el) return ''
  return el.children.map((c) => (isElement(c) ? textOf(c) : c)).join('')
}

const NAMED: Record<string, string> = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', shy: '',
  copy: '©', reg: '®', trade: '™', hellip: '…', mdash: '—', ndash: '–', minus: '−',
  lsquo: '‘', rsquo: '’', ldquo: '“', rdquo: '”', laquo: '«', raquo: '»', bull: '•', middot: '·',
  euro: '€', pound: '£', yen: '¥', cent: '¢', deg: '°', times: '×', divide: '÷', plusmn: '±',
  frac12: '½', frac14: '¼', frac34: '¾', sect: '§', para: '¶', larr: '←', rarr: '→', harr: '↔',
  le: '≤', ge: '≥', ne: '≠', check: '✓', aacute: 'á', agrave: 'à', acirc: 'â', auml: 'ä', aring: 'å',
  ccedil: 'ç', eacute: 'é', egrave: 'è', ecirc: 'ê', euml: 'ë', iacute: 'í', iuml: 'ï', ntilde: 'ñ',
  oacute: 'ó', ocirc: 'ô', ouml: 'ö', oslash: 'ø', uacute: 'ú', uuml: 'ü', szlig: 'ß',
  Aacute: 'Á', Auml: 'Ä', Eacute: 'É', Ouml: 'Ö', Uuml: 'Ü', Ntilde: 'Ñ', Ccedil: 'Ç',
}

export function decodeEntities(text: string): string {
  if (!text.includes('&')) return text
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z][a-z0-9]*);/gi, (whole, body: string) => {
    if (body[0] === '#') {
      const code = body[1] === 'x' || body[1] === 'X' ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10)
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole
    }
    return NAMED[body] ?? whole
  })
}
