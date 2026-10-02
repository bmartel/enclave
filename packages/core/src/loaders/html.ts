import { decodeEntities } from './xml.js'

/**
 * HTML → Markdown for retrieval: keeps headings, lists, tables and code, and
 * drops scripts, styles, navigation and other page chrome. When the page has
 * `<main>` or `<article>`, only that content is kept.
 *
 * A streaming converter rather than a DOM walk, so it tolerates malformed
 * markup and runs in workers and Node (no DOMParser).
 */
export function htmlToMarkdown(html: string): { title?: string; text: string } {
  const titleMatch = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)
  const contentOnly = /<(main|article)[\s>]/i.test(html)
  const c = new Converter(contentOnly)
  c.run(html)
  const text = c.result()
  const pageTitle = titleMatch ? clean(decodeEntities(titleMatch[1]!)) : undefined
  return { title: pageTitle || /^# (.+)$/m.exec(text)?.[1], text }
}

const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'])
/** Elements whose content is never text for the reader. */
const RAW = new Set(['script', 'style', 'textarea', 'title', 'noscript', 'template', 'xmp'])
/** Elements skipped with everything inside them. */
const SKIP = new Set(['head', 'nav', 'aside', 'button', 'select', 'dialog', 'svg', 'math', 'iframe', 'object', 'canvas', 'video', 'audio', 'map'])
const SKIP_ROLES = new Set(['navigation', 'banner', 'contentinfo', 'complementary', 'search', 'menu', 'menubar', 'toolbar', 'dialog', 'alert'])
const BLOCK = new Set([
  'p', 'div', 'section', 'article', 'main', 'header', 'footer', 'blockquote', 'figure', 'figcaption', 'address',
  'dl', 'dt', 'dd', 'details', 'summary', 'center', 'body', 'html', 'fieldset', 'legend', 'caption',
])
const TAG = /<\/?([a-zA-Z][\w:-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/y
const ATTR = /([^\s=/]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g

class Out {
  parts: string[] = []
  atLineStart = true
  text(t: string) {
    if (this.atLineStart) t = t.replace(/^ +/, '')
    if (!t) return
    this.parts.push(t)
    this.atLineStart = false
  }
  /** Markup the converter adds (markers, fences): never trimmed. */
  mark(t: string) {
    this.parts.push(t)
    this.atLineStart = true
  }
  line() {
    this.parts.push('\n')
    this.atLineStart = true
  }
  block() {
    this.parts.push('\n\n')
    this.atLineStart = true
  }
  toString() {
    return this.parts.join('')
  }
}

interface ListState {
  ordered: boolean
  n: number
}

class Converter {
  private outs = [new Out()]
  private skip: { name: string; depth: number } | undefined
  private contentDepth = 0
  private pre = 0
  private lists: ListState[] = []
  private row: string[] | undefined
  private rowHasHeader = false
  private tableRows = 0
  private headingLevel = 0

  constructor(private contentOnly: boolean) {}

  private get out() {
    return this.outs[this.outs.length - 1]!
  }
  private get active() {
    return !this.skip && (!this.contentOnly || this.contentDepth > 0)
  }

  run(html: string) {
    let i = 0
    while (i < html.length) {
      const lt = html.indexOf('<', i)
      if (lt < 0) {
        this.onText(html.slice(i))
        break
      }
      if (lt > i) this.onText(html.slice(i, lt))
      if (html.startsWith('<!--', lt)) {
        const end = html.indexOf('-->', lt + 4)
        i = end < 0 ? html.length : end + 3
        continue
      }
      if (html[lt + 1] === '!' || html[lt + 1] === '?') {
        const end = html.indexOf('>', lt)
        i = end < 0 ? html.length : end + 1
        continue
      }
      TAG.lastIndex = lt
      const m = TAG.exec(html)
      if (!m) {
        this.onText('<')
        i = lt + 1
        continue
      }
      i = TAG.lastIndex
      const name = m[1]!.toLowerCase()
      if (html[lt + 1] === '/') {
        this.onClose(name)
        continue
      }
      if (RAW.has(name)) {
        // Jump past the raw text; its content is never parsed as markup.
        const close = new RegExp(`</${name}\\s*>`, 'ig')
        close.lastIndex = i
        const end = close.exec(html)
        i = end ? close.lastIndex : html.length
        continue
      }
      const selfClosing = /\/\s*$/.test(m[2] ?? '')
      this.onOpen(name, parseAttrs(m[2] ?? ''), VOID.has(name) || selfClosing)
    }
  }

  private onOpen(name: string, attrs: Record<string, string>, isVoid: boolean) {
    if (this.skip) {
      if (name === this.skip.name && !isVoid) this.skip.depth++
      return
    }
    const hidden =
      'hidden' in attrs || attrs['aria-hidden'] === 'true' || SKIP_ROLES.has(attrs.role ?? '') || /display:\s*none/i.test(attrs.style ?? '')
    const chrome = name === 'footer' && this.contentDepth === 0
    if ((SKIP.has(name) || hidden || chrome) && !isVoid) {
      this.skip = { name, depth: 1 }
      return
    }
    if (name === 'main' || name === 'article') this.contentDepth++
    if (!this.active) return
    const out = this.out

    if (/^h[1-6]$/.test(name)) {
      if (this.row) return
      out.block()
      this.headingLevel = Number(name[1])
      out.mark('#'.repeat(this.headingLevel) + ' ')
    } else if (name === 'br') {
      if (this.pre) out.mark('\n')
      else if (this.row) out.text(' ')
      else out.line()
    } else if (name === 'hr') out.block()
    else if (name === 'pre') {
      out.block()
      out.mark('```\n')
      this.pre++
    } else if (name === 'code' && !this.pre) out.text('`')
    else if (name === 'ul' || name === 'ol') {
      if (!this.lists.length) out.block()
      this.lists.push({ ordered: name === 'ol', n: Number(attrs.start ?? 1) - 1 })
    } else if (name === 'li') {
      const list = this.lists[this.lists.length - 1] ?? { ordered: false, n: 0 }
      list.n++
      out.line()
      out.mark('  '.repeat(Math.max(0, this.lists.length - 1)) + (list.ordered ? `${list.n}. ` : '- '))
    } else if (name === 'table') {
      out.block()
      this.tableRows = 0
    } else if (name === 'tr') {
      this.row = []
      this.rowHasHeader = false
    } else if (name === 'td' || name === 'th') {
      if (name === 'th') this.rowHasHeader = true
      this.outs.push(new Out())
    } else if (BLOCK.has(name)) {
      if (name === 'dd') out.mark('  ')
      else out.block()
    }
  }

  private onClose(name: string) {
    if (this.skip) {
      if (name === this.skip.name && --this.skip.depth === 0) this.skip = undefined
      return
    }
    const wasActive = this.active
    if (name === 'main' || name === 'article') this.contentDepth = Math.max(0, this.contentDepth - 1)
    if (!wasActive) return
    const out = this.out

    if (/^h[1-6]$/.test(name)) {
      if (this.row) return
      this.headingLevel = 0
      out.block()
    } else if (name === 'pre') {
      if (!this.pre) return
      this.pre--
      if (!out.parts[out.parts.length - 1]?.endsWith('\n')) out.mark('\n')
      out.mark('```')
      out.block()
    } else if (name === 'code' && !this.pre) out.text('`')
    else if (name === 'ul' || name === 'ol') {
      this.lists.pop()
      if (!this.lists.length) out.block()
    } else if ((name === 'td' || name === 'th') && this.outs.length > 1) {
      const cell = this.outs.pop()!.toString().replace(/\s+/g, ' ').trim().replace(/\|/g, '\\|')
      this.row?.push(cell)
    } else if (name === 'tr' && this.row) {
      const cells = this.row
      this.row = undefined
      if (cells.every((c) => !c)) return
      const target = this.out
      target.mark(`| ${cells.join(' | ')} |`)
      target.line()
      if (this.tableRows === 0 && this.rowHasHeader) {
        target.mark(`| ${cells.map(() => '---').join(' | ')} |`)
        target.line()
      }
      this.tableRows++
    } else if (name === 'table') out.block()
    else if (BLOCK.has(name) && name !== 'dd') out.block()
  }

  private onText(raw: string) {
    if (!this.active) return
    const text = decodeEntities(raw)
    if (this.pre) this.out.mark(text)
    else this.out.text(text.replace(/[\s ]+/g, ' '))
  }

  result(): string {
    return this.outs[0]!.toString()
      .split('\n')
      .map((l) => l.replace(/[ \t]+$/, ''))
      .join('\n')
      .replace(/^(#{1,6}|-|\d+\.) *$/gm, '') // markers left empty
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  }
}

function parseAttrs(source: string): Record<string, string> {
  const attrs: Record<string, string> = {}
  for (const m of source.matchAll(ATTR)) attrs[m[1]!.toLowerCase()] = decodeEntities(m[2] ?? m[3] ?? m[4] ?? '')
  return attrs
}

function clean(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}
