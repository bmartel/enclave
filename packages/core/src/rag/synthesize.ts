import type { Citation, Message, Model } from '../types.js'
import { chunkText } from './chunk.js'

/** A document to write from (a note, a page of a PDF…). Long ones are split into passages. */
export interface SynthesisSource {
  id: string
  title?: string | null
  content: string
  collection?: string
  source?: string | null
  metadata?: Record<string, unknown>
}

export interface SynthesizeOptions {
  model: Model
  sources: SynthesisSource[]
  /**
   * What to write, e.g. "Write a study guide with key terms and practice
   * questions." The model is also told to cite passages inline as [n].
   */
  instruction: string
  /**
   * What matters when condensing many sources, e.g. "dates and events" for a
   * timeline. Defaults to the instruction.
   */
  focus?: string
  /** Context window in tokens when the model doesn't declare one. Default 4096. */
  contextWindow?: number
  /** Tokens kept free for the answer. Default 1200. */
  outputTokens?: number
  /** Passage size in characters. Default 1200. */
  chunkSize?: number
  signal?: AbortSignal
}

export type SynthesisEvent =
  /** `reading` passes over source batches, `condensing` merges notes, `writing` streams the result. */
  | { type: 'progress'; stage: 'reading' | 'condensing' | 'writing'; done: number; total: number }
  | { type: 'text-delta'; delta: string }
  /** The final text and the passages it cites (numbers match the text's [n]). */
  | { type: 'finish'; text: string; citations: Citation[]; passages: Citation[] }

const CHARS_PER_TOKEN = 3.2
const NOTES_SYSTEM = `You read numbered source passages and take notes for a writer.
- Write short bullet points of the facts, ideas, names, dates and numbers that matter for the writer's goal.
- End every bullet with the numbers of the passages it comes from, like [3] or [3][7]. Never invent numbers.
- Keep wording close to the sources. Skip anything irrelevant to the goal.
- If no passage is relevant, reply with just: NONE`
const MERGE_SYSTEM = `You merge notes taken from numbered sources into one shorter set of bullet points.
- Combine duplicates, keep every distinct fact, and keep the [n] citations on each bullet (merge their numbers).
- Never invent facts or numbers.`

/** Passage numbers cited in a text: [3], [3, 4], [3][4] and [[3]]. */
export function citedNumbers(text: string): number[] {
  const out = new Set<number>()
  for (const m of text.matchAll(/\[\[?(\d{1,4}(?:\s*,\s*\d{1,4})*)\]?\]/g)) for (const n of m[1]!.split(',')) out.add(Number(n))
  return [...out]
}

/**
 * Source-grounded writing over any number of documents (NotebookLM-style
 * study guides, briefings, FAQs…) with a small context window: passages are
 * numbered, read in batches that fit the model, condensed into cited notes
 * (map-reduce) when they don't all fit, then the result is streamed with
 * inline [n] citations that map back to the passages.
 */
export async function* synthesize(options: SynthesizeOptions): AsyncGenerator<SynthesisEvent> {
  const { model, instruction, signal } = options
  const window = model.contextWindow ?? options.contextWindow ?? 4096
  const output = options.outputTokens ?? 1200
  // Characters of source text one prompt can carry next to its instructions and the answer.
  const budget = Math.max(1500, Math.floor((window - output - 400) * CHARS_PER_TOKEN))
  const focus = options.focus ?? instruction

  const passages: Citation[] = []
  for (const s of options.sources) {
    const pieces = chunkText(s.content, { size: Math.min(options.chunkSize ?? 1200, Math.floor(budget / 2)) })
    pieces.forEach((content, chunkId) =>
      passages.push({
        n: passages.length + 1,
        documentId: s.id,
        chunkId,
        collection: s.collection,
        title: s.title ?? null,
        source: s.source ?? null,
        content,
        metadata: s.metadata,
      }),
    )
  }
  const render = (p: Citation) => `[${p.n}]${p.title ? ` (${p.title})` : ''}\n${p.content}`

  const ask = async (system: string, content: string): Promise<string> => {
    let text = ''
    const messages: Message[] = [{ role: 'user', content }]
    for await (const chunk of model.stream({ system, messages, tools: [], signal })) {
      if (chunk.type === 'text') text += chunk.delta
    }
    return text.trim()
  }

  // Pack items (rendered strings) into batches under the budget.
  const batches = (items: string[]) => {
    const out: string[][] = []
    let cur: string[] = []
    let size = 0
    for (const item of items) {
      if (cur.length && size + item.length > budget) out.push(cur), (cur = []), (size = 0)
      cur.push(item)
      size += item.length + 2
    }
    if (cur.length) out.push(cur)
    return out
  }

  let context = passages.map(render).join('\n\n')
  let kind = 'Sources'
  if (context.length > budget) {
    // Map: notes from each batch of passages.
    const groups = batches(passages.map(render))
    let notes: string[] = []
    for (const [i, group] of groups.entries()) {
      signal?.throwIfAborted()
      yield { type: 'progress', stage: 'reading', done: i, total: groups.length }
      const text = await ask(NOTES_SYSTEM, `Writer's goal: ${focus}\n\nPassages:\n\n${group.join('\n\n')}`)
      if (text && !/^NONE\.?$/i.test(text)) notes.push(text)
    }
    yield { type: 'progress', stage: 'reading', done: groups.length, total: groups.length }
    // Reduce: merge notes until they fit one prompt.
    let round = 0
    while (notes.join('\n').length > budget && notes.length > 1 && round++ < 6) {
      const merged = batches(notes)
      const next: string[] = []
      for (const [i, group] of merged.entries()) {
        signal?.throwIfAborted()
        yield { type: 'progress', stage: 'condensing', done: i, total: merged.length }
        next.push(group.length === 1 ? group[0]! : await ask(MERGE_SYSTEM, `Writer's goal: ${focus}\n\nNotes:\n\n${group.join('\n\n')}`))
      }
      notes = next
    }
    context = notes.join('\n').slice(0, budget)
    kind = 'Notes taken from the numbered sources (cite the numbers they carry)'
  }

  yield { type: 'progress', stage: 'writing', done: 0, total: 1 }
  const system = `${instruction}
Ground everything in the material below and cite it inline with its passage numbers, like [2] or [2][5]. Don't invent citations. If the material doesn't cover something, leave it out.`
  let text = ''
  const messages: Message[] = [{ role: 'user', content: `${kind}:\n\n${context}` }]
  for await (const chunk of model.stream({ system, messages, tools: [], signal })) {
    if (chunk.type !== 'text') continue
    text += chunk.delta
    yield { type: 'text-delta', delta: chunk.delta }
  }
  yield { type: 'progress', stage: 'writing', done: 1, total: 1 }
  const cited = new Set(citedNumbers(text))
  yield { type: 'finish', text, citations: passages.filter((p) => cited.has(p.n)), passages }
}
