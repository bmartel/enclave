import type { Model } from '../types.js'
import { fromTextModel } from './text-protocol.js'

// Minimal typings for Chrome's built-in Prompt API (Gemini Nano on-device).
interface LanguageModelSession {
  promptStreaming(input: string, options?: { signal?: AbortSignal }): ReadableStream<string>
  destroy(): void
}
interface LanguageModelStatic {
  availability(): Promise<'unavailable' | 'downloadable' | 'downloading' | 'available'>
  create(options: {
    initialPrompts?: { role: 'system' | 'user' | 'assistant'; content: string }[]
    temperature?: number
    topK?: number
    signal?: AbortSignal
    monitor?(m: EventTarget): void
  }): Promise<LanguageModelSession>
}

function api(): LanguageModelStatic | undefined {
  return (globalThis as { LanguageModel?: LanguageModelStatic }).LanguageModel
}

export interface ChromeAIOptions {
  temperature?: number
  topK?: number
  onDownloadProgress?(fraction: number): void
}

/** True when the browser ships a usable on-device model. */
export async function chromeAIAvailable(): Promise<boolean> {
  const lm = api()
  if (!lm) return false
  return (await lm.availability()) !== 'unavailable'
}

/**
 * Chrome's built-in on-device model. Zero download size for your app, fully
 * private; best for lightweight tasks and small tool sets.
 */
export function chromeAI(options: ChromeAIOptions = {}): Model {
  return fromTextModel({
    id: 'chrome:built-in',
    locality: 'device',
    async *streamText({ system, messages, signal }) {
      const lm = api()
      if (!lm) throw new Error('Chrome built-in AI (LanguageModel) is not available in this browser')
      const last = messages.at(-1)
      const prior = last?.role === 'user' ? messages.slice(0, -1) : messages
      const session = await lm.create({
        initialPrompts: [{ role: 'system', content: system }, ...prior],
        ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
        ...(options.topK !== undefined ? { topK: options.topK } : {}),
        ...(signal ? { signal } : {}),
        monitor(m) {
          m.addEventListener('downloadprogress', (e) => options.onDownloadProgress?.((e as ProgressEvent).loaded))
        },
      })
      try {
        const stream = session.promptStreaming(last?.role === 'user' ? last.content : 'Continue.', signal ? { signal } : {})
        const reader = stream.getReader()
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          if (value) yield value
        }
      } finally {
        session.destroy()
      }
    },
  })
}
