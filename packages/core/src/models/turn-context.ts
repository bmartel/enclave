import type { Message, ModelRequest } from '../types.js'

/**
 * Live context laid out for prefix-caching servers (Ollama, LM Studio,
 * llama.cpp, vLLM): the system prompt holds instructions only; the context
 * rides on the newest user message, frozen as first sent, so every step of a
 * turn extends the same prompt. Earlier turns carry no context (stale passages
 * would pile up uncounted); a new turn re-reads from its first user message.
 */
export class TurnContext {
  /** User messages as first sent, with the context they carried then. */
  private readonly sent = new WeakMap<Message, string>()

  /** The content to send for each message index, and the context to append after the last tool result, if it changed. */
  layout(request: ModelRequest): { userContent(message: Message, index: number): string; changedContext: string | undefined } {
    const messages = request.messages as Message[]
    const lastUser = messages.findLastIndex((m) => m.role === 'user')
    const latest = messages[lastUser]
    if (latest && !this.sent.has(latest)) {
      this.sent.set(latest, request.context ? wrap(request.context, latest.content) : latest.content)
    }
    const frozen = latest ? this.sent.get(latest)! : undefined
    // Context that changed after the newest user message was sent (a write
    // changed the schema, say) goes after the latest tool result instead.
    const changed = request.context && frozen !== undefined && !frozen.includes(request.context) && messages.at(-1)?.role === 'tool'
    return {
      userContent: (message, index) => (index === lastUser ? frozen! : message.content),
      changedContext: changed ? `\n\n${wrap(request.context!, '').trimEnd()}` : undefined,
    }
  }
}

const wrap = (context: string, text: string) => `<context>\n${context}\n</context>\n\n${text}`
