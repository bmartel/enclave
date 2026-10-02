import { reasoningLoops } from '../util.js'
import type { AppConfig, CompletionUsage, InitProgressReport, MLCEngineInterface } from '@mlc-ai/web-llm'
import type { Downloadable, Model, ToolSpec } from '../types.js'
import { findLLM } from '../web/catalog.js'
import { fromTextModel, type TextMessage } from './text-protocol.js'
import { absolute } from '../privacy/index.js'

export type ThinkingMode = boolean | 'auto'

export interface WebLLMOptions {
  /**
   * A WebLLM model id (e.g. `Qwen3-4B-q4f16_1-MLC`) or a catalog preset id
   * (e.g. `qwen3-4b`, resolved to the f16 build). Prefer `recommendLLM()`
   * from `enclave-ai/web` to pick one that fits the device.
   */
  model: string
  /**
   * Run inference in a dedicated worker (recommended). Its entry calls
   * `serveWebLLM()` from `enclave-ai/models/webllm-worker`. Models created
   * on the same worker share one engine: switching models unloads the old one.
   */
  worker?: Worker
  /** Tokens of context. WebLLM builds default to 4096; agents need more. Default 8192. */
  contextWindow?: number
  temperature?: number
  maxTokens?: number
  /**
   * Qwen3 thinking. `'auto'` (default): reason on new user requests, answer
   * directly after tool results. `true`: reason before every step. `false`:
   * never. On Qwen3 4B, thinking is what makes tool use reliable, and `'auto'`
   * matched `true` on quality while cutting p90 latency (see README).
   */
  thinking?: ThinkingMode
  /**
   * Most reasoning tokens per step. Past it, reasoning is closed and the step
   * is re-asked with thinking off, so a runaway deliberation can't stall a
   * turn. Keep it well above normal reasoning: Qwen3 4B without thinking
   * describes SQL instead of running it. In the evals, 17% of SQL turns used
   * more than 1024 tokens, and a 1024 budget broke them. Only 2 of 246
   * turns exceeded 2048. Default 2048; 0 disables.
   */
  thinkingBudget?: number
  /**
   * Constrain decoding so every `<tool_call>` is a real tool with schema-valid
   * arguments (xgrammar structural tags). Free text is unaffected. Default true.
   */
  constrainToolCalls?: boolean
  /**
   * Reasoning from earlier turns: `current-turn` (default) drops it like
   * Qwen3's template; `all` keeps it so the KV cache survives across turns;
   * `auto` keeps it until the conversation fills 60% of the window, then
   * compacts once.
   */
  reasoningHistory?: 'current-turn' | 'all' | 'auto'
  /**
   * Where to download weights and compiled model libraries from. Use
   * `selfHostedAppConfig()` to serve everything from your own origin.
   */
  appConfig?: AppConfig
  onProgress?(report: InitProgressReport): void
}

export interface WebLLMModel extends Model, Downloadable {
  readonly modelId: string
  readonly contextWindow: number
  unload(): Promise<void>
}

interface EngineSlot {
  engine: Promise<MLCEngineInterface>
  /** `modelId@contextWindow` currently loaded, or being loaded. */
  loaded: string | undefined
  loading: Promise<void>
  progress: Set<(report: InitProgressReport) => void>
  /** The conversation the engine's KV cache currently holds (as sent + generated). */
  transcript: string[] | undefined
  /** Structural tag JSON that failed to compile; skipped afterwards. */
  badGrammars: Set<string>
}

const slots = new WeakMap<object, EngineSlot>()
const inThreadKey = {}

function slotFor(worker: Worker | undefined, appConfig: AppConfig | undefined): EngineSlot {
  const key = worker ?? inThreadKey
  let slot = slots.get(key)
  if (!slot) {
    const progress = new Set<(report: InitProgressReport) => void>()
    const initProgressCallback = (report: InitProgressReport) => progress.forEach((l) => l(report))
    const config = { initProgressCallback, ...(appConfig ? { appConfig } : {}) }
    slot = {
      progress,
      loaded: undefined,
      loading: Promise.resolve(),
      transcript: undefined,
      badGrammars: new Set(),
      engine: import('@mlc-ai/web-llm').then((lib) =>
        worker ? new lib.WebWorkerMLCEngine(worker, config) : new lib.MLCEngine(config),
      ),
    }
    slots.set(key, slot)
  }
  return slot
}

/** Resolve a catalog preset id to a WebLLM model id; pass real ids through. */
export function resolveWebLLMId(model: string, shaderF16 = true): string {
  const preset = findLLM(model)
  if (!preset) return model
  if (preset.runtime !== 'webllm') throw new Error(`Preset "${model}" is not a WebLLM model`)
  return `${preset.model}-${shaderF16 ? 'q4f16_1' : 'q4f32_1'}-MLC`
}

/**
 * xgrammar structural tag: text is free, but once the model writes
 * `<tool_call>` it must complete a call to a real tool with arguments that
 * validate against that tool's JSON schema.
 */
export function toolCallStructuralTag(tools: ToolSpec[]): object {
  return {
    type: 'structural_tag',
    format: {
      type: 'triggered_tags',
      triggers: ['<tool_call>'],
      tags: tools.map((t) => ({
        type: 'tag',
        begin: `<tool_call>\n{"name": ${JSON.stringify(t.name)}, "arguments": `,
        content: { type: 'json_schema', json_schema: t.inputSchema },
        end: '}\n</tool_call>',
      })),
    },
  }
}

/**
 * Fully offline inference on the user's GPU via WebGPU. Tuned for agent loops:
 * - the prompt is laid out so WebLLM reuses its KV cache between steps and turns,
 * - tool calls are grammar-constrained to valid tools and arguments,
 * - per-step prefill/decode metrics are reported for tuning.
 */
export function webllm(options: WebLLMOptions): WebLLMModel {
  const modelId = resolveWebLLMId(options.model)
  const contextWindow = options.contextWindow ?? findLLM(options.model)?.contextWindow ?? 8192
  const key = `${modelId}@${contextWindow}`
  const slot = slotFor(options.worker, options.appConfig)
  const thinkingMode = options.thinking ?? 'auto'
  const budget = options.thinkingBudget ?? 2048
  const qwen3 = /^Qwen3/i.test(modelId)

  const load = async (): Promise<void> => {
    if (options.onProgress) slot.progress.add(options.onProgress)
    try {
      // Chain loads so concurrent callers never race `reload`.
      slot.loading = slot.loading.catch(() => undefined).then(async () => {
        if (slot.loaded === key) return
        const engine = await slot.engine
        slot.loaded = undefined
        slot.transcript = undefined
        await engine.reload(modelId, { context_window_size: contextWindow })
        slot.loaded = key
      })
      await slot.loading
    } finally {
      if (options.onProgress) slot.progress.delete(options.onProgress)
    }
  }

  const model = fromTextModel(
    {
      id: `webllm:${modelId}`,
      contextWindow,
      locality: 'device',
      async *streamText({ system, messages, tools, after, signal, stop }) {
        await load()
        const engine = await slot.engine
        const thinking = thinkingMode === 'auto' ? after === 'user' : thinkingMode
        const wire: { role: 'system' | 'user' | 'assistant'; content: string }[] = [
          { role: 'system', content: system },
          ...messages,
        ]
        const transcript = wire.map((m) => `${m.role}\u0000${m.content}`)
        const reuse = extendsTranscript(slot.transcript, transcript)

        let grammar: string | undefined
        if (tools.length && options.constrainToolCalls !== false) {
          grammar = JSON.stringify(toolCallStructuralTag(tools))
          if (slot.badGrammars.has(grammar)) grammar = undefined
        }

        const request = (withGrammar: boolean, think = thinking) =>
          engine.chat.completions.create({
            stream: true,
            stream_options: { include_usage: true },
            messages: wire,
            ...(stop?.length ? { stop } : {}),
            ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
            ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
            ...(qwen3 ? { extra_body: { enable_thinking: think } } : {}),
            ...(withGrammar && grammar
              ? { response_format: { type: 'structural_tag' as const, structural_tag: grammar } }
              : {}),
          })

        let chunks: Awaited<ReturnType<typeof request>>
        try {
          chunks = await request(true)
        } catch (error) {
          if (!grammar) throw error
          // A schema xgrammar can't compile: remember it and run unconstrained.
          slot.badGrammars.add(grammar)
          chunks = await request(false)
        }

        const onAbort = () => engine.interruptGenerate()
        signal?.addEventListener('abort', onAbort, { once: true })
        let output = ''
        let usage: CompletionUsage | undefined
        let reasoningTokens = 0
        let reasoningText = ''
        let checkedAt = 0
        let inThink: boolean | undefined // undefined until the reply's opening is seen
        let overBudget = false
        try {
          for await (const chunk of chunks) {
            // Once interrupted, drain the stream without using it. Breaking out
            // would strand a worker engine's generator (worker streams are
            // pulled chunk by chunk), and the next request would wait forever.
            if (overBudget) continue
            const delta = chunk.choices[0]?.delta?.content
            if (delta) {
              output += delta
              yield delta
              if (inThink === undefined && output.trimStart().length >= 7) inThink = output.trimStart().startsWith('<think>')
              if (inThink && delta.includes('</think>')) inThink = false
              if (thinking && inThink) {
                reasoningText += delta
                // Streamed chunks are single tokens; count those inside <think>.
                const overTokens = budget > 0 && ++reasoningTokens > budget
                // Every ~400 characters, check whether the reasoning is looping.
                const looping = reasoningText.length - checkedAt > 400 && ((checkedAt = reasoningText.length), reasoningLoops(reasoningText))
                if (overTokens || looping) {
                  overBudget = true
                  await engine.interruptGenerate()
                }
              }
            }
            if (chunk.usage) usage = chunk.usage
          }
          if (overBudget) {
            // Close the reasoning, then answer from the same prompt without thinking.
            yield '\n</think>\n\n'
            const answer = await request(!!grammar && !slot.badGrammars.has(grammar), false)
            output = ''
            for await (const chunk of answer) {
              const delta = chunk.choices[0]?.delta?.content
              if (delta) {
                output += delta
                yield delta
              }
              if (chunk.usage) usage = chunk.usage
            }
          }
        } finally {
          signal?.removeEventListener('abort', onAbort)
        }
        signal?.throwIfAborted()
        // What the engine now holds: this prompt plus the reply it generated.
        slot.transcript = [...transcript, `assistant\u0000${output}`]

        if (usage) {
          const extra = usage.extra
          yield {
            usage: { inputTokens: usage.prompt_tokens, outputTokens: usage.completion_tokens },
            metrics: {
              prefillTokens: usage.prompt_tokens,
              kvCacheReused: reuse,
              ...(overBudget ? { thinkingCutOff: true } : {}),
              ...(extra
                ? {
                    timeToFirstTokenMs: Math.round(extra.time_to_first_token_s * 1000),
                    prefillTokensPerSec: Math.round(extra.prefill_tokens_per_s),
                    decodeTokensPerSec: Math.round(extra.decode_tokens_per_s),
                    ...(extra.grammar_init_s !== undefined ? { grammarInitMs: Math.round(extra.grammar_init_s * 1000) } : {}),
                  }
                : {}),
            },
          }
        }
      },
    },
    { contextPlacement: 'message', ...(options.reasoningHistory ? { reasoningHistory: options.reasoningHistory } : {}) },
  )

  return {
    ...model,
    modelId,
    contextWindow,
    load,
    async unload() {
      if (slot.loaded !== key) return
      slot.loaded = undefined
      slot.transcript = undefined
      await (await slot.engine).unload()
    },
    // Checked against the same appConfig used to download, so self-hosted
    // weights (cached under your URLs) are reported correctly.
    isCached: () => isWebLLMCached(modelId, options.appConfig),
    async clearCache() {
      if (slot.loaded === key) {
        slot.loaded = undefined
        slot.transcript = undefined
        await (await slot.engine).unload()
      }
      await deleteWebLLMCache(modelId, options.appConfig)
    },
  }
}

/** WebLLM reuses its KV cache when everything but the newest message matches what it holds. */
function extendsTranscript(held: string[] | undefined, next: string[]): boolean {
  if (!held || next.length !== held.length + 1) return false
  return held.every((entry, i) => entry === next[i])
}

export type { TextMessage }

/**
 * An AppConfig that loads weights and compiled model libraries from your own
 * host instead of Hugging Face / GitHub. Mirror the files first with
 * `enclave-mirror --webllm <model-id> --out <dir>`.
 */
export async function selfHostedAppConfig(options: { baseUrl: string; models?: string[] }): Promise<AppConfig> {
  const { prebuiltAppConfig } = await import('@mlc-ai/web-llm')
  const base = absolute(options.baseUrl)
  const wanted = options.models?.map((m) => resolveWebLLMId(m))
  return {
    ...prebuiltAppConfig,
    model_list: prebuiltAppConfig.model_list
      .filter((record) => !wanted || wanted.includes(record.model_id))
      .map((record) => ({
        ...record,
        model: `${base}/mlc/${record.model_id}/`,
        model_lib: `${base}/mlc/libs/${record.model_lib.split('/').pop()}`,
      })),
  }
}

/**
 * Whether the model can load offline: weights and the compiled model library
 * are both in the browser cache. (WebLLM never caches libraries served from a
 * `localhost` URL, so those report false: fine in development, and the reason
 * to test offline behaviour on a real hostname.)
 */
export async function isWebLLMCached(model: string, appConfig?: AppConfig): Promise<boolean> {
  const lib = await import('@mlc-ai/web-llm')
  const id = resolveWebLLMId(model)
  if (!(await lib.hasModelInCache(id, appConfig))) return false
  const record = (appConfig ?? lib.prebuiltAppConfig).model_list.find((r) => r.model_id === id)
  if (!record?.model_lib) return false
  if (typeof caches === 'undefined') return true
  // Default Cache API backend; other backends (IndexedDB, cross-origin storage) only report weights.
  const backend = (appConfig as { cacheBackend?: string } | undefined)?.cacheBackend
  if (backend && backend !== 'cache') return true
  const cache = await caches.open('webllm/wasm')
  return (await cache.match(new Request(record.model_lib))) !== undefined
}

/** Free the disk space used by a downloaded model. */
export async function deleteWebLLMCache(model: string, appConfig?: AppConfig): Promise<void> {
  const lib = await import('@mlc-ai/web-llm')
  await lib.deleteModelAllInfoInCache(resolveWebLLMId(model), appConfig)
}
