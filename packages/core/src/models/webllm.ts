import type { AppConfig, CompletionUsage, InitProgressReport, MLCEngineInterface } from '@mlc-ai/web-llm'
import type { Model, ToolSpec } from '../types.js'
import { findLLM } from '../web/catalog.js'
import { fromTextModel, type TextMessage } from './text-protocol.js'
import { absolute } from '../privacy/index.js'

export type ThinkingMode = boolean | 'auto'

export interface WebLLMOptions {
  /**
   * A WebLLM model id (e.g. `Qwen3-4B-q4f16_1-MLC`) or a catalog preset id
   * (e.g. `qwen3-4b`, resolved to the f16 build). Prefer `recommendLLM()`
   * from `@enclave/core/web` to pick one that fits the device.
   */
  model: string
  /**
   * Run inference in a dedicated worker (recommended). Its entry calls
   * `serveWebLLM()` from `@enclave/core/models/webllm-worker`. Models created
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
   * Constrain decoding so every `<tool_call>` is a real tool with schema-valid
   * arguments (xgrammar structural tags). Free text is unaffected. Default true.
   */
  constrainToolCalls?: boolean
  /**
   * `current-turn` (default) drops reasoning from earlier turns like Qwen3's
   * template. `all` keeps it so the KV cache also survives across turns.
   */
  reasoningHistory?: 'current-turn' | 'all'
  /**
   * Where to download weights and compiled model libraries from. Use
   * `selfHostedAppConfig()` to serve everything from your own origin.
   */
  appConfig?: AppConfig
  onProgress?(report: InitProgressReport): void
}

export interface WebLLMModel extends Model {
  readonly modelId: string
  readonly contextWindow: number
  /** Download (first run) and compile the model. Called lazily on first use. */
  load(): Promise<void>
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

        const request = (withGrammar: boolean) =>
          engine.chat.completions.create({
            stream: true,
            stream_options: { include_usage: true },
            messages: wire,
            ...(stop?.length ? { stop } : {}),
            ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
            ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
            ...(qwen3 ? { extra_body: { enable_thinking: thinking } } : {}),
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
        try {
          for await (const chunk of chunks) {
            const delta = chunk.choices[0]?.delta?.content
            if (delta) {
              output += delta
              yield delta
            }
            if (chunk.usage) usage = chunk.usage
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

/** Whether the model's weights are already in the browser cache (loads offline). */
export async function isWebLLMCached(model: string, appConfig?: AppConfig): Promise<boolean> {
  const lib = await import('@mlc-ai/web-llm')
  return lib.hasModelInCache(resolveWebLLMId(model), appConfig)
}

/** Free the disk space used by a downloaded model. */
export async function deleteWebLLMCache(model: string, appConfig?: AppConfig): Promise<void> {
  const lib = await import('@mlc-ai/web-llm')
  await lib.deleteModelAllInfoInCache(resolveWebLLMId(model), appConfig)
}
