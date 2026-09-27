import type { Model, ModelChunk, ModelRequest } from '../types.js'
import { widest } from '../privacy/index.js'

/**
 * Try models in order, moving to the next only if one fails before producing
 * output. Typical use: a remote model when reachable, a local one otherwise.
 *
 * ```ts
 * fallback(anthropic({ ... }), webllm({ model: 'Qwen3-4B-q4f16_1-MLC' }))
 * ```
 */
export function fallback(...models: Model[]): Model {
  if (!models.length) throw new Error('fallback() needs at least one model')
  return {
    id: `fallback(${models.map((m) => m.id).join(', ')})`,
    locality: widest(...models.map((m) => m.locality)),
    async *stream(request: ModelRequest): AsyncGenerator<ModelChunk> {
      let lastError: unknown
      for (const model of models) {
        let started = false
        try {
          for await (const chunk of model.stream(request)) {
            started = true
            yield chunk
          }
          return
        } catch (error) {
          if (started || request.signal?.aborted) throw error
          lastError = error
        }
      }
      throw lastError
    },
  }
}
