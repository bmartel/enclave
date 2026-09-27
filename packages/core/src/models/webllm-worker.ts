import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

/**
 * Worker entry for `webllm({ worker })`:
 *
 * ```ts
 * // llm.worker.ts
 * import { serveWebLLM } from '@enclave/core/models/webllm-worker'
 * serveWebLLM()
 * ```
 */
export function serveWebLLM(): void {
  const handler = new WebWorkerMLCEngineHandler()
  self.onmessage = (event: MessageEvent) => handler.onmessage(event)
}
