import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'
import { installAdapterFallback } from '../web/device.js'

/**
 * Worker entry for `webllm({ worker })`:
 *
 * ```ts
 * // llm.worker.ts
 * import { serveWebLLM } from 'enclave-ai/models/webllm-worker'
 * serveWebLLM()
 * ```
 */
export function serveWebLLM(): void {
  // WebLLM asks for a high-performance adapter and fails on null; some browsers
  // only answer a request without powerPreference (see requestGpuAdapter).
  installAdapterFallback()
  const handler = new WebWorkerMLCEngineHandler()
  self.onmessage = (event: MessageEvent) => handler.onmessage(event)
}
