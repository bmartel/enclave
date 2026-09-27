import type { Locality } from '../types.js'

export type { Locality } from '../types.js'

const ORDER: Record<Locality, number> = { device: 0, 'local-network': 1, remote: 2 }

/**
 * Classify an endpoint: loopback is on this `device`; private addresses and
 * `.local` hosts are the `local-network`; anything else is `remote`.
 */
export function localityOfUrl(url: string): Locality {
  let host: string
  try {
    host = new URL(url, typeof location === 'undefined' ? 'http://localhost/' : location.href).hostname
  } catch {
    return 'remote'
  }
  host = host.replace(/^\[|\]$/g, '').toLowerCase()
  if (host === 'localhost' || host.endsWith('.localhost') || host === '::1' || /^127\./.test(host)) return 'device'
  if (
    host.endsWith('.local') ||
    host.endsWith('.internal') ||
    /^10\./.test(host) ||
    /^192\.168\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(host) ||
    /^169\.254\./.test(host) ||
    /^f[cd][0-9a-f]{2}:/.test(host) ||
    /^fe80:/.test(host)
  ) {
    return 'local-network'
  }
  return 'remote'
}

/** The least private of several localities. */
export function widest(...localities: (Locality | undefined)[]): Locality {
  return localities.reduce<Locality>((w, l) => (ORDER[l ?? 'remote'] > ORDER[w] ? (l ?? 'remote') : w), 'device')
}

export function isAllowed(locality: Locality | undefined, allow: Locality): boolean {
  return ORDER[locality ?? 'remote'] <= ORDER[allow]
}

export class PrivacyError extends Error {
  constructor(what: string, locality: Locality | undefined, allow: Locality) {
    super(
      `${what} runs ${locality ? `on the ${locality === 'remote' ? 'internet' : locality}` : 'at an unknown location'}, ` +
        `but this enclave only allows "${allow}". Pass privacy: { allow: '${locality ?? 'remote'}' } to opt in explicitly.`,
    )
    this.name = 'PrivacyError'
  }
}

export function assertLocality(what: string, locality: Locality | undefined, allow: Locality): void {
  if (!isAllowed(locality, allow)) throw new PrivacyError(what, locality, allow)
}

// ---------------------------------------------------------------------------
// Content Security Policy
// ---------------------------------------------------------------------------

/** Where model files come from by default (Hugging Face, WebLLM libraries, ONNX Runtime WASM). */
export const PUBLIC_MODEL_HOSTS = [
  'https://huggingface.co',
  'https://*.huggingface.co',
  'https://*.hf.co',
  'https://raw.githubusercontent.com',
  'https://cdn.jsdelivr.net',
]

export interface CSPOptions {
  /**
   * Origins allowed to serve model weights. Default: the public hubs. Set to
   * `[]` when you self-host everything (see `selfHostedModels`), so the page
   * can only talk to its own origin.
   */
  modelHosts?: string[]
  /** Allow Ollama / LM Studio on this machine. Default false. */
  localServers?: boolean | string[]
  /** Extra `connect-src` entries. */
  connect?: string[]
}

/**
 * A Content-Security-Policy that makes the browser itself refuse to send data
 * anywhere else: `connect-src` is limited to your origin and model hosts, and
 * forms, frames and beacons are locked down. Serve it as a response header.
 */
export function contentSecurityPolicy(options: CSPOptions = {}): string {
  const local =
    options.localServers === true
      ? ['http://localhost:11434', 'http://127.0.0.1:11434', 'http://localhost:1234', 'http://127.0.0.1:1234']
      : Array.isArray(options.localServers)
        ? options.localServers
        : []
  const connect = ["'self'", 'blob:', 'data:', ...(options.modelHosts ?? PUBLIC_MODEL_HOSTS), ...local, ...(options.connect ?? [])]
  return [
    "default-src 'self'",
    // WASM compilation (PGlite, ONNX Runtime, WebLLM) needs wasm-unsafe-eval.
    "script-src 'self' 'wasm-unsafe-eval' blob:",
    "worker-src 'self' blob:",
    `connect-src ${[...new Set(connect)].join(' ')}`,
    "img-src 'self' data: blob:",
    "style-src 'self' 'unsafe-inline'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'none'",
    "frame-ancestors 'self'",
  ].join('; ')
}

// ---------------------------------------------------------------------------
// Runtime network guard
// ---------------------------------------------------------------------------

export interface NetworkGuardOptions {
  /** Origins (or `(url) => boolean`) that may be contacted. `'self'` is always allowed. */
  allow?: string[] | ((url: URL) => boolean)
  /** `block` (default) throws; `report` only calls `onViolation`. */
  mode?: 'block' | 'report'
  onViolation?(url: string, api: string): void
}

/**
 * Defense in depth for the current page or worker: wraps `fetch`,
 * `XMLHttpRequest`, `WebSocket`, `EventSource` and `sendBeacon` so requests
 * to unlisted origins are blocked (or reported). Call it at the top of your
 * entry and worker files. CSP is the stronger guarantee; use both.
 */
export function guardNetwork(options: NetworkGuardOptions = {}): () => void {
  const g = globalThis as Record<string, any>
  const self = typeof location === 'undefined' ? undefined : location.origin
  const origins = Array.isArray(options.allow) ? options.allow : []
  const allowed = (raw: string | URL): boolean => {
    let url: URL
    try {
      url = new URL(String(raw), typeof location === 'undefined' ? undefined : location.href)
    } catch {
      return false
    }
    if (url.protocol === 'blob:' || url.protocol === 'data:') return true
    if (self && url.origin === self) return true
    if (typeof options.allow === 'function') return options.allow(url)
    return origins.some((o) => matchOrigin(o, url))
  }
  const check = (url: string | URL, api: string) => {
    if (allowed(url)) return
    options.onViolation?.(String(url), api)
    if ((options.mode ?? 'block') === 'block') throw new Error(`Network request to ${String(url)} blocked by enclave guardNetwork (${api})`)
  }

  const restore: (() => void)[] = []
  const wrap = (obj: any, key: string, make: (orig: any) => any) => {
    if (!obj || typeof obj[key] !== 'function') return
    const orig = obj[key]
    obj[key] = make(orig)
    restore.push(() => (obj[key] = orig))
  }

  wrap(g, 'fetch', (orig) =>
    function (this: unknown, input: RequestInfo | URL, init?: RequestInit) {
      const url = typeof input === 'string' || input instanceof URL ? input : input.url
      try {
        check(url, 'fetch')
      } catch (error) {
        return Promise.reject(error)
      }
      return orig.call(this, input, init)
    },
  )
  if (g.XMLHttpRequest) {
    wrap(g.XMLHttpRequest.prototype, 'open', (orig) =>
      function (this: unknown, method: string, url: string | URL, ...rest: unknown[]) {
        check(url, 'XMLHttpRequest')
        return orig.call(this, method, url, ...rest)
      },
    )
  }
  for (const name of ['WebSocket', 'EventSource'] as const) {
    const Orig = g[name]
    if (!Orig) continue
    const Guarded = function (url: string | URL, ...rest: unknown[]) {
      check(url, name)
      return new Orig(url, ...rest)
    } as unknown as typeof Orig
    Guarded.prototype = Orig.prototype
    g[name] = Guarded
    restore.push(() => (g[name] = Orig))
  }
  if (g.navigator?.sendBeacon) {
    wrap(g.navigator, 'sendBeacon', (orig) =>
      function (this: unknown, url: string | URL, data?: BodyInit | null) {
        check(url, 'sendBeacon')
        return orig.call(this, url, data)
      },
    )
  }
  return () => restore.reverse().forEach((r) => r())
}

function matchOrigin(pattern: string, url: URL): boolean {
  const [scheme, host] = pattern.includes('://') ? pattern.split('://') : [undefined, pattern]
  if (scheme && `${scheme}:` !== url.protocol) return false
  const hostPattern = host!.replace(/\/.*$/, '')
  if (hostPattern.startsWith('*.')) return url.host.endsWith(hostPattern.slice(1))
  return url.host === hostPattern
}

// ---------------------------------------------------------------------------
// Self-hosted models
// ---------------------------------------------------------------------------

export interface SelfHostOptions {
  /** Where `enclave-mirror` put the files, e.g. `/models` or `https://cdn.internal/models`. */
  baseUrl: string
}

export interface TransformersHosting {
  remoteHost: string
  remotePathTemplate: string
  wasmPaths: { mjs: string; wasm: string }
  allowRemoteModels: true
}

/**
 * Configuration that makes Transformers.js load models and the ONNX Runtime
 * WASM from your host (layout produced by `enclave-mirror`).
 */
export function selfHostedTransformers(options: SelfHostOptions): TransformersHosting {
  const base = absolute(options.baseUrl)
  return {
    remoteHost: `${base}/hf/`,
    remotePathTemplate: '{model}/resolve/{revision}/',
    wasmPaths: {
      mjs: `${base}/ort/ort-wasm-simd-threaded.asyncify.mjs`,
      wasm: `${base}/ort/ort-wasm-simd-threaded.asyncify.wasm`,
    },
    allowRemoteModels: true,
  }
}

/** Resolve relative bases against the page so workers get absolute URLs. */
export function absolute(baseUrl: string): string {
  const base = typeof location === 'undefined' ? new URL(baseUrl) : new URL(baseUrl, location.href)
  return base.href.replace(/\/$/, '')
}
