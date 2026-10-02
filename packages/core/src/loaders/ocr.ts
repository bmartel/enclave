import type { OcrFunction } from './pdf.js'

/** The parts of the tesseract.js module this adapter uses. */
export interface TesseractLike {
  createWorker(
    langs?: string | string[],
    oem?: number,
    options?: Record<string, unknown>,
  ): Promise<{
    recognize(image: unknown): Promise<{ data: { text: string; confidence: number } }>
    terminate(): Promise<unknown>
  }>
}

export interface TesseractOcrOptions {
  /** The tesseract.js module, or a function that imports it: `() => import('tesseract.js')`. */
  lib: TesseractLike | (() => Promise<TesseractLike>)
  /**
   * Where `enclave-mirror --ocr` put the files, e.g. `/models/ocr`. Expects
   * `worker.min.js`, `core/` and `lang/` under it.
   */
  baseUrl?: string
  /** Override individual locations instead of `baseUrl`. */
  workerPath?: string
  corePath?: string
  langPath?: string
  /**
   * Allow tesseract.js's defaults, which download the engine and language data
   * from jsdelivr. Images are still read locally, but the page contacts a CDN.
   * Default false: without `baseUrl` or explicit paths, creating the OCR function throws.
   */
  cdn?: boolean
  /** Tesseract language codes. Default `eng`. Mirror each one (`--ocr eng,deu`). */
  languages?: string | string[]
  /** Text below this mean confidence (0–100) is dropped as noise. Default 30. */
  minConfidence?: number
  /** Engine loading and recognition progress. */
  onProgress?(progress: { status: string; progress: number }): void
}

export type TesseractOcr = OcrFunction & {
  /** Free the worker (its WASM memory is large). The next call starts a new one. */
  terminate(): Promise<void>
}

/**
 * An `ocr` function for the loaders, backed by tesseract.js running in a web
 * worker. The engine starts on first use and is reused. Serve its files from
 * your own origin with `enclave-mirror --ocr eng`.
 *
 *   const ocr = tesseractOcr({ lib: () => import('tesseract.js'), baseUrl: '/models/ocr' })
 *   await loadFiles(files, { ocr, pdf })
 */
export function tesseractOcr(options: TesseractOcrOptions): TesseractOcr {
  const paths = resolvePaths(options)
  let worker: ReturnType<TesseractLike['createWorker']> | undefined

  const start = async () => {
    const lib = typeof options.lib === 'function' ? await options.lib() : options.lib
    return lib.createWorker(options.languages ?? 'eng', 1, {
      ...paths,
      // Self-hosted language files are mirrored gzipped, as tesseract.js expects by default.
      ...(options.onProgress ? { logger: (m: { status: string; progress: number }) => options.onProgress!({ status: m.status, progress: m.progress }) } : {}),
    })
  }

  const recognize: OcrFunction = async (image) => {
    worker ??= start().catch((error) => {
      worker = undefined
      throw error
    })
    const { data } = await (await worker).recognize(image)
    return data.confidence >= (options.minConfidence ?? 30) ? data.text.trim() : ''
  }
  return Object.assign(recognize, {
    async terminate() {
      const current = worker
      worker = undefined
      if (current) await (await current.catch(() => undefined))?.terminate()
    },
  })
}

function resolvePaths(options: TesseractOcrOptions): Record<string, string> {
  const base = options.baseUrl?.replace(/\/$/, '')
  const paths: Record<string, string> = {}
  const set = (key: string, explicit: string | undefined, fallback: string | undefined) => {
    const value = explicit ?? fallback
    if (value) paths[key] = absolute(value)
  }
  set('workerPath', options.workerPath, base && `${base}/worker.min.js`)
  set('corePath', options.corePath, base && `${base}/core`)
  set('langPath', options.langPath, base && `${base}/lang`)
  const missing = ['workerPath', 'corePath', 'langPath'].filter((k) => !paths[k])
  if (missing.length && !options.cdn) {
    throw new Error(
      `tesseractOcr needs self-hosted files (missing ${missing.join(', ')}). Run \`enclave-mirror --out public/models --ocr eng\` and pass baseUrl: '/models/ocr', or pass cdn: true to download from jsdelivr.`,
    )
  }
  return paths
}

/** tesseract.js loads its worker from a blob URL, so relative paths must be made absolute. */
function absolute(path: string): string {
  if (typeof location === 'undefined' || /^[a-z][a-z0-9+.-]*:/i.test(path)) return path
  return new URL(path, location.href).href
}
