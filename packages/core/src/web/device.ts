/**
 * Whether WebGPU can run models here, and if not why:
 * - `available`: a hardware adapter answered.
 * - `no-api`: no `navigator.gpu` (the browser, a worker without WebGPU, an insecure context).
 * - `no-adapter`: `navigator.gpu` exists but every `requestAdapter()` returned null
 *   (blocklisted driver, graphics acceleration turned off, GPU process blocked after crashes).
 * - `fallback-adapter`: only a software (CPU) adapter: slower than WASM for models.
 * - `error`: `requestAdapter()` threw.
 */
export type WebGpuStatus = 'available' | 'no-api' | 'no-adapter' | 'fallback-adapter' | 'error'

/** Which request found the adapter (see `requestGpuAdapter`). */
export type AdapterRequest = 'high-performance' | 'default' | 'retry'

export interface DeviceProfile {
  webgpu: boolean
  /** Why `webgpu` is what it is. Absent on profiles made by older versions. */
  webgpuStatus?: WebGpuStatus
  /** The error `requestAdapter()` threw (`webgpuStatus: 'error'`). */
  webgpuError?: string
  /** The request that found the adapter: a `null` from the high-performance request is retried. */
  adapterRequest?: AdapterRequest
  /**
   * WebGPU `shader-f16`: enables the faster, smaller q4f16 model builds.
   * Without it models still run on the GPU, from full-precision (f32) builds.
   */
  shaderF16: boolean
  adapter: { vendor: string; architecture: string; description: string } | undefined
  /** Largest single GPU storage buffer, in MB. Below ~1 GB only small models load. */
  maxStorageBufferMB: number
  /** Largest GPU buffer, in MB. */
  maxBufferMB?: number
  /**
   * A desktop graphics card with its own memory (NVIDIA, Intel Arc, Radeon RX):
   * WebGPU reports no memory size, so this lets f32 builds plan for more than
   * the RAM-based estimate. False when unsure (integrated GPUs, Apple, AMD APUs).
   */
  discreteGpu?: boolean
  /** `navigator.deviceMemory` (GB, capped at 8 by browsers), when exposed. */
  deviceMemoryGB: number | undefined
  mobile: boolean
  /** Estimated GPU memory we can safely use for model weights + KV cache, in MB. */
  gpuBudgetMB: number
  crossOriginIsolated: boolean
  storage: { quotaMB: number; usageMB: number; persisted: boolean } | undefined
}

export interface GPUAdapterLike {
  features: { has(name: string): boolean }
  limits: { maxStorageBufferBindingSize: number; maxBufferSize: number }
  info?: { vendor?: string; architecture?: string; description?: string; isFallbackAdapter?: boolean }
  /** Older browsers put it on the adapter. */
  isFallbackAdapter?: boolean
}

export interface GPULike {
  requestAdapter(options?: { powerPreference?: 'high-performance' | 'low-power' }): Promise<GPUAdapterLike | null>
}

export interface AdapterResult {
  adapter: GPUAdapterLike | null
  status: WebGpuStatus
  request?: AdapterRequest
  error?: string
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))
const isFallback = (a: GPUAdapterLike) => !!(a.info?.isFallbackAdapter ?? a.isFallbackAdapter)

/**
 * Ask for a GPU adapter the way that works across browsers:
 * 1. `powerPreference: 'high-performance'` (picks the discrete GPU on dual-GPU Macs
 *    and laptops); skipped on Windows, where Chrome ignores it with a console warning.
 * 2. No options: the browser's default adapter. A high-performance request can come
 *    back null where the default one doesn't.
 * 3. Once more after `retryDelayMs`: the first request can find the GPU process starting.
 * A software (fallback) adapter counts as none: models run faster on WASM.
 */
export async function requestGpuAdapter(gpu: GPULike | undefined, options: { windows?: boolean; retryDelayMs?: number } = {}): Promise<AdapterResult> {
  if (!gpu || typeof gpu.requestAdapter !== 'function') return { adapter: null, status: 'no-api' }
  const attempts: AdapterRequest[] = options.windows ? ['default', 'retry'] : ['high-performance', 'default', 'retry']
  let error: string | undefined
  let fallback = false
  for (const request of attempts) {
    if (request === 'retry') await wait(options.retryDelayMs ?? 200)
    try {
      const adapter = await (request === 'high-performance' ? gpu.requestAdapter({ powerPreference: 'high-performance' }) : gpu.requestAdapter())
      if (adapter && !isFallback(adapter)) return { adapter, status: 'available', request }
      if (adapter) fallback = true
    } catch (err) {
      error = String((err as Error)?.message ?? err)
    }
  }
  if (fallback) return { adapter: null, status: 'fallback-adapter' }
  return error ? { adapter: null, status: 'error', error } : { adapter: null, status: 'no-adapter' }
}

/** A desktop graphics card with its own memory, from what the adapter says about itself. */
export function isDiscreteGpu(adapter: DeviceProfile['adapter'], mobile: boolean): boolean {
  if (!adapter || mobile) return false
  const vendor = adapter.vendor.toLowerCase()
  const text = `${adapter.architecture} ${adapter.description}`.toLowerCase()
  if (vendor === 'nvidia' || /geforce|quadro|\brtx\b|\bgtx\b/.test(text)) return !/tegra/.test(text)
  // Intel Arc (Xe-HPG, Battlemage); Xe-LP/LPG are integrated.
  if (vendor === 'intel') return /hpg|arc\s?[ab]\d/.test(text)
  // AMD: APUs share an architecture name with Radeon RX cards, so only the description tells.
  if (vendor === 'amd') return /radeon\s+(rx|pro)/.test(text)
  return false
}

/**
 * Probe what this browser can run. Cheap (no device is created) and safe to
 * call from a page or a worker.
 */
export async function detectDevice(options: { gpuBudgetMB?: number; retryDelayMs?: number } = {}): Promise<DeviceProfile> {
  const nav = (typeof navigator === 'undefined' ? {} : navigator) as {
    gpu?: GPULike
    deviceMemory?: number
    userAgent?: string
    maxTouchPoints?: number
    userAgentData?: { mobile?: boolean; platform?: string }
    storage?: { estimate?(): Promise<{ quota?: number; usage?: number }>; persisted?(): Promise<boolean> }
  }

  // iPadOS Safari asks for desktop sites: its user agent says "Macintosh", but a Mac has no touch screen.
  const ua = nav.userAgent ?? ''
  const mobile = nav.userAgentData?.mobile ?? (/Android|iPhone|iPad|Mobile/i.test(ua) || (/Macintosh/.test(ua) && (nav.maxTouchPoints ?? 0) > 1))
  const windows = /windows/i.test(nav.userAgentData?.platform ?? '') || /Windows/.test(ua)

  const found = await requestGpuAdapter(nav.gpu, { windows, ...(options.retryDelayMs !== undefined ? { retryDelayMs: options.retryDelayMs } : {}) })
  const adapter = found.adapter
  const maxStorageBufferMB = adapter ? Math.round(adapter.limits.maxStorageBufferBindingSize / 2 ** 20) : 0
  const maxBufferMB = adapter ? Math.round(adapter.limits.maxBufferSize / 2 ** 20) : 0
  const deviceMemoryGB = nav.deviceMemory

  let storage: DeviceProfile['storage']
  try {
    const estimate = await nav.storage?.estimate?.()
    if (estimate) {
      storage = {
        quotaMB: Math.round((estimate.quota ?? 0) / 2 ** 20),
        usageMB: Math.round((estimate.usage ?? 0) / 2 ** 20),
        persisted: (await nav.storage?.persisted?.()) ?? false,
      }
    }
  } catch {
    storage = undefined
  }

  const info = adapter
    ? {
        vendor: adapter.info?.vendor ?? '',
        architecture: adapter.info?.architecture ?? '',
        description: adapter.info?.description ?? '',
      }
    : undefined
  const profile: DeviceProfile = {
    webgpu: !!adapter,
    webgpuStatus: found.status,
    ...(found.error ? { webgpuError: found.error } : {}),
    ...(found.request ? { adapterRequest: found.request } : {}),
    shaderF16: !!adapter?.features.has('shader-f16'),
    adapter: info,
    maxStorageBufferMB,
    maxBufferMB,
    discreteGpu: isDiscreteGpu(info, mobile),
    deviceMemoryGB,
    mobile,
    gpuBudgetMB: 0,
    crossOriginIsolated: typeof globalThis.crossOriginIsolated === 'boolean' ? globalThis.crossOriginIsolated : false,
    storage,
  }
  profile.gpuBudgetMB = options.gpuBudgetMB ?? estimateGpuBudget(profile)
  return profile
}

/**
 * The profile with some fields replaced, its GPU budget estimated again
 * (unless `gpuBudgetMB` is given). For trying another device's path on this
 * one, e.g. `{ shaderF16: false, discreteGpu: true }`: a desktop card that
 * runs WebGPU without half precision (Windows on Vulkan).
 */
export function overrideDevice(device: DeviceProfile, overrides: Partial<DeviceProfile>, gpuBudgetMB?: number): DeviceProfile {
  const next = { ...device, ...overrides }
  next.gpuBudgetMB = gpuBudgetMB ?? overrides.gpuBudgetMB ?? estimateGpuBudget(next)
  return next
}

/**
 * GPU memory full-precision (f32) builds may plan for on a discrete desktop
 * card, in MB. The 4 GB estimate fits a 4B model's f16 build but not its f32
 * one (weights ~25% bigger, KV cache doubled), which would drop such cards to
 * a 3B model. Cards that run WebGPU without `shader-f16` (Windows on Vulkan,
 * where D3D12 isn't usable) have 6 GB or more in practice.
 */
export const DISCRETE_F32_BUDGET_MB = 6144

/**
 * WebGPU does not report VRAM, so this is a conservative heuristic:
 * half of reported RAM (browsers cap it at 8 GB, so desktops land on 4 GB),
 * less on phones, and nothing without WebGPU. A discrete card without
 * `shader-f16` gets `DISCRETE_F32_BUDGET_MB`, so the same model tier runs
 * from its f32 build instead of dropping to a smaller model.
 */
export function estimateGpuBudget(
  profile: Pick<DeviceProfile, 'webgpu' | 'mobile' | 'deviceMemoryGB' | 'maxStorageBufferMB'> & Partial<Pick<DeviceProfile, 'shaderF16' | 'discreteGpu'>>,
): number {
  if (!profile.webgpu) return 0
  if (profile.mobile) return Math.min(1800, (profile.deviceMemoryGB ?? 4) * 400)
  const fromMemory = profile.deviceMemoryGB ? profile.deviceMemoryGB * 1024 * 0.5 : 3000
  // Adapters with small binding limits are usually integrated or older GPUs.
  if (profile.maxStorageBufferMB < 1024) return Math.min(fromMemory, 2000)
  if (profile.discreteGpu && profile.shaderF16 === false) return Math.max(fromMemory, DISCRETE_F32_BUDGET_MB)
  return fromMemory
}

/**
 * Wrap `navigator.gpu.requestAdapter` so a request that comes back null is
 * tried again the ways `requestGpuAdapter` does. For libraries that ask for a
 * high-performance adapter and give up on null (WebLLM); call it in their
 * worker before they start.
 */
export function installAdapterFallback(gpu: GPULike | undefined = (globalThis.navigator as { gpu?: GPULike } | undefined)?.gpu): boolean {
  if (!gpu || typeof gpu.requestAdapter !== 'function' || (gpu as { enclaveAdapterFallback?: boolean }).enclaveAdapterFallback) return false
  const request = gpu.requestAdapter.bind(gpu)
  const patched: GPULike['requestAdapter'] = async (options) => {
    const first = await request(options).catch(() => null)
    if (first) return first
    return (await requestGpuAdapter({ requestAdapter: request }, { windows: true })).adapter
  }
  Object.assign(gpu, { requestAdapter: patched, enclaveAdapterFallback: true })
  return true
}

/**
 * Ask the browser not to evict our database and model caches under storage
 * pressure. Essential for offline-first apps; granted silently for installed
 * PWAs and frequently-used sites.
 */
export async function persistStorage(): Promise<boolean> {
  try {
    const storage = (navigator as { storage?: { persist?(): Promise<boolean> } }).storage
    return (await storage?.persist?.()) ?? false
  } catch {
    return false
  }
}
