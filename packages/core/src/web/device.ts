export interface DeviceProfile {
  webgpu: boolean
  /** WebGPU `shader-f16`: enables the faster, smaller q4f16 model builds. */
  shaderF16: boolean
  adapter: { vendor: string; architecture: string; description: string } | undefined
  /** Largest single GPU storage buffer, in MB. Below ~1 GB only small models load. */
  maxStorageBufferMB: number
  /** `navigator.deviceMemory` (GB, capped at 8 by browsers), when exposed. */
  deviceMemoryGB: number | undefined
  mobile: boolean
  /** Estimated GPU memory we can safely use for model weights + KV cache, in MB. */
  gpuBudgetMB: number
  crossOriginIsolated: boolean
  storage: { quotaMB: number; usageMB: number; persisted: boolean } | undefined
}

interface GPUAdapterLike {
  features: { has(name: string): boolean }
  limits: { maxStorageBufferBindingSize: number; maxBufferSize: number }
  info?: { vendor?: string; architecture?: string; description?: string }
}

/**
 * Probe what this browser can run. Cheap (no device is created) and safe to
 * call from a page or a worker.
 */
export async function detectDevice(options: { gpuBudgetMB?: number } = {}): Promise<DeviceProfile> {
  const nav = (typeof navigator === 'undefined' ? {} : navigator) as {
    gpu?: { requestAdapter(o?: unknown): Promise<GPUAdapterLike | null> }
    deviceMemory?: number
    userAgent?: string
    maxTouchPoints?: number
    userAgentData?: { mobile?: boolean }
    storage?: { estimate?(): Promise<{ quota?: number; usage?: number }>; persisted?(): Promise<boolean> }
  }

  let adapter: GPUAdapterLike | null = null
  try {
    adapter = (await nav.gpu?.requestAdapter({ powerPreference: 'high-performance' })) ?? null
  } catch {
    adapter = null
  }

  // iPadOS Safari asks for desktop sites: its user agent says "Macintosh", but a Mac has no touch screen.
  const ua = nav.userAgent ?? ''
  const mobile = nav.userAgentData?.mobile ?? (/Android|iPhone|iPad|Mobile/i.test(ua) || (/Macintosh/.test(ua) && (nav.maxTouchPoints ?? 0) > 1))
  const maxStorageBufferMB = adapter ? Math.round(adapter.limits.maxStorageBufferBindingSize / 2 ** 20) : 0
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

  const profile: DeviceProfile = {
    webgpu: !!adapter,
    shaderF16: !!adapter?.features.has('shader-f16'),
    adapter: adapter
      ? {
          vendor: adapter.info?.vendor ?? '',
          architecture: adapter.info?.architecture ?? '',
          description: adapter.info?.description ?? '',
        }
      : undefined,
    maxStorageBufferMB,
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
 * WebGPU does not report VRAM, so this is a conservative heuristic:
 * half of reported RAM (browsers cap it at 8 GB, so desktops land on 4 GB),
 * less on phones, and nothing without WebGPU.
 */
export function estimateGpuBudget(profile: Pick<DeviceProfile, 'webgpu' | 'mobile' | 'deviceMemoryGB' | 'maxStorageBufferMB'>): number {
  if (!profile.webgpu) return 0
  if (profile.mobile) return Math.min(1800, (profile.deviceMemoryGB ?? 4) * 400)
  const fromMemory = profile.deviceMemoryGB ? profile.deviceMemoryGB * 1024 * 0.5 : 3000
  // Adapters with small binding limits are usually integrated or older GPUs.
  return profile.maxStorageBufferMB < 1024 ? Math.min(fromMemory, 2000) : fromMemory
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
