import { createDb } from '../src/store/pglite.js'

export const memoryDb = () => createDb({ dataDir: 'memory://' })

export async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = []
  for await (const item of iter) out.push(item)
  return out
}
