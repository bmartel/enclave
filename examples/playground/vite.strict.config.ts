import { mergeConfig } from 'vite'
import { contentSecurityPolicy } from 'enclave-ai/privacy'
import base from './vite.config'

/**
 * Strict privacy mode: models are served from /models (see `pnpm mirror`),
 * and the Content-Security-Policy lets the page and its workers talk only to
 * this origin. The browser blocks any other connection.
 */
const csp = contentSecurityPolicy({ modelHosts: [] })

export default mergeConfig(base, {
  server: { headers: { 'Content-Security-Policy': csp } },
  preview: { headers: { 'Content-Security-Policy': csp } },
  define: { 'import.meta.env.VITE_STRICT': JSON.stringify('1') },
})
