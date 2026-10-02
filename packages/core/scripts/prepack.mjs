/**
 * Runs before `npm pack` / `npm publish`: copies the repository README and
 * LICENSE into the package, with relative links made absolute so they work
 * on npmjs.com.
 */
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const pkg = join(dirname(fileURLToPath(import.meta.url)), '..')
const root = join(pkg, '..', '..')
const repo = 'https://github.com/bmartel/enclave'

const readme = readFileSync(join(root, 'README.md'), 'utf8').replace(
  /\]\((?!https?:|#|mailto:)([^)\s]+)\)/g,
  (_, path) => `](${repo}/${/\.[a-z]+$/i.test(path.split('#')[0]) ? 'blob' : 'tree'}/main/${path.replace(/^\.\//, '')})`,
)
writeFileSync(join(pkg, 'README.md'), readme)
copyFileSync(join(root, 'LICENSE'), join(pkg, 'LICENSE'))
console.log('prepack: README.md and LICENSE copied into the package')
