# enclave-ai

[![npm](https://img.shields.io/npm/v/enclave-ai.svg)](https://www.npmjs.com/package/enclave-ai)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**AI agents that run in your user's browser.** The language model, the search index, a Postgres database and the chat history all live on the user's device. Add an assistant that answers questions from your documents, works with your app's data and calls your app's functions, with no AI backend and nothing sent to a server.

```ts
const ai = await createWebEnclave({ workers, skills: [knowledgeSkill(), sqlSkill()] })
await ai.knowledge!.ingest({ title: 'IT handbook', content: handbookText })

for await (const event of ai.thread('support').send('What is the guest wifi password?')) {
  if (event.type === 'text-delta') render(event.delta)
}
// → "The guest wifi password is maple-harbor-42 [1]."
```

What you get:

- **Answers from your documents**, with citations. Drop in PDFs, Word, PowerPoint, Excel, EPUB, HTML, CSV or images.
- **Questions about data answered with SQL**, run against a real Postgres database in the browser.
- **Your own tools**: typed functions the assistant can call, with a confirmation step for anything risky.
- **Memory** of the user's preferences across conversations.
- **Works offline** once the model is downloaded.
- **Use a bigger model when one is available.** It also works with Ollama or LM Studio on the same machine, which is faster and more accurate.
- **Framework-free.** It's a plain async API, with examples below for React, Vue, Svelte, Angular, Alacris and plain JavaScript.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Use it with your framework](#use-it-with-your-framework)
- [Add documents and files](#add-documents-and-files)
- [Answer questions with SQL](#answer-questions-with-sql)
- [Give the assistant your app's tools](#give-the-assistant-your-apps-tools)
- [Choose where the model runs](#choose-where-the-model-runs)
- [Keep data private](#keep-data-private)
- [Build the chat UI](#build-the-chat-ui)
- [Memory](#memory)
- [Offline use and downloads](#offline-use-and-downloads)
- [Test your assistant](#test-your-assistant)
- [Benchmarks](#benchmarks)
- [API reference](#api-reference)
- [Browser support and limits](#browser-support-and-limits)
- [Contributing](#contributing)

## Install

```sh
npm install enclave-ai zod @electric-sql/pglite @electric-sql/pglite-pgvector @mlc-ai/web-llm @huggingface/transformers
```

Optional, for reading files:

```sh
npm install pdfjs-dist     # PDFs
npm install tesseract.js   # images and scanned PDFs
```

You'll need:

- **A browser with WebGPU** (Chrome or Edge 121+) to run models in the browser. Without WebGPU, smaller models still run on the CPU.
- **A bundler that supports module workers** (`new Worker(new URL('./x.ts', import.meta.url))`). Vite, webpack 5, Next.js, Angular and SvelteKit all do.
- **Recommended:** serve your app with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers. That turns on multi-threaded WebAssembly, which is noticeably faster.

## Quick start

### 1. Add three worker files

The model, the search index and the database each run in their own worker, so your UI never freezes. Each file is one line:

```ts
// src/db.worker.ts: the database (Postgres with vector search)
import { servePGlite } from 'enclave-ai/pglite-worker'; servePGlite()

// src/ml.worker.ts: embeddings for search
import { serveTransformers } from 'enclave-ai/transformers/worker'; serveTransformers()

// src/llm.worker.ts: the language model
import { serveWebLLM } from 'enclave-ai/models/webllm-worker'; serveWebLLM()
```

### 2. Create the assistant

```ts
import { createWebEnclave } from 'enclave-ai/web'
import { knowledgeSkill, sqlSkill, memorySkill } from 'enclave-ai/skills'

const ai = await createWebEnclave({
  workers: {
    db: new Worker(new URL('./db.worker.ts', import.meta.url), { type: 'module' }),
    ml: new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' }),
    llm: new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' }),
  },
  skills: [knowledgeSkill(), sqlSkill(), memorySkill()],
  onProgress: (p) => showProgress(p.text, p.progress),   // model downloads and indexing
})
```

`createWebEnclave` checks what the device can handle (WebGPU support and GPU memory) and picks a model to match. On a typical laptop that's Qwen3 4B, a 2.3 GB download. It downloads on the first message, or during setup with `preloadLLM: true`, and loads from the browser cache after that, offline included.

### 3. Add documents and ask

```ts
await ai.knowledge!.ingest([
  { title: 'Wifi', content: 'Guest network: NorthGuest. Password: maple-harbor-42.' },
  { title: 'Expenses', content: expensePolicyText },
])

for await (const e of ai.thread('support').send('What is the guest wifi password?')) {
  if (e.type === 'text-delta') append(e.delta)
}

// Or wait for the whole answer:
const answer = await ai.thread().send('How do I file an expense report?').text()
```

Threads are saved in the browser's database, so conversations survive a page reload.

A complete example app is in [`examples/playground`](examples/playground): model picker, download progress, file upload, streaming chat with approvals, and evals.

## Use it with your framework

Every framework uses the same pattern:

1. Create the assistant once, in the browser.
2. Send a message.
3. Append each `text-delta` event to your state.

Put the setup in a module that creates it on first use:

```ts
// enclave.ts
import { createWebEnclave, type WebEnclave } from 'enclave-ai/web'
import { knowledgeSkill, sqlSkill } from 'enclave-ai/skills'

let instance: Promise<WebEnclave> | undefined

export function getEnclave() {
  instance ??= createWebEnclave({
    workers: {
      db: new Worker(new URL('./db.worker.ts', import.meta.url), { type: 'module' }),
      ml: new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' }),
      llm: new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' }),
    },
    skills: [knowledgeSkill(), sqlSkill()],
  })
  return instance
}
```

Because `getEnclave()` is only called when the user sends a message, the module is safe to import in server-rendered apps.

The examples below were run in Chrome against the library, and each streams the answer into the page.

### React

```tsx
// useChat.ts
import { useCallback, useEffect, useRef, useState } from 'react'
import { getEnclave } from './enclave'

export type Message = { role: 'user' | 'assistant'; text: string }

export function useChat(threadId = 'main') {
  const [messages, setMessages] = useState<Message[]>([])
  const [busy, setBusy] = useState(false)
  const controller = useRef<AbortController | null>(null)

  useEffect(() => () => controller.current?.abort(), [])

  const send = useCallback(
    async (text: string) => {
      controller.current = new AbortController()
      setBusy(true)
      setMessages((m) => [...m, { role: 'user', text }, { role: 'assistant', text: '' }])
      try {
        const ai = await getEnclave()
        for await (const e of ai.thread(threadId).send(text, { signal: controller.current.signal })) {
          if (e.type === 'text-delta') {
            setMessages((m) => [...m.slice(0, -1), { role: 'assistant', text: m[m.length - 1]!.text + e.delta }])
          }
        }
      } finally {
        setBusy(false)
      }
    },
    [threadId],
  )

  const stop = useCallback(() => controller.current?.abort(), [])
  return { messages, busy, send, stop }
}
```

```tsx
// Chat.tsx
import { useState } from 'react'
import { useChat } from './useChat'

export function Chat() {
  const { messages, busy, send, stop } = useChat()
  const [draft, setDraft] = useState('')

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        send(draft)
        setDraft('')
      }}
    >
      {messages.map((m, i) => (
        <p key={i} className={m.role}>
          {m.text}
        </p>
      ))}
      <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Ask about your documents" />
      {busy ? <button type="button" onClick={stop}>Stop</button> : <button>Send</button>}
    </form>
  )
}
```

**Next.js:** the assistant needs browser APIs, so load the chat on the client only:

```tsx
'use client'
import dynamic from 'next/dynamic'
const Chat = dynamic(() => import('./Chat').then((m) => m.Chat), { ssr: false })
```

### Vue

```ts
// useChat.ts
import { onScopeDispose, ref } from 'vue'
import { getEnclave } from './enclave'

export function useChat(threadId = 'main') {
  const messages = ref<{ role: 'user' | 'assistant'; text: string }[]>([])
  const busy = ref(false)
  let controller: AbortController | undefined

  async function send(text: string) {
    controller = new AbortController()
    busy.value = true
    messages.value.push({ role: 'user', text }, { role: 'assistant', text: '' })
    const reply = messages.value[messages.value.length - 1]!
    try {
      const ai = await getEnclave()
      for await (const e of ai.thread(threadId).send(text, { signal: controller.signal })) {
        if (e.type === 'text-delta') reply.text += e.delta
      }
    } finally {
      busy.value = false
    }
  }

  const stop = () => controller?.abort()
  onScopeDispose(stop)
  return { messages, busy, send, stop }
}
```

```vue
<!-- Chat.vue -->
<script setup lang="ts">
import { ref } from 'vue'
import { useChat } from './useChat'

const { messages, busy, send, stop } = useChat()
const draft = ref('')

function submit() {
  send(draft.value)
  draft.value = ''
}
</script>

<template>
  <form @submit.prevent="submit">
    <p v-for="(m, i) in messages" :key="i" :class="m.role">{{ m.text }}</p>
    <input v-model="draft" placeholder="Ask about your documents" />
    <button v-if="busy" type="button" @click="stop">Stop</button>
    <button v-else>Send</button>
  </form>
</template>
```

**Nuxt:** wrap the component in `<ClientOnly>`, or name it `Chat.client.vue`.

### Svelte 5

```ts
// chat.svelte.ts
import { getEnclave } from './enclave'

export class Chat {
  messages = $state<{ role: 'user' | 'assistant'; text: string }[]>([])
  busy = $state(false)
  #controller: AbortController | undefined

  constructor(private threadId = 'main') {}

  async send(text: string) {
    this.#controller = new AbortController()
    this.busy = true
    this.messages.push({ role: 'user', text }, { role: 'assistant', text: '' })
    const reply = this.messages[this.messages.length - 1]!
    try {
      const ai = await getEnclave()
      for await (const e of ai.thread(this.threadId).send(text, { signal: this.#controller.signal })) {
        if (e.type === 'text-delta') reply.text += e.delta
      }
    } finally {
      this.busy = false
    }
  }

  stop() {
    this.#controller?.abort()
  }
}
```

```svelte
<!-- Chat.svelte -->
<script lang="ts">
  import { onDestroy } from 'svelte'
  import { Chat } from './chat.svelte'

  const chat = new Chat()
  let draft = $state('')
  onDestroy(() => chat.stop())

  function submit(e: SubmitEvent) {
    e.preventDefault()
    chat.send(draft)
    draft = ''
  }
</script>

<form onsubmit={submit}>
  {#each chat.messages as m, i (i)}
    <p class={m.role}>{m.text}</p>
  {/each}
  <input bind:value={draft} placeholder="Ask about your documents" />
  {#if chat.busy}
    <button type="button" onclick={() => chat.stop()}>Stop</button>
  {:else}
    <button>Send</button>
  {/if}
</form>
```

**SvelteKit:** this works with server rendering as is, because nothing touches the browser until the user sends a message.

### Angular

```ts
// chat.service.ts
import { Injectable, OnDestroy, signal } from '@angular/core'
import { getEnclave } from './enclave'

@Injectable({ providedIn: 'root' })
export class ChatService implements OnDestroy {
  readonly messages = signal<{ role: 'user' | 'assistant'; text: string }[]>([])
  readonly busy = signal(false)
  private controller?: AbortController

  async send(text: string, threadId = 'main') {
    this.controller = new AbortController()
    this.busy.set(true)
    this.messages.update((m) => [...m, { role: 'user', text }, { role: 'assistant', text: '' }])
    try {
      const ai = await getEnclave()
      for await (const e of ai.thread(threadId).send(text, { signal: this.controller.signal })) {
        if (e.type === 'text-delta') {
          this.messages.update((m) => [...m.slice(0, -1), { role: 'assistant', text: m[m.length - 1]!.text + e.delta }])
        }
      }
    } finally {
      this.busy.set(false)
    }
  }

  stop() {
    this.controller?.abort()
  }

  ngOnDestroy() {
    this.stop()
  }
}
```

```ts
// chat.component.ts
import { Component, inject } from '@angular/core'
import { FormsModule } from '@angular/forms'
import { ChatService } from './chat.service'

@Component({
  selector: 'app-chat',
  imports: [FormsModule],
  template: `
    <form (ngSubmit)="submit()">
      @for (m of chat.messages(); track $index) {
        <p [class]="m.role">{{ m.text }}</p>
      }
      <input [(ngModel)]="draft" name="draft" placeholder="Ask about your documents" />
      @if (chat.busy()) {
        <button type="button" (click)="chat.stop()">Stop</button>
      } @else {
        <button>Send</button>
      }
    </form>
  `,
})
export class ChatComponent {
  chat = inject(ChatService)
  draft = ''

  submit() {
    this.chat.send(this.draft)
    this.draft = ''
  }
}
```

### Alacris

[Alacris](https://github.com/bmartel/alacris) components are standard custom elements, so an `<ask-docs>` element built with it works on its own page or inside any of the frameworks above.

Each reply is its own signal, so a new token updates one text node and nothing else re-renders.

```ts
// ask-docs.ts
import { css, define, each, html, onCleanup, signal, type Signal } from '@alacris/core'
import { getEnclave } from './enclave'

type Message = { id: number; role: 'user' | 'assistant'; text: Signal<string> }

define('ask-docs', {
  props: { thread: 'main' },
  styles: css`
    :host { display: grid; gap: 8px; font: inherit }
    .user { font-weight: 600 }
  `,
  setup({ thread }, host) {
    const messages = signal<Message[]>([])
    const draft = signal('')
    const busy = signal(false)
    let controller: AbortController | undefined
    let nextId = 0
    onCleanup(() => controller?.abort())

    async function ask(question: string) {
      const reply = signal('')
      messages.update((list) => [
        ...list,
        { id: nextId++, role: 'user', text: signal(question) },
        { id: nextId++, role: 'assistant', text: reply },
      ])
      controller = new AbortController()
      busy(true)
      try {
        const ai = await getEnclave()
        for await (const e of ai.thread(thread()).send(question, { signal: controller.signal })) {
          if (e.type === 'text-delta') reply.update((t) => t + e.delta)
        }
        host.emit('answer', { question, answer: reply() })
      } finally {
        busy(false)
      }
    }

    return html`
      ${each(
        () => messages(),
        (m) => html`<p class=${m().role}>${m().text}</p>`,
        (m) => m.id,
      )}
      <form @submit.prevent=${() => (ask(draft()), draft(''))}>
        <input .value=${draft} @input=${(e: Event) => draft((e.target as HTMLInputElement).value)} placeholder="Ask about your documents" />
        ${() => (busy() ? html`<button type="button" @click=${() => controller?.abort()}>Stop</button>` : html`<button>Send</button>`)}
      </form>`
  },
})
```

```html
<ask-docs thread="support"></ask-docs>
<script type="module">
  document.querySelector('ask-docs').addEventListener('answer', (e) => console.log(e.detail.answer))
</script>
```

Removing the element from the page stops any reply in progress.

### Plain JavaScript

```ts
import { getEnclave } from './enclave'

const ai = await getEnclave()
form.addEventListener('submit', async (event) => {
  event.preventDefault()
  const reply = document.createElement('p')
  log.append(reply)
  for await (const e of ai.thread('main').send(input.value)) {
    if (e.type === 'text-delta') reply.textContent += e.delta
  }
})
```

## Add documents and files

### Ingest text

```ts
const { documents, chunks, skipped } = await ai.knowledge!.ingest(
  [{ id: 'hr-leave', title: 'Leave policy', content: text, source: 'hr/leave.md', metadata: { dept: 'hr' } }],
  { collection: 'handbook', onProgress: ({ done, total }) => bar(done / total) },
)
```

- **Chunking:** documents are split along headings and paragraphs into pieces of about 1,200 characters. Each piece is indexed for meaning (vectors) and for exact words (full-text search).
- **Re-ingesting is cheap.** Unchanged documents are skipped, and re-ingesting the same `id` replaces the old version.
- **Collections** group documents (`handbook`, `tickets`) so searches can target one or several.
- **Other operations:** `remove(id)`, `clear(collection)`, `collections()`, `reindex()`.

### Load files

`enclave-ai/loaders` turns files into documents, and spreadsheets into SQL tables. Files are read in the browser and never uploaded.

```ts
import { ACCEPT, importTable, loadFiles } from 'enclave-ai/loaders'
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'   // Vite syntax; served from your own site

fileInput.accept = ACCEPT
fileInput.onchange = async () => {
  const { documents, tables, warnings, errors } = await loadFiles(fileInput.files!, {
    collection: 'uploads',
    pdf: { lib: () => import('pdfjs-dist'), workerSrc: pdfWorkerSrc },
  })
  await ai.knowledge!.ingest(documents)
  for (const t of tables) await importTable(ai.db, t, { ifExists: 'replace' })
}
```

| Format | What the assistant gets | Extra package |
|---|---|---|
| PDF | Headings, paragraphs, lists and simple tables. Running headers, footers and page numbers are removed. | `pdfjs-dist` |
| Word (.docx) | Headings, lists and tables. Deleted tracked changes are left out. | |
| PowerPoint (.pptx) | One section per slide, in order, with bullets, tables and speaker notes | |
| Excel (.xlsx), CSV | Every sheet as a table, with dates and numbers as Excel shows them | |
| JSON | Lists of records become tables. Other JSON becomes readable text. | |
| HTML, EPUB | The main content, without navigation, scripts or footers | |
| Markdown, text, code | As is | |
| Images, scanned PDFs | Text read with OCR (below) | `tesseract.js` |

Options:
- `pdfSplit: 'page'` stores each PDF page separately, so answers can cite a page number.
- `slideNotes: false` leaves out speaker notes.
- `tablesAsText: false` keeps spreadsheets for SQL only.
- `maxBytes` limits file size (default 100 MB).
- `metadata` is added to every document.

`loadFiles` keeps going when a file can't be read and lists it in `errors`. Old `.doc`, `.xls` and `.ppt` files get a message asking for the newer format.

### Read images and scanned PDFs

`tesseractOcr()` reads text from images and scanned PDF pages. Serve its engine from your own site. By default tesseract.js downloads it from a CDN, and `tesseractOcr` refuses to do that unless you pass `cdn: true`.

```sh
npx enclave-mirror --out public/models --ocr eng      # or --ocr eng,deu,fra
```

```ts
import { loadFiles, tesseractOcr } from 'enclave-ai/loaders'

const ocr = tesseractOcr({ lib: () => import('tesseract.js'), baseUrl: '/models/ocr' })
const { documents } = await loadFiles(files, { ocr, pdf })
```

The engine starts on first use (about a second) and is reused after that. A typical screenshot of a printed invoice is read in under 200 ms. Low-confidence results are dropped as noise (`minConfidence`, default 30). You can also pass any `(image: Blob) => Promise<string>` function as `ocr`.

### How answers use your documents

Before the model answers, the user's message is searched against your documents. Only passages that are clearly relevant are passed to the model, and it cites them as `[1]`, `[2]`. Greetings, commands, and requests to rework an earlier answer ("summarize both of those") don't pull in passages. The model can also run more searches itself.

```ts
knowledgeSkill({
  collections: ['handbook'],   // limit what it searches
  autoRetrieve: { limit: 3 },  // passages per question; false to only search when the model asks
})
```

Search directly:

```ts
const hits = await ai.knowledge!.search('parental leave notice period', { limit: 5, collection: 'handbook' })
// [{ documentId, title, source, content, similarity, metadata }, ...]
```

Text inside documents is treated as data, not instructions. Paragraphs that try to instruct the assistant ("ignore previous instructions…") are removed before the model sees them.

## Answer questions with SQL

Every assistant has a Postgres 17 database, stored in the browser. Your app can use it directly:

```ts
await ai.db.exec(`create table if not exists books (id bigint generated always as identity primary key, title text, author text)`)
const { rows } = await ai.db.query('select * from books where author = $1', ['Le Guin'])
```

`sqlSkill()` lets the assistant query and change data. It sees your current schema, and it asks the user before any write.

```ts
sqlSkill({
  readOnly: false,       // true allows SELECT only
  approveWrites: true,   // ask before INSERT, UPDATE, DELETE and schema changes (default)
  maxRows: 100,
})
```

Imported spreadsheets work the same way. `importTable()` gives each column the right type, gives it a name you can write SQL against (`Unit Price (€)` becomes `unit_price`), and notes which file it came from. A question like "What's the total of Ana's approved expenses?" is then computed with SQL instead of estimated from text.

Two ways to make answers more accurate:

- **Write your business rules as comments.** The assistant reads them.
  ```sql
  comment on table orders is 'Revenue counts only orders whose status is not cancelled.';
  ```
- **Lookup values are shown automatically.** Values from small lookup columns (statuses, countries, product names) are shown to the model, so it can match the user's words to your data. Free-text and sensitive-looking columns (emails, phone numbers, passwords, tokens) never are.

## Give the assistant your app's tools

A skill bundles everything the assistant needs for one part of your app:
- instructions;
- typed tools;
- tables, if any;
- live context such as today's date.

```ts
import { z } from 'zod'
import { defineSkill, tool, dateContext } from 'enclave-ai'

export const invoices = defineSkill({
  name: 'invoices',
  description: 'Look up and annotate customer invoices.',
  instructions: 'Use find_invoices before answering billing questions. Amounts are in cents.',
  migrations: [`create table invoice_notes (invoice_id text primary key, note text not null)`],
  tools: {
    find_invoices: tool({
      description: 'Find invoices for a customer.',
      input: z.object({ customer: z.string(), status: z.enum(['open', 'paid']).optional() }),
      execute: ({ customer, status }) => myApp.invoices.search({ customer, status }),
      toModelOutput: (rows) => rows.slice(0, 20),   // your UI gets everything; the model gets 20
    }),
    annotate: tool({
      description: 'Attach a note to an invoice.',
      input: z.object({ invoiceId: z.string(), note: z.string() }),
      needsApproval: true,                          // the user confirms first
      execute: ({ invoiceId, note }, { db }) =>
        db.query('insert into invoice_notes values ($1, $2) on conflict (invoice_id) do update set note = $2', [invoiceId, note]),
    }),
  },
  context: async () => `${dateContext()}\nOpen invoices: ${await myApp.invoices.countOpen()}`,
})

const ai = await createWebEnclave({ workers, skills: [invoices, knowledgeSkill()] })
```

| Field | What it does |
|---|---|
| `instructions` | Added to the assistant's instructions |
| `tools` | Functions it can call. Inputs are checked against the Zod schema; invalid input or a thrown error goes back to the model so it can correct itself. |
| `migrations` | SQL run once per database, in order |
| `context` | Text refreshed before every step, such as the date or current counts |
| `lazy` | Keeps the skill's tools hidden until the assistant needs them. Smaller models do better with fewer tools on screen. |

Tools receive a context object with `db`, `knowledge`, `threadId`, `signal` and `emit(data)`. `emit` sends custom events to your UI. `needsApproval` can also be a function: `(input) => input.amount > 100`.

Tips for small models, each one learned from our tests:
- **Put guidance in tool results.** When a lookup matches several records, return `{ matches, note: 'Several contacts share this name. Ask which one before acting.' }`. The model reads results at the moment it decides what to do next. The same rule in the instructions didn't work.
- **Give it the date.** `dateContext()` lists today, the rest of this week and next week, so "this Friday" is read off, not counted.
- **Return less.** Send the model only the fields it needs, with `toModelOutput`.

## Choose where the model runs

### In the browser (default)

`createWebEnclave` picks a model that fits the device. You can also name one:

| Model | Download | Good for |
|---|---|---|
| `qwen3-4b` | 2.3 GB | The default on desktops with 8 GB+ of memory. 95% on our test suite. |
| `qwen3-8b` | 4.6 GB | Best quality in the browser. Needs about 6 GB of GPU memory. |
| `qwen3-1.7b` | 1.1 GB | Laptops with integrated graphics |
| `qwen3-0.6b` | 0.5 GB | Phones and demos |
| `tjs-qwen3-1.7b`, `tjs-qwen3-0.6b`, `tjs-granite-4-1b` | 0.5–1.4 GB | Devices without WebGPU (runs on the CPU) |

```ts
createWebEnclave({ llm: 'qwen3-1.7b', ... })
await ai.useModel('qwen3-8b')   // switch later; the old model is unloaded
```

Qwen3 models think before acting on new requests and answer directly after a tool result. This is the default (`thinking: 'auto'`). Without it, Qwen3 4B picked the right tool far less often. For plain chat with no tools, `thinking: false` is faster.

### On the same computer: Ollama or LM Studio

If the user runs Ollama or LM Studio, a larger local model is more accurate and about 2.5× faster than the in-browser one. Data still stays on the machine.

```ts
import { discoverLocalModels, recommendOllamaModel, ollama, lmstudio } from 'enclave-ai/models/local'

const pick = recommendOllamaModel(await discoverLocalModels())   // the best tested model that's installed

const ai = await createWebEnclave({
  workers,
  llm: pick ? ollama({ model: pick.tag, contextWindow: 32768 }) : 'auto',   // fall back to the browser
  skills: [knowledgeSkill(), sqlSkill()],
})
```

Tested on an Apple M2 Max (32 GB):

| Model | Test suite | Typical answer | Download |
|---|---|---|---|
| `qwen3.6:27b-q4_K_M` (Ollama, recommended) | 98.5% | 15 s | 17 GB |
| `qwen3.6-27b` GGUF (LM Studio) | 99.5% | 14 s | 17.5 GB |
| `qwen3.5:9b` (Ollama, faster) | 93% | 6 s | 6.6 GB |

- **Ollama** works with no setup when your app is served from `localhost`. For a deployed site, allow its origin when starting Ollama: `OLLAMA_ORIGINS=https://your-app.example ollama serve`.
- **LM Studio** must be started with `lms server start --cors`, so the browser is allowed to call it. That also lets any website you visit call it, so turn it on only while you need it. Load the model with the same context length you pass in code (`lms load <model> --context-length 32768`):

```ts
lmstudio({ model: 'qwen/qwen3.6-27b', contextWindow: 32768 })
```

### Cloud APIs (opt in)

```ts
import { anthropic } from 'enclave-ai/models/anthropic'
import { openaiCompatible } from 'enclave-ai/models/openai'

createWebEnclave({ privacy: { allow: 'remote' }, llm: anthropic({ apiKey }) })
```

Cloud models are refused unless you allow them, so data can't leave the device by accident.

## Keep data private

By default, user data stays on the device. Three settings let you guarantee it:

**1. Limit where models may run.** Every model declares where it processes data, and anything outside your policy is refused with a `PrivacyError`.

```ts
createWebEnclave({ privacy: { allow: 'device' } })   // this browser and this computer only
// 'local-network' (default): also servers on your private network
// 'remote': cloud APIs
```

**2. Host the model files yourself.** By default, models download from Hugging Face and a CDN. No user data is sent, but those services see the user's IP address. Mirror the files to your own site instead:

```sh
npx enclave-mirror --out public/models --webllm qwen3-4b --embedding embeddinggemma --ort --ocr eng
```

```ts
createWebEnclave({ selfHost: { baseUrl: '/models' }, ... })
```

**3. Let the browser enforce it.** Serve a Content-Security-Policy that only allows your own site. The browser then blocks every other connection, from the page and from its workers:

```ts
import { contentSecurityPolicy } from 'enclave-ai/privacy'

contentSecurityPolicy({ modelHosts: [] })        // only this site
contentSecurityPolicy({ localServers: true })    // also Ollama and LM Studio on this computer
```

We tested this with all three in place. On a first visit, the app downloaded its models, indexed a document and answered questions without contacting any other site. A separate run did the same for OCR on a scanned PDF.

Stored data (the database, chat history, search index and model cache) lives in the browser's storage for your site. `createWebEnclave` asks the browser to keep it rather than clear it under storage pressure.

## Build the chat UI

`thread.send()` streams events:

| Event | Use it to |
|---|---|
| `text-delta` | Append to the answer |
| `reasoning-delta` | Show the model's thinking (optional) |
| `tool-call`, `tool-result` | Show what the assistant is doing |
| `approval-request` | Ask the user to confirm an action |
| `custom` | Receive data a tool sent with `emit` |
| `finish` | Know it's done: `stop`, `max-steps` or `aborted` |

```ts
const controller = new AbortController()
const stream = thread.send(input, {
  signal: controller.signal,                      // wire this to a Stop button
  onApproval: (call) => confirmInApp(call.name, call.input),   // resolve true or false
})
for await (const e of stream) {
  if (e.type === 'text-delta') appendText(e.delta)
  if (e.type === 'tool-call') showTool(e.call.name)
}
const { text, steps, usage } = await stream.result()
```

**Approvals.** Tools marked `needsApproval` wait for your `onApproval` handler, which you can also set once in `createWebEnclave`. With no handler, the action is declined, and the assistant says so rather than claiming it was done.

**Threads.**

```ts
await ai.threads()                    // [{ id, title, updatedAt }, ...]
await ai.thread(id).messages()        // full history
await ai.thread(id).rename('Q3 revenue')
await ai.thread(id).delete()
await ai.run('One-off question').text()   // no history kept
```

Messages marked `synthetic: true` are internal nudges from the agent loop. Hide them in your UI.

## Memory

```ts
memorySkill({ recent: 10 })   // the 10 latest memories are included in every conversation
```

The assistant can remember, look up, update and forget facts about the user. It refuses to store passwords, API keys and card numbers. Forgetting only ever deletes memories, never documents.

## Offline use and downloads

```ts
await ai.modelCache.status()
// [{ kind: 'llm', id: 'qwen3-4b', cached: true, active: true, downloadMB: 2300 }, ...]
await ai.modelCache.clear('llm', 'qwen3-8b')
```

- `cached: true` means every file the device needs is stored, so it works offline.
- The model picked automatically on a first visit is capped at a 2.5 GB download (`maxDownloadMB`).
- Test offline behavior on `127.0.0.1` or a real hostname. WebLLM doesn't cache some files served from `localhost`.

Tuning for the in-browser model:

```ts
createWebEnclave({
  thinking: 'auto',                    // true | false | 'auto'
  webllm: {
    reasoningHistory: 'current-turn',  // 'auto' makes follow-up messages start faster, slightly less accurately
    thinkingBudget: 2048,              // thinking stops after this many tokens and the model answers
  },
})
```

Every `step-finish` event includes timing: time to first token, tokens per second, and how much of the prompt was reused from cache.

## Test your assistant

`enclave-ai/eval` runs test cases against your assistant, in the browser, on your data:

```ts
import { runEval, formatReport, numberNear } from 'enclave-ai/eval'

const report = await runEval(ai, [
  { name: 'wifi', input: 'Guest wifi password?', expect: { answer: 'maple-harbor-42' } },
  {
    name: 'adds books',
    input: 'Create a books table and add two novels',
    expect: { check: async ({ ai }) => (await ai.db.query('select * from books')).rows.length === 2 || 'expected 2 rows' },
  },
  { name: 'no tools for chat', input: 'Say good morning in Spanish', expect: { noTools: true, answer: /buenos/i } },
  {
    name: 'follow-up',
    turns: [
      { input: 'What was revenue in June?', expect: { answer: numberNear(34799) } },
      { input: 'And in May?', expect: { answer: numberNear(30020) } },
    ],
  },
], { repeats: 3 })

console.log(formatReport(report))   // pass rate with a confidence range, latency and failures
```

While fixing a failure, `{ failFast: true }` stops at the first failing case. `{ first: lastFailures }` runs the cases that failed last time first.

The suite we use for this library is in [`packages/evals`](packages/evals). It has 65 cases covering documents, SQL, a CRM-style app, memory, multi-turn conversations and safety, and runs against the browser model, Ollama or LM Studio.

## Benchmarks

The full test suite (65 cases, each run 3 times) on an Apple M2 Max (32 GB):

| Setup | Passed | Typical time per case |
|---|---|---|
| In the browser: Qwen3 4B | **95%** | 38 s |
| Ollama: `qwen3.6:27b` | 98.5% | 15 s |
| LM Studio: `qwen3.6-27b` GGUF | 99.5% | 14 s |

The in-browser model passes every safety, memory and approval case. The case it still misses is a four-message analysis that ends in a multi-row insert; the 27B models pass it.

Search finds the right passage in the top three results for 99.5% of 91 test questions, including questions in other languages, in about 50 ms each.

Methodology and every report: [`packages/evals`](packages/evals/README.md).

## API reference

| Import | Main exports |
|---|---|
| `enclave-ai` | `createEnclave`, `defineSkill`, `tool`, `dateContext`, `PrivacyError`, types |
| `enclave-ai/web` | `createWebEnclave`, `detectDevice`, `rankLLMs`, `recommendLLM`, `BROWSER_LLMS`, `EMBEDDING_PRESETS` |
| `enclave-ai/skills` | `knowledgeSkill`, `sqlSkill`, `memorySkill` |
| `enclave-ai/loaders` | `loadFiles`, `loadFile`, `importTable`, `tesseractOcr`, `ACCEPT`, `htmlToMarkdown`, `parseCsv` |
| `enclave-ai/models/local` | `ollama`, `lmstudio`, `discoverLocalModels`, `recommendOllamaModel`, `ollamaEmbedder` |
| `enclave-ai/models/anthropic`, `/models/openai` | `anthropic`, `openaiCompatible` |
| `enclave-ai/privacy` | `contentSecurityPolicy`, `guardNetwork` |
| `enclave-ai/eval` | `runEval`, `formatReport`, `evalRetrieval`, matchers (`anyOf`, `numberNear`, `declines`, …) |
| `enclave-ai/testing` | `mockModel`, `hashEmbedder` for fast tests without downloads |
| `enclave-ai/pglite-worker`, `/transformers/worker`, `/models/webllm-worker` | The worker entry points |

`createEnclave` from `enclave-ai` is the lower-level constructor. Bring your own database, model and embedder; it also runs in Node:

```ts
import { createEnclave } from 'enclave-ai'
import { createDb } from 'enclave-ai/pglite'
import { mockModel, hashEmbedder } from 'enclave-ai/testing'

const ai = await createEnclave({
  db: await createDb({ dataDir: 'memory://' }),
  model: mockModel(['Hello!']),
  embedder: hashEmbedder(),
})
```

## Browser support and limits

- **Before 1.0.** The API may still change between minor versions.
- **Tested on** Chrome with WebGPU on Apple silicon, and in Node for the unit tests. Windows and Linux GPUs and mobile browsers haven't been tested yet.
- **The in-browser model is small.** Qwen3 4B handles most tasks but still slips on long multi-step changes to data. For heavier work, use a 27B model through Ollama or LM Studio.
- **The first visit downloads 2–3 GB.** Show progress with `onProgress`, and check `modelCache` to see what's stored.
- **PDF reading is best-effort.** Multi-column layouts and complex tables may not come out perfectly. Handwriting isn't supported by OCR.

## Contributing

```sh
pnpm install
pnpm test                              # unit tests (real Postgres, scripted model)
pnpm dev                               # the playground at http://localhost:5173
pnpm --filter playground dev:strict    # self-hosted models with a strict Content-Security-Policy
pnpm --filter @enclave/evals eval      # the full test suite against real models (slow)
```

To release:

```sh
cd packages/core
npm version patch        # or minor
npm publish              # builds, typechecks and runs the tests first
```

## License

[MIT](LICENSE)
