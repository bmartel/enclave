import { anyOf, declines } from '@enclave/core/eval'
import { NEGATION, onlyQualified, OUTDATED, REPORTED } from '../graders.js'
import type { ProductionCase } from '../world.js'

/** A citation marker like [1] or [2]. */
const cites = /\[\d+\]/

/**
 * Private RAG over the Northwind knowledge base. Graded on the facts in the
 * answer, the absence of superseded or distractor facts, and citations.
 */
export const ragCases: ProductionCase[] = [
  {
    name: 'rag: simple fact',
    tags: ['rag', 'fact'],
    input: 'What is the guest wifi password at the Toronto office?',
    expect: { answer: ['maple-harbor-42', cites, onlyQualified(/letmein123/, REPORTED)], notAnswer: 'linden-river-17' },
    reference: [{ answer: 'The Toronto guest network NorthGuest uses the password maple-harbor-42 [1].' }],
    variants: {
      pass: [
        'maple-harbor-42 [2]',
        'Connect to NorthGuest using maple-harbor-42 [1]. Note: a vendor newsletter claims it is letmein123, but that is not an official source.',
      ],
      fail: ['The Toronto guest wifi password is letmein123 [1].', 'The guest password is linden-river-17 [1].', 'The password is maple-harbor-42.'],
    },
  },
  {
    name: 'rag: disambiguate by office',
    tags: ['rag', 'distractor'],
    input: "I'm visiting the Berlin office tomorrow. How do I get on the guest wifi?",
    expect: { answer: [/SpreeGuest/i, 'linden-river-17', onlyQualified(/maple-harbor-42|letmein123/, /toronto|claims?|says|not/i)] },
    reference: [{ answer: 'In Berlin, join SpreeGuest with the password linden-river-17 [1].' }],
    variants: {
      pass: ['Network: SpreeGuest, password: linden-river-17 [1]. (Toronto uses maple-harbor-42, which will not work in Berlin.)'],
      fail: ['Join SpreeGuest with maple-harbor-42 [1].', 'Use the NorthGuest network with password maple-harbor-42.'],
    },
  },
  {
    name: 'rag: current policy over superseded',
    tags: ['rag', 'versioning'],
    input: 'How many vacation days does a full-time employee get per year?',
    expect: { answer: [/\b20\b|twenty/i, onlyQualified(/\b15\b/, OUTDATED)] },
    reference: [{ answer: 'Full-time employees get 20 vacation days per year under the 2026 policy [1].' }],
    variants: {
      pass: ['20 days per year [1]. The older 2024 policy gave 15 days, but it was superseded.', 'You get twenty vacation days annually.'],
      fail: ['Full-time employees get 15 vacation days per year [1].', 'You receive 15 days, or 20 after five years.'],
    },
  },
  {
    name: 'rag: current limit over superseded',
    tags: ['rag', 'versioning'],
    input: 'What is the daily meal limit when I travel domestically for work?',
    expect: { answer: [/\$?\s?45\b/, onlyQualified(/\b35\b/, OUTDATED)] },
    reference: [{ answer: 'Domestic travel meals are reimbursed up to $45 per day [1].' }],
    variants: {
      pass: ['Up to 45 dollars per day [1] (it was $35 under the 2023 policy).', '$45/day for domestic trips.'],
      fail: ['The domestic meal limit is $35 per day [1].', 'You can spend up to $65 per day.'],
    },
  },
  {
    name: 'rag: retired system',
    tags: ['rag', 'versioning'],
    input: 'Which server should I connect my VPN client to?',
    // The old host may be mentioned only as retired/old.
    expect: { answer: ['vpn2.northwind.example', onlyQualified(/(?<![\w.])vpn\.northwind\.example/i, OUTDATED)] },
    variants: {
      pass: ['Use vpn2.northwind.example; the old vpn.northwind.example was retired in June 2026 [1].', 'vpn2.northwind.example [1]'],
      fail: ['Connect to vpn.northwind.example with OpenVPN [1].', 'Use vpn2.northwind.example or vpn.northwind.example, either works.'],
    },
    reference: [{ answer: 'Connect to vpn2.northwind.example with Northwind Connect v5+ [1].' }],
  },
  {
    name: 'rag: table lookup',
    tags: ['rag', 'table'],
    input: 'How long does the X200 scanner battery last?',
    expect: { answer: /\b14\s?(hours|h)\b/i },
    variants: { pass: ['About 14 hours (the X100 lasts 8 hours and the X300 20 hours) [1].', '14h [1]'], fail: ['The X200 battery lasts 20 hours [1].'] },
    reference: [{ answer: 'The X200 battery lasts 14 hours [1].' }],
  },
  {
    name: 'rag: multi-hop across documents',
    tags: ['rag', 'multi-hop'],
    input: 'I have a 750 dollar conference registration to expense. Who has to approve it? Give me their name.',
    expect: { answer: /Priya Raman/i },
    variants: { pass: ['Since it is over $500, the Director of Finance must approve it: Priya Raman [1][2].'], fail: ['Your manager approves expenses up to $500, so your manager.', 'Helena Brandt, the CEO.'] },
    reference: [{ answer: 'Expenses over $500 need the Director of Finance, Priya Raman [1][2].' }],
  },
  {
    name: 'rag: procedure first step',
    tags: ['rag', 'procedure'],
    input: 'My laptop is acting like it has malware. What should I do first?',
    // Wiping or reinstalling may only appear as something not to do.
    expect: { answer: [/disconnect|unplug|turn off (the )?wi-?fi|off the network/i, onlyQualified(/\b(wipe|reinstall)/i, NEGATION)] },
    variants: {
      pass: ["Disconnect it from the network right away (turn off wifi or unplug the cable), but don't shut it down [1]. Do not wipe or reinstall it.", 'Unplug the network cable or turn off wifi first [1].', 'The first step is to disconnect the device from the network [1]. Preserve evidence by avoiding rebooting, wiping, or reinstalling the device.'],
      fail: ['Run a full antivirus scan and reinstall the operating system.', 'First, wipe the device to remove the malware.'],
    },
    reference: [{ answer: 'First, disconnect it from the network (turn off wifi or unplug the cable) but do not shut it down [1]. Then report it to security within one hour.' }],
  },
  {
    name: 'rag: negation',
    tags: ['rag', 'negation'],
    input: 'Can I expense the drinks from a client dinner, like wine?',
    expect: { answer: anyOf(/\b(not|never)\b[^.]{0,40}reimburs/i, /^\W*no\b/i, /non-reimbursable/i, /\b(cannot|can'?t) (be )?expense/i) },
    variants: {
      pass: ['No. Alcohol is never reimbursable [1].', 'Unfortunately wine is not reimbursable under the expense policy [1].', "You can't expense alcohol, including wine [1]."],
      fail: ['Yes, drinks are reimbursable within the $45 daily meal limit [1].', 'Yes, you can expense wine as part of the client meal.'],
    },
    reference: [{ answer: 'No. Alcohol is never reimbursable under the expense policy [1].' }],
  },
  {
    name: 'rag: arithmetic over policy',
    tags: ['rag', 'reasoning'],
    input: 'I have been at Northwind for six years full time. How many vacation days do I get?',
    expect: { answer: /\b25\b|twenty-five/i },
    variants: { pass: ['20 + 5 = 25 days per year [1].'], fail: ['You get 20 days per year [1].'] },
    reference: [{ answer: 'After five years you get 5 extra days: 20 + 5 = 25 days [1].' }],
  },
  {
    name: 'rag: eligibility',
    tags: ['rag', 'negation'],
    input: 'I am a contractor. How much paid vacation do I get?',
    expect: { answer: anyOf(/not eligible/i, /\bno paid vacation\b/i, /\b(don'?t|do not) (get|receive|have)\b/i, /\bnone\b/i, /\b0 days\b/i) },
    variants: {
      pass: ['Contractors are not eligible for paid vacation [1].', "As a contractor you don't get paid vacation under this policy [1]."],
      fail: ['Contractors get 20 vacation days per year [1].'],
    },
    reference: [{ answer: 'Contractors are not eligible for paid vacation under the policy [1].' }],
  },
  {
    name: 'rag: fact buried in a long manual',
    tags: ['rag', 'long-doc'],
    input: 'How do I factory reset an X-series scanner?',
    expect: { answer: [/12\s?seconds/i, /volume/i] },
    reference: [{ answer: 'Hold power and volume-down together for 12 seconds, then choose "Wipe data" [1].' }],
  },
  {
    name: 'rag: paraphrased question',
    tags: ['rag', 'paraphrase'],
    input: 'When is the window to change my dental coverage for next year?',
    expect: { answer: [/Nov(ember)?\.? 3/i, /Nov(ember)?\.? 21|21(st)? (of )?Nov/i] },
    reference: [{ answer: 'Open enrollment runs from November 3 to November 21, 2026 [1].' }],
  },
  {
    name: 'rag: German document, English question',
    tags: ['rag', 'multilingual'],
    input: 'What are the cafeteria opening hours at the Berlin office?',
    expect: { answer: [/11[:.]30/, /14[:.]00|2(:00)?\s?pm/i] },
    reference: [{ answer: 'The Berlin cafeteria is open Monday to Friday from 11:30 to 14:00 [1].' }],
  },
  {
    name: 'rag: two facts in one question',
    tags: ['rag', 'multi-part'],
    input: 'For a customer visit to Toronto: where do they park, and what is the guest wifi password?',
    expect: { answer: [/\bP1\b/, 'maple-harbor-42'], notAnswer: 'letmein123' },
    reference: [{ answer: 'Visitors park on level P1 and register at reception [1]; the guest wifi password is maple-harbor-42 [2].' }],
  },
  {
    name: 'rag: unanswerable (not in corpus)',
    tags: ['rag', 'unanswerable'],
    input: "What is the CEO's personal phone number?",
    expect: { answer: declines, notAnswer: /\+?\d[\d\s().-]{6,}\d/ },
    variants: {
      pass: ['Personal phone numbers are not published; contact Helena Brandt through her assistant via the People portal [1].', "I couldn't find a phone number for the CEO in the documents."],
      fail: ["Helena Brandt's number is +1 416 555 0199.", 'You can reach the CEO at 416-555-0100.'],
    },
    reference: [{ answer: "The documents don't list personal phone numbers; they say to contact leaders through their assistants via the People portal [1]." }],
    nullPasses: true,
  },
  {
    name: 'rag: unanswerable (false premise)',
    tags: ['rag', 'unanswerable'],
    input: 'What is the guest wifi password at the Paris office?',
    expect: { answer: [anyOf(declines, /no (paris|other) office/i, /no (document|docs?|information|record)s? (for|about|on) (a |the )?paris/i, /only (two|2) offices/i), onlyQualified(/maple-harbor-42|linden-river-17|letmein123/, /toronto|berlin/i)] },
    variants: {
      pass: [
        'Northwind has no Paris office; only Toronto and Berlin [1].',
        'There is no Paris office. If you meant Toronto, the password is maple-harbor-42 [1].',
        // Real qwen3.8:27b answer.
        "There's no document for a Paris office in the knowledge base, so I can't tell you the guest wifi password there. The only office wifi docs I have are Berlin (linden-river-17) and Toronto (maple-harbor-42).",
      ],
      fail: ['The Paris office wifi password is maple-harbor-42.', 'Use linden-river-17 in Paris.'],
    },
    reference: [{ answer: 'There is no Paris office; Northwind only has offices in Toronto and Berlin [1].' }],
    nullPasses: true,
  },
  {
    name: 'rag: retention period',
    tags: ['rag', 'fact'],
    input: 'How long do we keep application logs?',
    expect: { answer: /\b90\s?days\b/i },
    reference: [{ answer: 'Application logs are retained for 90 days [1].' }],
  },
]
