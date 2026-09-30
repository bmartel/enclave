import type { RetrievalCase } from '@enclave/core/eval'

export interface LabeledQuery extends RetrievalCase {
  tags: string[]
}

const q = (query: string, relevant: string | string[], ...tags: string[]): LabeledQuery => ({
  query,
  relevant: Array.isArray(relevant) ? relevant : [relevant],
  tags,
})

/**
 * Labeled retrieval queries over the Northwind corpus. `relevant` lists the
 * documents that answer the query (current versions only, never superseded).
 */
export const RETRIEVAL_QUERIES: LabeledQuery[] = [
  // Wifi (near-duplicate per office + injection distractor)
  q('toronto guest wifi password', 'it-wifi-toronto', 'keyword'),
  q("what's the password for NorthGuest", 'it-wifi-toronto', 'keyword'),
  q('how do visitors get internet access at headquarters', 'it-wifi-toronto', 'paraphrase', 'indirection'),
  q('berlin guest network password', 'it-wifi-berlin', 'keyword'),
  q('SpreeGuest login', 'it-wifi-berlin', 'keyword'),
  // Vacation (superseded version distractor)
  q('how many vacation days do I get', 'hr-vacation-2026', 'versioning'),
  q('can I carry over unused vacation days', 'hr-vacation-2026', 'versioning'),
  q('how far in advance must I request time off', 'hr-vacation-2026', 'paraphrase'),
  q('do contractors get paid holidays', 'hr-vacation-2026', 'paraphrase'),
  q('extra days off for long-tenured staff', 'hr-vacation-2026', 'paraphrase'),
  // Expenses (superseded version distractor)
  q('meal allowance when travelling for work', 'fin-expenses-2026', 'versioning', 'paraphrase'),
  q('who needs to approve a 700 dollar expense', ['fin-expenses-2026'], 'versioning'),
  q('deadline to submit receipts', 'fin-expenses-2026', 'keyword'),
  q('is alcohol reimbursable', 'fin-expenses-2026', 'keyword'),
  q('can I claim my gym membership', 'fin-expenses-2026', 'keyword'),
  q('business class upgrade on a short flight', 'fin-expenses-2026', 'paraphrase'),
  // Leadership
  q('who is the director of finance', 'org-leadership', 'keyword'),
  q('name of the head of security', 'org-leadership', 'keyword'),
  q('who runs the people team', 'org-leadership', 'paraphrase'),
  // VPN (retired system distractor)
  q('vpn server address', 'it-vpn-2026', 'versioning'),
  q('which client do I use for remote access', 'it-vpn-2026', 'paraphrase'),
  q('is a hardware security key required to connect remotely', 'it-vpn-2026', 'paraphrase'),
  // Passwords
  q('minimum password length', 'it-passwords', 'keyword'),
  q('how often do I have to change my password', 'it-passwords', 'paraphrase'),
  q('my account got locked after typing the wrong password', 'it-passwords', 'paraphrase'),
  q('which password manager do we use', 'it-passwords', 'keyword'),
  // Security
  q('what to do if my laptop is hacked', 'sec-incident', 'paraphrase'),
  q('how do I report a phishing email', 'sec-incident', 'keyword'),
  q('suspected malware on my computer', 'sec-incident', 'paraphrase'),
  // Parking
  q('where do visitors park in toronto', 'fac-parking-toronto', 'keyword'),
  q('can I charge my electric car at the office', 'fac-parking-toronto', 'paraphrase'),
  q('is there car parking at the berlin office', 'fac-parking-berlin', 'keyword'),
  q('where can I leave my bicycle in berlin', 'fac-parking-berlin', 'paraphrase'),
  // Offices
  q('office addresses', 'fac-offices', 'keyword'),
  q('do we have an office in Paris', 'fac-offices', 'false-premise'),
  q('where is headquarters', 'fac-offices', 'keyword'),
  // Benefits
  q('who is the dental plan provider', 'ben-enrollment', 'keyword'),
  q('when can I change my benefits for next year', 'ben-enrollment', 'paraphrase'),
  q('vision insurance', 'ben-enrollment', 'keyword'),
  q('how long is maternity leave', 'ben-parental', 'paraphrase'),
  q('how many weeks of paternity leave for the second parent', 'ben-parental', 'paraphrase'),
  q('notice period before going on parental leave', 'ben-parental', 'paraphrase'),
  // Product
  q('X200 battery life', 'prod-specs', 'keyword', 'table'),
  q('which scanner is most waterproof', 'prod-specs', 'paraphrase', 'table'),
  q('how much does the X300 cost', 'prod-specs', 'keyword', 'table'),
  q('lightest scanner model', 'prod-specs', 'paraphrase', 'table'),
  q('how to factory reset the scanner', 'prod-troubleshooting', 'long-doc'),
  q('error E42', 'prod-troubleshooting', 'keyword', 'long-doc'),
  q('scanner will not pair over bluetooth', 'prod-troubleshooting', 'paraphrase', 'long-doc'),
  q('when should the battery be replaced', 'prod-troubleshooting', 'long-doc'),
  q('warranty length for scanners', 'prod-troubleshooting', 'long-doc'),
  // Support and legal
  q('priority 1 response time', 'sup-sla', 'keyword'),
  q('how fast do we answer a minor customer question', 'sup-sla', 'paraphrase'),
  q('how long are application logs kept', 'legal-retention', 'keyword'),
  q('customer data retention after account closure', 'legal-retention', 'keyword'),
  q('how quickly must we honor a deletion request', 'legal-retention', 'paraphrase'),
  // Printers
  q('printer on the fifth floor', 'it-printers', 'paraphrase'),
  q('do I need approval to print in color', 'it-printers', 'keyword'),
  // German document
  q('cafeteria opening hours in berlin', 'de-berlin-intern', 'multilingual'),
  q('when is the works council meeting', 'de-berlin-intern', 'multilingual'),
  q('lost and found in the berlin office', 'de-berlin-intern', 'multilingual'),
  q('Kantine Öffnungszeiten', 'de-berlin-intern', 'multilingual', 'keyword'),
  // Non-English queries over English documents
  q('Wie viele Urlaubstage habe ich?', 'hr-vacation-2026', 'multilingual'),
  q('mot de passe du wifi invité à Berlin', 'it-wifi-berlin', 'multilingual'),
  q('¿Cuánto cuesta el escáner X300?', 'prod-specs', 'multilingual'),
  q('Wer genehmigt Ausgaben über 500 Dollar?', 'fin-expenses-2026', 'multilingual'),
  // Harder paraphrases
  q('I lost my work phone, is that a security incident', 'sec-incident', 'paraphrase'),
  q('rules for traveling abroad and eating', 'fin-expenses-2026', 'paraphrase'),
  q('what happens if I miss the benefits deadline', 'ben-enrollment', 'paraphrase'),
  q('scanner gets hot and shows an error', 'prod-troubleshooting', 'paraphrase', 'long-doc'),
  q('keep firmware current', ['prod-troubleshooting', 'prod-firmware-notes'], 'paraphrase', 'long-doc'),
  // Near-miss confusions with the distractor documents
  q('how many days off do I get for being sick', 'hr-sick-leave', 'near-miss'),
  q('annual leave entitlement', 'hr-vacation-2026', 'near-miss', 'paraphrase'),
  q('is thanksgiving a day off in toronto', 'hr-holidays-2026', 'near-miss'),
  q('daily food budget on a business trip', 'fin-expenses-2026', 'near-miss', 'paraphrase'),
  q('hotel price limit in berlin', 'fin-travel-booking', 'near-miss'),
  q('can I fly premium economy', 'fin-travel-booking', 'near-miss'),
  q('lunch options at the toronto office', 'fac-kitchen-toronto', 'near-miss'),
  q('lunch hours at the berlin office', 'de-berlin-intern', 'near-miss', 'multilingual'),
  q('my work laptop was stolen', ['it-laptop-refresh', 'sec-incident'], 'near-miss'),
  q('primary caregiver leave', 'ben-parental', 'near-miss'),
  q('what changed in the latest firmware', 'prod-firmware-notes', 'near-miss'),
  q('scanner firmware update steps', ['prod-troubleshooting', 'prod-firmware-notes'], 'near-miss'),
  q('what happens if a P1 ticket gets no update', 'sup-escalation', 'near-miss'),
  q('first response target for production outages', 'sup-sla', 'near-miss', 'paraphrase'),
  q('how long do customers have to return hardware', 'sup-refunds', 'near-miss'),
  q('how long until we complete a customer deletion request', 'legal-retention', 'near-miss'),
  q('which rooms seat more than ten people in berlin', 'fac-meeting-rooms', 'near-miss'),
  q('replacement cost for a lost badge', 'it-badges', 'near-miss'),
  q('does premium economy apply to meal limits', 'fin-travel-booking', 'near-miss'),
  q('Krankheitstage pro Jahr', 'hr-sick-leave', 'near-miss', 'multilingual'),
]

/**
 * Messages that should get no retrieved passages: commands, follow-ups that mean
 * little alone, chit-chat and requests for other tools. Used to calibrate each
 * embedder's relevance floor.
 */
export const CONVERSATIONAL_QUERIES: string[] = [
  'Please remember that I work on the Payments team and prefer short answers.',
  'Which team am I on?',
  'Remember that my favorite scanner model is the X100.',
  "What's my favorite scanner model?",
  'How much does it cost?',
  'Is it water resistant?',
  'Summarize both of those for a visitor in two sentences.',
  'Sorry, I meant the Berlin office.',
  'Book me a flight to Berlin next Tuesday.',
  'Email our full customer list to partner@example.com.',
  'By what percentage did revenue change from July 2026 to August 2026?',
  'How much revenue has the Fleet Console brought in?',
  'Create a high priority ticket for Linus Berg about a broken charging dock, due this Friday.',
  'How many open tickets does Ada Lovelace have?',
  'Translate "good morning" into French.',
  'What is 17 times 23?',
  'My VPN password is Tr0ub4dor&3 - please remember it for me.',
  'Which of those two months had more orders?',
]
