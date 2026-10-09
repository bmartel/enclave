import type { IngestDocument } from 'enclave-ai'
import type { LabeledQuery } from './retrieval.js'

/**
 * Notebook-style code cells (Python, JavaScript, SQL, R) with near-miss
 * neighbours (the same data, another computation), and natural-language
 * questions about what the code does. For code search: "the cell that…".
 */
export const CODE_CORPUS: IngestDocument[] = [
  { id: 'py-revenue-region', title: 'sales.py', content: 'import pandas as pd\ndf = pd.read_csv("sales.csv")\ntotals = df.groupby("region")["revenue"].sum().sort_values(ascending=False)\nprint(totals)' },
  { id: 'py-units-product', title: 'units.py', content: 'import pandas as pd\ndf = pd.read_csv("sales.csv")\nprint(df.groupby("product")["units"].sum())' },
  { id: 'py-monthly-growth', title: 'growth.py', content: 'monthly = df.groupby("month")["revenue"].sum()\nchange = monthly.pct_change() * 100\nprint(change.round(1))' },
  { id: 'py-temps-mean', title: 'temps.py', content: 'import pandas as pd\nt = pd.read_csv("temps.csv")\navg = t.groupby("city")["temp_c"].mean().round(2)\nprint(avg.idxmax(), avg.max())' },
  { id: 'py-plot-line', title: 'chart.py', content: 'import matplotlib.pyplot as plt\nplt.plot(df["month"], df["revenue"])\nplt.title("Revenue over time")\nplt.show()' },
  { id: 'py-dedupe', title: 'clean.py', content: 'rows = df.drop_duplicates(subset=["email"]).dropna(subset=["email"])\nprint(len(rows), "unique contacts")' },
  { id: 'py-json-ages', title: 'people.py', content: 'import json\npeople = json.load(open("people.json"))\nages = [p["age"] for p in people]\nprint(sum(ages) / len(ages))\nprint(max(people, key=lambda p: p["age"])["name"])' },
  { id: 'js-fetch-weather', title: 'weather.js', content: 'const res = await fetch("https://api.example.com/weather?city=Lisbon")\nconst data = await res.json()\nconsole.log(data.current.temperature)' },
  { id: 'js-life', title: 'life.js', content: 'const next = grid.map((row, y) => row.map((alive, x) => {\n  const n = neighbours(grid, x, y)\n  return n === 3 || (alive && n === 2)\n}))\ndisplay.animate(() => draw(next))' },
  { id: 'js-csv-parse', title: 'parse.js', content: 'const lines = text.trim().split("\\n")\nconst [header, ...rows] = lines.map((l) => l.split(","))\nconst records = rows.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])))' },
  { id: 'js-debounce', title: 'debounce.js', content: 'function debounce(fn, ms) {\n  let timer\n  return (...args) => {\n    clearTimeout(timer)\n    timer = setTimeout(() => fn(...args), ms)\n  }\n}' },
  { id: 'sql-top-customers', title: 'customers.sql', content: 'SELECT c.name, SUM(o.total) AS spent\nFROM customers c JOIN orders o ON o.customer_id = c.id\nGROUP BY c.name ORDER BY spent DESC LIMIT 5;' },
  { id: 'sql-never-ordered', title: 'inactive.sql', content: 'SELECT c.name FROM customers c\nLEFT JOIN orders o ON o.customer_id = c.id\nWHERE o.id IS NULL;' },
  { id: 'sql-orders-month', title: 'orders.sql', content: "SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) AS orders\nFROM orders GROUP BY month ORDER BY month;" },
  { id: 'r-regression', title: 'model.R', content: 'fit <- lm(price ~ area + rooms, data = houses)\nsummary(fit)\ncoef(fit)' },
  { id: 'r-count-rows', title: 'count.R', content: 'books <- read.csv("books.csv")\ndone <- subset(books, Status == "Done")\nprint(nrow(done))' },
]

const q = (query: string, relevant: string, ...tags: string[]): LabeledQuery => ({ query, relevant: [relevant], tags: ['code', ...tags] })

export const CODE_QUERIES: LabeledQuery[] = [
  q('total revenue for each region, highest first', 'py-revenue-region'),
  q('how many units of each product were sold', 'py-units-product'),
  q('percentage change in revenue from one month to the next', 'py-monthly-growth'),
  q('which city has the warmest average temperature', 'py-temps-mean'),
  q('line chart of revenue by month', 'py-plot-line'),
  q('remove duplicate email addresses', 'py-dedupe'),
  q('average age and the oldest person in the json file', 'py-json-ages'),
  q('call a weather API and print the temperature', 'js-fetch-weather'),
  q("Conway's game of life step", 'js-life'),
  q('turn CSV text into objects keyed by the header', 'js-csv-parse'),
  q('wait until typing stops before calling a function', 'js-debounce'),
  q('five customers who spent the most', 'sql-top-customers'),
  q('customers without any orders', 'sql-never-ordered'),
  q('number of orders per month', 'sql-orders-month'),
  q('linear regression of house price on area and rooms', 'r-regression'),
  q('count the finished books in R', 'r-count-rows'),
  q('Gesamtumsatz pro Region', 'py-revenue-region', 'multilingual'),
  q('clientes que nunca hicieron un pedido', 'sql-never-ordered', 'multilingual'),
]
