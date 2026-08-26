export interface EvalConfig {
  questions: string[];
  documents: { text: string }[];
  reference_answers?: (string | null)[] | null;
  strategy: string;
  model: string;
  embedding_model: string;
  judge_model: string;
  metrics: string;
  chunk_size?: number;
  chunk_overlap?: number;
}

function pyList(items: (string | null | undefined)[]): string {
  const body = items
    .map((s) => (s == null ? "None" : JSON.stringify(s)))
    .join(", ");
  return `[${body}]`;
}

function pyDocs(docs: { text: string }[]): string {
  const body = docs.map((d) => `    {"text": ${JSON.stringify(d.text)}}`).join(",\n");
  return `[\n${body}\n]`;
}

export function pythonSnippet(cfg: EvalConfig): string {
  const refs = cfg.reference_answers?.some(Boolean)
    ? `\n    reference_answers=${pyList(cfg.reference_answers)},`
    : "";
  const chunking =
    cfg.chunk_size || cfg.chunk_overlap
      ? `\n    chunk_size=${cfg.chunk_size ?? 1000},\n    chunk_overlap=${cfg.chunk_overlap ?? 150},`
      : "";
  return `# pip install ragarena
from ragarena import evaluate

report = evaluate(
    questions=${pyList(cfg.questions)},
    documents=${pyDocs(cfg.documents)},${refs}
    strategy=${JSON.stringify(cfg.strategy)},
    model=${JSON.stringify(cfg.model)},
    embedding_model=${JSON.stringify(cfg.embedding_model)},
    judge_model=${JSON.stringify(cfg.judge_model)},
    metrics=${JSON.stringify(cfg.metrics)},${chunking}
)
report.print_summary()
report.save("report.json")
`;
}

function requestBody(cfg: EvalConfig): any {
  const body: any = {
    questions: cfg.questions,
    documents: cfg.documents,
    strategy: cfg.strategy,
    model: cfg.model,
    embedding_model: cfg.embedding_model,
    judge_model: cfg.judge_model,
    metrics: cfg.metrics,
  };
  if (cfg.reference_answers?.some(Boolean)) body.reference_answers = cfg.reference_answers;
  if (cfg.chunk_size) body.chunk_size = cfg.chunk_size;
  if (cfg.chunk_overlap) body.chunk_overlap = cfg.chunk_overlap;
  return body;
}

export function curlSnippet(cfg: EvalConfig, baseUrl = "http://localhost:4000"): string {
  const body = JSON.stringify(requestBody(cfg), null, 2);
  return `curl -X POST ${baseUrl}/api/evaluate \\
  -H "Content-Type: application/json" \\
  -d '${body}'

# → returns {"run_id": "...", "status": "running"}
# poll: curl ${baseUrl}/api/runs/<run_id>
`;
}

export function javascriptSnippet(cfg: EvalConfig, baseUrl = "http://localhost:4000"): string {
  const body = JSON.stringify(requestBody(cfg), null, 2);
  return `// npm install unnecessary — plain fetch against the RagArena REST API
const res = await fetch("${baseUrl}/api/evaluate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(${body.split("\n").join("\n  ")}),
});
const { run_id } = await res.json();

async function pollUntilDone(id) {
  for (let i = 0; i < 300; i++) {
    const r = await fetch(\`${baseUrl}/api/runs/\${id}\`).then((r) => r.json());
    if (r.status === "done" || r.status === "error") return r;
    await new Promise((res) => setTimeout(res, 1500));
  }
  throw new Error("timeout");
}

const report = await pollUntilDone(run_id);
console.log(report.aggregate);
`;
}

export type CodegenLanguage = "python" | "curl" | "javascript";

export function generateSnippet(lang: CodegenLanguage, cfg: EvalConfig, baseUrl?: string): string {
  if (lang === "python") return pythonSnippet(cfg);
  if (lang === "curl") return curlSnippet(cfg, baseUrl);
  return javascriptSnippet(cfg, baseUrl);
}
