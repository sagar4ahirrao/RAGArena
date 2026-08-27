"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { Chip, PageHeader, ScoreBadge, Spinner, StatCard } from "../components/ui";
import CodePanel from "../components/CodePanel";

type Doc = { text: string; metadata?: Record<string, any> };

export default function PlaygroundPage() {
  const [opts, setOpts] = useState<any>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [docSource, setDocSource] = useState<string>("");
  const [questionsText, setQuestionsText] = useState("");
  const [strategy, setStrategy] = useState("naive");
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [embedModel, setEmbedModel] = useState("openai/text-embedding-3-small");
  const [judgeModel, setJudgeModel] = useState("openai/gpt-4o-mini");
  const [metricsPreset, setMetricsPreset] = useState("production");
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(150);
  const [status, setStatus] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.options().then(setOpts).catch(() => {});
    const pending = typeof window !== "undefined" ? sessionStorage.getItem("ragarena:dataset") : null;
    if (pending) {
      sessionStorage.removeItem("ragarena:dataset");
      loadDataset(pending);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onUpload(files: FileList | null) {
    if (!files || files.length === 0) return;
    setStatus("parsing uploaded files…");
    try {
      const res = await api.upload(Array.from(files));
      setDocs(res.documents);
      setDocSource(`${files.length} file(s) → ${res.n_documents} chunks parsed`);
      setStatus("");
    } catch (e: any) {
      setStatus(String(e.message || e));
    }
  }

  async function loadDataset(name: string) {
    setStatus(`loading dataset "${name}"…`);
    try {
      const res = await api.dataset(name, 20, true);
      setDocs(res.documents);
      setQuestionsText(
        res.questions
          .map((q: string, i: number) => (res.reference_answers[i] ? `${q}::${res.reference_answers[i]}` : q))
          .join("\n")
      );
      setDocSource(`dataset "${name}" → ${res.n_documents} documents, ${res.n_questions} questions`);
      setStatus("");
    } catch (e: any) {
      setStatus(String(e.message || e));
    }
  }

  function pasteDocs(text: string) {
    const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
    setDocs(lines.map((t) => ({ text: t })));
    setDocSource(`${lines.length} pasted line(s)`);
  }

  function parseQuestions() {
    const lines = questionsText.split("\n").map((l) => l.trim()).filter(Boolean);
    const questions: string[] = [];
    const refs: (string | null)[] = [];
    for (const l of lines) {
      const [q, r] = l.split("::");
      questions.push(q.trim());
      refs.push(r?.trim() || null);
    }
    return { questions, refs };
  }

  async function run() {
    if (docs.length === 0) {
      setStatus("add documents first — upload a file, pick a dataset, or paste text");
      return;
    }
    const { questions, refs } = parseQuestions();
    if (questions.length === 0) {
      setStatus("add at least one question");
      return;
    }
    setRunning(true);
    setResult(null);
    setStatus("submitting…");
    try {
      const { run_id } = await api.evaluate({
        questions,
        documents: docs,
        reference_answers: refs.some(Boolean) ? refs : null,
        strategy,
        model,
        embedding_model: embedModel,
        judge_model: judgeModel,
        metrics: metricsPreset,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
      });
      const final = await api.pollUntilDone(run_id, (s) => setStatus(`retrieving → generating → judging… (${s.status})`));
      if (final.status === "error") throw new Error(final.error || "evaluation failed");
      setResult(final);
      setStatus(`done in ${final.wall_time_s ?? "?"}s`);
    } catch (e: any) {
      setStatus(String(e.message || e));
    } finally {
      setRunning(false);
    }
  }

  const chatModels = opts?.chat_models || [];
  const embModels = opts?.embedding_models || [];
  const strategies = opts?.strategies || [];
  const datasets = opts?.datasets || [];

  return (
    <div>
      <PageHeader
        title="Playground"
        sub="Combine a parser, chunking strategy, RAG strategy and model to find your best-performing RAG stack."
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Left: corpus */}
        <div className="card lg:col-span-1">
          <h3 className="mb-3 text-sm font-semibold text-fg">1. Corpus</h3>
          <div className="flex flex-wrap gap-2">
            <button className="btn-ghost" onClick={() => fileInput.current?.click()}>
              ⬆ Upload files
            </button>
            <input
              ref={fileInput}
              type="file"
              multiple
              className="hidden"
              accept=".txt,.md,.pdf,.docx,.pptx,.html,.htm,.csv,.json,.jsonl,.xlsx,.xls,.png,.jpg,.jpeg"
              onChange={(e) => onUpload(e.target.files)}
            />
          </div>
          <p className="mt-2 text-[11px] text-fg-muted">
            pdf · docx · pptx · html · csv · json · xlsx · md · txt · images
          </p>

          <div className="my-3 border-t border-line/5" />
          <label className="label">Or pick a bundled/benchmark dataset</label>
          <select className="input" onChange={(e) => e.target.value && loadDataset(e.target.value)} defaultValue="">
            <option value="" disabled>
              choose a dataset…
            </option>
            {datasets.map((d: any) => (
              <option key={d.name} value={d.name}>
                {d.name} — {d.description}
              </option>
            ))}
          </select>

          <div className="my-3 border-t border-line/5" />
          <label className="label">Or paste raw text (one document per line)</label>
          <textarea
            className="input min-h-[90px]"
            placeholder="Paris is the capital of France.&#10;The Eiffel Tower was built in 1889."
            onBlur={(e) => e.target.value.trim() && pasteDocs(e.target.value)}
          />

          {docSource && <p className="mt-3 text-xs text-emerald-400">✓ {docSource}</p>}

          <div className="my-3 border-t border-line/5" />
          <label className="label">Chunk size / overlap</label>
          <div className="flex gap-2">
            <select className="input" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))}>
              {(opts?.chunk_sizes || [256, 512, 1000, 1500]).map((n: number) => (
                <option key={n} value={n}>
                  {n} chars
                </option>
              ))}
            </select>
            <select className="input" value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))}>
              {(opts?.chunk_overlaps || [0, 128, 200]).map((n: number) => (
                <option key={n} value={n}>
                  {n} overlap
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Middle: strategy + models */}
        <div className="card lg:col-span-1">
          <h3 className="mb-3 text-sm font-semibold text-fg">2. Strategy &amp; models</h3>
          <label className="label">RAG strategy</label>
          <div className="mb-3 flex flex-wrap gap-1.5">
            {strategies.map((s: any) => (
              <Chip key={s.name} active={strategy === s.name} onClick={() => setStrategy(s.name)}>
                {s.name}
              </Chip>
            ))}
          </div>
          <p className="mb-3 text-xs text-fg-muted">
            {strategies.find((s: any) => s.name === strategy)?.description}
          </p>

          <label className="label">Generator model</label>
          <select className="input mb-3" value={model} onChange={(e) => setModel(e.target.value)}>
            {chatModels.map((m: any) => (
              <option key={m.id} value={m.id}>
                {m.id}
              </option>
            ))}
          </select>

          <label className="label">Embedding model</label>
          <select className="input mb-3" value={embedModel} onChange={(e) => setEmbedModel(e.target.value)}>
            {embModels.map((m: any) => (
              <option key={m.id} value={m.id}>
                {m.id}
              </option>
            ))}
          </select>

          <label className="label">Judge model (LLM-as-judge)</label>
          <select className="input mb-3" value={judgeModel} onChange={(e) => setJudgeModel(e.target.value)}>
            {chatModels.map((m: any) => (
              <option key={m.id} value={m.id}>
                {m.id}
              </option>
            ))}
          </select>

          <label className="label">Metric preset</label>
          <select className="input" value={metricsPreset} onChange={(e) => setMetricsPreset(e.target.value)}>
            {(opts?.metrics_presets || ["production", "quality", "quick", "full"]).map((p: string) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>

        {/* Right: questions + run */}
        <div className="card lg:col-span-1">
          <h3 className="mb-3 text-sm font-semibold text-fg">3. Questions</h3>
          <textarea
            className="input min-h-[160px]"
            placeholder={"What is the capital of France?::Paris\nWhen was the Eiffel Tower built?"}
            value={questionsText}
            onChange={(e) => setQuestionsText(e.target.value)}
          />
          <p className="mt-1 text-[11px] text-fg-muted">one per line — suffix with ::answer for a reference</p>

          <div className="mt-4 flex gap-2">
            <button className="btn flex-1 justify-center" disabled={running} onClick={run}>
              {running ? (
                <>
                  <Spinner /> Running…
                </>
              ) : (
                "▶ Run evaluation"
              )}
            </button>
          </div>
          {status && <p className="mt-2 text-xs text-fg-muted">{status}</p>}

          {docs.length > 0 && questionsText.trim() && (
            <div className="mt-3">
              <CodePanel
                config={{
                  questions: parseQuestions().questions,
                  documents: docs.map((d) => ({ text: d.text })),
                  reference_answers: parseQuestions().refs,
                  strategy,
                  model,
                  embedding_model: embedModel,
                  judge_model: judgeModel,
                  metrics: metricsPreset,
                  chunk_size: chunkSize,
                  chunk_overlap: chunkOverlap,
                }}
              />
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="mt-6">
          <h3 className="mb-3 text-sm font-semibold text-fg">Results</h3>
          <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
            {["faithfulness", "answer_relevance", "context_precision", "avg_latency_s", "total_cost_usd"]
              .filter((k) => result.aggregate?.[k] !== undefined)
              .map((k) => (
                <StatCard
                  key={k}
                  label={k.replace(/_/g, " ")}
                  value={
                    k.includes("cost")
                      ? `$${Number(result.aggregate[k]).toFixed(5)}`
                      : k.includes("latency")
                      ? `${result.aggregate[k].toFixed(2)}s`
                      : Number(result.aggregate[k]).toFixed(3)
                  }
                />
              ))}
          </div>
          <div className="card overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-fg-muted">
                  <th className="pb-2 pr-4">Question</th>
                  <th className="pb-2 pr-4">Answer</th>
                  {Object.keys(result.samples?.[0]?.metrics || {}).map((m) => (
                    <th key={m} className="pb-2 pr-4">
                      {m}
                    </th>
                  ))}
                  <th className="pb-2">Latency</th>
                </tr>
              </thead>
              <tbody>
                {(result.samples || []).map((s: any, i: number) => (
                  <tr key={i} className="border-t border-line/5">
                    <td className="max-w-[220px] truncate py-2 pr-4">{s.question}</td>
                    <td className="max-w-[280px] truncate py-2 pr-4 text-fg-muted">{s.answer}</td>
                    {Object.keys(result.samples?.[0]?.metrics || {}).map((m) => (
                      <td key={m} className="py-2 pr-4">
                        <ScoreBadge value={s.metrics?.[m]?.score} />
                      </td>
                    ))}
                    <td className="py-2 font-mono text-fg-muted">{s.latency_s?.toFixed(2)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
