"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { Chip, PageHeader, ScoreBadge, Spinner } from "../components/ui";

type Doc = { text: string; metadata?: Record<string, any> };

export default function RecommendPage() {
  const [opts, setOpts] = useState<any>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [docSource, setDocSource] = useState<string>("");
  const [questionsText, setQuestionsText] = useState("");
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [embedModel, setEmbedModel] = useState("openai/text-embedding-3-small");
  const [judgeModel, setJudgeModel] = useState("openai/gpt-4o-mini");
  const [metricsPreset, setMetricsPreset] = useState("quality");
  const [qualityWeight, setQualityWeight] = useState(0.7);
  const [costWeight, setCostWeight] = useState(0.15);
  const [latencyWeight, setLatencyWeight] = useState(0.15);
  const [status, setStatus] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .options()
      .then((o) => {
        setOpts(o);
        setSelectedStrategies((o.strategies || []).map((s: any) => s.name));
      })
      .catch(() => {});
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

  function toggleStrategy(name: string) {
    setSelectedStrategies((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]));
  }

  function selectAllStrategies() {
    setSelectedStrategies((opts?.strategies || []).map((s: any) => s.name));
  }
  function selectNoStrategies() {
    setSelectedStrategies([]);
  }

  async function run() {
    if (docs.length === 0) {
      setStatus("add documents first — upload a file, pick a dataset, or paste text");
      return;
    }
    const lines = questionsText.split("\n").map((l) => l.trim()).filter(Boolean);
    const questions: string[] = [];
    const refs: (string | null)[] = [];
    for (const l of lines) {
      const [q, r] = l.split("::");
      questions.push(q.trim());
      refs.push(r?.trim() || null);
    }
    if (questions.length === 0) {
      setStatus("add at least one question");
      return;
    }
    if (selectedStrategies.length === 0) {
      setStatus("select at least one strategy to evaluate");
      return;
    }
    setRunning(true);
    setResult(null);
    setStatus("submitting…");
    try {
      const allStrategies = (opts?.strategies || []).map((s: any) => s.name);
      const { run_id } = await api.recommend({
        questions,
        documents: docs,
        reference_answers: refs.some(Boolean) ? refs : null,
        strategies: selectedStrategies.length === allStrategies.length ? undefined : selectedStrategies,
        model,
        embedding_model: embedModel,
        judge_model: judgeModel,
        metrics: metricsPreset,
        quality_weight: qualityWeight,
        cost_weight: costWeight,
        latency_weight: latencyWeight,
      });
      const final = await api.pollUntilDone(run_id, (s) =>
        setStatus(`running every strategy against your corpus… (${s.status})`)
      );
      if (final.status === "error") throw new Error(final.error || "recommendation failed");
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
  const leaderboard = result?.leaderboard || [];

  return (
    <div>
      <PageHeader
        title="Recommend"
        sub="Run every RAG strategy against the same corpus and questions to find the best-performing configuration."
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Left: corpus */}
        <div className="card lg:col-span-1">
          <h3 className="mb-3 text-sm font-semibold text-slate-200">1. Corpus</h3>
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
          <p className="mt-2 text-[11px] text-slate-500">
            pdf · docx · pptx · html · csv · json · xlsx · md · txt · images
          </p>

          <div className="my-3 border-t border-white/5" />
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

          <div className="my-3 border-t border-white/5" />
          <label className="label">Or paste raw text (one document per line)</label>
          <textarea
            className="input min-h-[90px]"
            placeholder="Paris is the capital of France.&#10;The Eiffel Tower was built in 1889."
            onBlur={(e) => e.target.value.trim() && pasteDocs(e.target.value)}
          />

          {docSource && <p className="mt-3 text-xs text-emerald-400">✓ {docSource}</p>}

          <div className="my-3 border-t border-white/5" />
          <label className="label">Questions</label>
          <textarea
            className="input min-h-[120px]"
            placeholder={"What is the capital of France?::Paris\nWhen was the Eiffel Tower built?"}
            value={questionsText}
            onChange={(e) => setQuestionsText(e.target.value)}
          />
          <p className="mt-1 text-[11px] text-slate-500">one per line — suffix with ::answer for a reference</p>
        </div>

        {/* Middle: strategies */}
        <div className="card lg:col-span-1">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200">2. Strategies to evaluate</h3>
            <span className="text-[11px] text-slate-500">{selectedStrategies.length}/{strategies.length}</span>
          </div>
          <div className="mb-2 flex gap-2">
            <button className="btn-ghost text-[11px]" onClick={selectAllStrategies}>
              select all
            </button>
            <button className="btn-ghost text-[11px]" onClick={selectNoStrategies}>
              clear
            </button>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {strategies.map((s: any) => (
              <Chip key={s.name} active={selectedStrategies.includes(s.name)} onClick={() => toggleStrategy(s.name)}>
                {s.name}
              </Chip>
            ))}
          </div>

          <div className="my-3 border-t border-white/5" />
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
          <select className="input" value={judgeModel} onChange={(e) => setJudgeModel(e.target.value)}>
            {chatModels.map((m: any) => (
              <option key={m.id} value={m.id}>
                {m.id}
              </option>
            ))}
          </select>
        </div>

        {/* Right: metrics + weights + run */}
        <div className="card lg:col-span-1">
          <h3 className="mb-3 text-sm font-semibold text-slate-200">3. Ranking</h3>
          <label className="label">Metric preset</label>
          <select className="input mb-4" value={metricsPreset} onChange={(e) => setMetricsPreset(e.target.value)}>
            {(opts?.metrics_presets || ["production", "quality", "quick", "full"]).map((p: string) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>

          <label className="label flex items-center justify-between">
            <span>Quality weight</span>
            <span className="font-mono text-slate-400">{qualityWeight.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={qualityWeight}
            onChange={(e) => setQualityWeight(Number(e.target.value))}
            className="mb-3 w-full"
          />

          <label className="label flex items-center justify-between">
            <span>Cost weight</span>
            <span className="font-mono text-slate-400">{costWeight.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={costWeight}
            onChange={(e) => setCostWeight(Number(e.target.value))}
            className="mb-3 w-full"
          />

          <label className="label flex items-center justify-between">
            <span>Latency weight</span>
            <span className="font-mono text-slate-400">{latencyWeight.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={latencyWeight}
            onChange={(e) => setLatencyWeight(Number(e.target.value))}
            className="mb-4 w-full"
          />

          <button className="btn w-full justify-center" disabled={running} onClick={run}>
            {running ? (
              <>
                <Spinner /> Running…
              </>
            ) : (
              "★ Get recommendation"
            )}
          </button>
          {status && <p className="mt-2 text-xs text-slate-400">{status}</p>}
        </div>
      </div>

      {result && (
        <div className="mt-6">
          {result.best && (
            <div className="card mb-4 border-brand-500/30 bg-brand-500/10">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-brand-400">
                ★ Recommended strategy
              </div>
              <div className="mt-1 text-2xl font-bold text-white">{result.best}</div>
              {result.reasoning && <p className="mt-2 text-sm text-slate-300">{result.reasoning}</p>}
            </div>
          )}

          <h3 className="mb-3 text-sm font-semibold text-slate-200">Leaderboard</h3>
          <div className="card overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-slate-500">
                  <th className="pb-2 pr-4">Rank</th>
                  <th className="pb-2 pr-4">Strategy</th>
                  <th className="pb-2 pr-4">Quality</th>
                  <th className="pb-2 pr-4">Cost</th>
                  <th className="pb-2 pr-4">Latency</th>
                  <th className="pb-2">Composite</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row: any, i: number) => (
                  <tr
                    key={row.strategy}
                    className={`border-t border-white/5 ${row.strategy === result.best ? "bg-brand-500/5" : ""}`}
                  >
                    <td className="py-2 pr-4 font-mono text-slate-500">#{i + 1}</td>
                    <td className="py-2 pr-4 font-medium">
                      {row.strategy}
                      {row.strategy === result.best && <span className="ml-1.5 text-brand-400">★</span>}
                    </td>
                    <td className="py-2 pr-4">
                      <ScoreBadge value={row.quality_score} />
                    </td>
                    <td className="py-2 pr-4">${Number(row.total_cost_usd || 0).toFixed(5)}</td>
                    <td className="py-2 pr-4">{Number(row.avg_latency_s || 0).toFixed(2)}s</td>
                    <td className="py-2">
                      <ScoreBadge value={row.composite_score} />
                    </td>
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
