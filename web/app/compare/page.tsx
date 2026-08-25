"use client";

import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { PageHeader, ScoreBadge, Spinner } from "../components/ui";

type Cfg = { strategy: string; model: string };

export default function ComparePage() {
  const [opts, setOpts] = useState<any>(null);
  const [docsText, setDocsText] = useState("");
  const [questionsText, setQuestionsText] = useState("");
  const [configs, setConfigs] = useState<Cfg[]>([
    { strategy: "naive", model: "openai/gpt-4o-mini" },
    { strategy: "hybrid", model: "openai/gpt-4o-mini" },
  ]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    api.options().then(setOpts).catch(() => {});
  }, []);

  function addConfig() {
    setConfigs((c) => [...c, { strategy: "naive", model: "openai/gpt-4o-mini" }]);
  }
  function removeConfig(i: number) {
    setConfigs((c) => c.filter((_, idx) => idx !== i));
  }
  function updateConfig(i: number, patch: Partial<Cfg>) {
    setConfigs((c) => c.map((cfg, idx) => (idx === i ? { ...cfg, ...patch } : cfg)));
  }

  async function run() {
    const docs = docsText.split("\n").map((l) => l.trim()).filter(Boolean).map((t) => ({ text: t }));
    const lines = questionsText.split("\n").map((l) => l.trim()).filter(Boolean);
    const questions: string[] = [];
    const refs: (string | null)[] = [];
    for (const l of lines) {
      const [q, r] = l.split("::");
      questions.push(q.trim());
      refs.push(r?.trim() || null);
    }
    if (docs.length === 0 || questions.length === 0 || configs.length < 2) {
      setStatus("need documents, questions, and at least 2 configs");
      return;
    }
    setRunning(true);
    setResult(null);
    setStatus("submitting battle…");
    try {
      const { run_id } = await api.compare({
        questions,
        documents: docs,
        configs,
        reference_answers: refs.some(Boolean) ? refs : null,
      });
      const final = await api.pollUntilDone(run_id, (s) => setStatus(`battle in progress… (${s.status})`));
      if (final.status === "error") throw new Error(final.error || "comparison failed");
      setResult(final);
      setStatus("");
    } catch (e: any) {
      setStatus(String(e.message || e));
    } finally {
      setRunning(false);
    }
  }

  const strategies = opts?.strategies || [];
  const chatModels = opts?.chat_models || [];
  const rows = result ? Object.entries(result.matrix).map(([name, v]: any) => ({ config: name, ...v })) : [];
  const cols = rows.length ? Object.keys(rows[0]).filter((c) => c !== "config") : [];

  return (
    <div>
      <PageHeader title="Compare" sub="Head-to-head leaderboard across strategy × model configurations on the same corpus." />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          <label className="label">Documents (one per line)</label>
          <textarea className="input min-h-[140px]" value={docsText} onChange={(e) => setDocsText(e.target.value)} />
        </div>
        <div>
          <label className="label">Questions (one per line, ::answer optional)</label>
          <textarea className="input min-h-[140px]" value={questionsText} onChange={(e) => setQuestionsText(e.target.value)} />
        </div>
      </div>

      <div className="card mt-4">
        <label className="label mb-2">Configurations to compare</label>
        <div className="flex flex-col gap-2">
          {configs.map((cfg, i) => (
            <div key={i} className="flex items-center gap-2">
              <select
                className="input flex-1"
                value={cfg.strategy}
                onChange={(e) => updateConfig(i, { strategy: e.target.value })}
              >
                {strategies.map((s: any) => (
                  <option key={s.name} value={s.name}>
                    {s.name}
                  </option>
                ))}
              </select>
              <select
                className="input flex-[2]"
                value={cfg.model}
                onChange={(e) => updateConfig(i, { model: e.target.value })}
              >
                {chatModels.map((m: any) => (
                  <option key={m.id} value={m.id}>
                    {m.id}
                  </option>
                ))}
              </select>
              <button className="btn-ghost" onClick={() => removeConfig(i)}>
                ✕
              </button>
            </div>
          ))}
        </div>
        <button className="btn-ghost mt-3" onClick={addConfig}>
          + Add config
        </button>
      </div>

      <div className="mt-4 flex items-center gap-3">
        <button className="btn" disabled={running} onClick={run}>
          {running ? (
            <>
              <Spinner /> Running…
            </>
          ) : (
            "⚔ Run comparison"
          )}
        </button>
        {status && <span className="text-xs text-slate-400">{status}</span>}
      </div>

      {rows.length > 0 && (
        <div className="card mt-6 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-slate-500">
                <th className="pb-2 pr-4">Config</th>
                {cols.map((c) => (
                  <th key={c} className="pb-2 pr-4">
                    {c.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r: any) => (
                <tr key={r.config} className="border-t border-white/5">
                  <td className="py-2 pr-4 font-medium">{r.config}</td>
                  {cols.map((c) => (
                    <td key={c} className="py-2 pr-4">
                      {c.includes("cost") ? (
                        `$${Number(r[c] || 0).toFixed(5)}`
                      ) : c.includes("latency") ? (
                        `${Number(r[c] || 0).toFixed(2)}s`
                      ) : typeof r[c] === "number" ? (
                        <ScoreBadge value={r[c]} />
                      ) : (
                        String(r[c])
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
