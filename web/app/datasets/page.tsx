"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "../lib/api";
import { PageHeader } from "../components/ui";

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [preview, setPreview] = useState<any>(null);
  const [loading, setLoading] = useState("");
  const router = useRouter();

  useEffect(() => {
    api.datasets().then((r) => setDatasets(r.datasets)).catch(() => {});
  }, []);

  async function open(name: string) {
    setLoading(name);
    try {
      const d = await api.dataset(name, 10, true);
      setPreview(d);
    } catch (e: any) {
      setPreview({ name, error: String(e.message || e) });
    } finally {
      setLoading("");
    }
  }

  function sendToPlayground(name: string) {
    sessionStorage.setItem("ragarena:dataset", name);
    router.push("/playground");
  }

  return (
    <div>
      <PageHeader
        title="Datasets"
        sub="Bundled offline QA sets and popular benchmark loaders (SQuAD, HotpotQA, Natural Questions, TriviaQA, MS MARCO)."
      />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="flex flex-col gap-3">
          {datasets.map((d) => (
            <div key={d.name} className="card">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{d.name}</span>
                    <span
                      className={`chip ${d.offline ? "chip-on" : ""}`}
                      title={d.offline ? "no network required" : "requires `pip install ragarena[datasets]`"}
                    >
                      {d.offline ? "offline" : "HuggingFace"}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-fg-muted">{d.description}</p>
                  <p className="mt-1 text-[11px] text-fg-muted">size: {String(d.size)}</p>
                </div>
                <div className="flex shrink-0 flex-col gap-1.5">
                  <button className="btn-ghost text-xs" onClick={() => open(d.name)}>
                    {loading === d.name ? "loading…" : "preview"}
                  </button>
                  <button className="btn text-xs" onClick={() => sendToPlayground(d.name)}>
                    use in playground →
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="card min-h-[200px]">
          <h3 className="mb-2 text-sm font-semibold text-fg">Preview</h3>
          {!preview && <p className="text-sm text-fg-muted">Click "preview" on a dataset to inspect it here.</p>}
          {preview?.error && <p className="text-sm text-red-400">{preview.error}</p>}
          {preview && !preview.error && (
            <div className="text-xs">
              <p className="mb-2 text-fg-muted">
                {preview.n_documents} documents · {preview.n_questions} questions
              </p>
              <p className="label">Sample document</p>
              <p className="mb-3 rounded-lg bg-ink-900/60 p-2 text-fg">{preview.sample_document}</p>
              <p className="label">Questions</p>
              <ul className="flex flex-col gap-1">
                {preview.questions?.slice(0, 8).map((q: string, i: number) => (
                  <li key={i} className="rounded-lg bg-ink-900/40 p-2 text-fg">
                    {q}
                    {preview.reference_answers?.[i] && (
                      <span className="text-fg-muted"> → {preview.reference_answers[i]}</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
