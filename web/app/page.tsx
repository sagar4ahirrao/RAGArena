"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "./lib/api";
import { StatCard, Chip, PageHeader } from "./components/ui";

export default function OverviewPage() {
  const [opts, setOpts] = useState<any>(null);
  const [catalog, setCatalog] = useState<any>(null);
  const [runs, setRuns] = useState<any[]>([]);
  const [envStatus, setEnvStatus] = useState<any>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    Promise.all([api.options(), api.catalog(), api.runs()])
      .then(([o, c, r]) => {
        setOpts(o);
        setCatalog(c);
        setRuns(r);
      })
      .catch((e) => setErr(String(e.message || e)));
    api.envStatus().then(setEnvStatus).catch(() => {});
  }, []);

  const providers = envStatus
    ? Object.entries(envStatus.providers || {}).sort(([, a]: any, [, b]: any) => Number(b) - Number(a))
    : [];

  return (
    <div>
      <PageHeader
        title="Overview"
        sub="Evaluate & discover the best RAG strategy, model and parser combination for your data."
      />

      {err && (
        <div className="card mb-5 border-red-500/30 bg-red-500/10 text-sm text-red-300">
          Couldn&apos;t reach the RagArena API — is <code>ragarena serve</code> running? ({err})
        </div>
      )}

      {providers.length > 0 && (
        <div className="card mb-6">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200">Provider API keys</h3>
            <span className="text-[11px] text-slate-500">
              {envStatus.configured_count} of {providers.length} configured
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {providers.map(([name, configured]: any) => (
              <span
                key={name}
                className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold ${
                  configured
                    ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-400"
                    : "border-white/10 bg-ink-900/40 text-slate-600"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${configured ? "bg-emerald-400" : "bg-slate-600"}`}
                />
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Runs stored" value={runs.length || "–"} />
        <StatCard label="Chat models" value={catalog?.counts?.chat ?? "–"} />
        <StatCard label="Embedding models" value={catalog?.counts?.embedding ?? "–"} />
        <StatCard label="Strategies" value={opts?.strategies?.length ?? "–"} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="card">
          <h3 className="mb-3 text-sm font-semibold text-slate-200">Recent runs</h3>
          {runs.length === 0 ? (
            <p className="text-sm text-slate-500">No runs yet — start one in the Playground.</p>
          ) : (
            <div className="flex flex-col gap-2">
              {[...runs].reverse().slice(0, 6).map((r) => (
                <Link
                  key={r.run_id}
                  href="/runs"
                  className="flex items-center justify-between rounded-lg border border-white/5 bg-ink-900/40 px-3 py-2 text-xs hover:border-brand-500/30"
                >
                  <span className="font-mono text-slate-400">{r.run_id}</span>
                  <span className="text-slate-300">{r.kind}</span>
                  <span className="text-slate-500">{r.strategy}</span>
                </Link>
              ))}
            </div>
          )}
          <Link href="/playground" className="btn mt-4 w-full justify-center">
            ▶ New evaluation
          </Link>
        </div>

        <div className="card">
          <h3 className="mb-3 text-sm font-semibold text-slate-200">Strategies available</h3>
          <div className="flex flex-wrap gap-2">
            {(opts?.strategies || []).map((s: any) => (
              <Chip key={s.name}>{s.name}</Chip>
            ))}
          </div>
          <h3 className="mb-3 mt-5 text-sm font-semibold text-slate-200">Metric presets</h3>
          <div className="flex flex-wrap gap-2">
            {(opts?.metrics_presets || []).map((m: string) => (
              <Chip key={m}>{m}</Chip>
            ))}
          </div>
          <h3 className="mb-3 mt-5 text-sm font-semibold text-slate-200">Bundled &amp; benchmark datasets</h3>
          <div className="flex flex-wrap gap-2">
            {(opts?.datasets || []).map((d: any) => (
              <Chip key={d.name}>{d.name}</Chip>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
