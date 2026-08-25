"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { PageHeader, ProviderBadge, Chip } from "../components/ui";

const FILTERS = ["all", "chat", "embedding", "rerank"];

export default function CatalogPage() {
  const [catalog, setCatalog] = useState<any>(null);
  const [filter, setFilter] = useState("all");
  const [q, setQ] = useState("");

  useEffect(() => {
    api.catalog().then(setCatalog).catch(() => {});
  }, []);

  const rows = useMemo(() => {
    const models = catalog?.models || [];
    const ql = q.toLowerCase();
    return models.filter(
      (m: any) =>
        (filter === "all" || m.modality === filter) &&
        (!ql || m.id.toLowerCase().includes(ql) || (m.description || "").toLowerCase().includes(ql))
    );
  }, [catalog, filter, q]);

  return (
    <div>
      <PageHeader
        title="Model catalog"
        sub={catalog ? `${catalog.counts.total} models · ${catalog.counts.providers} providers` : "Loading…"}
      />
      <input
        className="input mb-3 max-w-sm"
        placeholder="Search models…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      <div className="mb-4 flex flex-wrap gap-2">
        {FILTERS.map((f) => (
          <Chip key={f} active={filter === f} onClick={() => setFilter(f)}>
            {f}
          </Chip>
        ))}
      </div>
      <div className="card overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="text-slate-500">
              <th className="pb-2 pr-4">Model ID</th>
              <th className="pb-2 pr-4">Type</th>
              <th className="pb-2 pr-4">Context</th>
              <th className="pb-2 pr-4">$ In / 1M</th>
              <th className="pb-2 pr-4">$ Out / 1M</th>
              <th className="pb-2">Notes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m: any) => (
              <tr key={m.id} className="border-t border-white/5">
                <td className="py-2 pr-4">
                  <div className="flex items-center gap-2">
                    <ProviderBadge provider={m.provider} />
                    <span className="font-mono text-slate-300">{m.model_name}</span>
                  </div>
                </td>
                <td className="py-2 pr-4 text-slate-400">{m.modality}</td>
                <td className="py-2 pr-4 font-mono text-slate-400">
                  {m.modality === "chat" && m.context_window ? `${Math.round(m.context_window / 1000)}k` : "—"}
                </td>
                <td className="py-2 pr-4 font-mono text-slate-400">
                  {m.input_cost ? `$${m.input_cost}` : "free"}
                </td>
                <td className="py-2 pr-4 font-mono text-slate-400">{m.output_cost ? `$${m.output_cost}` : "—"}</td>
                <td className="py-2 text-slate-500">{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
