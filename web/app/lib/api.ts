export interface Metric {
  name: string;
  requires_reference: boolean;
  llm_judged: boolean;
}
export interface StrategyInfo {
  name: string;
  description: string;
}
export interface Options {
  chat_models: any[];
  embedding_models: any[];
  strategies: StrategyInfo[];
  metrics_presets: string[];
  datasets: string[];
  chunk_sizes: number[];
  chunk_overlaps: number[];
}

async function getJSON(url: string) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url} → ${r.status}`);
  return r.json();
}
async function postJSON(url: string, body: any) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = "";
    try {
      detail = (await r.json()).detail || "";
    } catch {}
    throw new Error(`POST ${url} → ${r.status} ${detail}`);
  }
  return r.json();
}

export const api = {
  options: () => getJSON("/api/options"),
  catalog: () => getJSON("/api/catalog"),
  datasets: () => getJSON("/api/datasets"),
  dataset: (name: string, n = 20, bundled = false) =>
    getJSON(`/api/datasets/${name}?n=${n}&bundled=${bundled}`),
  upload: async (files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`upload → ${r.status}`);
    return r.json();
  },
  ingest: (text: string) => postJSON("/api/ingest", { text }),
  evaluate: (body: any) => postJSON("/api/evaluate", body),
  compare: (body: any) => postJSON("/api/compare", body),
  recommend: (body: any) => postJSON("/api/recommend", body),
  run: (id: string) => getJSON(`/api/runs/${id}`),
  runs: () => getJSON("/api/runs"),
  envStatus: () => getJSON("/api/env-status"),
  pollUntilDone: async (id: string, onTick?: (s: any) => void) => {
    for (let i = 0; i < 600; i++) {
      const s = await getJSON(`/api/runs/${id}`);
      if (s.status === "done" || s.status === "error") return s;
      onTick?.(s);
      await new Promise((r) => setTimeout(r, 1500));
    }
    throw new Error("timeout polling run " + id);
  },
};
