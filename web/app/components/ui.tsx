"use client";

export function StatCard({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="card">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1.5 text-2xl font-bold">{value}</div>
    </div>
  );
}

export function Chip({
  active,
  onClick,
  children,
}: {
  active?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`chip ${active ? "chip-on" : ""} ${onClick ? "cursor-pointer" : "cursor-default"}`}
    >
      {children}
    </button>
  );
}

const PROVIDER_COLORS: Record<string, string> = {
  openai: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
  anthropic: "text-orange-400 bg-orange-400/10 border-orange-400/20",
  google: "text-blue-400 bg-blue-400/10 border-blue-400/20",
  groq: "text-red-400 bg-red-400/10 border-red-400/20",
  mistral: "text-amber-400 bg-amber-400/10 border-amber-400/20",
  cohere: "text-teal-400 bg-teal-400/10 border-teal-400/20",
  voyage: "text-violet-400 bg-violet-400/10 border-violet-400/20",
};

export function ProviderBadge({ provider }: { provider: string }) {
  const cls = PROVIDER_COLORS[provider] || "text-slate-400 bg-slate-400/10 border-slate-400/20";
  return (
    <span className={`inline-flex rounded-full border px-2 py-0.5 text-[10.5px] font-semibold ${cls}`}>
      {provider}
    </span>
  );
}

export function ScoreBadge({ value }: { value: number | undefined | null }) {
  if (value === undefined || value === null) return <span className="text-slate-600">—</span>;
  const cls = value >= 0.8 ? "text-emerald-400" : value >= 0.5 ? "text-amber-400" : "text-red-400";
  return <span className={`font-mono font-semibold ${cls}`}>{value.toFixed(3)}</span>;
}

export function Spinner() {
  return (
    <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white align-[-2px]" />
  );
}

export function PageHeader({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
      {sub && <p className="mt-1 text-sm text-slate-400">{sub}</p>}
    </div>
  );
}
