"use client";

import { useState } from "react";
import { EvalConfig, CodegenLanguage, generateSnippet } from "../lib/codegen";

const TABS: { id: CodegenLanguage; label: string }[] = [
  { id: "python", label: "Python" },
  { id: "curl", label: "cURL" },
  { id: "javascript", label: "JavaScript" },
];

export default function CodePanel({ config }: { config: EvalConfig }) {
  const [open, setOpen] = useState(false);
  const [lang, setLang] = useState<CodegenLanguage>("python");
  const [copied, setCopied] = useState(false);

  const code = generateSnippet(lang, config, typeof window !== "undefined" ? window.location.origin : undefined);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  }

  if (!open) {
    return (
      <button className="btn-ghost" onClick={() => setOpen(true)}>
        {"</>"} Get code
      </button>
    );
  }

  return (
    <div className="card mt-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex gap-1.5">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setLang(t.id)}
              className={`chip ${lang === t.id ? "chip-on" : ""}`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <button className="btn-ghost" onClick={copy}>
            {copied ? "copied!" : "copy"}
          </button>
          <button className="btn-ghost" onClick={() => setOpen(false)}>
            ✕ close
          </button>
        </div>
      </div>
      <pre className="mono max-h-[420px] overflow-auto rounded-xl bg-ink-900/60 p-4 text-[12px] leading-relaxed text-fg">
        <code>{code}</code>
      </pre>
    </div>
  );
}
