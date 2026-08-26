"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ThemeToggle from "./ThemeToggle";

const NAV = [
  { href: "/", label: "Overview", icon: "◆" },
  { href: "/playground", label: "Playground", icon: "▶" },
  { href: "/compare", label: "Compare", icon: "⚔" },
  { href: "/recommend", label: "Recommend", icon: "★" },
  { href: "/datasets", label: "Datasets", icon: "▤" },
  { href: "/catalog", label: "Model Catalog", icon: "◈" },
  { href: "/runs", label: "Runs", icon: "▦" },
];

export default function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="flex min-h-screen">
      <aside className="flex w-60 shrink-0 flex-col border-r border-line/10 bg-ink-800/60 backdrop-blur">
        <div className="px-5 py-6">
          <div className="text-lg font-bold tracking-tight text-fg">
            ⚡ Rag<span className="text-brand-400">Arena</span>
          </div>
          <div className="mt-1 text-[11px] text-fg-muted">
            RAG strategy &amp; model playground
          </div>
        </div>
        <nav className="flex-1 px-3">
          {NAV.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`mb-1 flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition ${
                  active
                    ? "bg-brand-500/15 text-fg border border-brand-500/30"
                    : "text-fg-muted hover:bg-ink-700 hover:text-fg"
                }`}
              >
                <span className="w-4 text-center opacity-80">{item.icon}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="border-t border-line/10 px-3 py-3">
          <ThemeToggle />
        </div>
        <div className="border-t border-line/10 px-5 py-4 text-[11px] leading-relaxed text-fg-muted">
          18 strategies · 100+ models · any provider
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto px-8 py-7">{children}</main>
    </div>
  );
}
