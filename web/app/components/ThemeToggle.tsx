"use client";

import { useEffect, useState } from "react";

type ThemeChoice = "system" | "light" | "dark";

function applyTheme(choice: ThemeChoice) {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
}

export default function ThemeToggle() {
  const [choice, setChoice] = useState<ThemeChoice>("system");

  useEffect(() => {
    try {
      const stored = localStorage.getItem("ragarena:theme") as ThemeChoice | null;
      if (stored) setChoice(stored);
    } catch {}
  }, []);

  function cycle() {
    const order: ThemeChoice[] = ["system", "light", "dark"];
    const next = order[(order.indexOf(choice) + 1) % order.length];
    setChoice(next);
    applyTheme(next);
    try {
      localStorage.setItem("ragarena:theme", next);
    } catch {}
  }

  const icon = choice === "system" ? "🖥" : choice === "light" ? "☀" : "🌙";
  const label = choice === "system" ? "System" : choice === "light" ? "Light" : "Dark";

  return (
    <button
      onClick={cycle}
      className="btn-ghost w-full justify-start gap-2.5 text-xs"
      title="Cycle theme: system → light → dark"
    >
      <span className="w-4 text-center">{icon}</span>
      {label}
    </button>
  );
}
