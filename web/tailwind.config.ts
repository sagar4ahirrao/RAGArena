export default {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: { 900: "#0a0a0f", 800: "#111118", 700: "#1a1a24", 600: "#23232f" },
        brand: { 400: "#7c8cff", 500: "#5b6cff", 600: "#4a55e0" },
        accent: { 400: "#36e0c0", 500: "#1fc7a8" },
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: { glow: "0 0 0 1px rgba(124,140,255,0.25), 0 8px 30px rgba(91,108,255,0.15)" },
    },
  },
  plugins: [],
};
