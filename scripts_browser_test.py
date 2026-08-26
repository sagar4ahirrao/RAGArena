"""Real browser smoke test for the RagArena playground UI (Playwright)."""
import sys
import time
from playwright.sync_api import sync_playwright

BASE = "http://localhost:4000"
PAGES = ["/", "/playground", "/compare", "/recommend", "/datasets", "/catalog", "/runs"]

results = []

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    console_errors = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: console_errors.append(f"pageerror: {exc}"))

    for path in PAGES:
        console_errors.clear()
        try:
            page.goto(BASE + path, wait_until="networkidle", timeout=15000)
            page.wait_for_timeout(500)
            title = page.title()
            body_text = page.inner_text("body")
            has_content = len(body_text.strip()) > 20
            errs = list(console_errors)
            results.append((path, "OK" if has_content and not errs else "WARN",
                             f"title={title!r} len={len(body_text)} console_errors={errs[:3]}"))
        except Exception as e:
            results.append((path, "FAIL", str(e)[:200]))

    # Deeper check: Overview page shows provider status pills
    try:
        page.goto(BASE + "/", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(800)
        body = page.inner_text("body")
        has_providers = "configured" in body.lower()
        results.append(("/ (provider panel)", "OK" if has_providers else "WARN", f"has_providers={has_providers}"))
    except Exception as e:
        results.append(("/ (provider panel)", "FAIL", str(e)[:200]))

    # Deeper check: Playground page has strategy chips and model selects
    try:
        page.goto(BASE + "/playground", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(800)
        chips = page.locator(".chip").count()
        selects = page.locator("select").count()
        results.append(("/playground (widgets)", "OK" if chips > 5 and selects >= 3 else "WARN",
                         f"chips={chips} selects={selects}"))
    except Exception as e:
        results.append(("/playground (widgets)", "FAIL", str(e)[:200]))

    browser.close()

print(f"\n{'PATH':<28}{'STATUS':<8}DETAILS")
for path, status, detail in results:
    print(f"{path:<28}{status:<8}{detail}")

fails = [r for r in results if r[1] == "FAIL"]
print(f"\n{len(results)-len(fails)}/{len(results)} checks passed")
sys.exit(1 if fails else 0)
