"""Interactive browser E2E: pick a bundled dataset, run naive strategy with live providers."""
import sys
from playwright.sync_api import sync_playwright

BASE = "http://localhost:4000"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    console_errors = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: console_errors.append(f"pageerror: {exc}"))

    page.goto(BASE + "/playground", wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(500)

    # Pick the bundled "capitals" dataset
    page.select_option("select:below(:text('Or pick a bundled/benchmark dataset'))", label=lambda v: True) if False else None
    selects = page.locator("select")
    # first select after the "choose a dataset" label is the dataset picker (2nd select overall: model is 1st? use text match instead)
    dataset_select = page.locator("select").filter(has=page.locator("option", has_text="capitals"))
    dataset_select.select_option(label=[o for o in dataset_select.locator("option").all_inner_texts() if o.startswith("capitals")][0])
    page.wait_for_timeout(1500)

    corpus_ok = "capitals" in page.inner_text("body")
    print("dataset loaded into playground:", corpus_ok)

    # pick model dropdowns that have groq/gemini options if present, else leave defaults
    def try_select(label_substr, value_substr):
        try:
            sel = page.locator("select").filter(has=page.locator(f"option:has-text('{value_substr}')")).first
            opt = [o for o in sel.locator("option").all_inner_texts() if value_substr in o][0]
            sel.select_option(label=opt)
            return True
        except Exception as e:
            return False

    try_select("model", "groq/openai/gpt-oss-20b")
    try_select("embedding", "google/gemini-embedding-001")

    page.screenshot(path="scratch_playground_before_run.png", full_page=True)

    run_btn = page.locator("button", has_text="Run evaluation")
    run_btn.click()

    # wait for either results table or an error message, up to 30s
    try:
        page.wait_for_selector("text=Results", timeout=30000)
        got_results = True
    except Exception:
        got_results = False

    page.wait_for_timeout(1000)
    page.screenshot(path="scratch_playground_after_run.png", full_page=True)
    body = page.inner_text("body")

    print("got results section:", got_results)
    print("console errors during run:", console_errors[:5])
    print("body snippet around status:", [l for l in body.splitlines() if l.strip()][-15:])

    browser.close()
    sys.exit(0 if got_results and not console_errors else 1)
