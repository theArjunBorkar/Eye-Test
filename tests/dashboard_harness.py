"""
Offline render harness for the dashboard (used by tests/test_dashboard_render.py
and for taking screenshots while iterating on the design).

Needs: pip install playwright && playwright install chromium
If that's unavailable, callers should skip rather than fail.

WHAT IS STUBBED, AND WHY: the page loads Chart.js and Google Fonts from CDNs,
which aren't reachable in an offline sandbox. Chart.js is replaced with a tiny
stand-in that records each chart's config and draws plain bars on the canvas;
fonts are blocked (so the browser falls back to system fonts). That's enough to
verify the page's own code - data bindings, table rendering, layout, colours,
runtime errors - but charts in screenshots are APPROXIMATIONS, not real
Chart.js output, and webfonts are not shown.
"""

import json

CHART_STUB = r"""
window.__charts = [];
class Chart {
  constructor(canvas, cfg) {
    this.cfg = cfg; window.__charts.push(cfg);
    const horizontal = cfg.options && cfg.options.indexAxis === 'y';
    const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight || (horizontal ? 300 : 260);
    canvas.width = w; canvas.height = h; canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    const ds = cfg.data.datasets[0], labels = cfg.data.labels, data = ds.data;
    const max = Math.max(...data) * 1.1, n = data.length;
    const title = cfg.options && cfg.options.plugins && cfg.options.plugins.title && cfg.options.plugins.title.text;
    ctx.font = '12px sans-serif';
    const textColor = (Chart.defaults && Chart.defaults.color) || '#666';
    ctx.fillStyle = textColor;
    if (title) ctx.fillText(title, 8, 16);
    for (let i = 0; i < n; i++) {
      const col = Array.isArray(ds.backgroundColor) ? ds.backgroundColor[i] : ds.backgroundColor;
      ctx.fillStyle = col || '#999';
      if (horizontal) {
        const bh = (h - 30) / n, bw = (data[i] / max) * (w * 0.55);
        ctx.fillRect(w * 0.4, 24 + i * bh + 3, bw, bh - 6);
        ctx.fillStyle = textColor; ctx.fillText(String(labels[i]).slice(0, 38), 6, 24 + i * bh + bh / 2 + 4);
      } else {
        const bw = (w - 20) / n, bh = (data[i] / max) * (h - 60);
        ctx.fillRect(10 + i * bw + 8, h - 28 - bh, bw - 16, bh);
        ctx.fillStyle = textColor; ctx.fillText(String(labels[i]).slice(0, 14), 10 + i * bw + 8, h - 10);
      }
    }
  }
}
Chart.defaults = { font: {}, plugins: {} };
"""


def render(html_path, width=1280, height=900, color_scheme="light", shot_path=None,
           full_page=True, timeout_ms=15000):
    """
    Opens the dashboard offline, returns {"errors": [...], "counts": {...}, "charts": [...]}.
    Optionally writes a screenshot.
    """
    from playwright.sync_api import sync_playwright

    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": width, "height": height},
                                      color_scheme=color_scheme)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        # "Failed to load resource" is just the webfont / CDN requests this harness
        # deliberately blocks to stay offline - not a page bug, so it isn't reported.
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" and "Failed to load resource" not in m.text else None)

        def handle(route):
            url = route.request.url
            if "cdnjs.cloudflare.com" in url and "Chart" in url:
                route.fulfill(status=200, content_type="application/javascript", body=CHART_STUB)
            elif url.startswith("file://") or url.startswith("data:"):
                route.continue_()
            else:
                route.abort()  # fonts and anything else external: offline

        page.route("**/*", handle)
        page.goto("file://" + str(html_path), wait_until="load", timeout=timeout_ms)
        page.wait_for_timeout(300)

        counts = page.evaluate("""() => ({
            stat_cards: document.querySelectorAll('#stat-strip .stat').length,
            player_rows: document.querySelectorAll('#player-table tbody tr').length,
            model_rows: document.querySelectorAll('#model-metrics-table tbody tr').length,
            lomo_rows: document.querySelectorAll('#lomo-table tbody tr').length,
            loadings_rows: document.querySelectorAll('#loadings-table tbody tr').length,
            video_rows: document.querySelectorAll('#video-table tbody tr').length,
            finding_filled: document.getElementById('video-finding').innerText.length > 80,
            team_split_present: !!document.getElementById('team-split'),
            limit_efa_filled: document.getElementById('limit-efa').innerText.length > 40,
            limit_leak_filled: document.getElementById('limit-leak').innerText.length > 40,
            unbound_counts: Array.from(document.querySelectorAll('[data-bind]')).filter(e => !/^[0-9]+$/.test(e.textContent.trim())).length,
            charts: (window.__charts || []).length,
            scree_loaded: !!(document.getElementById('scree-img') && document.getElementById('scree-img').naturalWidth > 0),
            nan_text: /NaN|undefined/.test(document.body.innerText),
            h_overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
        })""")
        charts = page.evaluate("() => (window.__charts || []).map(c => ({type: c.type, "
                               "colors: c.data.datasets[0].backgroundColor}))")
        if shot_path:
            page.screenshot(path=str(shot_path), full_page=full_page)
        browser.close()
    return {"errors": errors, "counts": counts, "charts": charts}


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    scheme = sys.argv[3] if len(sys.argv) > 3 else "light"
    width = int(sys.argv[4]) if len(sys.argv) > 4 else 1280
    print(json.dumps(render(path, width=width, color_scheme=scheme, shot_path=out), indent=2, default=str))
