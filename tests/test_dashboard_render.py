"""
Render test for the dashboard: opens the built page in headless Chromium and
checks it actually works - no JavaScript errors, every table/chart populated
from the data, no "NaN"/"undefined" text, no sideways overflow on desktop or a
phone-width screen, and the white/red theme in place.

This exists because earlier dashboard bugs (a metrics-file format change that
silently broke the JS, a stale hard-coded sentence) were invisible to every
other test. See tests/dashboard_harness.py for what is stubbed offline (Chart.js
and webfonts) and why.

Skips, rather than fails, if Playwright or its Chromium isn't available.

Run with:
    python -m tests.test_dashboard_render
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.dashboard_harness import render


def _playwright_usable():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


PLAYWRIGHT_OK = _playwright_usable()


def _build():
    from src.dashboard.build_dashboard import build_dashboard
    path, _ = build_dashboard()
    return path


def test_renders_without_errors_and_fully_populated():
    import pandas as pd
    from config.config import PROCESSED_DATA_DIR

    path = _build()
    r = render(path, width=1280, color_scheme="light")
    assert r["errors"] == [], f"JavaScript errors: {r['errors']}"

    c = r["counts"]
    n_rows = len(pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "player_match_with_video.csv")))
    assert c["stat_cards"] == 5
    assert c["player_rows"] == n_rows, "player table doesn't show every row in the data"
    assert c["lomo_rows"] == 5 and c["model_rows"] == 6
    assert c["loadings_rows"] > 0 and c["video_rows"] > 0
    assert c["charts"] == 2 and c["scree_loaded"]
    assert c["finding_filled"] and c["limit_efa_filled"] and c["limit_leak_filled"]
    assert c["unbound_counts"] == 0, "a data-bound count was left unfilled"
    assert not c["nan_text"], "page text contains NaN or undefined"


def test_no_sideways_overflow_on_desktop_dark_and_phone():
    path = _build()
    for width, scheme in [(1280, "dark"), (390, "light")]:
        r = render(path, width=width, color_scheme=scheme)
        assert r["errors"] == [], f"errors at {width}px/{scheme}: {r['errors']}"
        assert not r["counts"]["h_overflow"], f"horizontal overflow at {width}px/{scheme}"


def test_chart_colours_use_the_red_accent_for_video_variants():
    path = _build()
    r = render(path, width=1280, color_scheme="light")
    mae_chart = r["charts"][0]
    # bars alternate grey (stats only / dummy) and red (+ video): 5 bars, 2 of them red
    reds = [c for c in mae_chart["colors"] if c.lower() == "#d2122e"]
    assert len(reds) == 2, f"expected 2 red '+ video' bars, got colours {mae_chart['colors']}"


def test_theme_is_white_and_red_not_the_old_green():
    path = _build()
    html = open(path, encoding="utf-8").read()
    assert "#D2122E" in html and "--accent" in html
    assert "--pine" not in html and "#2F5233" not in html, "old green palette is still present"


if __name__ == "__main__":
    if not PLAYWRIGHT_OK:
        print("SKIPPED  Playwright/Chromium not available in this environment - "
              "pip install playwright && playwright install chromium to run these tests")
    else:
        tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
        passed, failed = 0, 0
        for t in tests:
            try:
                t()
                print(f"PASS  {t.__name__}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL  {t.__name__}: {e}")
                failed += 1
            except Exception as e:
                print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
