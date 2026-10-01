"""Open the built page in a headless browser and fail if chart text collides.

Checks every SVG at desktop and phone width: no two labels may overlap, and no label may run past
the chart's edge. The Python build cannot see this; only a browser lays the text out.

    pip install playwright && playwright install chromium
    python site/check_layout.py            # checks site/build/index.html
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGE = Path(__file__).parent / "build" / "index.html"
WIDTHS = (1280, 390)

FIND = """() => {
  const out = [];
  document.querySelectorAll('svg').forEach(svg => {
    const where = (svg.closest('[id]') || {}).id || '?';
    const ts = [...svg.querySelectorAll('text')].filter(t => t.textContent.trim());
    const bb = ts.map(t => t.getBoundingClientRect());
    const sb = svg.getBoundingClientRect();
    for (let i = 0; i < ts.length; i++) {
      if (bb[i].width === 0) continue;                  // hidden (e.g. an unselected tab)
      if (bb[i].right > sb.right + 1 || bb[i].left < sb.left - 1)
        out.push(`${where}: "${ts[i].textContent}" runs past the chart edge`);
      for (let j = i + 1; j < ts.length; j++) {
        const a = bb[i], b = bb[j];
        const ox = Math.min(a.right, b.right) - Math.max(a.left, b.left);
        const oy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
        if (ox > 1 && oy > 2)
          out.push(`${where}: "${ts[i].textContent}" overlaps "${ts[j].textContent}"`);
      }
    }
  });
  return out;
}"""


def main():
    bad = []
    with sync_playwright() as p:
        br = p.chromium.launch()
        for w in WIDTHS:
            pg = br.new_page(viewport={"width": w, "height": 900})
            pg.goto(PAGE.resolve().as_uri())
            pg.wait_for_load_state("load")
            bad += [f"[{w}px] {x}" for x in pg.evaluate(FIND)]
            pg.close()
        br.close()
    for x in bad:
        print("  FAIL", x)
    print(f"{len(bad)} text collisions at widths {', '.join(map(str, WIDTHS))}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
