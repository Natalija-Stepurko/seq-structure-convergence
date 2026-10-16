"""Open the built page in a headless browser and fail if chart text collides.

Checks every SVG at desktop and phone width: no two labels may overlap, and no label may run past
the chart's edge or into a bar. The Python build cannot see this; only a browser lays the text out.
Readers' fonts differ (the serif labels are Palatino on Windows, wider than the Linux fallback),
so every serif label is measured 12% wider than it renders here. Monospace labels are not padded:
the Linux monospace fallback is already the widest of the stack. Text drawn wholly inside a bar
(the counts in the stitching chart) is deliberate and allowed.

    pip install playwright && playwright install chromium
    python site/check_layout.py            # checks site/build/index.html
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGE = Path(__file__).parent / "build" / "index.html"
WIDTHS = (1280, 390)
PAD = 0.12                  # headroom for wider fonts on readers' machines

FIND = """pad => {
  const out = [];
  document.querySelectorAll('svg').forEach(svg => {
    const where = (svg.closest('[id]') || {}).id || '?';
    const ts = [...svg.querySelectorAll('text')].filter(t => t.textContent.trim());
    const bb = ts.map(t => {
      const r = t.getBoundingClientRect();
      const g = getComputedStyle(t).fontFamily.includes('monospace') ? 0 : r.width * pad;
      const anc = getComputedStyle(t).textAnchor;
      const l = anc === 'end' ? r.left - g : anc === 'middle' ? r.left - g / 2 : r.left;
      return {left: l, right: l + r.width + g, top: r.top, bottom: r.bottom, width: r.width, raw: r};
    });
    // filled bars: not the pale full-width tracks, legend swatches or outlines
    const bars = [...svg.querySelectorAll('rect')].filter(r => {
      const f = (r.getAttribute('fill') || '').toUpperCase();
      return f && f !== 'NONE' && f !== '#EBF0F2' && +r.getAttribute('width') > 12;
    }).map(r => r.getBoundingClientRect());
    const sb = svg.getBoundingClientRect();
    for (let i = 0; i < ts.length; i++) {
      if (bb[i].width === 0) continue;                  // hidden (e.g. an unselected tab)
      if (bb[i].right > sb.right + 1 || bb[i].left < sb.left - 1)
        out.push(`${where}: "${ts[i].textContent}" runs past the chart edge`);
      for (const b of bars) {
        const ox = Math.min(bb[i].right, b.right) - Math.max(bb[i].left, b.left);
        const oy = Math.min(bb[i].bottom, b.bottom) - Math.max(bb[i].top, b.top);
        const r = bb[i].raw, inside = r.left >= b.left && r.right <= b.right &&
                                      r.top >= b.top - 1 && r.bottom <= b.bottom + 1;
        if (ox > 1 && oy > 2 && !inside) { out.push(`${where}: "${ts[i].textContent}" runs into a bar`); break; }
      }
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
            bad += [f"[{w}px] {x}" for x in pg.evaluate(FIND, PAD)]
            pg.close()
        br.close()
    for x in bad:
        print("  FAIL", x)
    print(f"{len(bad)} text collisions at widths {', '.join(map(str, WIDTHS))}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
