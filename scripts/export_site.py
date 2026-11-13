#!/usr/bin/env python3
"""Export the project page for the personal site (https://ozzey.github.io/papers/eddies/).

    python scripts/export_site.py /path/to/ozzey.github.io/papers/eddies [--video-base URL]

Copies docs/index.html, styles.css, script.js, favicon.svg, paper.pdf and supplementary.pdf, plus the
small assets the page uses (architecture SVGs, result figures, posters, the certified-stop frames) into
the target folder, rewriting the page's ../assets/ paths to assets/. The mp4 clips are not copied: the
page streams them from the EDDIES repository on GitHub (--video-base, default the raw URL of main), which
keeps the site repository small. Everything else on the page stays relative.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
VIDEO_BASE = "https://raw.githubusercontent.com/Ozzey/EDDIES/main/assets/videos/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="folder to write, e.g. ozzey.github.io/papers/eddies")
    ap.add_argument("--video-base", default=VIDEO_BASE, help="URL prefix for assets/videos/*.mp4")
    ap.add_argument("--copy-videos", action="store_true", help="copy the mp4s instead of streaming them")
    a = ap.parse_args()
    out = Path(a.target)
    out.mkdir(parents=True, exist_ok=True)

    html = (DOCS / "index.html").read_text()
    refs = sorted(set(re.findall(r'(?:src|href|poster)="\.\./(assets/[^"#?]+)"', html)))
    copied, streamed = [], []
    for rel in refs:
        src = ROOT / rel
        if not src.exists():
            print("missing:", rel)
            continue
        if rel.endswith(".mp4") and not a.copy_videos:
            html = html.replace(f'"../{rel}"', f'"{a.video_base}{rel[len("assets/videos/"):]}"')
            streamed.append(rel)
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    html = html.replace('"../assets/', '"assets/')
    (out / "index.html").write_text(html)
    for name in ("styles.css", "script.js", "favicon.svg", "paper.pdf", "supplementary.pdf"):
        if (DOCS / name).exists():
            shutil.copy2(DOCS / name, out / name)
    (out / ".nojekyll").touch()
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {out}: {len(copied)} assets copied, {len(streamed)} videos streamed from {a.video_base}, {total:.1f} MB")


if __name__ == "__main__":
    main()
