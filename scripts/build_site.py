"""Build a static GitHub Pages site from the generated documents.

Usage: python scripts/build_site.py --out site
"""

import argparse
import re
import shutil
from pathlib import Path

from rie.report import to_html

ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "README.md": ("index.html", "Retention incentive evaluation", "en"),
    "causal/RESULTS.md": ("causal/RESULTS.html", "NSW estimates vs benchmark", "en"),
    "design/pre_analysis_plan.md": ("design/pre_analysis_plan.html", "Pre-analysis plan", "en"),
    "brief/beslutningsoplaeg.md": ("brief/beslutningsoplaeg.html", "Beslutningsoplæg", "da"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    out = ROOT / ap.parse_args().out
    shutil.rmtree(out, ignore_errors=True)
    for src, (dst, title, lang) in PAGES.items():
        md = (ROOT / src).read_text()
        md = re.sub(r"\]\(([^)#:]+)\.md\)", r"](\1.html)", md)
        path = out / dst
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_html(md, title, lang))
    for folder in ("causal", "design"):
        for png in (ROOT / folder).glob("*.png"):
            shutil.copy(png, out / folder / png.name)
    shutil.copy(ROOT / "design" / "simulated_pilot_data.csv", out / "design")
    (out / ".nojekyll").write_text("")
    print(f"site written to {out}")


if __name__ == "__main__":
    main()
