#!/usr/bin/env python3
"""
Build the Devpost upload archive (item 8 on the handoff sheet).

The zip is deliberately NOT committed: an archive of the repository, stored in
the repository, is stale the moment anything else changes. Regenerate it instead
-- it is built from `git ls-files`, so it contains exactly what is published,
plus the demo video, and never anything untracked.

    python scripts/make_submission_zip.py

Writes docs/track3-submission.zip. Devpost's limit is 35 MB.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "track3-submission.zip")
LIMIT = 35_000_000

# The gallery stills are uploaded separately as the image gallery, and the
# report page template is a build input rather than something to review.
SKIP_PREFIX = ("docs/video/gallery/", "report/report_page_template.html")
SKIP_EXACT = {"docs/track3-submission.zip"}


def main():
    tracked = subprocess.run(["git", "ls-files"], cwd=HERE, check=True,
                             capture_output=True, text=True,
                             encoding="utf-8").stdout.split()
    files = [f for f in tracked
             if not f.startswith(SKIP_PREFIX) and f not in SKIP_EXACT
             and os.path.exists(os.path.join(HERE, f))]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(os.path.join(HERE, f), os.path.join("track3", f))

    size = os.path.getsize(OUT)
    print(f"wrote {OUT}")
    print(f"  {size / 1e6:.1f} MB, {len(files)} files")
    if size > LIMIT:
        print(f"  OVER the {LIMIT / 1e6:.0f} MB Devpost limit -- drop the video "
              f"from the archive and link it instead")
        return 1
    print(f"  within the {LIMIT / 1e6:.0f} MB Devpost limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
