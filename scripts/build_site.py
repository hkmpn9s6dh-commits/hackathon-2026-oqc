# -*- coding: utf-8 -*-
"""Build the self-contained report page.

    python scripts/build_site.py

Reads out/results.json + out/predictions_qec.sha256, injects them into site/template.html,
subsets the two Noto SC fonts down to the characters actually used, embeds them as base64
WOFF2, and writes site/index.html.  No external request at view time: the page must render
identically offline and on GitHub Pages.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_SANS = r"C:\Windows\Fonts\NotoSansSC-VF.ttf"
FONT_SERIF = r"C:\Windows\Fonts\NotoSerifSC-VF.ttf"


def used_chars(text: str) -> str:
    chars = set(text)
    keep = {c for c in chars if 0x20 <= ord(c) <= 0x7E or 0x2000 <= ord(c) <= 0x206F
            or 0x3000 <= ord(c) <= 0x303F or 0x4E00 <= ord(c) <= 0x9FFF
            or 0xFF00 <= ord(c) <= 0xFF65 or c in "·—…“”‘’≤≥−×"}
    return "".join(sorted(keep))


def subset_woff2(path: str, chars: str) -> str:
    with open(path, "rb") as fh:
        data = fh.read()
    font = TTFont(io.BytesIO(data))
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.desubroutinize = True
    opts.layout_features = ["*"]
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    subsetter = subset.Subsetter(options=opts)
    subsetter.populate(text=chars)
    subsetter.subset(font)
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    tpl_path = os.path.join(HERE, "site", "template.html")
    with open(tpl_path, "r", encoding="utf-8") as fh:
        tpl = fh.read()

    with open(os.path.join(HERE, "out", "results.json"), "r", encoding="utf-8") as fh:
        results = json.load(fh)
    digest_path = os.path.join(HERE, "out", "predictions_qec.sha256")
    digest = open(digest_path, "r", encoding="utf-8").read().strip()

    payload = {
        "pooled": results["pooled"],
        "per_seed": results["per_seed"],
        "board": results["board"],
        "digest": digest,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    page = tpl.replace("__DATA__", data_json)

    chars = used_chars(page)
    # the serif face only sets headings, so subset it to the heading text (+ digits/punctuation)
    heads = " ".join(re.findall(r"<h[123][^>]*>(.*?)</h[123]>", page, flags=re.S))
    head_chars = used_chars(re.sub(r"<[^>]+>", " ", heads))
    print("unique glyphs to embed: sans %d, serif %d" % (len(chars), len(head_chars)))
    sans = subset_woff2(FONT_SANS, chars)
    serif = subset_woff2(FONT_SERIF, head_chars)
    print("sans woff2 %.0f KB, serif woff2 %.0f KB"
          % (len(sans) * 3 / 4 / 1024, len(serif) * 3 / 4 / 1024))
    page = page.replace("__FONT_SANS__", sans).replace("__FONT_SERIF__", serif)

    out_dir = os.path.join(HERE, "site")
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote %s (%.0f KB)" % (out_path, os.path.getsize(out_path) / 1024))
    # GitHub Pages serves /docs on main, so the built page also lands there
    import shutil
    pages_dir = os.path.join(HERE, "docs")
    shutil.copyfile(out_path, os.path.join(pages_dir, "index.html"))
    print("copied to docs/index.html for GitHub Pages")

    # cheap structural checks: no placeholders left, no external requests
    left = re.findall(r"__[A-Z_]+__", page)
    assert not left, "unreplaced placeholders: %s" % left
    for bad in ("http://", "https://", "//cdn", "src=\"//"):
        assert bad not in page, "external reference found: %s" % bad
    assert page.count("<svg") == 1 and "viewBox" in page
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
