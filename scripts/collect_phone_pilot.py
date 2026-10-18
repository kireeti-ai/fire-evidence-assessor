"""Collect a PILOT set of openly licensed phone photos from Wikimedia Commons for the phone-photo evaluation.

This is not the consented, human-annotated phone-photo study the readiness document requires. It builds a labelled-by-review
pilot from files whose EXIF Make/Model is a phone, in fire, smoke and hard-negative categories, with licence and author kept
for attribution. Labels are NOT taken from categories: they come from the annotation step (docs/validation/phone_pilot/).

Politeness: one request per ~1.2 s, honours Retry-After on 429, identifies itself. Only CC0, public domain, CC BY and CC BY-SA files.
Usage: python scripts/collect_phone_pilot.py --out-dir DIR [--per-category 60]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "TerraWatch-phone-pilot/0.1 (research evaluation; contact kireetiv2005@gmail.com)"
PHONE_MAKES = ("apple", "samsung", "xiaomi", "oneplus", "google", "huawei", "honor", "oppo", "vivo", "realme", "motorola", "nokia", "sony mobile", "lge", "lg electronics", "asus", "redmi", "poco", "tecno", "infinix", "itel", "micromax", "lava", "htc", "zte", "meizu", "fairphone", "nothing")
PHONE_MODEL_HINT = re.compile(r"(iphone|sm-[a-z]\d|galaxy|pixel|redmi|mi \d|poco|oneplus|nord|cph\d|rmx\d|vivo|realme|moto|nokia|xt\d{4}|lm-|honor|huawei|infinix|tecno)", re.I)
OK_LICENSE = re.compile(r"^(cc0|public domain|pd|cc[ -]by(?!-nc)(?!-nd)|cc[ -]by-sa)", re.I)
# category -> coarse expectation (NOT a label): used only to spread the sample across scenes
CATEGORIES = {
    "fire_flames": ["Category:Forest fires", "Category:Wildfires", "Category:Bushfires in Australia", "Category:Grass fires", "Category:Fires in India", "Category:Wildfires in India"],
    "smoke": ["Category:Smoke", "Category:Smoke from forest fires", "Category:Wildfire smoke", "Category:Smoke plumes"],
    "hard_negative_sky": ["Category:Sunsets", "Category:Sunrises", "Category:Red sky", "Category:Clouds", "Category:Orange sky"],
    "hard_negative_haze": ["Category:Fog", "Category:Haze", "Category:Mist", "Category:Smog", "Category:Air pollution in India", "Category:Dust storms"],
    "hard_negative_other": ["Category:Chimneys", "Category:Cooking", "Category:Steam", "Category:Bonfires", "Category:Burnt vegetation", "Category:Dry forests", "Category:Dry grass", "Category:Fireworks", "Category:Tea stalls in India"],
}


def call(params, retries=6):
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                time.sleep(1.2)
                return json.load(r)
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After", "0") or 0) or min(120, 5 * 2 ** attempt)
            print(f"HTTP {e.code}; waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            print(f"{type(e).__name__}; retry", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("API unavailable")


def is_phone(meta: dict) -> bool:
    make, model = (meta.get("Make") or "").strip().lower(), (meta.get("Model") or "").strip()
    return bool(make) and (any(make.startswith(m) for m in PHONE_MAKES) and (PHONE_MODEL_HINT.search(model) or make.startswith(("apple", "oneplus", "xiaomi", "redmi", "realme", "oppo", "vivo", "tecno", "infinix", "itel", "micromax", "lava"))))


def category_files(cat, limit):
    out, cont = [], {}
    while len(out) < limit:
        d = call({"action": "query", "generator": "categorymembers", "gcmtitle": cat, "gcmtype": "file", "gcmlimit": 50, "prop": "imageinfo",
                  "iiprop": "metadata|extmetadata|size|url|mime", "iiurlwidth": 1024, "iiextmetadatafilter": "LicenseShortName|Artist|ImageDescription", **cont})
        for pg in (d.get("query", {}).get("pages", {}) or {}).values():
            ii = (pg.get("imageinfo") or [{}])[0]
            if ii.get("mime") not in ("image/jpeg",):
                continue
            meta = {m["name"]: m["value"] for m in (ii.get("metadata") or []) if isinstance(m.get("value"), (str, int, float))}
            lic = (ii.get("extmetadata", {}).get("LicenseShortName", {}) or {}).get("value", "")
            if not (is_phone(meta) and OK_LICENSE.match(lic or "")):
                continue
            out.append({"title": pg["title"], "category": cat, "make": meta.get("Make"), "model": meta.get("Model"), "license": lic,
                        "author": re.sub(r"<[^>]+>", "", (ii["extmetadata"].get("Artist", {}) or {}).get("value", ""))[:120],
                        "description": re.sub(r"<[^>]+>", "", (ii["extmetadata"].get("ImageDescription", {}) or {}).get("value", ""))[:200],
                        "width": ii.get("width"), "height": ii.get("height"), "thumb_url": ii.get("thumburl") or ii.get("url"), "page_url": ii.get("descriptionurl")})
        if "continue" not in d:
            break
        cont = d["continue"]
    return out[:limit]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--per-category", type=int, default=60, help="max phone files per Commons category")
    ap.add_argument("--max-scan", type=int, default=600, help="max files scanned per category before stopping")
    a = ap.parse_args(argv)
    (a.out_dir / "img").mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for group, cats in CATEGORIES.items():
        for cat in cats:
            try:
                found = category_files(cat, a.per_category)
            except RuntimeError:
                print("skipping", cat, file=sys.stderr)
                continue
            n_new = 0
            for f in found:
                if f["title"] in seen:
                    continue
                seen.add(f["title"])
                f["group"] = group
                rows.append(f)
                n_new += 1
            print(f"{group:22s} {cat:42s} phone+licence files: {len(found):3d} new: {n_new}", flush=True)
    with open(a.out_dir / "candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["title"])
        w.writeheader()
        w.writerows(rows)
    ok = 0
    for i, r in enumerate(rows):
        target = a.out_dir / "img" / f"p{i:04d}.jpg"
        r["file"] = target.name
        if target.exists():
            ok += 1
            continue
        try:
            with urllib.request.urlopen(urllib.request.Request(r["thumb_url"], headers={"User-Agent": UA}), timeout=90) as resp:
                target.write_bytes(resp.read())
            ok += 1
            time.sleep(0.6)
        except urllib.error.HTTPError as e:
            print("download", e.code, r["title"], file=sys.stderr)
            time.sleep(10 if e.code == 429 else 1)
        except Exception as e:  # noqa: BLE001
            print("download", type(e).__name__, r["title"], file=sys.stderr)
    with open(a.out_dir / "candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"candidates {len(rows)}, downloaded {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
