"""Collect manga/manhwa/manhua metadata from the official MangaDex API.

For each original language, titles are fetched in order of follower count
(most popular first), then their ratings/follows are fetched in batches from
the statistics endpoint. Output is a JSON list of records that
preprocess_mangadex.py reads directly.

Usage:
    python data_processing/scrape_mangadex.py --out data_processing/mangadex_data.json
    python data_processing/scrape_mangadex.py --languages ko --per-language 3000

API docs: https://api.mangadex.org/docs/
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

API = "https://api.mangadex.org"
COVER_BASE = "https://uploads.mangadex.org/covers"
USER_AGENT = "mc-suggests-scraper/2.0 (+https://github.com/michaelnkema1/mc-suggests)"

PAGE_SIZE = 100          # max allowed by /manga
STATS_BATCH = 100        # ids per /statistics/manga request
MAX_WINDOW = 10_000      # MangaDex rejects offset + limit > 10000
REQUEST_INTERVAL = 0.25  # MangaDex allows ~5 req/s per IP; stay under it

TITLE_LANG_PREFERENCE = ["en", "ko-ro", "ja-ro", "zh-ro", "ko", "ja", "zh"]


class MangaDexClient:
    def __init__(self, session: Optional[requests.Session] = None, interval: float = REQUEST_INTERVAL):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.interval = interval
        self._last = 0.0

    def get(self, path: str, params: Any = None, retries: int = 5) -> Dict[str, Any]:
        for attempt in range(retries):
            wait = self.interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(f"{API}{path}", params=params, timeout=30)
            except requests.RequestException as e:
                backoff = 2 ** attempt
                print(f"  network error ({e}); retrying in {backoff}s", file=sys.stderr)
                time.sleep(backoff)
                continue
            if resp.status_code == 429:
                retry_at = resp.headers.get("X-RateLimit-Retry-After")
                backoff = max(1.0, float(retry_at) - time.time()) if retry_at else 2 ** attempt
                print(f"  rate limited; sleeping {backoff:.0f}s", file=sys.stderr)
                time.sleep(backoff)
                continue
            if resp.status_code >= 500:
                backoff = 2 ** attempt
                print(f"  server error {resp.status_code}; retrying in {backoff}s", file=sys.stderr)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Giving up on {path} after {retries} attempts")


def pick_localized(values: Any, prefer: Iterable[str] = TITLE_LANG_PREFERENCE, fallback: bool = True) -> Optional[str]:
    """Pick a string from a {lang: text} map (or list of such maps).

    Languages in `prefer` are tried in order; with `fallback`, any other language is accepted.
    """
    if isinstance(values, list):
        merged: Dict[str, str] = {}
        for v in values:
            if isinstance(v, dict):
                for k, s in v.items():
                    merged.setdefault(k, s)
        values = merged
    if not isinstance(values, dict) or not values:
        return None
    for lang in prefer:
        s = values.get(lang)
        if isinstance(s, str) and s.strip():
            return s
    if not fallback:
        return None
    for s in values.values():
        if isinstance(s, str) and s.strip():
            return s
    return None


def parse_manga(item: Dict[str, Any]) -> Dict[str, Any]:
    attrs = item.get("attributes", {})
    manga_id = item["id"]

    title = (
        pick_localized(attrs.get("title"), ["en"], fallback=False)
        or pick_localized(attrs.get("altTitles"), ["en"], fallback=False)
        or pick_localized(attrs.get("title"))
    )
    description = pick_localized(attrs.get("description"), ["en"])

    tags = sorted({
        name
        for t in attrs.get("tags", [])
        if (name := pick_localized(t.get("attributes", {}).get("name"), ["en"]))
    })

    cover_url = None
    for rel in item.get("relationships", []):
        if rel.get("type") == "cover_art":
            file_name = (rel.get("attributes") or {}).get("fileName")
            if file_name:
                cover_url = f"{COVER_BASE}/{manga_id}/{file_name}.256.jpg"
            break

    return {
        "id": manga_id,
        "title": title,
        "description": description,
        "tags": tags,
        "demographic": attrs.get("publicationDemographic"),
        "status": attrs.get("status"),
        "content_rating": attrs.get("contentRating"),
        "year": attrs.get("year"),
        "original_language": attrs.get("originalLanguage"),
        "last_chapter": attrs.get("lastChapter") or None,
        "cover_url": cover_url,
        "created_at": attrs.get("createdAt"),
        "updated_at": attrs.get("updatedAt"),
    }


def fetch_language(client: MangaDexClient, lang: str, limit: int, content_ratings: List[str]) -> List[Dict[str, Any]]:
    limit = min(limit, MAX_WINDOW)
    out: List[Dict[str, Any]] = []
    offset = 0
    while offset < limit:
        page = min(PAGE_SIZE, limit - offset)
        params = [
            ("limit", page),
            ("offset", offset),
            ("originalLanguage[]", lang),
            ("order[followedCount]", "desc"),
            ("includes[]", "cover_art"),
        ] + [("contentRating[]", cr) for cr in content_ratings]
        data = client.get("/manga", params=params)
        items = data.get("data", [])
        out.extend(parse_manga(it) for it in items)
        total = data.get("total", 0)
        offset += len(items)
        print(f"  [{lang}] {offset}/{min(limit, total)}", file=sys.stderr)
        if not items or offset >= total:
            break
    return out


def fetch_statistics(client: MangaDexClient, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(ids), STATS_BATCH):
        batch = ids[start:start + STATS_BATCH]
        data = client.get("/statistics/manga", params=[("manga[]", i) for i in batch])
        for manga_id, s in (data.get("statistics") or {}).items():
            rating = s.get("rating") or {}
            stats[manga_id] = {
                "rating": rating.get("bayesian") if rating.get("bayesian") is not None else rating.get("average"),
                "follows": s.get("follows"),
            }
        print(f"  [stats] {min(start + STATS_BATCH, len(ids))}/{len(ids)}", file=sys.stderr)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Scrape MangaDex titles via the official API")
    parser.add_argument("--out", default="data_processing/mangadex_data.json")
    parser.add_argument("--languages", nargs="+", default=["ko", "ja", "zh"],
                        help="Original languages: ko=manhwa, ja=manga, zh=manhua")
    parser.add_argument("--per-language", type=int, default=6000,
                        help=f"Most-followed titles to fetch per language (max {MAX_WINDOW})")
    parser.add_argument("--content-ratings", nargs="+", default=["safe", "suggestive", "erotica"])
    args = parser.parse_args()

    client = MangaDexClient()
    records: Dict[str, Dict[str, Any]] = {}
    for lang in args.languages:
        print(f"Fetching most-followed titles for originalLanguage={lang}", file=sys.stderr)
        for rec in fetch_language(client, lang, args.per_language, args.content_ratings):
            records.setdefault(rec["id"], rec)

    print(f"Fetching statistics for {len(records)} titles", file=sys.stderr)
    stats = fetch_statistics(client, list(records))
    for manga_id, rec in records.items():
        rec.update(stats.get(manga_id, {"rating": None, "follows": None}))

    rows = sorted(records.values(), key=lambda r: r.get("follows") or 0, reverse=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(json.dumps({"rows": len(rows), "out": args.out}))


if __name__ == "__main__":
    main()
