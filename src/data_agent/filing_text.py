"""Data Agent, part 5: the TEXT of 8-K filings (main document + press-release exhibit).

An 8-K is a short "something happened" report. For earnings (item 2.02) the numbers sit in
an attached press release (exhibit 99.1), so for each 8-K we fetch two things:
  - main_text    : the 8-K itself (item headings and a few sentences)
  - exhibit_text : the press-release exhibit, when there is one
10-K and 10-Q sections are a later step.

Look-ahead protection: nothing new here. Every row keeps `effective_date` from the filings
index (the first trading day the text may be used).

SEC rules: User-Agent with name and email (SEC_USER_AGENT), under 10 requests per second
(we use about 5), retry politely when the SEC says "slow down".

Caching: one file per ticker in raw/, saved every few filings, so a Colab disconnect only
costs the last few filings. Rerunning continues where it stopped.
"""
import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config
from src.data_agent.edgar import _headers

PAUSE = 0.2            # seconds between requests
MAX_CHARS = 150_000    # keep at most this much text per document
SAVE_EVERY = 25
MAX_FAILS_IN_A_ROW = 5
EXHIBIT_RE = re.compile(r"(?:^|[^a-z])(?:ex|exhibit)[-_]?0?99", re.I)
SKIP_RE = re.compile(r"^R\d+\.htm$|index|FilingSummary|\.xml$|\.xsd$", re.I)


def _get(url, as_json=False):
    """GET with a polite pause and retries. Returns None for 404."""
    for k in range(4):
        time.sleep(PAUSE)
        resp = requests.get(url, headers=_headers(), timeout=60)
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (k + 1))
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json() if as_json else resp.content.decode("utf-8", errors="replace")
    raise RuntimeError(f"SEC kept refusing: {url}")


def html_to_text(raw):
    """HTML (or plain text) -> clean readable text. Tables become 'a | b | c' lines."""
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # inline-XBRL hidden facts and anything hidden by style
    for tag in soup.find_all(lambda t: t.name and t.name.startswith("ix:") and t.name.endswith("header")):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        tr.replace_with(" | ".join(c for c in cells if c) + "\n")
    text = soup.get_text("\n").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)[:MAX_CHARS]


def pick_exhibit(files, main_doc, items):
    """Choose the press-release file from the filing's file list (name + size)."""
    cands = []
    for f in files:
        name = f.get("name", "")
        if name.lower().endswith((".htm", ".html")) and name != main_doc and not SKIP_RE.search(name):
            try:
                size = int(str(f.get("size") or "0").replace(",", ""))
            except ValueError:
                size = 0
            cands.append((name, size))
    for name, _ in cands:                       # normal case: file name says exhibit 99
        if EXHIBIT_RE.search(name):
            return name
    if "2.02" in str(items) and cands:          # e.g. NVIDIA names it q4fy24pr.htm
        name, size = max(cands, key=lambda c: c[1])
        if size > 5000:
            return name
    return None


def fetch_one(row):
    """Download the text of one 8-K."""
    base = row["url"].rsplit("/", 1)[0]
    main_text = html_to_text(_get(row["url"]))
    idx = _get(base + "/index.json", as_json=True) or {}
    files = idx.get("directory", {}).get("item", [])
    ex = pick_exhibit(files, row["document"], row["items"])
    exhibit_text = html_to_text(_get(f"{base}/{ex}")) if ex else ""
    return {"accession": row["accession"], "exhibit_file": ex or "",
            "main_text": main_text, "exhibit_text": exhibit_text, "error": ""}


def _cache_path(ticker):
    return os.path.join(config.RAW_DIR, f"filing8k_{ticker}.parquet")


def run(tickers=None, limit=None, force=False):
    """Fetch 8-K text for every ticker. limit=N does only N new filings per ticker (for testing)."""
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    filings = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "filings.parquet"))
    eight_k = filings[filings["form"] == "8-K"]

    for ticker in (tickers or config.TICKERS):
        rows = eight_k[eight_k["ticker"] == ticker].sort_values("filing_date")
        path = _cache_path(ticker)
        have = pd.read_parquet(path) if (os.path.exists(path) and not force) else pd.DataFrame()
        if len(have):
            have = have[have["error"] == ""]            # failed filings are retried on rerun
        done = set(have["accession"]) if len(have) else set()
        todo = rows[~rows["accession"].isin(done)]
        if limit:
            todo = todo.head(limit)
        print(f"  {ticker}: {len(rows)} 8-Ks, {len(done)} cached, fetching {len(todo)}")

        new, fails = [], 0
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            try:
                new.append(fetch_one(row))
                fails = 0
            except Exception as e:                                  # keep going, remember why
                new.append({"accession": row["accession"], "exhibit_file": "", "main_text": "",
                            "exhibit_text": "", "error": f"{type(e).__name__}: {e}"[:200]})
                fails += 1
                if fails >= MAX_FAILS_IN_A_ROW:
                    pd.concat([have, pd.DataFrame(new)], ignore_index=True).to_parquet(path, index=False)
                    raise RuntimeError(f"{ticker}: {fails} failures in a row, last: {new[-1]['error']}")
            if i % SAVE_EVERY == 0:
                pd.concat([have, pd.DataFrame(new)], ignore_index=True).to_parquet(path, index=False)
        if new:
            have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
            have.to_parquet(path, index=False)

    # combine all tickers with the index columns (effective_date etc.)
    parts = [pd.read_parquet(_cache_path(t)) for t in (tickers or config.TICKERS) if os.path.exists(_cache_path(t))]
    text = pd.concat(parts, ignore_index=True)
    meta = eight_k[["ticker", "accession", "form", "filing_date", "effective_date", "items", "url"]]
    out = meta.merge(text, on="accession", how="inner").sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    out["text_chars"] = out["main_text"].str.len() + out["exhibit_text"].str.len()
    path = os.path.join(config.PROCESSED_DIR, "filing_text.parquet")
    out.to_parquet(path, index=False)

    g = out.groupby("ticker")
    print(pd.DataFrame({"filings": g.size(),
                        "with_exhibit": g["exhibit_file"].apply(lambda s: (s != "").sum()),
                        "errors": g["error"].apply(lambda s: (s != "").sum()),
                        "median_chars": g["text_chars"].median().astype(int)}))
    print(f"Saved {len(out)} filings to {path}")
    return out