"""Data Agent, part 4: company-focused English news from GDELT (GKG table in BigQuery).

Why BigQuery: the GDELT web API blocks Colab (HTTP 429) and may not reach back to 2018.
Google hosts a free public copy of the same data (read-only, ~1 TB/month free queries).

What we keep: English articles that mention one of our companies at least MIN_MENTIONS
times. Each article is assigned to the company it mentions MOST (ties are dropped), so
"fund X bought shares of Y" lists are not counted for every company in them.
We keep at most TOP_PER_DAY articles per company per day.

Look-ahead protection: GDELT `DATE` is when GDELT first saw the article (UTC), which is
never earlier than publication. `effective_date` is the first trading day on which a
prediction made at the close may use the article: after 4 pm New York time the article
counts from the next day, and weekends roll forward to Monday (same rule as filings).

Setup: the Google Cloud project id goes in env var ASI_BQ_PROJECT (not in the repo).
In Colab run `auth.authenticate_user()` once first.
"""
import os
import re

import pandas as pd

import config

TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
MIN_MENTIONS = 3
TOP_PER_DAY = 15
MAX_TOTAL_GB = 800          # refuse to run if the 8-year scan would exceed this

# Names as GDELT writes them (lower-case compared). Observed: "Exxonmobil", "Jpmorgan Chase Co".
ALIASES = {
    "AAPL": ["apple", "apple inc"],
    "MSFT": ["microsoft", "microsoft corp", "microsoft corporation"],
    "NVDA": ["nvidia", "nvidia corp", "nvidia corporation"],
    "JPM":  ["jpmorgan", "jpmorgan chase", "jpmorgan chase co", "jpmorgan chase & co", "jp morgan"],
    "XOM":  ["exxon", "exxonmobil", "exxon mobil", "exxon mobil corp", "exxon mobil corporation"],
}


def _sql(first_day, last_day):
    """One query for a date range (BigQuery bills by columns scanned, not by filters)."""
    tickers = [t for t in config.TICKERS if t in ALIASES]
    counts = ",\n    ".join(
        f"ARRAY_LENGTH(REGEXP_EXTRACT_ALL(LOWER(V2Organizations), "
        f"r'(?:^|;)(?:{'|'.join(ALIASES[t])}),[0-9]+')) AS m_{t}" for t in tickers)
    best = ", ".join(f"m_{t}" for t in tickers)
    return f"""
SELECT * FROM (
  SELECT DATE, SourceCommonName AS source, DocumentIdentifier AS url, V2Tone AS tone,
    {counts}
  FROM `{TABLE}`
  WHERE DATE(_PARTITIONTIME) BETWEEN '{first_day}' AND '{last_day}'
    AND TranslationInfo IS NULL
)
WHERE GREATEST({best}) >= {MIN_MENTIONS}"""


def _slug_to_headline(url):
    """Readable headline from the URL: .../exxon-claims-climate-hypocrisy/683977620 -> words."""
    path = re.sub(r"[?#].*$", "", str(url)).rstrip("/")
    parts = [p for p in path.split("/")[3:] if p]          # drop 'http:', '', domain
    for seg in reversed(parts):                              # last segment that looks like words
        seg = re.sub(r"\.(html?|aspx?|php|cms|ece)$", "", seg, flags=re.I)
        words = [w for w in re.split(r"[-_+]", seg) if w]
        words = [w for w in words if not re.fullmatch(r"[0-9a-f]{6,}|[0-9]+", w, flags=re.I)]
        if len(words) >= 4:
            return " ".join(words).lower()
    return ""


def process(raw):
    """Turn the raw BigQuery rows into the clean news table (pure pandas, testable)."""
    tickers = [t for t in config.TICKERS if f"m_{t}" in raw.columns]
    m = raw[[f"m_{t}" for t in tickers]].to_numpy()
    top = m.max(axis=1)
    n_top = (m == top[:, None]).sum(axis=1)
    keep = (top >= MIN_MENTIONS) & (n_top == 1)             # one clear winner only
    df = raw[keep].copy()
    df["ticker"] = [tickers[i] for i in m[keep].argmax(axis=1)]
    df["mentions"] = top[keep]

    seen = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d%H%M%S", utc=True)
    eastern = seen.dt.tz_convert("America/New_York")
    day = eastern.dt.tz_localize(None).dt.normalize()
    eff = day.where(eastern.dt.hour < 16, day + pd.Timedelta(days=1))
    df["seen_utc"] = seen.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["effective_date"] = eff + pd.offsets.BDay(0)         # weekend -> Monday
    df["tone"] = pd.to_numeric(df["tone"].astype(str).str.split(",").str[0], errors="coerce")
    df["headline"] = df["url"].map(_slug_to_headline)

    df["_key"] = df["headline"].str[:60].where(df["headline"] != "", df["url"])
    df = df.sort_values("mentions", ascending=False)
    df = df.drop_duplicates(["ticker", "effective_date", "_key"])
    df = df.groupby(["ticker", "effective_date"], group_keys=False).head(TOP_PER_DAY)

    lo, hi = pd.Timestamp(config.START_DATE), pd.Timestamp(config.END_DATE)
    df = df[(df["effective_date"] >= lo) & (df["effective_date"] <= hi)]
    cols = ["ticker", "effective_date", "seen_utc", "source", "headline", "url", "tone", "mentions"]
    return df[cols].sort_values(["ticker", "effective_date", "mentions"],
                                ascending=[True, True, False]).reset_index(drop=True)


def _year_ranges():
    y0, y1 = int(config.START_DATE[:4]), int(config.END_DATE[:4])
    return [(y, f"{y}-01-01", f"{y}-12-31") for y in range(y0, y1 + 1)]


def run(dry_run=False, force=False):
    """dry_run=True only prints how much data each year would scan (free, nothing downloaded)."""
    from google.cloud import bigquery
    project = os.environ.get("ASI_BQ_PROJECT")
    if not project:
        raise RuntimeError("Set os.environ['ASI_BQ_PROJECT'] to your Google Cloud project id.")
    client = bigquery.Client(project=project)
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    plan, total = [], 0.0
    for year, a, b in _year_ranges():
        sql = _sql(a, b)
        job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
        gb = job.total_bytes_processed / 1e9
        cached = os.path.exists(os.path.join(config.RAW_DIR, f"gdelt_{year}.parquet")) and not force
        plan.append((year, sql, gb, cached))
        total += 0 if cached else gb
        print(f"  {year}: scans {gb:6.1f} GB {'(cached, skipped)' if cached else ''}")
    print(f"  total to scan now: {total:.1f} GB  (free allowance 1000 GB per month)")
    if dry_run:
        return None
    if total > MAX_TOTAL_GB:
        raise RuntimeError(f"Would scan {total:.0f} GB, above the {MAX_TOTAL_GB} GB safety limit.")

    frames = []
    for year, sql, gb, cached in plan:
        path = os.path.join(config.RAW_DIR, f"gdelt_{year}.parquet")
        if cached:
            raw = pd.read_parquet(path)
        else:
            cfg = bigquery.QueryJobConfig(maximum_bytes_billed=int(gb * 1.1e9) + 200_000_000)
            raw = client.query(sql, job_config=cfg).to_dataframe()
            raw.to_parquet(path, index=False)
            print(f"  {year}: downloaded {len(raw)} candidate articles")
        frames.append(raw)

    news = process(pd.concat([f for f in frames if not f.empty], ignore_index=True))
    out = os.path.join(config.PROCESSED_DIR, "news.parquet")
    news.to_parquet(out, index=False)
    print(news.groupby(["ticker", news.effective_date.dt.year]).size().unstack(fill_value=0))
    print(f"Saved {len(news)} articles to {out}")
    return news