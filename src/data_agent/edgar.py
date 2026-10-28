"""Data Agent, part 3: list SEC EDGAR filings (10-K, 10-Q, 8-K) for our tickers.

This step builds an INDEX of filings (what, when, link). Text is downloaded later.

SEC rules: send a User-Agent with your name and email, and stay under 10
requests per second. Put it in the environment variable SEC_USER_AGENT,
for example:  "Jane Doe jane@example.com"

Look-ahead protection: `effective_date` is the first trading day on which a
prediction made at the close may use the filing. A filing accepted after the
4 pm New York close only counts from the next business day.
"""
import os
import time
import warnings

import pandas as pd
import requests

import config

FORMS = {"10-K", "10-Q", "8-K"}
PAUSE = 0.2   # seconds between requests (SEC limit is 10 per second)

# Exxon moved to a new holding company in 2026 (new CIK 2115436, filings from
# July 2026 only). All 2018-2025 filings stay under the original Exxon Mobil
# Corp number, so we point XOM there. Mention this in the final report.
CIK_OVERRIDES = {"XOM": 34088}
KEEP = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime",
        "form", "items", "primaryDocument", "primaryDocDescription"]


def _headers():
    ua = os.environ.get("SEC_USER_AGENT", "")
    if "@" not in ua:
        raise RuntimeError('SEC_USER_AGENT is missing. Add it in Colab Secrets, '
                           'e.g. "Your Name your_email@example.com".')
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def _get_json(url):
    time.sleep(PAUSE)
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def ticker_to_cik():
    """Map tickers like AAPL to the SEC's company number (CIK)."""
    data = _get_json("https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def _add_dates(df):
    """Add accepted_utc and effective_date (see module docstring)."""
    accepted = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce")
    eastern = accepted.dt.tz_convert("America/New_York")
    day = eastern.dt.tz_localize(None).dt.normalize()
    after_close = eastern.dt.hour >= 16
    effective = day.where(~after_close, day + pd.offsets.BDay(1))
    # if the timestamp is missing, be conservative: filing date + 1 business day
    fallback = pd.to_datetime(df["filingDate"]) + pd.offsets.BDay(1)
    df["accepted_utc"] = accepted.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["effective_date"] = effective.fillna(fallback)
    return df


def list_filings(ticker, cik):
    """All 10-K / 10-Q / 8-K filings for one company inside our date window."""
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    print(f"  {ticker}: CIK {cik} = {sub.get('name')}")
    for extra in sub["filings"].get("files", []):          # older filings live in extra files
        if extra["filingTo"] >= config.START_DATE and extra["filingFrom"] <= config.END_DATE:
            frames.append(pd.DataFrame(_get_json(f"https://data.sec.gov/submissions/{extra['name']}")))

    frames = [f for f in frames if not f.empty]           # avoids pandas FutureWarning
    if not frames:
        raise ValueError(f"{ticker}: SEC returned no filings for CIK {cik}")
       with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)   # harmless: some SEC files have empty columns
        df = pd.concat(frames, ignore_index=True).reindex(columns=KEEP)
    df = df[df["form"].isin(FORMS)]
    df = df[(df["filingDate"] >= config.START_DATE) & (df["filingDate"] <= config.END_DATE)]
    df = df.drop_duplicates(subset="accessionNumber").copy()

    df = _add_dates(df)
    acc = df["accessionNumber"].str.replace("-", "", regex=False)
    df["url"] = ("https://www.sec.gov/Archives/edgar/data/" + str(cik) + "/"
                 + acc + "/" + df["primaryDocument"].fillna(""))
    df.insert(0, "cik", cik)
    df.insert(0, "ticker", ticker)
    df = df.rename(columns={"filingDate": "filing_date", "reportDate": "report_date",
                            "accessionNumber": "accession", "primaryDocument": "document",
                            "primaryDocDescription": "description"})
    df = df.drop(columns=["acceptanceDateTime"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df.sort_values("filing_date").reset_index(drop=True)


def run(force=False):
    """Build the filings index for all tickers and save one combined file."""
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    cik_map = None
    frames = []
    for ticker in config.TICKERS:
        path = os.path.join(config.RAW_DIR, f"edgar_{ticker}.csv")
        if os.path.exists(path) and not force:
            df = pd.read_csv(path, parse_dates=["filing_date", "effective_date"])
        else:
            if cik_map is None:
                cik_map = ticker_to_cik()
            if ticker not in cik_map:
                raise ValueError(f"No SEC company number found for {ticker}")
            cik = CIK_OVERRIDES.get(ticker, cik_map[ticker])
            df = list_filings(ticker, cik)
            if df.empty:
                raise ValueError(f"{ticker}: 0 filings in {config.START_DATE} to {config.END_DATE}")
            df.to_csv(path, index=False)
            print(f"  {ticker}: downloaded index of {len(df)} filings")
        frames.append(df)

    filings = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    print(filings.groupby(["ticker", "form"]).size().unstack(fill_value=0))
    out = os.path.join(config.PROCESSED_DIR, "filings.parquet")
    filings.to_parquet(out, index=False)
    print(f"Saved {len(filings)} filings to {out}")
    return filings