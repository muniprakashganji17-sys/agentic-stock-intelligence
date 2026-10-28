"""Data Agent, part 2: macroeconomic series from FRED.

Key idea: every value carries an `available_date`, the first day the public
could have known it. We only use a value on or after that day, so the model
never peeks into the future.

Needs a free FRED API key in the environment variable FRED_API_KEY.
"""
import os

import pandas as pd

import config

# series id -> (frequency, description)
SERIES = {
    "DGS10":    ("daily",   "10-year Treasury yield (%)"),
    "DGS2":     ("daily",   "2-year Treasury yield (%)"),
    "DFF":      ("daily",   "Effective federal funds rate (%)"),
    "VIXCLS":   ("daily",   "VIX volatility index"),
    "CPIAUCSL": ("monthly", "Consumer price index"),
    "UNRATE":   ("monthly", "Unemployment rate (%)"),
}

LOOKBACK_START = "2017-01-01"   # a little before START_DATE so early days have a value


def _client():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set. Add it in Colab Secrets first.")
    from fredapi import Fred
    return Fred(api_key=key)


def _download_daily(fred, sid):
    """Daily market series: value for day D is published the next business day."""
    s = fred.get_series(sid, observation_start=LOOKBACK_START,
                        observation_end=config.END_DATE).dropna()
    df = s.rename("value").rename_axis("date").reset_index()
    df["available_date"] = df["date"] + pd.offsets.BDay(1)
    return df[["date", "available_date", "value"]]


def _download_monthly(fred, sid):
    """Monthly series: use the FIRST-RELEASE date, so later revisions can't leak in."""
    rel = fred.get_series_all_releases(sid)          # columns: realtime_start, date, value
    rel["date"] = pd.to_datetime(rel["date"])
    rel["realtime_start"] = pd.to_datetime(rel["realtime_start"])
    rel["value"] = pd.to_numeric(rel["value"], errors="coerce")
    rel = rel.dropna(subset=["value"])
    first = (rel.sort_values(["date", "realtime_start"])
                .groupby("date", as_index=False).first())
    first = first[first["date"] >= LOOKBACK_START]
    first = first.rename(columns={"realtime_start": "available_date"})
    return first[["date", "available_date", "value"]]


def get_series(sid, force=False):
    """One FRED series as (date, available_date, value), cached in data/raw."""
    os.makedirs(config.RAW_DIR, exist_ok=True)
    path = os.path.join(config.RAW_DIR, f"fred_{sid}.csv")
    if os.path.exists(path) and not force:
        return pd.read_csv(path, parse_dates=["date", "available_date"])

    freq = SERIES[sid][0]
    fred = _client()
    df = _download_daily(fred, sid) if freq == "daily" else _download_monthly(fred, sid)
    df.to_csv(path, index=False)
    print(f"  {sid}: downloaded {len(df)} observations ({freq})")
    return df


def build_daily(trading_dates, force=False):
    """One row per trading day: the latest value that was publicly known that day."""
    base = pd.DataFrame({"Date": pd.to_datetime(sorted(set(trading_dates)))})
    base["Date"] = base["Date"].astype("datetime64[ns]")
    for sid in SERIES:
        df = get_series(sid, force=force)
        df = df.sort_values(["available_date", "date"])
        df = df.rename(columns={"available_date": "Date", "value": sid})[["Date", sid]]
        df["Date"] = df["Date"].astype("datetime64[ns]")
        base = pd.merge_asof(base, df, on="Date", direction="backward")
    return base


def run(force=False):
    """Build the macro table for every trading day in prices.parquet and save it."""
    prices_path = os.path.join(config.PROCESSED_DIR, "prices.parquet")
    dates = pd.read_parquet(prices_path, columns=["Date"])["Date"].unique()

    macro = build_daily(dates, force=force)

    for sid, (_, desc) in SERIES.items():
        print(f"  {sid:9s} {desc:36s} missing days: {int(macro[sid].isna().sum())}")

    out = os.path.join(config.PROCESSED_DIR, "macro.parquet")
    macro.to_parquet(out, index=False)
    print(f"Saved {len(macro)} rows to {out}")
    return macro