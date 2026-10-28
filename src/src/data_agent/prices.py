"""Data Agent, part 1: download daily stock prices and cache them.

Run from the project root, e.g.:
    from src.data_agent import prices
    df = prices.run()
"""
import io
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

import config

COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _from_stooq(ticker):
    """Download one ticker from Stooq as a DataFrame."""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    text = resp.text
    if not text.startswith("Date"):
        raise ValueError(f"Stooq gave no CSV for {ticker}: {text[:80]!r}")
    return pd.read_csv(io.StringIO(text), parse_dates=["Date"])


def _from_yfinance(ticker):
    """Backup source: Yahoo Finance via the yfinance library."""
    import yfinance as yf

    end = (datetime.strptime(config.END_DATE, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=config.START_DATE, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"yfinance gave no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df[COLUMNS]


def _clean(df):
    """Keep our columns, sort by date, drop duplicates and bad rows, trim dates."""
    df = df[COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    for col in COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df.drop_duplicates(subset="Date").sort_values("Date")
    mask = (df["Date"] >= config.START_DATE) & (df["Date"] <= config.END_DATE)
    return df.loc[mask].reset_index(drop=True)


def validate(df, ticker):
    """Print simple health checks so problems are visible early."""
    gaps = df["Date"].diff().dt.days.dropna()
    print(f"  {ticker}: {len(df)} rows | {df['Date'].min().date()} to {df['Date'].max().date()}"
          f" | longest gap {int(gaps.max())} days | bad prices: {int((df['Close'] <= 0).sum())}")


def get_prices(ticker, force=False):
    """Return clean prices for one ticker, using the cache in data/raw if present."""
    os.makedirs(config.RAW_DIR, exist_ok=True)
    path = os.path.join(config.RAW_DIR, f"prices_{ticker}.csv")
    if os.path.exists(path) and not force:
        return pd.read_csv(path, parse_dates=["Date"])

    try:
        df, source = _from_stooq(ticker), "stooq"
    except Exception as err:
        print(f"  {ticker}: Stooq failed ({err}); trying yfinance")
        df, source = _from_yfinance(ticker), "yfinance"

    df = _clean(df)
    df.to_csv(path, index=False)
    print(f"  {ticker}: downloaded from {source}")
    return df


def run(force=False):
    """Download all tickers, validate, and save one combined processed file."""
    frames = []
    for ticker in config.TICKERS:
        df = get_prices(ticker, force=force)
        validate(df, ticker)
        df.insert(0, "Ticker", ticker)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out = os.path.join(config.PROCESSED_DIR, "prices.parquet")
    combined.to_parquet(out, index=False)
    print(f"Saved {len(combined)} rows to {out}")
    return combined