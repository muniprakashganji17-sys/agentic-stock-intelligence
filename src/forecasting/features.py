"""Phase 2.1: the modelling table - features from prices + macro, and the two targets.

One row = one stock on one trading day t, "standing at the close of day t".

FEATURES use only data up to and including the close of day t (rolling windows look back).
TARGETS look forward from t:
  y_dir : 1 if the close on day t+1 is higher than the close on day t, else 0 (next-day direction)
  y_vol : realised volatility over the NEXT 5 trading days (t+1 ... t+5), annualised
  ret_next : the log return t -> t+1 (kept for the backtest, never a feature)

Split by date (config.TRAIN_END / VAL_END), never random. A row only counts if its whole target
window (t+1 ... t+5) also lies inside the same split, so no training target "sees" validation prices.

Feature groups (for the ablation study):
  PRICE_FEATURES : price/volume history only
  MACRO_FEATURES : rates, VIX, inflation, unemployment (already aligned to what was known on day t)
  (event features from news and filings are added in a later phase)
"""
import os

import numpy as np
import pandas as pd

import config

HORIZON = 5          # trading days in the "next week" volatility target
TRADING_DAYS = 252

PRICE_FEATURES = ["ret_1", "ret_5", "ret_10", "ret_21", "vol_5", "vol_21", "sma_gap_10",
                  "sma_gap_50", "rsi_14", "vol_ratio", "range_5", "dow"]
MACRO_FEATURES = ["dgs10", "dgs2", "dff", "vix", "term_spread", "vix_chg_5", "dgs10_chg_5",
                  "cpi_chg_63", "unrate"]
TARGETS = ["y_dir", "y_vol", "ret_next"]


def _price_block(g):
    """Features and targets for ONE ticker (g sorted by Date)."""
    c = g["Close"]
    lc = np.log(c)
    r = lc.diff()                                              # daily log return

    f = pd.DataFrame({"Ticker": g["Ticker"], "Date": g["Date"]})
    for n in (1, 5, 10, 21):
        f[f"ret_{n}"] = lc.diff(n)
    f["vol_5"] = r.rolling(5).std()
    f["vol_21"] = r.rolling(21).std()
    f["sma_gap_10"] = c / c.rolling(10).mean() - 1
    f["sma_gap_50"] = c / c.rolling(50).mean() - 1
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = (100 - 100 / (1 + up / down)).replace([np.inf, -np.inf], np.nan).fillna(100.0)
    f["vol_ratio"] = np.log(g["Volume"].clip(lower=1) / g["Volume"].clip(lower=1).rolling(20).mean())
    f["range_5"] = ((g["High"] - g["Low"]) / c).rolling(5).mean()
    f["dow"] = g["Date"].dt.dayofweek

    # ---- targets (look FORWARD) ----
    f["ret_next"] = lc.shift(-1) - lc
    f["y_dir"] = (f["ret_next"] > 0).astype(float).where(f["ret_next"].notna())
    f["y_vol"] = np.sqrt((r ** 2).rolling(HORIZON).mean().shift(-HORIZON)) * np.sqrt(TRADING_DAYS)
    f["target_end"] = g["Date"].shift(-HORIZON)                # last day used by the targets
    return f


def _macro_block(macro):
    m = macro.sort_values("Date").reset_index(drop=True)
    out = pd.DataFrame({"Date": m["Date"]})
    out["dgs10"], out["dgs2"], out["dff"], out["vix"] = m["DGS10"], m["DGS2"], m["DFF"], m["VIXCLS"]
    out["term_spread"] = m["DGS10"] - m["DGS2"]
    out["vix_chg_5"] = m["VIXCLS"].diff(5)
    out["dgs10_chg_5"] = m["DGS10"].diff(5)
    out["cpi_chg_63"] = m["CPIAUCSL"].pct_change(63)           # about 3 months
    out["unrate"] = m["UNRATE"]
    return out


def split_of(dates):
    """'train' / 'val' / 'test' by calendar date (NaT -> None)."""
    d = pd.to_datetime(dates)
    out = pd.Series(np.where(d <= pd.Timestamp(config.TRAIN_END), "train",
                    np.where(d <= pd.Timestamp(config.VAL_END), "val", "test")), index=d.index, dtype=object)
    return out.where(d.notna())


def build(prices, macro):
    """Pure function: prices + macro -> modelling table (no files, easy to test)."""
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    macro = macro.copy()
    macro["Date"] = pd.to_datetime(macro["Date"])

    blocks = [_price_block(g.sort_values("Date")) for _, g in prices.groupby("Ticker")]
    df = pd.concat(blocks, ignore_index=True).merge(_macro_block(macro), on="Date", how="left")

    df["split"] = split_of(df["Date"])
    same_split = df["split"] == split_of(df["target_end"])     # whole target window inside the split
    df = df[same_split & df[PRICE_FEATURES + MACRO_FEATURES + TARGETS].notna().all(axis=1)]
    df = df[(df["Date"] >= pd.Timestamp(config.START_DATE)) & (df["Date"] <= pd.Timestamp(config.END_DATE))]
    cols = ["Ticker", "Date", "split"] + PRICE_FEATURES + MACRO_FEATURES + TARGETS + ["target_end"]
    return df[cols].sort_values(["Ticker", "Date"]).reset_index(drop=True)


def run():
    prices = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "prices.parquet"))
    macro = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "macro.parquet"))
    df = build(prices, macro)
    out = os.path.join(config.PROCESSED_DIR, "dataset.parquet")
    df.to_parquet(out, index=False)

    print(df.groupby(["split", "Ticker"]).size().unstack(fill_value=0))
    print("\nShare of 'up' days (y_dir=1) by split:")
    print(df.groupby("split")["y_dir"].mean().round(3).to_string())
    print("\nNext-week volatility (annualised) by split:")
    print(df.groupby("split")["y_vol"].describe()[["mean", "50%", "max"]].round(3).to_string())
    print(f"\nSaved {len(df)} rows x {df.shape[1]} columns to {out}")
    return df