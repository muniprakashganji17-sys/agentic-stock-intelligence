"""Phase 2.4 (step 1): build the LSTM input windows.

Idea in simple words
  An LSTM does not look at one day only. For a row (stock, day t) it gets the last WINDOW trading days
  of price features (days t-20 ... t) as a little table, and later learns to predict the target of day t.

This step only BUILDS and CHECKS the windows (no training yet).
  * A window ends on day t and uses only data up to day t  -> no look-ahead.
  * A window may reach back into an earlier split (e.g. a validation window starts in late 2022).
    That is fine: those are past facts known at day t. The TARGET of every row stays inside its own split.
  * Rows whose window is incomplete (first ~3 months of a stock's history) are dropped and counted.
"""
import os

import numpy as np
import pandas as pd

import config
from src.forecasting import features as F

WINDOW = 21


def make_windows(prices, df, feature_cols=None, window=WINDOW):
    """Pure function. prices + dataset table -> (X, meta).

    X    : float array, shape (rows, window, n_features)
    meta : DataFrame (Ticker, Date, split, y_dir, y_vol, ret_next), same row order as X
    """
    feature_cols = list(feature_cols or F.PRICE_FEATURES)
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    X_parts, meta_parts, dropped = [], [], 0
    for tic, g in prices.groupby("Ticker"):
        pb = F._price_block(g.sort_values("Date")).reset_index(drop=True)   # features for EVERY day
        feats = pb[feature_cols].to_numpy(dtype="float64")
        pos = pd.Series(np.arange(len(pb)), index=pb["Date"].to_numpy())      # date -> row number
        rows = df[df["Ticker"] == tic].sort_values("Date")
        for _, row in rows.iterrows():
            i = pos[row["Date"]]
            if i - window + 1 < 0:
                dropped += 1
                continue
            w = feats[i - window + 1: i + 1]                                  # days t-window+1 ... t
            if np.isnan(w).any():
                dropped += 1
                continue
            X_parts.append(w)
            meta_parts.append(row[["Ticker", "Date", "split", "y_dir", "y_vol", "ret_next"]])
    X = np.stack(X_parts).astype("float32")
    meta = pd.DataFrame(meta_parts).reset_index(drop=True)
    meta["Date"] = pd.to_datetime(meta["Date"])
    print(f"windows built: {len(X)}   dropped (incomplete window): {dropped}")
    return X, meta


def run():
    prices = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "prices.parquet"))
    df = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "dataset.parquet"))
    X, meta = make_windows(prices, df)
    print("X shape (rows, days, features):", X.shape)
    print("rows per split:", meta["split"].value_counts().to_dict())
    print("NaNs in X:", int(np.isnan(X).sum()))
    # the LAST day of every window must equal the dataset row's own features
    last = X[:, -1, :]
    ref = df.set_index(["Ticker", "Date"]).loc[list(zip(meta["Ticker"], meta["Date"])), F.PRICE_FEATURES].to_numpy()
    print("last window day equals dataset features:", bool(np.allclose(last, ref, atol=1e-5)))
    return None