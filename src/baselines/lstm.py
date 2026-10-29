"""Phase 2.4: LSTM baseline (step 1: input windows, step 2: training and scoring).

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
from sklearn.metrics import roc_auc_score

import config
from src.baselines.price_ml import direction_scores, volatility_scores
from src.forecasting import features as F

WINDOW = 21


def make_windows(prices, df, feature_cols=None, window=WINDOW):
    """Pure function. prices + dataset table -> (X, meta).

    X    : float array, shape (rows, window, n_features)
    meta : DataFrame (Ticker, Date, split, y_dir, y_vol, ret_next, target_end), same row order as X
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
            meta_parts.append(row[["Ticker", "Date", "split", "y_dir", "y_vol", "ret_next", "target_end"]])
    X = np.stack(X_parts).astype("float32")
    meta = pd.DataFrame(meta_parts).reset_index(drop=True)
    meta["Date"] = pd.to_datetime(meta["Date"])
    meta["target_end"] = pd.to_datetime(meta["target_end"])
    print(f"windows built: {len(X)}   dropped (incomplete window): {dropped}")
    return X, meta


# ---------------------------------------------------------------------------------------------
# Step 2: training and scoring
#   * fit rows   : train rows whose whole target window ends before ES_START (2022-01-01)
#   * stop rows  : train rows dated from ES_START on -> only used to decide WHEN TO STOP training
#   * eval rows  : validation rows (never used for fitting or stopping) -> the reported score
# ---------------------------------------------------------------------------------------------
ES_START = pd.Timestamp("2022-01-01")
HIDDEN, DROPOUT, LR, BATCH, MAX_EPOCHS, PATIENCE = 32, 0.2, 1e-3, 128, 60, 8


def split_rows(meta, eval_split="val", confirm_test=False):
    """Boolean masks (fit, stop, eval) over the rows of `meta`."""
    if eval_split == "test" and not confirm_test:
        raise RuntimeError("The test years are locked until the very end. Use eval_split='val'.")
    is_train = (meta["split"] == "train").to_numpy()
    fit = is_train & (meta["target_end"] < ES_START).to_numpy()
    stop = is_train & (meta["Date"] >= ES_START).to_numpy()
    ev = (meta["split"] == eval_split).to_numpy()
    return fit, stop, ev


def standardise(X_fit, *others):
    """Mean/std per feature from the FIT windows only; applied to everything else."""
    flat = X_fit.reshape(-1, X_fit.shape[-1]).astype("float64")     # 64-bit for accurate mean/std
    mu, sd = flat.mean(axis=0), flat.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [((a.astype("float64") - mu) / sd).astype("float32") for a in (X_fit,) + others]


def _make_net(n_features):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, HIDDEN, batch_first=True)
            self.drop = nn.Dropout(DROPOUT)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)    # last day of the window

    return Net()


def _train_one(task, X_fit, y_fit, X_stop, y_stop, seed):
    """Train one LSTM; keep the weights of the epoch with the best loss on the STOP rows."""
    import copy
    import torch
    import torch.nn as nn

    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = _make_net(X_fit.shape[-1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss() if task == "dir" else nn.MSELoss()
    Xf, yf = torch.tensor(X_fit).to(dev), torch.tensor(y_fit, dtype=torch.float32).to(dev)
    Xs, ys = torch.tensor(X_stop).to(dev), torch.tensor(y_stop, dtype=torch.float32).to(dev)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        net.train()
        order = torch.randperm(len(Xf), device=dev)
        for i in range(0, len(Xf), BATCH):
            idx = order[i:i + BATCH]
            opt.zero_grad()
            loss_fn(net(Xf[idx]), yf[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(Xs), ys))
        if val_loss < best - 1e-5:
            best, best_state, bad = val_loss, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, dev


def fit_predict(task, X_fit, y_fit, X_stop, y_stop, X_eval, seeds=(0, 1, 2)):
    """Pure array function. task 'dir' -> P(up) for X_eval; task 'vol' -> volatility level for X_eval.
    Prediction = average over several random seeds (LSTMs are noisy)."""
    import torch
    X_fit, X_stop, X_eval = standardise(X_fit, X_stop, X_eval)
    if task == "vol":                                           # learn log-volatility, scaled using FIT rows only
        ly_fit, ly_stop = np.log(np.clip(y_fit, 1e-4, None)), np.log(np.clip(y_stop, 1e-4, None))
        m, s = ly_fit.mean(), max(ly_fit.std(), 1e-8)
        y_fit, y_stop = (ly_fit - m) / s, (ly_stop - m) / s
    preds = []
    for seed in seeds:
        net, dev = _train_one(task, X_fit, y_fit, X_stop, y_stop, seed)
        with torch.no_grad():
            out = net(torch.tensor(X_eval).to(dev)).cpu().numpy()
        preds.append(1 / (1 + np.exp(-out)) if task == "dir" else out)
    p = np.mean(preds, axis=0)
    return p if task == "dir" else np.exp(p * s + m)


def run_lstm(X, meta, eval_split="val", confirm_test=False, seeds=(0, 1, 2)):
    fit, stop, ev = split_rows(meta, eval_split, confirm_test)
    print(f"fit rows {fit.sum()} | stop rows {stop.sum()} | scored rows ({eval_split}) {ev.sum()}")
    p = fit_predict("dir", X[fit], meta.loc[fit, "y_dir"].to_numpy(), X[stop], meta.loc[stop, "y_dir"].to_numpy(), X[ev], seeds)
    v = fit_predict("vol", X[fit], meta.loc[fit, "y_vol"].to_numpy(), X[stop], meta.loc[stop, "y_vol"].to_numpy(), X[ev], seeds)
    y_dir = meta.loc[ev, "y_dir"].astype(int).to_numpy()
    y_vol = meta.loc[ev, "y_vol"].to_numpy()
    dtab = pd.DataFrame({"lstm": direction_scores(y_dir, p)}).T.round(3)
    vtab = pd.DataFrame({"lstm": volatility_scores(y_vol, v)}).T.round(3)
    return dtab, vtab, p


def self_test():
    """Run this in Colab BEFORE trusting the LSTM. Uses fake data with known answers."""
    rng = np.random.default_rng(0)
    n, L_, f = 3000, WINDOW, len(F.PRICE_FEATURES)
    X = rng.normal(size=(n, L_, f)).astype("float32")
    fit, stop, ev = np.arange(n) < 1800, (np.arange(n) >= 1800) & (np.arange(n) < 2400), np.arange(n) >= 2400
    res = {}
    # 1. pure noise labels -> chance
    y = rng.integers(0, 2, n).astype(float)
    p = fit_predict("dir", X[fit], y[fit], X[stop], y[stop], X[ev], seeds=(0,))
    res["noise_auc (want ~0.5)"] = round(roc_auc_score(y[ev], p), 3)
    # 2. planted signal: label depends on feature 0 of the last day -> must be found
    z = X[:, -1, 0] + 0.5 * rng.normal(size=n)
    y2 = (z > 0).astype(float)
    p2 = fit_predict("dir", X[fit], y2[fit], X[stop], y2[stop], X[ev], seeds=(0,))
    res["planted_auc (want > 0.85)"] = round(roc_auc_score(y2[ev], p2), 3)
    # 3. planted volatility signal
    yv = np.exp(0.5 * X[:, -1, 4] - 1.5 + 0.1 * rng.normal(size=n))
    pv = fit_predict("vol", X[fit], yv[fit], X[stop], yv[stop], X[ev], seeds=(0,))
    res["planted_vol_corr (want > 0.9)"] = round(float(np.corrcoef(yv[ev], pv)[0, 1]), 3)
    # 4. changing the SCORED rows' labels must not change the predictions (they are never used in training)
    y3 = y2.copy(); y3[ev] = 1 - y3[ev]
    p3 = fit_predict("dir", X[fit], y3[fit], X[stop], y3[stop], X[ev], seeds=(0,))
    res["labels_of_scored_rows_changed -> max prediction change (want 0)"] = float(np.abs(p2 - p3).max())
    for k, v in res.items():
        print(k, ":", v)
    ok = (abs(res["noise_auc (want ~0.5)"] - 0.5) < 0.07 and res["planted_auc (want > 0.85)"] > 0.85
          and res["planted_vol_corr (want > 0.9)"] > 0.9
          and res["labels_of_scored_rows_changed -> max prediction change (want 0)"] < 1e-6)
    print("SELF-TEST", "PASSED" if ok else "FAILED")


def run(eval_split="val"):
    prices = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "prices.parquet"))
    df = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "dataset.parquet"))
    X, meta = make_windows(prices, df)
    dtab, vtab, p = run_lstm(X, meta, eval_split=eval_split)
    print(f"\n=== LSTM | next-day direction | scored on {eval_split} ===")
    print(dtab.to_string())
    print("Share of days called up:", round(float((p >= 0.5).mean()), 3), "| always_up reference: 0.551 / 0.711 / 0.500")
    print(f"\n=== LSTM | next-week volatility | scored on {eval_split} ===")
    print(vtab.to_string())
    print("(compare: arima 0.087 / 0.134 / 0.522, xgboost price-only 0.078 / 0.131 / 0.551)")
    out_dir = os.path.join(config.DATA_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    dtab.to_csv(os.path.join(out_dir, f"baseline_direction_lstm_{eval_split}.csv"))
    vtab.to_csv(os.path.join(out_dir, f"baseline_volatility_lstm_{eval_split}.csv"))
    return None