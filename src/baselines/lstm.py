"""Phase 2.4: LSTM baseline (small neural network that reads the last 21 days).

Input for a row (stock, day t): the PRICE features of days t-20 ... t (21 days, a "window").
Output: two numbers at once - probability that the next day is up, and next-week volatility.
Same rows and same metrics as the other baselines, scored on VALIDATION only.

Discipline (no look-ahead):
  * features of day t use data up to t, so a window only contains information known at t;
  * scaling statistics (mean/std) come from TRAIN rows only;
  * "when to stop training" is decided on the LAST 15% of the training period (a time-ordered
    slice with a 10-day gap before it), NOT on validation, so validation stays a clean exam;
  * averaged over 3 random seeds, because a single small network is noisy;
  * data after config.VAL_END is never used, so the test years stay locked.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

import config
from src.baselines.price_ml import direction_scores, volatility_scores
from src.forecasting import features as F

WINDOW = 21
SEEDS = (0, 1, 2)
MAX_EPOCHS = 60
PATIENCE = 8
GAP_DAYS = 10          # calendar days skipped between the inner-train and early-stopping slices


def build_windows(prices, dataset, feature_cols=F.PRICE_FEATURES, last_day=None):
    """One window per dataset row. Returns (rows_df, X) with X shaped (rows, WINDOW, features).

    Windows are cut from the FULL daily feature table (not from the filtered dataset), so they
    are contiguous trading days even across split borders. Rows whose window has gaps/NaN are
    dropped (only the very first rows of the history).
    """
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    if last_day is not None:
        prices = prices[prices["Date"] <= last_day]
    keys = dataset[["Ticker", "Date"]]
    rows, mats = [], []
    for tk, g in prices.groupby("Ticker"):
        feat = F._price_block(g.sort_values("Date")).reset_index(drop=True)
        arr = feat[feature_cols].to_numpy(dtype=float)
        pos = {d: i for i, d in enumerate(feat["Date"])}
        for d in keys.loc[keys["Ticker"] == tk, "Date"]:
            i = pos.get(d)
            if i is None or i < WINDOW - 1:
                continue
            w = arr[i - WINDOW + 1:i + 1]
            if np.isnan(w).any():
                continue
            rows.append((tk, d))
            mats.append(w)
    idx = pd.DataFrame(rows, columns=["Ticker", "Date"])
    return idx, np.stack(mats)


class Net(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, 2)                       # [direction logit, standardised log-vol]

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :]))


def _train_one(Xtr, ydir, yvol, Xes, ydir_es, yvol_es, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Xtr_t, yd_t, yv_t = T(Xtr), T(ydir), T(yvol)
    Xes_t, yde_t, yve_t = T(Xes), T(ydir_es), T(yvol_es)

    def loss_fn(x, yd, yv):
        o = net(x)
        return bce(o[:, 0], yd) + mse(o[:, 1], yv)

    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for k in range(0, len(perm), 128):
            b = perm[k:k + 128].to(device)
            opt.zero_grad()
            loss_fn(Xtr_t[b], yd_t[b], yv_t[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(Xes_t, yde_t, yve_t).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def run_lstm(prices, dataset, eval_split="val", confirm_test=False, seeds=SEEDS, device="cpu"):
    """Returns (direction_table, volatility_table, n_rows_scored, n_rows_total)."""
    if eval_split == "test" and not confirm_test:
        raise RuntimeError("The test years are locked until the very end. Use eval_split='val'.")
    last_day = pd.Timestamp(config.VAL_END if eval_split == "val" else config.END_DATE)
    ds = dataset[dataset["split"].isin(["train", eval_split])]
    idx, X = build_windows(prices, ds, last_day=last_day)
    m = idx.merge(ds, on=["Ticker", "Date"], how="left")
    is_train = (m["split"] == "train").to_numpy()
    is_ev = (m["split"] == eval_split).to_numpy()

    # scaling from TRAIN rows only
    flat = X[is_train].reshape(-1, X.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    X = (X - mu) / sd

    lv = np.log(m["y_vol"].to_numpy())
    lmu, lsd = lv[is_train].mean(), lv[is_train].std() + 1e-9
    lvz = (lv - lmu) / lsd
    yd = m["y_dir"].to_numpy(dtype=float)

    # time-ordered inner split of TRAIN: early-stopping slice = last 15% of train dates
    tr_dates = m.loc[is_train, "Date"]
    cut = tr_dates.quantile(0.85, interpolation="nearest")
    inner = is_train & (m["Date"] < cut - pd.Timedelta(days=GAP_DAYS)).to_numpy()
    es = is_train & (m["Date"] >= cut).to_numpy()

    probs, vols = [], []
    for s in seeds:
        net = _train_one(X[inner], yd[inner], lvz[inner], X[es], yd[es], lvz[es], s, device)
        with torch.no_grad():
            o = net(torch.tensor(X[is_ev], dtype=torch.float32, device=device)).cpu().numpy()
        probs.append(1 / (1 + np.exp(-o[:, 0])))
        vols.append(o[:, 1] * lsd + lmu)
    prob, logvol = np.mean(probs, axis=0), np.mean(vols, axis=0)

    ev = m[is_ev]
    d = direction_scores(ev["y_dir"].astype(int).to_numpy(), prob)
    v = volatility_scores(ev["y_vol"].to_numpy(), np.exp(logvol))
    n_total = int((dataset["split"] == eval_split).sum())
    out = ev[["Ticker", "Date"]].copy()
    out["prob_up"], out["vol_fc"] = prob, np.exp(logvol)
    return (pd.DataFrame({"lstm": d}).T.round(3), pd.DataFrame({"lstm": v}).T.round(3),
            len(ev), n_total, out)


def run(eval_split="val"):
    prices = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "prices.parquet"))
    dataset = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "dataset.parquet"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, v, n, n_total, _ = run_lstm(prices, dataset, eval_split=eval_split, device=device)
    print(f"=== LSTM (window {WINDOW} days, {len(SEEDS)} seeds, device {device}) | "
          f"scored on {eval_split} ({n} of {n_total} rows) ===")
    print("Next-day direction (higher is better):\n", d.to_string())
    print("\nNext-week volatility (lower MAE/RMSE is better, corr higher is better):\n", v.to_string())
    out_dir = os.path.join(config.DATA_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    d.to_csv(os.path.join(out_dir, f"baseline_direction_lstm_{eval_split}.csv"))
    v.to_csv(os.path.join(out_dir, f"baseline_volatility_lstm_{eval_split}.csv"))