"""Phase 2.3: ARIMA baseline (classic time-series model, one model per stock).

Two jobs, same as the other baselines, scored on the SAME validation rows as price_ml.py:

  NEXT-DAY DIRECTION: ARIMA(1,0,1) on daily log returns. Forecast for day t+1 > 0 -> "up".
                      The forecast itself is used as the score for AUC.
  NEXT-WEEK VOLATILITY: ARIMA(1,0,1) on log(vol_5), where vol_5 = std of the last 5 daily returns.
                      The target y_vol (days t+1..t+5) is almost exactly vol_5 five days later,
                      so we forecast that series 5 steps ahead.

Discipline (no look-ahead):
  * parameters are fitted on TRAIN dates only (up to config.TRAIN_END), once per stock;
  * for each validation day t the model only RE-READS data up to t (parameters stay fixed);
  * data after config.VAL_END is never loaded, so the test years stay locked.
"""
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import config
from src.baselines.price_ml import ANNUALISE, direction_scores, volatility_scores
from src.forecasting import features as F

ORDER = (1, 0, 1)


def _fit(series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # harmless convergence chatter
        return ARIMA(series.to_numpy(), order=ORDER, trend="c").fit()


def _log_vol5(r):
    v = r.rolling(5).std() * ANNUALISE
    return np.log(v.clip(lower=1e-4))


def forecast_ticker(g, eval_dates, train_end, last_day):
    """g: one ticker's prices (Date, Close). Returns a DataFrame with Date, ret_fc, logvol_fc.

    ret_fc[t]    : forecast of the return on day t+1, made with data up to t.
    logvol_fc[t] : forecast of log annualised vol over t+1..t+5, made with data up to t.
    Only data with Date <= last_day is used.
    """
    g = g.sort_values("Date")
    g = g[g["Date"] <= last_day].reset_index(drop=True)
    r = np.log(g["Close"]).diff()
    lv = _log_vol5(r)
    is_train = g["Date"] <= train_end

    res_r = _fit(r[is_train].dropna())
    res_v = _fit(lv[is_train].dropna())

    r_ok, lv_ok = r.iloc[1:], lv.iloc[5:]                      # drop warm-up NaNs
    # Return model: fixed parameters, filter through everything once. fittedvalues[k] predicts the
    # return of day k+1 using only returns up to day k, so it is stored at position k.
    pred_r = res_r.apply(r_ok.to_numpy(), refit=False).fittedvalues
    ret_fc = pd.Series(np.nan, index=g.index)
    ret_fc.iloc[:len(g) - 1] = pred_r

    # Volatility model: for each evaluation day, read data up to t and forecast 5 steps ahead.
    pos = {d: i for i, d in enumerate(g["Date"])}
    vol_fc = {}
    for d in eval_dates:
        i = pos.get(d)
        if i is None or i < 5:
            continue
        hist = lv.iloc[5:i + 1].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vol_fc[d] = float(res_v.apply(hist, refit=False).forecast(5)[-1])
    out = pd.DataFrame({"Date": g["Date"], "ret_fc": ret_fc.to_numpy()})
    out["logvol_fc"] = out["Date"].map(vol_fc)
    return out


def run_arima(prices, dataset, eval_split="val", confirm_test=False):
    """Returns (direction_table, volatility_table), one row 'arima' each."""
    if eval_split == "test" and not confirm_test:
        raise RuntimeError("The test years are locked until the very end. Use eval_split='val'.")
    last_day = pd.Timestamp(config.VAL_END if eval_split == "val" else config.END_DATE)
    train_end = pd.Timestamp(config.TRAIN_END)
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    ev = dataset[dataset["split"] == eval_split]

    parts = []
    for tk, g in prices.groupby("Ticker"):
        dates = set(ev.loc[ev["Ticker"] == tk, "Date"])
        fc = forecast_ticker(g, dates, train_end, last_day)
        fc["Ticker"] = tk
        parts.append(fc)
    fc = pd.concat(parts, ignore_index=True)
    m = ev.merge(fc, on=["Ticker", "Date"], how="left")
    m = m.dropna(subset=["ret_fc", "logvol_fc"])

    y = m["y_dir"].astype(int).to_numpy()
    score = m["ret_fc"].to_numpy()
    pred_up = (score > 0).astype(float)
    d = direction_scores(y, pred_up)
    d["auc"] = direction_scores(y, score)["auc"]               # AUC from the raw forecast, not 0/1
    v = volatility_scores(m["y_vol"].to_numpy(), np.exp(m["logvol_fc"].to_numpy()))
    n = len(m)
    return pd.DataFrame({"arima": d}).T.round(3), pd.DataFrame({"arima": v}).T.round(3), n


def run(eval_split="val"):
    prices = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "prices.parquet"))
    dataset = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "dataset.parquet"))
    d, v, n = run_arima(prices, dataset, eval_split=eval_split)
    print(f"=== ARIMA{ORDER} | scored on {eval_split} ({n} of {(dataset['split'] == eval_split).sum()} rows) ===")
    print("Next-day direction (higher is better):\n", d.to_string())
    print("\nNext-week volatility (lower MAE/RMSE is better, corr higher is better):\n", v.to_string())
    out_dir = os.path.join(config.DATA_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    d.to_csv(os.path.join(out_dir, f"baseline_direction_arima_{eval_split}.csv"))
    v.to_csv(os.path.join(out_dir, f"baseline_volatility_arima_{eval_split}.csv"))