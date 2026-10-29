"""Phase 2.3: ARIMA baselines - NEXT-DAY DIRECTION (step 1) and NEXT-WEEK VOLATILITY (step 2).

Idea in simple words
  ARIMA looks only at a stock's own past daily returns and predicts tomorrow's return.
  If the predicted return is above zero we say "up", otherwise "down".

How we keep it honest
  1. The model (its order and numbers) is chosen and fitted on TRAIN returns only.
  2. Then the fitted numbers are FROZEN and the model walks day by day through the
     validation year. On day t it sees returns up to day t only and forecasts day t+1.
     (It is not refitted every day - that keeps it fast and leak-free.)
  3. Scored on the same validation rows as price_ml.py, so the tables can be compared.
  Volatility (step 2): the same idea on the series log("volatility of the last 5 days").
  Params are fitted on TRAIN only and frozen; on each day t the model forecasts 5 days ahead,
  which is exactly the span of the target y_vol (days t+1 ... t+5).
  The TEST years are locked: eval_split='test' is refused unless confirm_test=True.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.arima.model import ARIMA

import config
from src.baselines.price_ml import direction_scores, volatility_scores
from src.forecasting import features as F

ORDER_GRID = [(p, 0, q) for p in range(3) for q in range(3)]   # d=0: returns are already "differenced"


def _end_date(eval_split):
    return pd.Timestamp(config.VAL_END if eval_split == "val" else config.END_DATE)


def forecast_next_returns(prices, eval_split="val", confirm_test=False):
    """For every ticker: forecast of the NEXT day's log return, made at the close of each day t.

    Returns a DataFrame (Ticker, Date, arima_forecast, order). Row (ticker, t) = forecast for t+1,
    using returns up to t only. Parameters are fitted on dates <= config.TRAIN_END only.
    """
    if eval_split == "test" and not confirm_test:
        raise RuntimeError("The test years are locked until the very end. Use eval_split='val'.")
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    end = _end_date(eval_split)
    out = []
    for tic, g in prices.groupby("Ticker"):
        g = g.sort_values("Date")
        r = np.log(g["Close"]).diff()
        r.index = g["Date"].to_numpy()
        r = r.dropna()
        r = r[r.index <= end]                                   # nothing after the scored period
        train = r[r.index <= pd.Timestamp(config.TRAIN_END)].to_numpy()

        best, best_aic = None, np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in ORDER_GRID:                            # pick the order by AIC on TRAIN only
                try:
                    res = ARIMA(train, order=order, trend="c").fit()
                except Exception:
                    continue
                if res.aic < best_aic:
                    best, best_aic = (order, res.params), res.aic
            order, params = best
            frozen = ARIMA(r.to_numpy(), order=order, trend="c").smooth(params)
        fitted = np.asarray(frozen.fittedvalues)                # fitted[i] = forecast of r[i] from r[:i]
        forecast_at_t = np.append(fitted[1:], np.nan)           # shift: value at t = forecast of t+1
        out.append(pd.DataFrame({"Ticker": tic, "Date": r.index, "arima_forecast": forecast_at_t,
                                 "order": str(order)}))
    return pd.concat(out, ignore_index=True)


def score_direction(df, fc, eval_split="val"):
    """Score ARIMA forecasts on the dataset rows of `eval_split`. Returns a one-row-per-model table."""
    ev = df[df["split"] == eval_split].merge(fc, on=["Ticker", "Date"], how="left")
    if ev["arima_forecast"].isna().any():
        raise RuntimeError(f"{ev['arima_forecast'].isna().sum()} rows have no ARIMA forecast")
    y = ev["y_dir"].astype(int).to_numpy()
    f = ev["arima_forecast"].to_numpy()
    s = direction_scores(y, (f > 0).astype(float))              # accuracy and F1 from the up/down call
    s["auc"] = roc_auc_score(y, f) if len(set(y)) > 1 and np.ptp(f) > 0 else 0.5   # AUC from the size of the forecast
    return pd.DataFrame({"arima": s}).T.round(3), ev


# ---------------------------------------------------------------------------------------------
# Step 2: next-week volatility
# ---------------------------------------------------------------------------------------------
def forecast_volatility(prices, eval_split="val", confirm_test=False):
    """For every ticker: forecast of volatility over days t+1..t+5, made at the close of day t.

    Series modelled: z_t = log( sqrt(mean of the last 5 squared daily log returns) * sqrt(252) ),
    i.e. the same recipe as the target y_vol but looking BACKWARD. Order is chosen by AIC on TRAIN
    only; the fitted numbers are frozen; on each day t the model sees z up to t and forecasts z(t+5).
    Returns DataFrame (Ticker, Date, arima_vol, order); arima_vol is in the same units as y_vol.
    """
    if eval_split == "test" and not confirm_test:
        raise RuntimeError("The test years are locked until the very end. Use eval_split='val'.")
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    end = _end_date(eval_split)
    out = []
    for tic, g in prices.groupby("Ticker"):
        g = g.sort_values("Date")
        r = np.log(g["Close"]).diff()
        z = np.log(np.sqrt((r ** 2).rolling(F.HORIZON).mean()) * np.sqrt(F.TRADING_DAYS))
        z.index = g["Date"].to_numpy()
        z = z.replace([np.inf, -np.inf], np.nan).dropna()
        z = z[z.index <= end]
        zv = z.to_numpy()
        n_train = int((z.index <= pd.Timestamp(config.TRAIN_END)).sum())

        best, best_aic = None, np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in ORDER_GRID:                            # order chosen on TRAIN only
                try:
                    res = ARIMA(zv[:n_train], order=order, trend="c").fit()
                except Exception:
                    continue
                if res.aic < best_aic:
                    best, best_aic = (order, res), res.aic
            order, train_res = best
            preds = np.full(len(zv), np.nan)
            for i in range(n_train - 1, len(zv)):               # day t = index i: sees zv[:i+1] only
                frozen = train_res.apply(zv[:i + 1], refit=False)
                preds[i] = frozen.forecast(F.HORIZON)[-1]       # forecast of z at t+5
        out.append(pd.DataFrame({"Ticker": tic, "Date": z.index, "arima_vol": np.exp(preds),
                                 "order": str(order)}))
    return pd.concat(out, ignore_index=True)


def score_volatility(df, fv, eval_split="val"):
    ev = df[df["split"] == eval_split].merge(fv, on=["Ticker", "Date"], how="left")
    if ev["arima_vol"].isna().any():
        raise RuntimeError(f"{ev['arima_vol'].isna().sum()} rows have no ARIMA volatility forecast")
    s = volatility_scores(ev["y_vol"].to_numpy(), ev["arima_vol"].to_numpy())
    return pd.DataFrame({"arima": s}).T.round(3), ev


def run(eval_split="val"):
    prices = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "prices.parquet"))
    df = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "dataset.parquet"))
    fc = forecast_next_returns(prices, eval_split=eval_split)
    table, ev = score_direction(df, fc, eval_split=eval_split)

    print("ARIMA order chosen per ticker (p, d, q), by AIC on train:")
    print(fc.groupby("Ticker")["order"].first().to_string())
    print(f"\n=== ARIMA | next-day direction | scored on {eval_split} ({len(ev)} rows) ===")
    print(table.to_string())
    print("\nShare of days ARIMA called 'up':", round(float((ev['arima_forecast'] > 0).mean()), 3))
    out_dir = os.path.join(config.DATA_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, f"baseline_direction_arima_{eval_split}.csv"))

    fv = forecast_volatility(prices, eval_split=eval_split)
    vtable, vev = score_volatility(df, fv, eval_split=eval_split)
    print("\nVolatility ARIMA order per ticker:")
    print(fv.groupby("Ticker")["order"].first().to_string())
    print(f"\n=== ARIMA | next-week volatility | scored on {eval_split} ({len(vev)} rows) ===")
    print(vtable.to_string())
    print("(compare: persistence_21d 0.096 / 0.149 / 0.487, xgboost price-only 0.078 / 0.131 / 0.551)")
    vtable.to_csv(os.path.join(out_dir, f"baseline_volatility_arima_{eval_split}.csv"))
    return None