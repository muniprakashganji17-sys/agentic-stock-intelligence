"""Phase 2.2: first baselines on the modelling table (dataset.parquet).

Two questions, each with simple "do nothing clever" models and one machine-learning model:

  NEXT-DAY DIRECTION (y_dir, up=1)
    always_up      : always predict "up" (what the training years did most often)
    logistic       : logistic regression on standardised features
    xgboost        : gradient-boosted trees
  NEXT-WEEK VOLATILITY (y_vol, annualised)
    persistence_5d : "next week looks like the last 5 days"
    persistence_21d: "next week looks like the last 21 days"
    xgboost        : gradient-boosted trees on log volatility

Discipline: models are fitted on TRAIN only (scaling too). We score on VALIDATION while we are
still making choices. The TEST years are locked: they are scored once, at the very end of the
project, so they stay an honest exam.

`features` chooses the feature group, so the same code is reused for the ablation study:
  features.PRICE_FEATURES                          -> price only
  features.PRICE_FEATURES + features.MACRO_FEATURES -> price + macro
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

import config
from src.forecasting import features as F

SEED = 42
ANNUALISE = np.sqrt(F.TRADING_DAYS)


def _xgb_params():
    return dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5, random_state=SEED, n_jobs=2)


def direction_scores(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y_true, pred), "f1_up": f1_score(y_true, pred, zero_division=0),
            "auc": roc_auc_score(y_true, prob) if len(set(y_true)) > 1 and np.ptp(prob) > 0 else 0.5}


def volatility_scores(y_true, pred):
    return {"mae": mean_absolute_error(y_true, pred), "rmse": float(np.sqrt(np.mean((y_true - pred) ** 2))),
            "corr": float(np.corrcoef(y_true, pred)[0, 1]) if np.ptp(pred) > 0 else 0.0}


def run_baselines(df, feature_cols, eval_split="val", confirm_test=False):
    """Fit on train, score on `eval_split`. Returns (direction_table, volatility_table)."""
    if eval_split == "test" and not confirm_test:
        raise RuntimeError("The test years are locked until the very end. Use eval_split='val'.")
    tr, ev = df[df["split"] == "train"], df[df["split"] == eval_split]

    # ---------------- direction ----------------
    ytr, yev = tr["y_dir"].astype(int).to_numpy(), ev["y_dir"].astype(int).to_numpy()
    rows = {}
    majority = int(ytr.mean() >= 0.5)
    rows["always_up" if majority else "always_down"] = direction_scores(yev, np.full(len(yev), float(majority)))

    scaler = StandardScaler().fit(tr[feature_cols])                     # statistics from TRAIN only
    logit = LogisticRegression(max_iter=1000).fit(scaler.transform(tr[feature_cols]), ytr)
    rows["logistic"] = direction_scores(yev, logit.predict_proba(scaler.transform(ev[feature_cols]))[:, 1])

    xgb = XGBClassifier(**_xgb_params()).fit(tr[feature_cols], ytr)
    rows["xgboost"] = direction_scores(yev, xgb.predict_proba(ev[feature_cols])[:, 1])
    dir_table = pd.DataFrame(rows).T

    # ---------------- volatility ----------------
    vtr, vev = tr["y_vol"].to_numpy(), ev["y_vol"].to_numpy()
    vrows = {"persistence_5d": volatility_scores(vev, ev["vol_5"].to_numpy() * ANNUALISE),
             "persistence_21d": volatility_scores(vev, ev["vol_21"].to_numpy() * ANNUALISE)}
    reg = XGBRegressor(**_xgb_params()).fit(tr[feature_cols], np.log(vtr))
    vrows["xgboost"] = volatility_scores(vev, np.exp(reg.predict(ev[feature_cols])))
    vol_table = pd.DataFrame(vrows).T
    return dir_table.round(3), vol_table.round(3)


def run(eval_split="val"):
    df = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "dataset.parquet"))
    out_dir = os.path.join(config.DATA_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    sets = {"price only": F.PRICE_FEATURES, "price + macro": F.PRICE_FEATURES + F.MACRO_FEATURES}
    for name, cols in sets.items():
        d, v = run_baselines(df, cols, eval_split=eval_split)
        print(f"\n=== {name.upper()} | scored on {eval_split} ({(df['split'] == eval_split).sum()} rows) ===")
        print("Next-day direction (higher is better):\n", d.to_string())
        print("\nNext-week volatility (lower MAE/RMSE is better, corr higher is better):\n", v.to_string())
        d.to_csv(os.path.join(out_dir, f"baseline_direction_{name.replace(' ', '_').replace('+', 'and')}_{eval_split}.csv"))
        v.to_csv(os.path.join(out_dir, f"baseline_volatility_{name.replace(' ', '_').replace('+', 'and')}_{eval_split}.csv"))
    return None