import argparse
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

LABEL_TO_CLASS = {0: 0, 1: 1, -1: 2}
CLASS_TO_LABEL = {0: 0, 1: 1, 2: -1}


def run_vectorized_backtest(
    df_test: pd.DataFrame,
    pred_actions: np.ndarray,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0002,
    risk_per_trade: float = 0.01,
    take_profit: float = 0.015,
    stop_loss: float = 0.010,
) -> Dict[str, float]:
    """
    Out-of-sample vectorized backtest on test predictions.
    Position size is scaled so a `stop_loss` (1.0%) move equals `risk_per_trade` (1.0%) of equity.
    Fees (0.05%) and slippage (0.02%) are charged only on position entries/reversals.
    """
    forward_raw_ret = np.exp(df_test["ret_1"].shift(-1).fillna(0.0).to_numpy() / 100.0) - 1.0
    candle_vol = df_test["candle_range_pct"].shift(-1).fillna(0.0).to_numpy()

    actions = pred_actions.astype(float)
    active_mask = actions != 0.0

    # Gross directional return per step
    gross_ret = actions * forward_raw_ret

    # Apply intrabar TP/SL caps using directional return and candle range envelope
    hit_tp = active_mask & ((gross_ret >= take_profit) | ((gross_ret > 0) & (candle_vol >= take_profit)))
    hit_sl = active_mask & ((gross_ret <= -stop_loss) | ((gross_ret < 0) & (candle_vol >= stop_loss)))
    capped_ret = np.where(hit_tp, take_profit, np.where(hit_sl, -stop_loss, gross_ret))

    # Group consecutive identical signals into discrete trades so friction is only paid on entry/exit
    prev_actions = np.roll(actions, 1)
    prev_actions[0] = 0.0
    new_trade_mask = active_mask & (actions != prev_actions)
    trade_ids = np.cumsum(new_trade_mask)

    notional_leverage = risk_per_trade / stop_loss
    per_side_friction = fee_rate + slippage_rate
    turnover = np.abs(actions - prev_actions)
    step_friction = turnover * per_side_friction
    net_step_rets = np.where(active_mask, notional_leverage * (capped_ret - step_friction), -notional_leverage * step_friction)

    if not np.any(new_trade_mask):
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "net_return_pct": 0.0,
        }

    # Aggregate PnL by discrete trade episode
    active_ids = trade_ids[active_mask]
    active_step_rets = net_step_rets[active_mask]
    unique_ids = np.unique(active_ids)
    trade_pnls = np.array([np.prod(1.0 + active_step_rets[active_ids == tid]) - 1.0 for tid in unique_ids])
    total_trades = len(trade_pnls)

    wins = trade_pnls[trade_pnls > 0.0]
    losses = trade_pnls[trade_pnls < 0.0]
    win_rate_pct = float((len(wins) / total_trades) * 100.0)

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-12 else (99.99 if gross_profit > 0 else 0.0)

    equity_curve = np.cumprod(1.0 + net_step_rets)
    running_peak = np.maximum.accumulate(np.insert(equity_curve, 0, 1.0))[1:]
    drawdowns = (running_peak - equity_curve) / running_peak
    max_drawdown_pct = float(np.max(drawdowns) * 100.0)
    net_return_pct = float((equity_curve[-1] - 1.0) * 100.0)

    return {
        "total_trades": total_trades,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "net_return_pct": net_return_pct,
    }


def train_and_evaluate(
    data_path: str = "data/abstracted_trades.parquet",
    model_path: str = "models/btc_mimic_model.pkl",
) -> Dict[str, Any]:
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c != "target_action"]

    X = df[feature_cols]
    y = df["target_action"].map(LABEL_TO_CLASS).astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    split_point = int(len(df) * 0.8)
    if abs(len(train_idx) - split_point) > max(5, int(len(df) * 0.05)):
        train_idx = np.arange(0, split_point)
        test_idx = np.arange(split_point, len(df))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    df_test = df.iloc[test_idx].reset_index(drop=True)

    model = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
    )
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    # Boost active trade classes (1=LONG, 2=SHORT) so directional signals achieve high confidence (>0.65)
    sample_weights = np.where(y_train != 0, sample_weights * 1.4, sample_weights)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    probs = model.predict_proba(X_test)
    pred_classes = np.argmax(probs, axis=1)
    pred_actions = np.vectorize(CLASS_TO_LABEL.get)(pred_classes)

    metrics = run_vectorized_backtest(df_test, pred_actions)

    print(f"Total Trades:     {metrics['total_trades']}")
    print(f"Win Rate %:       {metrics['win_rate_pct']:.2f}%")
    print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown %:   {metrics['max_drawdown_pct']:.2f}%")
    print(f"Net Return %:     {metrics['net_return_pct']:.2f}%")

    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "features": feature_cols,
        "label_to_class": LABEL_TO_CLASS,
        "class_to_label": CLASS_TO_LABEL,
        "metrics": metrics,
    }
    joblib.dump(artifact, out_path)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost behavioral clone and run OOS backtest.")
    parser.add_argument("--data", type=str, default="data/abstracted_trades.parquet")
    parser.add_argument("--model", type=str, default="models/btc_mimic_model.pkl")
    args = parser.parse_args()
    train_and_evaluate(data_path=args.data, model_path=args.model)
