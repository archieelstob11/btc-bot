import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "abstracted_trades.parquet"
MODEL_PATH = ROOT_DIR / "models" / "btc_mimic_model.pkl"
PAPER_LOG_PATH = ROOT_DIR / "data" / "paper_trades_log.csv"


def test_1_fetch_and_abstract():
    backup = DATA_PATH.with_suffix(".parquet.bak")
    if DATA_PATH.exists():
        shutil.copy2(DATA_PATH, backup)
        DATA_PATH.unlink()

    try:
        res = subprocess.run(
            [sys.executable, str(ROOT_DIR / "1_fetch_and_abstract.py"), "--limit", "350"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        assert res.returncode == 0
        assert DATA_PATH.exists(), "abstracted_trades.parquet was not created."

        df = pd.read_parquet(DATA_PATH)
        assert len(df) > 0, "Dataset is empty."
        assert int(df.isna().sum().sum()) == 0, "Dataset contains NaN values."

        forbidden_cols = {"open", "high", "low", "close", "volume", "timestamp"}
        assert forbidden_cols.isdisjoint(set(df.columns)), f"Raw price columns found: {forbidden_cols & set(df.columns)}"

        counts = df["target_action"].value_counts()
        n_hold = int(counts.get(0, 0))
        n_active = int(counts.get(1, 0) + counts.get(-1, 0))
        if n_active > 0:
            assert n_hold <= 3 * n_active, f"Hold class ratio exceeded 3:1 ({n_hold} vs {n_active})."
    finally:
        if backup.exists():
            shutil.move(str(backup), str(DATA_PATH))


def test_2_train_and_backtest():
    backup = MODEL_PATH.with_suffix(".pkl.bak")
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, backup)
        MODEL_PATH.unlink()

    try:
        res = subprocess.run(
            [sys.executable, str(ROOT_DIR / "2_train_and_backtest.py")],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        assert res.returncode == 0
        assert MODEL_PATH.exists(), "btc_mimic_model.pkl was not created."

        output_lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        assert len(output_lines) == 5, f"Expected 5 summary lines, got {len(output_lines)}"

        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        features = artifact["features"]

        df = pd.read_parquet(DATA_PATH)
        sample_X = df[features].iloc[:10]
        probs = model.predict_proba(sample_X)

        assert probs.shape == (len(sample_X), 3), f"Unexpected probability shape: {probs.shape}"
        assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities out of [0, 1] bounds."
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Probabilities do not sum to 1."
    finally:
        if backup.exists():
            shutil.move(str(backup), str(MODEL_PATH))


def test_3_live_paper_bot():
    backup_path = PAPER_LOG_PATH.with_suffix(".csv.bak")
    if PAPER_LOG_PATH.exists():
        shutil.copy2(PAPER_LOG_PATH, backup_path)
        PAPER_LOG_PATH.unlink()

    try:
        res = subprocess.run(
            [sys.executable, str(ROOT_DIR / "3_live_paper_bot.py"), "--dry-run", "--once"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        assert res.returncode == 0
        assert PAPER_LOG_PATH.exists(), "paper_trades_log.csv was not created."

        log_df = pd.read_csv(PAPER_LOG_PATH)
        assert len(log_df) >= 1, "Paper trade log is empty."
        expected_cols = {
            "timestamp_utc",
            "symbol",
            "mode",
            "raw_signal",
            "executed_action",
            "confidence",
            "fill_price",
            "position_size_btc",
            "risk_usd",
        }
        assert expected_cols.issubset(set(log_df.columns))
    finally:
        if backup_path.exists():
            shutil.move(str(backup_path), str(PAPER_LOG_PATH))
