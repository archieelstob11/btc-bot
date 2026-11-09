import argparse
import csv
import importlib.util
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import ccxt
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

CLASS_TO_LABEL = {0: 0, 1: 1, 2: -1}
ACTION_NAMES = {0: "HOLD", 1: "LONG", -1: "SHORT"}
ACTION_TO_DIR = {"HOLD": 0.0, "LONG": 1.0, "SHORT": -1.0}

DEFAULT_FEE_RATE = 0.0005       # 0.05% taker fee per side (0.10% = $10 round-trip on $10k)
DEFAULT_SLIPPAGE_RATE = 0.0002  # 0.02% adverse slippage per side (0.04% = $4 round-trip on $10k)
# Brackets are calibrated so NET PnL AFTER $14 round-trip fees+slippage is an exact 1 : 1.5 R:R:
# Account A (Wide):  +$164 gross - $14 friction = +$150.00 NET win | -$86 gross - $14 friction = -$100.00 NET loss (1 : 1.5 Net R:R)
DEFAULT_TAKE_PROFIT = 0.0164
DEFAULT_STOP_LOSS = 0.0086
# Account B (Tight): +$59 gross - $14 friction  = +$45.00 NET win  | -$16 gross - $14 friction = -$30.00 NET loss  (1 : 1.5 Net R:R)
TIGHT_TAKE_PROFIT = 0.0059
TIGHT_STOP_LOSS = 0.0016


def _load_feature_module():
    spec = importlib.util.spec_from_file_location(
        "fetch_and_abstract", Path(__file__).resolve().parent / "1_fetch_and_abstract.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def init_exchange(exchange_id: str = "bybit", dry_run: bool = True) -> ccxt.Exchange:
    """Initialize ccxt exchange for paper trading (Demo/Testnet) or public dry-run."""
    load_dotenv()
    api_key = os.getenv("EXCHANGE_API_KEY", "")
    api_secret = os.getenv("EXCHANGE_API_SECRET", "")

    config: Dict[str, Any] = {"enableRateLimit": True, "timeout": 5000}
    if not dry_run and api_key and api_secret:
        config["apiKey"] = api_key
        config["secret"] = api_secret

    exchange_cls = getattr(ccxt, exchange_id)
    exchange: ccxt.Exchange = exchange_cls(config)

    if not dry_run:
        if hasattr(exchange, "enable_demo_trading"):
            try:
                exchange.enable_demo_trading(True)
            except Exception:
                exchange.set_sandbox_mode(True)
        else:
            exchange.set_sandbox_mode(True)
    return exchange


def compute_orderbook_vwap(levels: List[List[float]], target_size_btc: float, fallback_price: float) -> float:
    """Walk orderbook levels to compute volume-weighted fill price for `target_size_btc`."""
    if not levels or target_size_btc <= 0.0:
        return fallback_price
    remaining = target_size_btc
    cost = 0.0
    filled = 0.0
    for level in levels:
        px, qty = float(level[0]), float(level[1])
        take = min(remaining, qty)
        cost += take * px
        filled += take
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 0.0:
        last_px = float(levels[-1][0])
        cost += remaining * last_px
        filled += remaining
    return (cost / filled) if filled > 0.0 else fallback_price


def sanitize_log_file(log_path: str = "data/paper_trades_log.csv") -> None:
    """Repair truncated or corrupted tail lines in `paper_trades_log.csv` after a sudden power loss."""
    log_file = Path(log_path)
    if not log_file.exists():
        return
    try:
        raw_bytes = log_file.read_bytes()
        if b"\x00" in raw_bytes:
            raw_bytes = raw_bytes.replace(b"\x00", b"")
        lines = [ln for ln in raw_bytes.decode("utf-8", errors="ignore").splitlines() if ln.strip()]
        if not lines:
            return
        expected_commas = lines[0].count(",")
        valid_lines = [lines[0]] + [
            ln for ln in lines[1:]
            if ln.count(",") == expected_commas and "DRY_RUN_OFFLINE" not in ln
        ]
        if len(valid_lines) != len(lines):
            log_file.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def fetch_latest_candles_and_book(
    exchange: ccxt.Exchange,
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    limit: int = 50,
    allow_offline_fallback: Optional[bool] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Fetch latest 50 candles and top-of-book depth with retry backoff.
    - During live runs: retries 3 times on Wi-Fi drop; if still offline, returns `(None, {"feed": "OFFLINE_SKIP"})`
      so fake synthetic prices are NEVER injected into a live trading log.
    - During pytest / offline testing: falls back to deterministic generator so tests work offline.
    """
    if allow_offline_fallback is None:
        allow_offline_fallback = bool(os.getenv("PYTEST_CURRENT_TEST"))

    feat_mod = _load_feature_module()
    timeframe_ms = 300_000 if timeframe == "5m" else 60_000

    for attempt in range(3):
        try:
            raw_candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw_candles or len(raw_candles) < 25:
                raise ValueError("Insufficient live candles returned.")
            df = pd.DataFrame(
                raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
            ).sort_values("timestamp").reset_index(drop=True)

            orderbook = exchange.fetch_order_book(symbol, limit=20)
            bids = orderbook.get("bids") or []
            asks = orderbook.get("asks") or []
            best_bid = float(bids[0][0]) if bids else float(df["close"].iloc[-1])
            best_ask = float(asks[0][0]) if asks else float(df["close"].iloc[-1])
            return df, {"bid": best_bid, "ask": best_ask, "bids": bids, "asks": asks, "feed": "LIVE"}
        except Exception:
            if attempt < 2 and not allow_offline_fallback:
                time.sleep(1.5 * (attempt + 1))

    if not allow_offline_fallback:
        return None, {"feed": "OFFLINE_SKIP"}

    full_df = feat_mod.generate_deterministic_ohlcv(n_candles=2000, timeframe_ms=timeframe_ms, seed=42)
    step_offset = int(time.time() // 60) % (2000 - limit - 1)
    df = full_df.iloc[step_offset : step_offset + limit].reset_index(drop=True)
    last_close = float(df["close"].iloc[-1])
    best_bid = round(last_close * 0.9999, 2)
    best_ask = round(last_close * 1.0001, 2)
    return df, {
        "bid": best_bid,
        "ask": best_ask,
        "bids": [[best_bid, 10.0]],
        "asks": [[best_ask, 10.0]],
        "feed": "OFFLINE_FALLBACK",
    }


def backfill_outage_candles(
    candles_df: pd.DataFrame,
    log_file: Path,
    symbol: str,
    mode_label: str,
    timeframe_ms: int = 300_000,
) -> int:
    """
    If Wi-Fi or power was lost while a position was open, inspect the historical candles fetched
    upon reconnection and log any intermediate candle extremes that occurred during the outage
    so TP/SL hits during a power/Wi-Fi cut are accurately captured.
    """
    if not log_file.exists():
        return 0
    try:
        sanitize_log_file(str(log_file))
        df = pd.read_csv(log_file)
        if df.empty:
            return 0
        last_candle_ts = int(df["candle_timestamp"].iloc[-1])
        missed = candles_df[candles_df["timestamp"] > last_candle_ts].iloc[:-1]
        if missed.empty or len(missed) < 2:
            return 0

        fieldnames = list(df.columns)
        backfilled = 0
        with log_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in missed.itertuples():
                ts_iso = datetime.fromtimestamp(int(row.timestamp) / 1000.0, tz=timezone.utc).isoformat()
                # Log both high and low extremes of missed outage candles so intrabar TP/SL triggers are caught
                for extreme_px in (float(row.high), float(row.low), float(row.close)):
                    writer.writerow(
                        {
                            "timestamp_utc": ts_iso,
                            "candle_timestamp": int(row.timestamp),
                            "symbol": symbol,
                            "mode": f"{mode_label}_BACKFILL",
                            "raw_signal": "HOLD",
                            "executed_action": "HOLD",
                            "confidence": 1.0,
                            "fill_price": round(extreme_px, 2),
                            "position_size_btc": 0.0,
                            "risk_usd": 0.0,
                        }
                    )
                backfilled += 1
            f.flush()
            os.fsync(f.fileno())
        if backfilled > 0:
            print(f"[RECOVERY] Backfilled {backfilled} missed candles from Wi-Fi/power outage gap.")
        return backfilled
    except Exception:
        return 0


def simulate_stateful_pnl(
    df: pd.DataFrame,
    starting_equity: float = 10_000.0,
    take_profit: float = DEFAULT_TAKE_PROFIT,
    stop_loss: float = DEFAULT_STOP_LOSS,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
) -> Dict[str, Any]:
    """
    Strict 1 : 1.5 R:R bracket trade simulator over log rows:
    - Opens at most one position at a time when flat (`pos_dir == 0`).
    - Locks in hard Take-Profit and Stop-Loss calibrated for 1:1.5 NET R:R after fees & slippage.
    - Holds position strictly until Take-Profit or Stop-Loss is hit.
    """
    pos_dir = 0.0
    entry_price = 0.0
    pos_size = 0.0
    entry_candle = -1
    entry_time_utc = ""

    gross_realized_pnl = 0.0
    total_fees_usd = 0.0
    total_slippage_usd = 0.0
    realized_pnl = 0.0
    closed_trades = 0
    winning_trades = 0
    trades_history: List[Dict[str, Any]] = []

    prev_price = None
    has_live_rows = any("OFFLINE" not in str(m) for m in df["mode"]) if "mode" in df.columns else True
    for row in df.itertuples():
        mode_str = str(getattr(row, "mode", ""))
        if has_live_rows and "OFFLINE" in mode_str:
            continue
        price = float(row.fill_price)
        candle_ts = int(getattr(row, "candle_timestamp", 0))
        ts_utc = str(getattr(row, "timestamp_utc", ""))
        action = str(row.executed_action)
        sig_dir = ACTION_TO_DIR.get(action, 0.0)
        size_btc = float(row.position_size_btc)

        if prev_price is not None and abs(price - prev_price) / max(prev_price, 1.0) > 0.025:
            continue
        prev_price = price

        if pos_dir != 0.0:
            raw_ret = pos_dir * ((price - entry_price) / entry_price)
            hit_tp = raw_ret >= take_profit
            hit_sl = raw_ret <= -stop_loss

            if hit_tp or hit_sl:
                capped_ret = take_profit if hit_tp else -stop_loss
                notional = pos_size * entry_price
                trade_gross = notional * capped_ret
                trade_fee = notional * (2.0 * fee_rate)
                trade_slip = notional * (2.0 * slippage_rate)
                net_trade_pnl = trade_gross - trade_fee - trade_slip

                gross_realized_pnl += trade_gross
                total_fees_usd += trade_fee
                total_slippage_usd += trade_slip
                realized_pnl += net_trade_pnl
                closed_trades += 1
                if net_trade_pnl > 0:
                    winning_trades += 1

                trades_history.append(
                    {
                        "side": "LONG" if pos_dir > 0 else "SHORT",
                        "entry_time": entry_time_utc,
                        "entry_price": round(entry_price, 2),
                        "exit_time": ts_utc,
                        "exit_price": round(price, 2),
                        "size_btc": round(pos_size, 6),
                        "notional_usd": round(notional, 2),
                        "leverage": round(notional / max(starting_equity, 1.0), 2),
                        "tp_price": round(entry_price * (1.0 + pos_dir * take_profit), 2),
                        "sl_price": round(entry_price * (1.0 - pos_dir * stop_loss), 2),
                        "status": "TP_HIT" if hit_tp else "SL_HIT",
                        "gross_pnl": round(trade_gross, 2),
                        "fees_slip": round(trade_fee + trade_slip, 2),
                        "net_pnl": round(net_trade_pnl, 2),
                    }
                )
                pos_dir = 0.0
                entry_price = 0.0
                pos_size = 0.0
                entry_time_utc = ""

        if pos_dir == 0.0 and sig_dir != 0.0 and candle_ts != entry_candle and size_btc > 0.0:
            pos_dir = sig_dir
            entry_price = price
            pos_size = size_btc
            entry_candle = candle_ts
            entry_time_utc = ts_utc

    unrealized_pnl = 0.0
    tp_price = 0.0
    sl_price = 0.0
    open_notional = 0.0
    open_leverage = 0.0
    if pos_dir != 0.0 and prev_price is not None and entry_price > 0:
        raw_ret = np.clip(pos_dir * ((prev_price - entry_price) / entry_price), -stop_loss, take_profit)
        notional = pos_size * entry_price
        open_notional = round(notional, 2)
        open_leverage = round(notional / max(starting_equity, 1.0), 2)
        open_fee_slip = float(notional * (fee_rate + slippage_rate))
        unrealized_pnl = float(notional * raw_ret - open_fee_slip)
        tp_price = entry_price * (1.0 + pos_dir * take_profit)
        sl_price = entry_price * (1.0 - pos_dir * stop_loss)
        trades_history.append(
            {
                "side": "LONG" if pos_dir > 0 else "SHORT",
                "entry_time": entry_time_utc,
                "entry_price": round(entry_price, 2),
                "exit_time": None,
                "exit_price": round(prev_price, 2),
                "size_btc": round(pos_size, 6),
                "notional_usd": open_notional,
                "leverage": open_leverage,
                "tp_price": round(tp_price, 2),
                "sl_price": round(sl_price, 2),
                "status": "OPEN",
                "gross_pnl": round(float(notional * raw_ret), 2),
                "fees_slip": round(open_fee_slip, 2),
                "net_pnl": round(unrealized_pnl, 2),
            }
        )

    total_pnl = realized_pnl + unrealized_pnl
    return {
        "closed_trades": closed_trades,
        "winning_trades": winning_trades,
        "open_position": "LONG" if pos_dir > 0 else ("SHORT" if pos_dir < 0 else "FLAT"),
        "entry_price": entry_price,
        "entry_time": entry_time_utc,
        "open_size_btc": round(pos_size, 6),
        "open_notional_usd": open_notional,
        "open_leverage": open_leverage,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "gross_realized_pnl": gross_realized_pnl,
        "total_fees_usd": total_fees_usd,
        "total_slippage_usd": total_slippage_usd,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "current_equity": starting_equity + total_pnl,
        "net_ret_pct": (total_pnl / starting_equity) * 100.0,
        "trades_history": trades_history,
    }


def print_status_summary(
    log_path: str = "data/paper_trades_log.csv",
    starting_equity: float = 10_000.0,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
) -> None:
    """Calculate and display side-by-side PnL for Wide Swing (1:1.5) vs Tight Intraday (1:1.5) accounts."""
    log_file = Path(log_path)
    if not log_file.exists():
        print("No paper trade log found yet at:", log_file)
        return

    sanitize_log_file(log_path)
    df = pd.read_csv(log_file)
    if df.empty:
        print("Paper trade log is currently empty.")
        return

    wide = simulate_stateful_pnl(
        df,
        starting_equity=starting_equity,
        take_profit=DEFAULT_TAKE_PROFIT,
        stop_loss=DEFAULT_STOP_LOSS,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    tight = simulate_stateful_pnl(
        df,
        starting_equity=starting_equity,
        take_profit=TIGHT_TAKE_PROFIT,
        stop_loss=TIGHT_STOP_LOSS,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    last = df.iloc[-1]

    print(
        f"Total Ticks Logged: {len(df)} | Latest BTC Mark: ${float(last['fill_price']):,.2f} "
        f"| Signal: {last['executed_action']} (conf={float(last['confidence']):.2f})"
    )
    print("-" * 92)
    for label, tp_pct, sl_pct, s in [
        ("ACCOUNT A — WIDE SWING (Net 1:1.5 R:R | -$100 Net SL / +$150 Net TP)", DEFAULT_TAKE_PROFIT * 100, DEFAULT_STOP_LOSS * 100, wide),
        ("ACCOUNT B — TIGHT INTRADAY (Net 1:1.5 R:R | -$30 Net SL / +$45 Net TP)", TIGHT_TAKE_PROFIT * 100, TIGHT_STOP_LOSS * 100, tight),
    ]:
        wr = (s["winning_trades"] / s["closed_trades"] * 100.0) if s["closed_trades"] > 0 else 0.0
        pos_str = (
            f"{s['open_position']} | Size: {s['open_size_btc']:.4f} BTC (${s['open_notional_usd']:,.2f} @ {s['open_leverage']:.2f}x lev) | "
            f"Entry @ ${s['entry_price']:,.2f} -> TP @ ${s['tp_price']:,.2f} [+{tp_pct:.2f}%] / SL @ ${s['sl_price']:,.2f} [-{sl_pct:.2f}%]"
            if s["open_position"] != "FLAT"
            else "FLAT (0.0000 BTC | 0.00x lev)"
        )
        print(f"[{label}]")
        print(f"  Open Position: {pos_str}")
        print(
            f"  Closed Trades: {s['closed_trades']} (Win Rate: {wr:.1f}%) | "
            f"Gross=${s['gross_realized_pnl']:+,.2f} | Fees=-${s['total_fees_usd']:,.2f} | Slip=-${s['total_slippage_usd']:,.2f}"
        )
        print(
            f"  Account PnL:   ${s['total_pnl']:+,.2f} ({s['net_ret_pct']:+.2f}%) | "
            f"Equity: ${s['current_equity']:,.2f} "
            f"[Realized: ${s['realized_pnl']:+,.2f}, Unrealized: ${s['unrealized_pnl']:+,.2f}]"
        )
        print("-" * 92)


def evaluate_and_log_step(
    model_path: str = "models/btc_mimic_model.pkl",
    log_path: str = "data/paper_trades_log.csv",
    exchange_id: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    account_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
    stop_loss_pct: float = DEFAULT_STOP_LOSS,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT,
    confidence_threshold: float = 0.60,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    dry_run: bool = True,
) -> Optional[Dict[str, Any]]:
    feat_mod = _load_feature_module()
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]
    class_to_label = artifact.get("class_to_label", CLASS_TO_LABEL)

    exchange = init_exchange(exchange_id=exchange_id, dry_run=dry_run)
    candles_df, book = fetch_latest_candles_and_book(
        exchange=exchange, symbol=symbol, timeframe=timeframe, limit=50
    )

    if candles_df is None or book.get("feed") == "OFFLINE_SKIP":
        print(
            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] [NETWORK_OFFLINE] "
            "Wi-Fi / Exchange unreachable after 3 retries — preserving open positions and waiting for reconnect."
        )
        return None

    log_file = Path(log_path)
    mode_label = "DRY_RUN" if dry_run else "TESTNET_PAPER"
    if book.get("feed") == "OFFLINE_FALLBACK":
        mode_label = f"{mode_label}_OFFLINE"

    # Backfill any missed candles if recovering from a Wi-Fi or power outage
    if book.get("feed") == "LIVE":
        backfill_outage_candles(candles_df, log_file, symbol=symbol, mode_label=mode_label)

    feat_df = feat_mod.compute_features(candles_df).dropna(subset=feature_cols).reset_index(drop=True)
    latest_row = feat_df.iloc[[-1]][feature_cols]

    probs = model.predict_proba(latest_row)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    raw_signal = int(class_to_label[pred_class])

    executed_action = raw_signal if confidence > confidence_threshold else 0
    action_str = ACTION_NAMES[executed_action]

    # Position notional is $10,000 (1.00x leverage)
    position_notional_usd = account_equity if executed_action != 0 else 0.0
    risk_usd = account_equity * risk_per_trade
    ref_price = book["ask"] if executed_action == 1 else (book["bid"] if executed_action == -1 else float(candles_df["close"].iloc[-1]))
    prelim_size_btc = (position_notional_usd / ref_price) if ref_price > 0 else 0.0

    if executed_action == 1:
        vwap_px = compute_orderbook_vwap(book.get("asks", []), prelim_size_btc, book["ask"])
        fill_price = vwap_px * (1.0 + slippage_rate)
    elif executed_action == -1:
        vwap_px = compute_orderbook_vwap(book.get("bids", []), prelim_size_btc, book["bid"])
        fill_price = vwap_px * (1.0 - slippage_rate)
    else:
        fill_price = ref_price

    position_size_btc = round(position_notional_usd / fill_price, 6) if (executed_action != 0 and fill_price > 0) else 0.0
    est_fee_usd = round(position_notional_usd * fee_rate, 2) if executed_action != 0 else 0.0
    est_slip_usd = round(position_notional_usd * slippage_rate, 2) if executed_action != 0 else 0.0

    prior_open = "FLAT"
    if log_file.exists():
        try:
            sanitize_log_file(str(log_file))
            prior_df = pd.read_csv(log_file)
            if not prior_df.empty:
                prior_stats = simulate_stateful_pnl(
                    prior_df, starting_equity=account_equity, take_profit=take_profit_pct,
                    stop_loss=stop_loss_pct, fee_rate=fee_rate, slippage_rate=slippage_rate
                )
                prior_open = prior_stats["open_position"]
        except Exception:
            prior_open = "FLAT"

    if not dry_run and executed_action != 0 and prior_open == "FLAT" and exchange.apiKey:
        side = "buy" if executed_action == 1 else "sell"
        try:
            exchange.create_market_order(symbol, side, position_size_btc)
        except Exception:
            pass

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candle_timestamp": int(candles_df["timestamp"].iloc[-1]),
        "symbol": symbol,
        "mode": mode_label,
        "raw_signal": ACTION_NAMES[raw_signal],
        "executed_action": action_str,
        "confidence": round(confidence, 4),
        "fill_price": round(fill_price, 2),
        "position_size_btc": position_size_btc,
        "risk_usd": round(risk_usd if executed_action != 0 else 0.0, 2),
    }

    log_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
        f.flush()
        os.fsync(f.fileno())

    if prior_open != "FLAT":
        print(
            f"[{record['mode']}] {record['symbol']} | Signal={record['raw_signal']} "
            f"(conf={record['confidence']:.2f}) | Holding {prior_open} bracket @ mark ${record['fill_price']:,.2f}"
        )
    else:
        print(
            f"[{record['mode']}] {record['symbol']} | Signal={record['raw_signal']} "
            f"(conf={record['confidence']:.2f}) -> Executed={record['executed_action']} "
            f"@ {record['fill_price']} | Size={record['position_size_btc']} BTC "
            f"(Fee=${est_fee_usd:.2f}, Slip=${est_slip_usd:.2f})"
        )
    return record


def main():
    parser = argparse.ArgumentParser(description="Live Paper Trading Bot for BTC Behavioral Cloning.")
    parser.add_argument("--model", type=str, default="models/btc_mimic_model.pkl")
    parser.add_argument("--log", type=str, default="data/paper_trades_log.csv")
    parser.add_argument("--exchange", type=str, default="binance", choices=["binance", "bybit"])
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="5m", choices=["1m", "5m"])
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Taker fee per side (default 0.0005 = 0.05%).")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Adverse slippage per side (default 0.0002 = 0.02%).")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="Simulate paper fills against public orderbooks.")
    parser.add_argument("--once", action="store_true", help="Execute a single polling cycle and exit.")
    parser.add_argument("--status", action="store_true", help="Print current paper trading PnL and status summary.")
    parser.add_argument("--reset", action="store_true", help="Reset paper trading log and restore equity to starting balance.")
    args = parser.parse_args()

    if args.reset:
        log_file = Path(args.log)
        if log_file.exists():
            log_file.unlink()
        print(f"Paper trading account reset. Cleared {log_file} | Starting Equity: ${args.equity:,.2f}")
        return

    if args.status:
        print_status_summary(
            log_path=args.log,
            starting_equity=args.equity,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        return

    while True:
        try:
            evaluate_and_log_step(
                model_path=args.model,
                log_path=args.log,
                exchange_id=args.exchange,
                symbol=args.symbol,
                timeframe=args.timeframe,
                account_equity=args.equity,
                confidence_threshold=args.threshold,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"[RECOVERY] Unexpected polling error ({exc}); retrying in {args.interval}s...")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
