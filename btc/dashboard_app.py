import argparse
import importlib.util
import json
import re
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
LOG_PATH = ROOT_DIR / "data" / "paper_trades_log.csv"
PUBLIC_TUNNEL_URL: Optional[str] = None


def _load_bot_module():
    spec = importlib.util.spec_from_file_location(
        "live_paper_bot", ROOT_DIR / "3_live_paper_bot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _format_account_stats(stats: Dict[str, Any], tp_pct: float, sl_pct: float) -> Dict[str, Any]:
    win_rate = (stats["winning_trades"] / stats["closed_trades"] * 100.0) if stats["closed_trades"] > 0 else 0.0
    return {
        "tp_pct": round(tp_pct * 100.0, 2),
        "sl_pct": round(sl_pct * 100.0, 2),
        "current_equity": round(float(stats["current_equity"]), 2),
        "total_pnl": round(float(stats["total_pnl"]), 2),
        "net_ret_pct": round(float(stats["net_ret_pct"]), 2),
        "realized_pnl": round(float(stats["realized_pnl"]), 2),
        "unrealized_pnl": round(float(stats["unrealized_pnl"]), 2),
        "gross_realized_pnl": round(float(stats["gross_realized_pnl"]), 2),
        "total_fees_usd": round(float(stats["total_fees_usd"]), 2),
        "total_slippage_usd": round(float(stats["total_slippage_usd"]), 2),
        "closed_trades": int(stats["closed_trades"]),
        "win_rate_pct": round(float(win_rate), 1),
        "open_position": stats["open_position"],
        "entry_price": round(float(stats["entry_price"]), 2),
        "open_size_btc": round(float(stats.get("open_size_btc", 0.0)), 6),
        "open_notional_usd": round(float(stats.get("open_notional_usd", 0.0)), 2),
        "open_leverage": round(float(stats.get("open_leverage", 0.0)), 2),
        "tp_price": round(float(stats["tp_price"]), 2),
        "sl_price": round(float(stats["sl_price"]), 2),
        "trades": list(reversed(stats.get("trades_history", []))),
    }


def build_dashboard_payload(starting_equity: float = 10_000.0) -> Dict[str, Any]:
    bot_mod = _load_bot_module()
    if not LOG_PATH.exists():
        return {"has_data": False, "starting_equity": starting_equity, "series": [], "recent_ticks": []}

    df = pd.read_csv(LOG_PATH)
    if df.empty:
        return {"has_data": False, "starting_equity": starting_equity, "series": [], "recent_ticks": []}

    wide_raw = bot_mod.simulate_stateful_pnl(
        df,
        starting_equity=starting_equity,
        take_profit=bot_mod.DEFAULT_TAKE_PROFIT,
        stop_loss=bot_mod.DEFAULT_STOP_LOSS,
    )
    tight_raw = bot_mod.simulate_stateful_pnl(
        df,
        starting_equity=starting_equity,
        take_profit=bot_mod.TIGHT_TAKE_PROFIT,
        stop_loss=bot_mod.TIGHT_STOP_LOSS,
    )

    wide_acc = _format_account_stats(wide_raw, bot_mod.DEFAULT_TAKE_PROFIT, bot_mod.DEFAULT_STOP_LOSS)
    tight_acc = _format_account_stats(tight_raw, bot_mod.TIGHT_TAKE_PROFIT, bot_mod.TIGHT_STOP_LOSS)

    if "mode" in df.columns and any("OFFLINE" not in str(m) for m in df["mode"]):
        df = df[~df["mode"].astype(str).str.contains("OFFLINE")].reset_index(drop=True)

    prices = df["fill_price"].astype(float).to_numpy()
    keep_mask = []
    last_valid_px = None
    for px in prices:
        if last_valid_px is None or abs(px - last_valid_px) / max(last_valid_px, 1.0) <= 0.025:
            keep_mask.append(True)
            last_valid_px = px
        else:
            keep_mask.append(False)
    clean_df = df[keep_mask].tail(500).reset_index(drop=True)

    series = [
        {
            "t": str(row.timestamp_utc),
            "p": round(float(row.fill_price), 2),
            "sig": str(row.executed_action),
            "conf": round(float(row.confidence), 4),
        }
        for row in clean_df.itertuples()
    ]

    recent_ticks = [
        {
            "timestamp_utc": str(row.timestamp_utc),
            "raw_signal": str(row.raw_signal),
            "executed_action": str(row.executed_action),
            "confidence": round(float(row.confidence), 4),
            "fill_price": round(float(row.fill_price), 2),
            "position_size_btc": float(row.position_size_btc),
        }
        for row in clean_df.tail(20).iloc[::-1].itertuples()
    ]

    last = clean_df.iloc[-1]
    return {
        "has_data": True,
        "total_ticks": int(len(df)),
        "starting_equity": starting_equity,
        "latest_price": round(float(last["fill_price"]), 2),
        "latest_signal": str(last["executed_action"]),
        "latest_confidence": round(float(last["confidence"]), 4),
        "wide": wide_acc,
        "tight": tight_acc,
        "series": series,
        "recent_ticks": recent_ticks,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
<meta name="theme-color" content="#0b0f17" />
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
<title>BTC Paper Bot — Live Terminal & Trading Academy</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131a26;
    --panel-alt: #182232;
    --border: #222d40;
    --text: #e5e9f0;
    --muted: #8b9bb4;
    --green: #10b981;
    --red: #ef4444;
    --amber: #f59e0b;
    --blue: #3b82f6;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    padding: 16px 20px 40px;
    max-width: 1600px;
    margin: 0 auto;
    line-height: 1.5;
  }

  /* Top Page Navigation Tabs */
  .page-nav {
    display: flex;
    gap: 10px;
    margin-bottom: 16px;
    background: var(--panel);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid var(--border);
  }
  .page-nav button {
    flex: 1;
    padding: 10px 16px;
    font-size: 13.5px;
    font-weight: 700;
    border-radius: 7px;
    border: none;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  .page-nav button.active {
    background: var(--blue);
    color: #fff;
    box-shadow: 0 2px 10px rgba(59,130,246,0.35);
  }

  /* Sticky mobile quick-profit bar */
  .mobile-sticky-bar {
    display: none;
    position: sticky;
    top: 0;
    z-index: 50;
    background: rgba(11, 15, 23, 0.94);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    margin: -12px -12px 12px -12px;
    padding: 10px 12px;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .m-pill {
    display: flex;
    flex-direction: column;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 9px;
    flex: 1;
    min-width: 0;
  }
  .m-pill-lbl { font-size: 9px; color: var(--muted); text-transform: uppercase; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .m-pill-val { font-size: 13px; font-weight: 800; margin-top: 2px; white-space: nowrap; }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    gap: 10px;
  }
  .title-wrap h1 { font-size: 19px; font-weight: 700; letter-spacing: 0.2px; }
  .title-wrap p { font-size: 12px; color: var(--muted); margin-top: 4px; line-height: 1.35; }
  .controls {
    display: flex;
    gap: 7px;
    align-items: center;
    overflow-x: auto;
    padding-bottom: 2px;
    max-width: 100%;
    scrollbar-width: none;
  }
  .controls::-webkit-scrollbar { display: none; }
  button {
    background: #1e293b;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
    transition: all 0.15s;
  }
  button:hover { background: #2d3d56; }
  button.active { border-color: var(--blue); background: rgba(59,130,246,0.22); color: #93c5fd; }
  button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
  button.danger { background: rgba(239,68,68,0.15); border-color: var(--red); color: #fca5a5; }

  .top-bar {
    display: grid;
    grid-template-columns: 245px 1fr 1fr;
    gap: 14px;
    margin-bottom: 16px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    min-width: 0;
  }
  .card-wide { border-top: 3px solid var(--amber); }
  .card-tight { border-top: 3px solid var(--purple); }
  .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; gap: 6px; }
  .card-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); font-weight: 700; }
  .card-value-row { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-top: 2px; }
  .card-value { font-size: 23px; font-weight: 800; }
  .card-pnl-pill { font-size: 14px; font-weight: 700; padding: 3px 8px; border-radius: 6px; }
  .pnl-pos { background: rgba(16,185,129,0.15); color: var(--green); }
  .pnl-neg { background: rgba(239,68,68,0.15); color: var(--red); }
  .card-sub { font-size: 12px; color: var(--muted); margin-top: 6px; line-height: 1.45; word-break: break-word; }

  .pos-long { color: var(--green); }
  .pos-short { color: var(--red); }
  .pos-flat { color: var(--muted); }

  .chart-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 16px;
    position: relative;
  }
  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 8px;
  }
  .legend { display: flex; gap: 12px; font-size: 11px; color: var(--muted); align-items: center; flex-wrap: wrap; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; }
  #scrub-info {
    font-size: 11px;
    color: #93c5fd;
    font-family: monospace;
    min-height: 16px;
    margin-bottom: 4px;
  }
  canvas { width: 100%; height: 380px; display: block; cursor: crosshair; touch-action: pan-y; }

  .mobile-tabs {
    display: none;
    gap: 6px;
    margin-bottom: 12px;
  }
  .mobile-tabs button { flex: 1; text-align: center; padding: 10px 6px; font-size: 12px; }

  .tables-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  .table-scroll {
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  table { width: 100%; border-collapse: collapse; font-size: 12px; min-width: 520px; }
  th, td {
    text-align: left;
    padding: 8px 8px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  th { color: var(--muted); font-weight: 600; font-size: 10.5px; text-transform: uppercase; }
  .badge {
    padding: 2px 7px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 11px;
    display: inline-block;
  }
  .badge-long { background: rgba(16,185,129,0.18); color: var(--green); }
  .badge-short { background: rgba(239,68,68,0.18); color: var(--red); }
  .badge-hold { background: rgba(139,155,180,0.15); color: var(--muted); }
  .badge-open { background: rgba(59,130,246,0.2); color: #60a5fa; }
  .badge-lev { background: rgba(245,158,11,0.16); color: #fbbf24; font-family: monospace; }
  .badge-lead { background: rgba(16,185,129,0.22); color: #34d399; font-size: 10px; padding: 2px 6px; border-radius: 4px; flex-shrink: 0; }

  /* GUIDE & ACADEMY PAGE STYLES */
  .guide-section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 20px;
  }
  .guide-section h2 {
    font-size: 18px;
    font-weight: 800;
    color: #fff;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .guide-section h3 {
    font-size: 14.5px;
    font-weight: 700;
    color: #93c5fd;
    margin: 16px 0 8px;
  }
  .guide-section p {
    font-size: 13.5px;
    color: #cbd5e1;
    margin-bottom: 10px;
  }
  .guide-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-top: 14px;
  }
  .visual-box {
    background: #0d131f;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    overflow: hidden;
  }
  .visual-box h4 {
    font-size: 13.5px;
    font-weight: 700;
    margin-bottom: 6px;
  }
  .visual-box svg {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 6px;
    background: #090d14;
    border: 1px solid #1a2436;
    margin: 10px 0;
  }
  .checklist {
    list-style: none;
    margin-top: 8px;
  }
  .checklist li {
    font-size: 12.5px;
    color: #cbd5e1;
    padding: 5px 0;
    border-bottom: 1px dashed #1e293b;
  }
  .checklist li strong { color: #fff; }
  .callout {
    background: rgba(59, 130, 246, 0.1);
    border-left: 4px solid var(--blue);
    padding: 12px 14px;
    border-radius: 6px;
    font-size: 13px;
    color: #dbeafe;
    margin: 12px 0;
  }
  .callout.warn {
    background: rgba(245, 158, 11, 0.1);
    border-left-color: var(--amber);
    color: #fef3c7;
  }

  @media (max-width: 1100px) {
    .top-bar { grid-template-columns: 1fr; }
    .tables-grid { grid-template-columns: 1fr; }
    .guide-grid-2 { grid-template-columns: 1fr; }
  }
  @media (max-width: 768px) {
    body { padding: 12px 12px 32px; }
    .mobile-sticky-bar { display: flex; }
    .title-wrap h1 { font-size: 16px; }
    .title-wrap p { font-size: 11px; }
    .card { padding: 12px 13px; }
    .card-value { font-size: 20px; }
    canvas { height: 270px; }
    .mobile-tabs { display: flex; }
    .mobile-hidden { display: none !important; }
    .guide-section { padding: 15px 14px; }
  }
</style>
</head>
<body>

  <!-- Sticky Mobile Top Bar -->
  <div class="mobile-sticky-bar">
    <div class="m-pill">
      <span class="m-pill-lbl">BTC/USDT</span>
      <span class="m-pill-val" id="m-btc">$--</span>
    </div>
    <div class="m-pill" style="border-top: 2px solid var(--amber);">
      <span class="m-pill-lbl">Acct A (Wide)</span>
      <span class="m-pill-val" id="m-wide-pnl">$0.00</span>
    </div>
    <div class="m-pill" style="border-top: 2px solid var(--purple);">
      <span class="m-pill-lbl">Acct B (Tight)</span>
      <span class="m-pill-val" id="m-tight-pnl">$0.00</span>
    </div>
  </div>

  <!-- Top Navigation Switcher: Live Terminal vs Trading Academy & Guide -->
  <div class="page-nav">
    <button id="nav-terminal" class="active" onclick="switchPage('terminal')">📊 Live Trading Monitor</button>
    <button id="nav-guide" onclick="switchPage('guide')">📖 Bot Guide & Trading Academy</button>
  </div>

  <!-- ================= PAGE 1: LIVE TRADING MONITOR ================= -->
  <div id="page-terminal">
    <div class="header">
      <div class="title-wrap">
        <h1>BTC/USDT Paper Bot — Dual Net 1:1.5 R:R Monitor</h1>
        <p>Acct A (Wide Net 1:1.5: -$100 SL / +$150 TP) vs Acct B (Tight Net 1:1.5: -$30 SL / +$45 TP) • $10k Notional (1.0x Lev)</p>
      </div>
      <div class="controls">
        <button id="mode-both" class="active" onclick="setOverlayMode('both')">Chart: Both</button>
        <button id="mode-wide" onclick="setOverlayMode('wide')">Acct A (Wide)</button>
        <button id="mode-tight" onclick="setOverlayMode('tight')">Acct B (Tight)</button>
        <button onclick="setZoom(80)">80m</button>
        <button onclick="setZoom(250)">250m</button>
        <button onclick="setZoom(500)">All</button>
        <button class="primary" onclick="triggerTick()">Poll Now</button>
      </div>
    </div>

    <div class="top-bar">
      <div class="card">
        <div class="card-label">Live BTC/USDT Feed</div>
        <div class="card-value" id="kpi-price" style="margin-top:4px;">$--</div>
        <div class="card-sub" id="kpi-signal">Latest Signal: --</div>
        <div class="card-sub" id="kpi-sizing">Sizing: $10,000 Notional (1.00x Lev)</div>
        <div class="card-sub" id="kpi-ticks">Total Ticks: --</div>
      </div>

      <div class="card card-wide">
        <div class="card-header">
          <span class="card-label" style="color:#f59e0b">Account A — Wide Swing (Net 1:1.5 | -$100 Net SL / +$150 Net TP)</span>
          <span id="lead-wide" class="badge-lead" style="display:none">LEADING</span>
        </div>
        <div class="card-value-row">
          <div class="card-value" id="wide-equity">$10,000.00</div>
          <div class="card-pnl-pill pnl-pos" id="wide-pnl-pill">+$0.00 (0.00%)</div>
        </div>
        <div class="card-sub" id="wide-pnl">Realized: $0.00 | Unrealized: $0.00</div>
        <div class="card-sub" id="wide-pos">Open: FLAT</div>
        <div class="card-sub" id="wide-lev">Size & Leverage: --</div>
        <div class="card-sub" id="wide-stats">Closed Trades: 0 (0.0% Win) | Fees: $0.00 | Slip: $0.00</div>
      </div>

      <div class="card card-tight">
        <div class="card-header">
          <span class="card-label" style="color:#c084fc">Account B — Tight Intraday (Net 1:1.5 | -$30 Net SL / +$45 Net TP)</span>
          <span id="lead-tight" class="badge-lead" style="display:none">LEADING</span>
        </div>
        <div class="card-value-row">
          <div class="card-value" id="tight-equity">$10,000.00</div>
          <div class="card-pnl-pill pnl-pos" id="tight-pnl-pill">+$0.00 (0.00%)</div>
        </div>
        <div class="card-sub" id="tight-pnl">Realized: $0.00 | Unrealized: $0.00</div>
        <div class="card-sub" id="tight-pos">Open: FLAT</div>
        <div class="card-sub" id="tight-lev">Size & Leverage: --</div>
        <div class="card-sub" id="tight-stats">Closed Trades: 0 (0.0% Win) | Fees: $0.00 | Slip: $0.00</div>
      </div>
    </div>

    <div class="chart-panel">
      <div class="chart-header">
        <strong style="font-size:13.5px;">Live BTC/USDT Price & Positions (Touch/Hover to Inspect)</strong>
        <div class="legend">
          <span><i class="dot" style="background:#3b82f6"></i>BTC Price</span>
          <span><i class="dot" style="background:#f59e0b"></i>Acct A (Wide)</span>
          <span><i class="dot" style="background:#a855f7"></i>Acct B (Tight)</span>
          <span><i class="dot" style="background:#10b981"></i>TP Exit</span>
          <span><i class="dot" style="background:#ef4444"></i>SL Exit</span>
        </div>
      </div>
      <div id="scrub-info">Tap or drag across the chart to inspect price & signal history</div>
      <canvas id="btcChart"></canvas>
    </div>

    <div class="mobile-tabs">
      <button id="tab-btn-wide" class="active" onclick="setMobileTab('wide')">Acct A Trades</button>
      <button id="tab-btn-tight" onclick="setMobileTab('tight')">Acct B Trades</button>
      <button id="tab-btn-ticks" onclick="setMobileTab('ticks')">Live 60s Polls</button>
    </div>

    <div class="tables-grid">
      <div class="card card-wide" id="panel-wide">
        <strong style="font-size:13px; display:block; margin-bottom:10px; color:#f59e0b;">Account A Positions — Wide Swing (Net -$100 SL / +$150 TP)</strong>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Side</th>
                <th>Time</th>
                <th>Size (BTC / $)</th>
                <th>Lev</th>
                <th>Entry Px</th>
                <th>TP / SL</th>
                <th>Status</th>
                <th>Net PnL</th>
              </tr>
            </thead>
            <tbody id="wide-trades-tbody"></tbody>
          </table>
        </div>
      </div>

      <div class="card card-tight" id="panel-tight">
        <strong style="font-size:13px; display:block; margin-bottom:10px; color:#c084fc;">Account B Positions — Tight Intraday (Net -$30 SL / +$45 TP)</strong>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Side</th>
                <th>Time</th>
                <th>Size (BTC / $)</th>
                <th>Lev</th>
                <th>Entry Px</th>
                <th>TP / SL</th>
                <th>Status</th>
                <th>Net PnL</th>
              </tr>
            </thead>
            <tbody id="tight-trades-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card" id="panel-ticks">
      <strong style="font-size:13px; display:block; margin-bottom:10px;">Recent 60s Model Polling Ticks (Raw Market Scan)</strong>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Time (UTC)</th>
              <th>Model Signal</th>
              <th>Confidence</th>
              <th>BTC Mark Price</th>
              <th>Target Size (BTC)</th>
              <th>Target Notional ($)</th>
              <th>Leverage</th>
            </tr>
          </thead>
          <tbody id="ticks-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ================= PAGE 2: BOT GUIDE & TRADING ACADEMY ================= -->
  <div id="page-guide" style="display:none;">

    <!-- Section 1: VisualLONG vs SHORT Setups -->
    <div class="guide-section">
      <h2>🎯 1. What to Look For: LONG vs. SHORT Setups (With Visual Diagrams)</h2>
      <p>
        Your bot trades <strong>5-minute BTC/USDT Perpetual Futures</strong> using a combination of <strong>Mean-Reversion (buying oversold pullbacks / selling overbought spikes)</strong> and <strong>Momentum Confirmation</strong>. Below are the exact visual patterns the bot looks for when entering a <strong>LONG</strong> or <strong>SHORT</strong> trade.
      </p>

      <div class="guide-grid-2">
        <!-- VISUAL 1: LONG SETUP SVG -->
        <div class="visual-box" style="border-top: 3px solid var(--green);">
          <h4 class="pos-long">▲ Anatomy of a High-Confidence LONG Setup (Buy Dip & Reclaim)</h4>
          <p style="font-size:12px; color:var(--muted);">Price stretches below the 20-EMA ("rubber band" pullback), RSI dips into oversold (&lt;38–48), selling exhausts on volume, and price turns back up.</p>
          <svg viewBox="0 0 560 320" xmlns="http://www.w3.org/2000/svg">
            <!-- Grid -->
            <line x1="30" y1="45" x2="530" y2="45" stroke="#182336" stroke-width="1"/>
            <line x1="30" y1="105" x2="530" y2="105" stroke="#182336" stroke-width="1"/>
            <line x1="30" y1="165" x2="530" y2="165" stroke="#182336" stroke-width="1"/>
            <line x1="30" y1="225" x2="530" y2="225" stroke="#182336" stroke-width="1"/>

            <!-- Shaded TP & SL Zones -->
            <rect x="240" y="32" width="280" height="143" fill="rgba(16,185,129,0.09)" />
            <rect x="240" y="175" width="280" height="75" fill="rgba(239,68,68,0.09)" />

            <!-- TP, Entry, SL Dashed Lines -->
            <line x1="220" y1="32" x2="525" y2="32" stroke="#10b981" stroke-width="1.8" stroke-dasharray="6,4"/>
            <text x="250" y="26" fill="#10b981" font-family="monospace" font-size="11" font-weight="bold">TAKE-PROFIT (+1.5R): +$150 Net (Acct A) / +$45 Net (Acct B)</text>

            <line x1="220" y1="175" x2="525" y2="175" stroke="#f59e0b" stroke-width="1.8" stroke-dasharray="4,3"/>
            <text x="335" y="170" fill="#f59e0b" font-family="monospace" font-size="11" font-weight="bold">LONG ENTRY @ Ask + 0.02% Slip</text>

            <line x1="220" y1="250" x2="525" y2="250" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="6,4"/>
            <text x="250" y="264" fill="#ef4444" font-family="monospace" font-size="11" font-weight="bold">STOP-LOSS (-1.0R): -$100 Net (Acct A) / -$30 Net (Acct B)</text>

            <!-- 20 EMA Curve (Blue) -->
            <path d="M 40 95 Q 140 120, 230 140 T 490 55" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
            <text x="45" y="85" fill="#60a5fa" font-family="sans-serif" font-size="11" font-weight="bold">20-Candle EMA (Fair Value)</text>

            <!-- Candlesticks: Pullback then Reversal -->
            <!-- Red candles dropping below EMA -->
            <line x1="70" y1="90" x2="70" y2="130" stroke="#ef4444" stroke-width="1.5"/><rect x="64" y="98" width="12" height="24" fill="#ef4444"/>
            <line x1="110" y1="115" x2="110" y2="160" stroke="#ef4444" stroke-width="1.5"/><rect x="104" y="122" width="12" height="30" fill="#ef4444"/>
            <line x1="150" y1="145" x2="150" y2="195" stroke="#ef4444" stroke-width="1.5"/><rect x="144" y="152" width="12" height="34" fill="#ef4444"/>
            <!-- Hammer exhaustion candle stretched below EMA -->
            <line x1="190" y1="175" x2="190" y2="232" stroke="#10b981" stroke-width="1.8"/><rect x="184" y="180" width="12" height="14" fill="#10b981"/>
            <!-- Double arrow showing stretch dist_ema20 -->
            <line x1="190" y1="136" x2="190" y2="175" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="2,2"/>
            <text x="80" y="218" fill="#fbbf24" font-family="monospace" font-size="10">Stretch below EMA (dist_ema20 &lt; 0)</text>

            <!-- Trigger Green Candle (Entry) -->
            <line x1="230" y1="155" x2="230" y2="195" stroke="#10b981" stroke-width="1.8"/><rect x="224" y="162" width="12" height="24" fill="#10b981"/>
            <polygon points="230,198 221,214 239,214" fill="#10b981"/>
            <text x="200" y="228" fill="#10b981" font-family="sans-serif" font-size="11" font-weight="bold">▲ BOT BUYS (Conf &gt; 60%)</text>

            <!-- Follow-through Green Candles to TP -->
            <line x1="285" y1="125" x2="285" y2="168" stroke="#10b981" stroke-width="1.5"/><rect x="279" y="132" width="12" height="28" fill="#10b981"/>
            <line x1="340" y1="95" x2="340" y2="140" stroke="#10b981" stroke-width="1.5"/><rect x="334" y="102" width="12" height="30" fill="#10b981"/>
            <line x1="395" y1="68" x2="395" y2="112" stroke="#10b981" stroke-width="1.5"/><rect x="389" y="75" width="12" height="28" fill="#10b981"/>
            <line x1="450" y1="25" x2="450" y2="80" stroke="#10b981" stroke-width="1.8"/><rect x="444" y="32" width="12" height="42" fill="#10b981"/>
            <circle cx="450" cy="32" r="7" fill="none" stroke="#10b981" stroke-width="2.5"/>
            <text x="462" y="48" fill="#10b981" font-family="sans-serif" font-size="11" font-weight="bold">★ TP HIT</text>

            <!-- Bottom Sub-strip: RSI & RVOL -->
            <rect x="30" y="278" width="500" height="32" rx="4" fill="#111827" stroke="#1f2937"/>
            <text x="42" y="298" fill="#34d399" font-family="monospace" font-size="11">Signals at Entry: ret_5 &lt; -q25 | dist_ema20 &lt; -q25 | RSI_14 = 36 (Oversold) | RVOL = 1.4x</text>
          </svg>
          <ul class="checklist">
            <li><strong>1. EMA Rubber-Band Stretch (`dist_ema20 &lt; 0`):</strong> Price drops sharply below the 20-period EMA, creating a discount relative to recent fair value.</li>
            <li><strong>2. Oversold RSI (`rsi_14 &lt; 48` or `&lt; 38`):</strong> Sellers have pushed momentum to a short-term extreme where bears begin taking profit.</li>
            <li><strong>3. 1:1.5 Net Bracket Locked:</strong> Enters at Ask + Slippage (`0.133 BTC` / `$10k`) and holds until `+1.5R` TP or `-1.0R` SL.</li>
          </ul>
        </div>

        <!-- VISUAL 2: SHORT SETUP SVG -->
        <div class="visual-box" style="border-top: 3px solid var(--red);">
          <h4 class="pos-short">▼ Anatomy of a High-Confidence SHORT Setup (Fade Rip & Breakdown)</h4>
          <p style="font-size:12px; color:var(--muted);">Price spikes too far above the 20-EMA, RSI reaches overbought (&gt;52–62), buyers exhaust on upper wicks, and price rolls over.</p>
          <svg viewBox="0 0 560 320" xmlns="http://www.w3.org/2000/svg">
            <!-- Grid -->
            <line x1="30" y1="45" x2="530" y2="45" stroke="#182336" stroke-width="1"/>
            <line x1="30" y1="105" x2="530" y2="105" stroke="#182336" stroke-width="1"/>
            <line x1="30" y1="165" x2="530" y2="165" stroke="#182336" stroke-width="1"/>
            <line x1="30" y1="225" x2="530" y2="225" stroke="#182336" stroke-width="1"/>

            <!-- Shaded SL (Top) & TP (Bottom) Zones -->
            <rect x="240" y="30" width="280" height="70" fill="rgba(239,68,68,0.09)" />
            <rect x="240" y="100" width="280" height="145" fill="rgba(16,185,129,0.09)" />

            <!-- SL, Entry, TP Dashed Lines -->
            <line x1="220" y1="30" x2="525" y2="30" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="6,4"/>
            <text x="250" y="24" fill="#ef4444" font-family="monospace" font-size="11" font-weight="bold">STOP-LOSS (-1.0R): -$100 Net (Acct A) / -$30 Net (Acct B)</text>

            <line x1="220" y1="100" x2="525" y2="100" stroke="#f59e0b" stroke-width="1.8" stroke-dasharray="4,3"/>
            <text x="330" y="94" fill="#f59e0b" font-family="monospace" font-size="11" font-weight="bold">SHORT ENTRY @ Bid - 0.02% Slip</text>

            <line x1="220" y1="245" x2="525" y2="245" stroke="#10b981" stroke-width="1.8" stroke-dasharray="6,4"/>
            <text x="250" y="260" fill="#10b981" font-family="monospace" font-size="11" font-weight="bold">TAKE-PROFIT (+1.5R): +$150 Net (Acct A) / +$45 Net (Acct B)</text>

            <!-- 20 EMA Curve (Blue) -->
            <path d="M 40 190 Q 140 160, 230 140 T 490 220" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
            <text x="45" y="208" fill="#60a5fa" font-family="sans-serif" font-size="11" font-weight="bold">20-Candle EMA (Fair Value)</text>

            <!-- Green candles spiking above EMA -->
            <line x1="70" y1="150" x2="70" y2="190" stroke="#10b981" stroke-width="1.5"/><rect x="64" y="158" width="12" height="24" fill="#10b981"/>
            <line x1="110" y1="118" x2="110" y2="162" stroke="#10b981" stroke-width="1.5"/><rect x="104" y="125" width="12" height="30" fill="#10b981"/>
            <line x1="150" y1="82" x2="150" y2="132" stroke="#10b981" stroke-width="1.5"/><rect x="144" y="90" width="12" height="34" fill="#10b981"/>
            <!-- Shooting star exhaustion candle stretched above EMA -->
            <line x1="190" y1="45" x2="190" y2="100" stroke="#ef4444" stroke-width="1.8"/><rect x="184" y="80" width="12" height="14" fill="#ef4444"/>
            <line x1="190" y1="95" x2="190" y2="142" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="2,2"/>
            <text x="65" y="65" fill="#fbbf24" font-family="monospace" font-size="10">Overextended above EMA (dist_ema20 &gt; 0)</text>

            <!-- Trigger Red Candle (Short Entry) -->
            <line x1="230" y1="82" x2="230" y2="122" stroke="#ef4444" stroke-width="1.8"/><rect x="224" y="88" width="12" height="26" fill="#ef4444"/>
            <polygon points="230,76 221,60 239,60" fill="#ef4444"/>
            <text x="195" y="54" fill="#ef4444" font-family="sans-serif" font-size="11" font-weight="bold">▼ BOT SHORTS (Conf &gt; 60%)</text>

            <!-- Follow-through Red Candles dropping to TP -->
            <line x1="285" y1="110" x2="285" y2="155" stroke="#ef4444" stroke-width="1.5"/><rect x="279" y="116" width="12" height="30" fill="#ef4444"/>
            <line x1="340" y1="142" x2="340" y2="188" stroke="#ef4444" stroke-width="1.5"/><rect x="334" y="148" width="12" height="32" fill="#ef4444"/>
            <line x1="395" y1="175" x2="395" y2="218" stroke="#ef4444" stroke-width="1.5"/><rect x="389" y="182" width="12" height="28" fill="#ef4444"/>
            <line x1="450" y1="202" x2="450" y2="252" stroke="#ef4444" stroke-width="1.8"/><rect x="444" y="206" width="12" height="39" fill="#ef4444"/>
            <circle cx="450" cy="245" r="7" fill="none" stroke="#10b981" stroke-width="2.5"/>
            <text x="462" y="242" fill="#10b981" font-family="sans-serif" font-size="11" font-weight="bold">★ TP HIT</text>

            <!-- Bottom Sub-strip -->
            <rect x="30" y="278" width="500" height="32" rx="4" fill="#111827" stroke="#1f2937"/>
            <text x="42" y="298" fill="#f87171" font-family="monospace" font-size="11">Signals at Entry: ret_5 &gt; +q25 | dist_ema20 &gt; +q25 | RSI_14 = 64 (Overbought) | RVOL = 1.3x</text>
          </svg>
          <ul class="checklist">
            <li><strong>1. EMA Overextension (`dist_ema20 &gt; 0`):</strong> Price rallies too fast above the 20-period EMA, creating an unsustainable premium.</li>
            <li><strong>2. Overbought RSI (`rsi_14 &gt; 52` or `&gt; 62`):</strong> Buying impulse is exhausted; upper candle wicks show sellers stepping in.</li>
            <li><strong>3. Profit from Falling Prices:</strong> Selling `0.133 BTC` high and buying back `1.64%` (Acct A) or `0.59%` (Acct B) lower banks `+1.5R` net profit.</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Section 2: How the AI Learns & The 8 Features -->
    <div class="guide-section">
      <h2>🧠 2. How the Bot Learns (Behavioral Cloning & Stationary Features)</h2>
      <p>
        Instead of trying to guess a raw dollar price like <em>"Bitcoin will be $77,500 tomorrow"</em>—which breaks whenever Bitcoin enters a new price range—the bot uses <strong>Behavioral Cloning</strong>. It converts raw 5-minute Binance candles into <strong>8 normalized percentages and ratios</strong>, aligns them with profitable perpetual trader actions (`LONG`, `SHORT`, `HOLD`), and trains an <strong>XGBoost 3-Class Probability Model</strong> (`180` trees, depth `5`).
      </p>

      <!-- Visual 3: ML Pipeline Diagram -->
      <div class="visual-box" style="margin: 14px 0;">
        <h4 style="color:#60a5fa;">⚙️ Visual Pipeline: From Raw Binance Candles to Live Bracket Execution</h4>
        <svg viewBox="0 0 920 155" xmlns="http://www.w3.org/2000/svg">
          <!-- Step 1 -->
          <rect x="15" y="22" width="195" height="110" rx="8" fill="#131c2e" stroke="#3b82f6" stroke-width="1.5"/>
          <text x="28" y="46" fill="#60a5fa" font-family="sans-serif" font-size="12" font-weight="bold">1. LIVE 5M CANDLES</text>
          <text x="28" y="68" fill="#cbd5e1" font-family="monospace" font-size="11">Binance BTC/USDT</text>
          <text x="28" y="86" fill="#8b9bb4" font-family="monospace" font-size="10.5">Last 50 Candles +</text>
          <text x="28" y="102" fill="#8b9bb4" font-family="monospace" font-size="10.5">Top-20 Orderbook</text>

          <path d="M 215 77 L 245 77" stroke="#60a5fa" stroke-width="2"/>
          <polygon points="245,77 237,72 237,82" fill="#60a5fa"/>

          <!-- Step 2 -->
          <rect x="250" y="22" width="210" height="110" rx="8" fill="#131c2e" stroke="#a855f7" stroke-width="1.5"/>
          <text x="263" y="46" fill="#c084fc" font-family="sans-serif" font-size="12" font-weight="bold">2. 8 STATIONARY FEATURES</text>
          <text x="263" y="68" fill="#cbd5e1" font-family="monospace" font-size="10.5">ret_1, ret_5, ret_15 (%)</text>
          <text x="263" y="85" fill="#cbd5e1" font-family="monospace" font-size="10.5">dist_ema20, rvol, rsi_14</text>
          <text x="263" y="102" fill="#8b9bb4" font-family="monospace" font-size="10.5">candle_range_pct, atr_pct</text>

          <path d="M 465 77 L 495 77" stroke="#c084fc" stroke-width="2"/>
          <polygon points="495,77 487,72 487,82" fill="#c084fc"/>

          <!-- Step 3 -->
          <rect x="500" y="22" width="195" height="110" rx="8" fill="#131c2e" stroke="#f59e0b" stroke-width="1.5"/>
          <text x="513" y="46" fill="#fbbf24" font-family="sans-serif" font-size="12" font-weight="bold">3. XGBOOST CLASSIFIER</text>
          <text x="513" y="68" fill="#cbd5e1" font-family="monospace" font-size="10.5">Outputs Probabilities:</text>
          <text x="513" y="86" fill="#10b981" font-family="monospace" font-size="10.5">P(LONG) = 0.84</text>
          <text x="513" y="102" fill="#8b9bb4" font-family="monospace" font-size="10.5">Filter: Must be &gt; 0.60</text>

          <path d="M 700 77 L 730 77" stroke="#fbbf24" stroke-width="2"/>
          <polygon points="730,77 722,72 722,82" fill="#fbbf24"/>

          <!-- Step 4 -->
          <rect x="735" y="22" width="170" height="110" rx="8" fill="#131c2e" stroke="#10b981" stroke-width="1.5"/>
          <text x="748" y="46" fill="#34d399" font-family="sans-serif" font-size="12" font-weight="bold">4. 1:1.5 EXECUTION</text>
          <text x="748" y="68" fill="#cbd5e1" font-family="monospace" font-size="10.5">$10,000 (1.00x Lev)</text>
          <text x="748" y="85" fill="#f59e0b" font-family="monospace" font-size="10.5">Acct A: -$100 / +$150</text>
          <text x="748" y="102" fill="#c084fc" font-family="monospace" font-size="10.5">Acct B: -$30  / +$45</text>
        </svg>
      </div>

      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>What It Measures</th>
              <th>How the Bot Uses It</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><strong>ret_1, ret_5, ret_15</strong></td><td>5m, 25m, and 75m log percentage returns</td><td>Detects short-term impulse spikes vs broader multi-candle trend direction.</td></tr>
            <tr><td><strong>dist_ema20</strong></td><td>% distance from the 20-candle Exponential Moving Average</td><td>Measures how far price is stretched above or below fair value ("rubber-band effect").</td></tr>
            <tr><td><strong>rsi_14</strong></td><td>14-period Relative Strength Index (0 to 100)</td><td>Identifies overbought (&gt;60) or oversold (&lt;40) exhaustion zones.</td></tr>
            <tr><td><strong>rvol</strong></td><td>Current candle volume divided by 20-candle average volume</td><td>Confirms whether a move has real institutional participation (&gt;1.0x) or low-volume drift.</td></tr>
            <tr><td><strong>candle_range_pct &amp; atr_pct</strong></td><td>High-Low range and 14-period Average True Range (% of price)</td><td>Measures current market volatility regime.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Section 3: R:R, Slippage, Fees & Orderbook Depth -->
    <div class="guide-section">
      <h2>⚖️ 3. Risk-to-Reward (1:1.5), Fees, Slippage & Orderbook Depth</h2>
      <p>
        In real-world crypto trading, you never get filled at the mid-price for free. Every trade pays <strong>Exchange Taker Fees (0.05% per side)</strong> and <strong>Orderbook / Latency Slippage (0.02% per side)</strong>. On a <strong>$10,000 position (1.00x leverage)</strong>, that equals <strong>$14.00 in round-trip friction</strong> ($10 fee + $4 slippage) on every trade.
      </p>

      <div class="callout">
        <strong>Why 1 : 1.5 Net R:R Only Needs a 40% Win Rate to Break Even:</strong><br/>
        Because every winning trade makes <strong>1.5x</strong> what a losing trade loses, your break-even win rate is <code>1 / (1 + 1.5) = 40.0%</code>. Winning just <strong>5 out of 10 trades (50%)</strong> produces a strong net profit!
      </div>

      <div class="callout warn">
        <strong>Why Both Accounts Offset the $14.00 Friction:</strong><br/>
        If a tight scalping bracket only targeted <code>+$24</code> gross profit and <code>-$16</code> gross loss, subtracting <code>-$14</code> in fees and slippage would shrink wins to <strong>+$10</strong> and inflate losses to <strong>-$30</strong>! To prevent fees from inverting the Risk-to-Reward ratio, both Account A and Account B calibrate their gross TP/SL levels so the <strong>NET profit after fees and slippage is an exact 1 : 1.5 ratio</strong>:
        <br/>• <strong>Account A (Wide Swing):</strong> <code>+1.64%</code> Gross TP (<code>+$164 - $14 = +$150.00 Net Win</code>) vs <code>-0.86%</code> Gross SL (<code>-$86 - $14 = -$100.00 Net Loss</code>).
        <br/>• <strong>Account B (Tight Intraday):</strong> <code>+0.59%</code> Gross TP (<code>+$59 - $14 = +$45.00 Net Win</code>) vs <code>-0.16%</code> Gross SL (<code>-$16 - $14 = -$30.00 Net Loss</code>).
      </div>
    </div>

  </div>

<script>
let zoomWindow = 250;
let overlayMode = "both";
let mobileTab = "wide";
let activePage = "terminal";
let latestPayload = null;
let hoverIndex = null;

function switchPage(page) {
  activePage = page;
  document.getElementById("page-terminal").style.display = page === "terminal" ? "block" : "none";
  document.getElementById("page-guide").style.display = page === "guide" ? "block" : "none";
  document.getElementById("nav-terminal").classList.toggle("active", page === "terminal");
  document.getElementById("nav-guide").classList.toggle("active", page === "guide");
  if (page === "terminal" && latestPayload) {
    drawChart(latestPayload);
  }
}

function setZoom(n) {
  zoomWindow = n;
  hoverIndex = null;
  if (latestPayload) drawChart(latestPayload);
}

function setOverlayMode(m) {
  overlayMode = m;
  ["both", "wide", "tight"].forEach(k => {
    document.getElementById("mode-" + k).classList.toggle("active", k === m);
  });
  if (latestPayload) drawChart(latestPayload);
}

function setMobileTab(tab) {
  mobileTab = tab;
  ["wide", "tight", "ticks"].forEach(k => {
    document.getElementById("tab-btn-" + k).classList.toggle("active", k === tab);
  });
  applyMobileTabVisibility();
}

function applyMobileTabVisibility() {
  const isMobile = window.innerWidth <= 768;
  const pWide = document.getElementById("panel-wide");
  const pTight = document.getElementById("panel-tight");
  const pTicks = document.getElementById("panel-ticks");
  if (!isMobile) {
    pWide.classList.remove("mobile-hidden");
    pTight.classList.remove("mobile-hidden");
    pTicks.classList.remove("mobile-hidden");
    return;
  }
  pWide.classList.toggle("mobile-hidden", mobileTab !== "wide");
  pTight.classList.toggle("mobile-hidden", mobileTab !== "tight");
  pTicks.classList.toggle("mobile-hidden", mobileTab !== "ticks");
}

function fmtUsd(v, sign=false) {
  const s = v >= 0 ? (sign ? "+" : "") : "-";
  return s + "$" + Math.abs(v).toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2});
}

async function fetchDashboard() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    latestPayload = data;
    renderDashboard(data);
  } catch (e) {
    console.error(e);
  }
}

async function triggerTick() {
  await fetch("/api/tick", {method: "POST"});
  await fetchDashboard();
}

function renderAccountCard(prefix, acc) {
  document.getElementById(prefix + "-equity").textContent = fmtUsd(acc.current_equity);

  const pillEl = document.getElementById(prefix + "-pnl-pill");
  const signStr = acc.net_ret_pct >= 0 ? "+" : "";
  pillEl.textContent = `${fmtUsd(acc.total_pnl, true)} (${signStr}${acc.net_ret_pct.toFixed(2)}%)`;
  pillEl.className = "card-pnl-pill " + (acc.total_pnl >= 0 ? "pnl-pos" : "pnl-neg");

  const mPnlEl = document.getElementById("m-" + prefix + "-pnl");
  if (mPnlEl) {
    mPnlEl.textContent = `${fmtUsd(acc.total_pnl, true)} (${signStr}${acc.net_ret_pct.toFixed(2)}%)`;
    mPnlEl.className = "m-pill-val " + (acc.total_pnl >= 0 ? "pos-long" : "pos-short");
  }

  document.getElementById(prefix + "-pnl").textContent =
    `Realized: ${fmtUsd(acc.realized_pnl, true)} | Unrealized: ${fmtUsd(acc.unrealized_pnl, true)}`;

  const posEl = document.getElementById(prefix + "-pos");
  const levEl = document.getElementById(prefix + "-lev");
  if (acc.open_position !== "FLAT") {
    posEl.innerHTML = `Open: <strong class="${acc.open_position === 'LONG' ? 'pos-long' : 'pos-short'}">${acc.open_position} @ ${fmtUsd(acc.entry_price)}</strong> → TP: ${fmtUsd(acc.tp_price)} | SL: ${fmtUsd(acc.sl_price)}`;
    levEl.innerHTML = `Size: <strong>${acc.open_size_btc.toFixed(4)} BTC (${fmtUsd(acc.open_notional_usd)})</strong> • Lev: <span class="badge badge-lev">${acc.open_leverage.toFixed(2)}x</span>`;
  } else {
    posEl.textContent = "Open: FLAT (Waiting for next signal)";
    levEl.textContent = "Size: 0.0000 BTC ($0.00) • Lev: 0.00x";
  }

  document.getElementById(prefix + "-stats").textContent =
    `Closed: ${acc.closed_trades} (${acc.win_rate_pct.toFixed(1)}% Win) | Gross: ${fmtUsd(acc.gross_realized_pnl, true)} | Fees: -${fmtUsd(acc.total_fees_usd)} | Slip: -${fmtUsd(acc.total_slippage_usd)}`;

  const tbody = document.getElementById(prefix + "-trades-tbody");
  tbody.innerHTML = acc.trades.length ? acc.trades.map(t => {
    const sideBadge = t.side === "LONG" ? "badge-long" : "badge-short";
    const statusBadge = t.status === "OPEN" ? "badge-open" : (t.status === "TP_HIT" ? "badge-long" : "badge-short");
    const pnlClass = t.net_pnl >= 0 ? "pos-long" : "pos-short";
    const timeShort = (t.entry_time || "").replace("T", " ").slice(11, 19);
    const notional = t.notional_usd || (t.size_btc * t.entry_price);
    const lev = t.leverage || (notional / 10000.0);
    return `<tr>
      <td><span class="badge ${sideBadge}">${t.side}</span></td>
      <td>${timeShort}</td>
      <td><strong>${t.size_btc.toFixed(4)} BTC</strong><br><span style="color:#8b9bb4;font-size:10.5px">${fmtUsd(notional)}</span></td>
      <td><span class="badge badge-lev">${lev.toFixed(2)}x</span></td>
      <td>${fmtUsd(t.entry_price)}</td>
      <td style="font-size:11px"><span class="pos-long">TP: ${fmtUsd(t.tp_price)}</span><br><span class="pos-short">SL: ${fmtUsd(t.sl_price)}</span></td>
      <td><span class="badge ${statusBadge}">${t.status}</span></td>
      <td class="${pnlClass}" style="font-weight:700">${fmtUsd(t.net_pnl, true)}</td>
    </tr>`;
  }).join("") : "<tr><td colspan='8' style='color:#8b9bb4'>No positions opened yet</td></tr>";
}

function renderDashboard(d) {
  applyMobileTabVisibility();
  if (!d.has_data) {
    document.getElementById("kpi-price").textContent = "No Ticks Yet";
    if (activePage === "terminal") drawChart(d);
    return;
  }

  document.getElementById("kpi-price").textContent = fmtUsd(d.latest_price);
  document.getElementById("m-btc").textContent = fmtUsd(d.latest_price);
  document.getElementById("kpi-signal").textContent = `Latest Scan: ${d.latest_signal} (${(d.latest_confidence*100).toFixed(1)}% conf)`;
  document.getElementById("kpi-ticks").textContent = `Total Market Scans: ${d.total_ticks}`;

  renderAccountCard("wide", d.wide);
  renderAccountCard("tight", d.tight);

  document.getElementById("lead-wide").style.display = d.wide.total_pnl >= d.tight.total_pnl ? "inline-block" : "none";
  document.getElementById("lead-tight").style.display = d.tight.total_pnl > d.wide.total_pnl ? "inline-block" : "none";

  const ticksHtml = d.recent_ticks.map(r => {
    const b = r.executed_action === "LONG" ? "badge-long" : (r.executed_action === "SHORT" ? "badge-short" : "badge-hold");
    const ts = r.timestamp_utc.replace("T", " ").slice(11, 19);
    const notional = r.position_size_btc * r.fill_price;
    const lev = notional / (d.starting_equity || 10000);
    return `<tr>
      <td>${ts}</td>
      <td><span class="badge ${b}">${r.executed_action}</span></td>
      <td>${(r.confidence * 100).toFixed(1)}%</td>
      <td>${fmtUsd(r.fill_price)}</td>
      <td>${r.position_size_btc.toFixed(4)} BTC</td>
      <td>${fmtUsd(notional)}</td>
      <td><span class="badge badge-lev">${lev.toFixed(2)}x</span></td>
    </tr>`;
  }).join("");
  document.getElementById("ticks-tbody").innerHTML = ticksHtml;

  if (activePage === "terminal") drawChart(d);
}

function drawChart(d) {
  const canvas = document.getElementById("btcChart");
  if (!canvas || activePage !== "terminal") return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0) return;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  if (!d.has_data || !d.series.length) {
    ctx.fillStyle = "#8b9bb4";
    ctx.font = "13px sans-serif";
    ctx.fillText("Waiting for BTC price ticks...", W / 2 - 75, H / 2);
    return;
  }

  const isMobile = W < 600;
  const series = d.series.slice(-zoomWindow);
  const padL = isMobile ? 8 : 18;
  const padR = isMobile ? 72 : 92;
  const padT = 20;
  const padB = 24;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  let prices = series.map(s => s.p);
  if ((overlayMode === "both" || overlayMode === "tight") && d.tight.open_position !== "FLAT") {
    prices.push(d.tight.tp_price, d.tight.sl_price);
  }
  let minP = Math.min(...prices);
  let maxP = Math.max(...prices);
  const span = Math.max(maxP - minP, minP * 0.0012);
  minP -= span * 0.15;
  maxP += span * 0.15;

  const xOf = i => padL + (series.length === 1 ? plotW / 2 : (i / (series.length - 1)) * plotW);
  const yOf = p => padT + (1 - (p - minP) / (maxP - minP)) * plotH;

  ctx.strokeStyle = "#1c2638";
  ctx.lineWidth = 1;
  const steps = isMobile ? 4 : 5;
  for (let g = 0; g <= steps; g++) {
    const gy = padT + (g / steps) * plotH;
    const gp = maxP - (g / steps) * (maxP - minP);
    ctx.beginPath();
    ctx.moveTo(padL, gy);
    ctx.lineTo(W - padR, gy);
    ctx.stroke();

    ctx.fillStyle = "#8b9bb4";
    ctx.font = (isMobile ? "10px" : "11px") + " monospace";
    ctx.fillText("$" + gp.toFixed(isMobile ? 0 : 2), W - padR + 6, gy + 4);
  }

  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  series.forEach((pt, idx) => {
    const x = xOf(idx), y = yOf(pt.p);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const timeIndex = {};
  series.forEach((pt, idx) => { timeIndex[pt.t] = idx; });

  function drawStrategyOverlay(acc, color, tag, yOffsetDir) {
    if (acc.open_position !== "FLAT" && acc.entry_price > 0) {
      const drawLevel = (px, col, lbl) => {
        if (px < minP || px > maxP) return;
        const y = yOf(px);
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();
        ctx.fillStyle = col;
        ctx.font = "bold " + (isMobile ? "9.5px" : "10px") + " monospace";
        ctx.fillText(`${lbl}: $${px.toFixed(2)}`, padL + 6, y - 4);
        ctx.restore();
      };
      const entryLbl = isMobile
        ? `[${tag}] ${acc.open_position}`
        : `[${tag}] OPEN ${acc.open_position} (${acc.open_size_btc.toFixed(3)} BTC @ ${acc.open_leverage.toFixed(1)}x)`;
      drawLevel(acc.entry_price, color, entryLbl);
      drawLevel(acc.tp_price, "#10b981", `[${tag}] TP`);
      drawLevel(acc.sl_price, "#ef4444", `[${tag}] SL`);
    }

    acc.trades.forEach(tr => {
      const eIdx = timeIndex[tr.entry_time];
      if (eIdx !== undefined) {
        const x = xOf(eIdx);
        const y = yOf(tr.entry_price);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, isMobile ? 4.5 : 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = "bold " + (isMobile ? "9px" : "10px") + " sans-serif";
        ctx.fillText(`${tag}:${tr.side}`, Math.max(padL, x - 16), y + (yOffsetDir * 13));
      }
      if (tr.exit_time && timeIndex[tr.exit_time] !== undefined) {
        const xIdx = timeIndex[tr.exit_time];
        const x = xOf(xIdx);
        const y = yOf(tr.exit_price);
        ctx.strokeStyle = tr.status === "TP_HIT" ? "#10b981" : "#ef4444";
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.arc(x, y, isMobile ? 5.5 : 6.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
  }

  if (overlayMode === "both" || overlayMode === "wide") {
    drawStrategyOverlay(d.wide, "#f59e0b", "A", -1);
  }
  if (overlayMode === "both" || overlayMode === "tight") {
    drawStrategyOverlay(d.tight, "#c084fc", "B", 1);
  }

  if (hoverIndex !== null && hoverIndex >= 0 && hoverIndex < series.length) {
    const pt = series[hoverIndex];
    const cx = xOf(hoverIndex);
    const cy = yOf(pt.p);
    ctx.save();
    ctx.strokeStyle = "rgba(147, 197, 253, 0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, padT);
    ctx.lineTo(cx, padT + plotH);
    ctx.stroke();

    ctx.fillStyle = "#60a5fa";
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    const ts = pt.t.replace("T", " ").slice(11, 19);
    document.getElementById("scrub-info").textContent =
      `[${ts} UTC] BTC: $${pt.p.toLocaleString("en-US", {minimumFractionDigits:2})} • Signal: ${pt.sig} (${(pt.conf*100).toFixed(1)}% conf)`;
  }
}

function bindChartScrubbing() {
  const canvas = document.getElementById("btcChart");
  function updateScrub(clientX) {
    if (!latestPayload || !latestPayload.series || !latestPayload.series.length) return;
    const rect = canvas.getBoundingClientRect();
    const isMobile = rect.width < 600;
    const padL = isMobile ? 8 : 18;
    const padR = isMobile ? 72 : 92;
    const plotW = rect.width - padL - padR;
    const series = latestPayload.series.slice(-zoomWindow);
    const relX = Math.max(0, Math.min(plotW, clientX - rect.left - padL));
    hoverIndex = Math.round((relX / Math.max(plotW, 1)) * (series.length - 1));
    drawChart(latestPayload);
  }
  canvas.addEventListener("mousemove", e => updateScrub(e.clientX));
  canvas.addEventListener("mouseleave", () => {
    hoverIndex = null;
    document.getElementById("scrub-info").textContent = "Tap or drag across the chart to inspect price & signal history";
    if (latestPayload) drawChart(latestPayload);
  });
  canvas.addEventListener("touchstart", e => {
    if (e.touches.length > 0) updateScrub(e.touches[0].clientX);
  }, {passive: true});
  canvas.addEventListener("touchmove", e => {
    if (e.touches.length > 0) updateScrub(e.touches[0].clientX);
  }, {passive: true});
}

window.addEventListener("resize", () => {
  applyMobileTabVisibility();
  if (latestPayload && activePage === "terminal") drawChart(latestPayload);
});
bindChartScrubbing();
fetchDashboard();
setInterval(fetchDashboard, 5000);
</script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(build_dashboard_payload())
            return

        body = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        bot_mod = _load_bot_module()
        if parsed.path == "/api/tick":
            rec = bot_mod.evaluate_and_log_step(dry_run=True)
            self._send_json({"ok": True, "record": rec})
            return
        self._send_json({"ok": False, "error": "Not found"}, status=404)


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def start_public_tunnel(port: int = 8050) -> None:
    """Maintain an auto-reconnecting public HTTPS tunnel via localhost.run."""
    global PUBLIC_TUNNEL_URL

    def _tunnel_worker():
        global PUBLIC_TUNNEL_URL
        cmd = [
            "ssh",
            "-T",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=NUL",
            "-o",
            "ServerAliveInterval=20",
            "-o",
            "ExitOnForwardFailure=yes",
            "-R",
            f"80:127.0.0.1:{port}",
            "nokey@localhost.run",
        ]
        while True:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    match = re.search(r"(https://[a-zA-Z0-9.-]+\.lhr\.life)", line)
                    if match:
                        url = match.group(1)
                        if url != PUBLIC_TUNNEL_URL:
                            PUBLIC_TUNNEL_URL = url
                            print(f"Dashboard (Anywhere 4G/5G): {PUBLIC_TUNNEL_URL}", flush=True)
                proc.wait()
            except Exception:
                pass
            time.sleep(3)

    t = threading.Thread(target=_tunnel_worker, daemon=True)
    t.start()


def run_server(host: str = "0.0.0.0", port: int = 8050, enable_tunnel: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), DashboardRequestHandler)
    lan_ip = get_lan_ip()
    print(f"Dashboard (This PC):        http://127.0.0.1:{port}", flush=True)
    print(f"Dashboard (Same Wi-Fi):     http://{lan_ip}:{port}", flush=True)
    if enable_tunnel:
        start_public_tunnel(port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live BTC Paper Trading Web Dashboard.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-tunnel", action="store_true", help="Disable automatic public HTTPS tunnel.")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, enable_tunnel=not args.no_tunnel)
