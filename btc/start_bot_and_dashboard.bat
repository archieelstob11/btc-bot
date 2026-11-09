@echo off
title BTC Paper Bot & Dashboard Watchdog
cd /d "%~dp0"

echo Starting Live Dashboard Server (Port 8050 + Auto 4G/5G Tunnel)...
start "BTC Dashboard Server" cmd /c "for /L %%i in (1,0,2) do (.\.venv\Scripts\python.exe -u dashboard_app.py --port 8050 & timeout /t 5)"

echo Starting Live Paper Trading Bot (Auto-Reconnect & Outage Backfill Enabled)...
:bot_loop
.\.venv\Scripts\python.exe -u 3_live_paper_bot.py --dry-run --interval 60
echo Bot process exited or interrupted. Restarting in 5 seconds...
timeout /t 5
goto bot_loop

