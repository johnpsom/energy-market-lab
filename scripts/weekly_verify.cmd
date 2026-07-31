@echo off
REM Weekly walk-forward verification refresh (heavier — retrains ~35x). Registered with Task Scheduler.
cd /d D:\energy-market-lab
set PYTHONPATH=.
.\.venv\Scripts\python.exe scripts\walk_forward.py >> data\walk_forward.log 2>&1
echo [%date% %time%] exit=%errorlevel% >> data\walk_forward.log
