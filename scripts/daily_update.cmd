@echo off
REM Daily live refresh for Energy Market Lab. Registered with Windows Task Scheduler.
REM Pulls latest ENTSO-E actuals + day-ahead forecasts, refreshes weather, extends the outlook.
cd /d D:\energy-market-lab
set PYTHONPATH=.
.\.venv\Scripts\python.exe scripts\update_live.py >> data\update_live.log 2>&1
REM refresh the deployed model on the latest trailing window (tracks the current regime)
.\.venv\Scripts\python.exe -c "from eml.models.price_forecast import train; print('deploy:', train())" >> data\update_live.log 2>&1
echo [%date% %time%] exit=%errorlevel% >> data\update_live.log
