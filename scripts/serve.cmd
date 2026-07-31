@echo off
REM Always serve the dashboard on the SAME port. Frees the port first, then starts uvicorn.
set PORT=8010
cd /d D:\energy-market-lab
set PYTHONPATH=.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do taskkill /PID %%p /F >nul 2>&1
echo Serving Energy Market Lab on http://127.0.0.1:%PORT%
.\.venv\Scripts\python.exe -m uvicorn eml.api.main:app --host 127.0.0.1 --port %PORT%
