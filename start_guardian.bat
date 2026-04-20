@echo off
echo Starting Guardian Tri-Modal Fraud Detection System...
echo.

:: Start Fusion Layer
start "Fusion Layer :8080" cmd /k "cd /d "D:\Classes\Winter_2026\COMP385 (AI CAPSTONE)\Guardian" && call "D:\Classes\AI\Scripts\activate.bat" && uvicorn fusion.api:app --reload --port 8080"

timeout /t 2 /nobreak >nul

:: Start NLP Stream
start "NLP Stream :8001" cmd /k "cd /d "D:\Classes\Winter_2026\COMP385 (AI CAPSTONE)\Guardian" && call "D:\Classes\AI\Scripts\activate.bat" && python -m uvicorn NPL.api.api:app --reload --port 8001"

timeout /t 2 /nobreak >nul

:: Start Voice Stream
start "Voice Stream :8000" cmd /k "cd /d "D:\Classes\Winter_2026\COMP385 (AI CAPSTONE)\Guardian" && call "D:\Classes\AI\Scripts\activate.bat" && uvicorn Voice.api:app --reload --port 8000"

timeout /t 2 /nobreak >nul

:: Start Frontend
start "Frontend :3000" cmd /k "cd /d "D:\Classes\Winter_2026\COMP385 (AI CAPSTONE)\Guardian\guardian-dashboard" && npm run dev"

echo.
echo All services started!
echo.
echo   Fusion Layer  ->  http://localhost:8080
echo   NLP Stream    ->  http://localhost:8001
echo   Voice Stream  ->  http://localhost:8000
echo   Frontend      ->  http://localhost:3000
echo.
pause