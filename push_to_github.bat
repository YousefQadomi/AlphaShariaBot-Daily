@echo off
title AlphaShariaBot - GitHub Auto Pusher
cd /d "%~dp0"
echo.
echo ============================================================
echo   AlphaShariaBot - GitHub Auto Pusher
echo ============================================================
echo.
echo   Choose a mode:
echo.
echo   [1] Push now (interactive - shows changes, asks to confirm)
echo   [2] Push now (automatic - no confirmation)
echo   [3] Watch mode (auto-push every 5 minutes)
echo   [4] Watch mode (auto-push every 10 minutes)
echo.
set /p choice="  Enter choice [1-4]: "

if "%choice%"=="1" (
    python auto_push.py
) else if "%choice%"=="2" (
    python auto_push.py --auto
) else if "%choice%"=="3" (
    python auto_push.py --watch 5
) else if "%choice%"=="4" (
    python auto_push.py --watch 10
) else (
    echo   Invalid choice. Running interactive mode...
    python auto_push.py
)

echo.
pause
