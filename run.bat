@echo off
REM Windows startup script for Offroad AI Web App v2.0 (WebSocket)

echo.
echo ╔════════════════════════════════════════════════════╗
echo ║   OFFROAD AI v2.0 - WebSocket Live Streaming       ║
echo ║          Starting Application Server...             ║
echo ╚════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "terrain_classifier.pth" (
    echo WARNING: terrain_classifier.pth not found in project root
    echo The app will fail when trying to load the model
    echo.
    pause
)

REM Install/upgrade dependencies
echo Installing dependencies...
python -m pip install -r requirements.txt --quiet

REM Run the Flask-SocketIO app
echo.
echo ✓ Starting WebSocket Server...
echo.
echo 🌐 Open your browser and go to: http://localhost:10000
echo.
echo Features:
echo   - Real-time WebSocket video streaming
echo   - Frame-by-frame terrain classification
echo   - Last 3 results history
echo   - Lightweight frame compression
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

if errorlevel 1 (
    echo.
    echo ❌ ERROR: Server failed to start
    echo Please check the error message above
    pause
    exit /b 1
)
    echo.
    echo ERROR: Flask app failed to start
    pause
    exit /b 1
)
