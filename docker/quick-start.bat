@echo off
REM SpeechScribe Docker Quick Start Script for Windows

setlocal enabledelayedexpansion

echo ================================
echo SpeechScribe Docker Quick Start
echo ================================
echo.

REM Check prerequisites
echo Checking prerequisites...

docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

echo ✓ Docker is installed
echo ✓ Docker Compose is installed
echo.

REM Check for GPU support
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ NVIDIA GPU detected
    for /f "delims=" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader') do (
        echo   %%i
    )
) else (
    echo ⚠ No NVIDIA GPU detected. CPU-only mode will be used.
)

echo.

REM Get the directory where this script is located
for %%i in ("%~dp0.") do set "PROJECT_ROOT=%%~fi"

cd /d "%PROJECT_ROOT%"

REM Create .env if it doesn't exist
if not exist "docker\.env" (
    echo Creating .env file from template...
    copy "docker\.env.example" "docker\.env" >nul
    echo ✓ Created docker\.env (please edit if needed)
)

echo.
echo Starting SpeechScribe services...
echo.

REM Build and start services
docker compose -f docker/docker-compose.yml up -d

echo.
echo ================================
echo Services Starting...
echo ================================
echo.

REM Wait for services to be ready
echo Waiting for services to be ready...
timeout /t 10 /nobreak

echo.
echo Checking API health...
set api_ready=0
for /l %%i in (1,1,30) do (
    curl -s http://localhost:8000/docs >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✓ API is ready at http://localhost:8000
        set api_ready=1
        goto api_done
    )
    timeout /t 1 /nobreak >nul
)
:api_done
if !api_ready! equ 0 (
    echo ⚠ API might not be ready yet. Check logs with:
    echo   docker compose -f docker/docker-compose.yml logs speechscribe-api
)

echo.
echo Checking UI...
set ui_ready=0
for /l %%i in (1,1,15) do (
    curl -s http://localhost:3000 >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✓ UI is ready at http://localhost:3000
        set ui_ready=1
        goto ui_done
    )
    timeout /t 1 /nobreak >nul
)
:ui_done
if !ui_ready! equ 0 (
    echo ⚠ UI might not be ready yet. Check logs with:
    echo   docker compose -f docker/docker-compose.yml logs speechscribe-ui
)

echo.
echo Checking Ollama...
set ollama_ready=0
for /l %%i in (1,1,30) do (
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✓ Ollama is ready at http://localhost:11434
        set ollama_ready=1
        goto ollama_done
    )
    timeout /t 1 /nobreak >nul
)
:ollama_done
if !ollama_ready! equ 0 (
    echo ⚠ Ollama might not be ready yet. Check logs with:
    echo   docker compose -f docker/docker-compose.yml logs ollama
)

echo.
echo ================================
echo Deployment Complete!
echo ================================
echo.
echo Services:
echo   Web UI:            http://localhost:3000
echo   API Docs:          http://localhost:8000/docs
echo   API ReDoc:         http://localhost:8000/redoc
echo   Ollama API:        http://localhost:11434
echo.
echo Next steps:
echo   1. Open http://localhost:3000 in your browser
echo   2. Pull Ollama models:
echo      docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5
echo   3. Start chatting!
echo.
echo View logs:
echo   All services:  docker compose -f docker/docker-compose.yml logs -f
echo   API only:      docker compose -f docker/docker-compose.yml logs -f speechscribe-api
echo   UI only:       docker compose -f docker/docker-compose.yml logs -f speechscribe-ui
echo   Ollama only:   docker compose -f docker/docker-compose.yml logs -f ollama
echo.
echo Stop services:
echo   docker compose -f docker/docker-compose.yml down
echo.

pause
