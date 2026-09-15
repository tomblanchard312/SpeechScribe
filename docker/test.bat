@echo off
REM SpeechScribe Docker Testing Script for Windows

setlocal enabledelayedexpansion

echo ================================
echo SpeechScribe Docker Test Suite
echo ================================
echo.

set /a TESTS_PASSED=0
set /a TESTS_FAILED=0

REM Test 1: Check Docker and Docker Compose
echo Test 1: Prerequisites
echo ====================

docker --version >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Docker is installed
    set /a TESTS_PASSED+=1
) else (
    echo ✗ Docker is not installed
    set /a TESTS_FAILED+=1
)

docker-compose --version >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Docker Compose is installed
    set /a TESTS_PASSED+=1
) else (
    echo ✗ Docker Compose is not installed
    set /a TESTS_FAILED+=1
)

echo.

REM Test 2: Check containers are running
echo Test 2: Container Status
echo =======================

docker ps | findstr /r "speechscribe-api" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ API container is running
    set /a TESTS_PASSED+=1
) else (
    echo ✗ API container is not running
    set /a TESTS_FAILED+=1
)

docker ps | findstr /r "speechscribe-ui" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ UI container is running
    set /a TESTS_PASSED+=1
) else (
    echo ✗ UI container is not running
    set /a TESTS_FAILED+=1
)

docker ps | findstr /r "speechscribe-ollama" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Ollama container is running
    set /a TESTS_PASSED+=1
) else (
    echo ✗ Ollama container is not running
    set /a TESTS_FAILED+=1
)

echo.

REM Test 3: Check network connectivity
echo Test 3: Network Connectivity
echo ============================

curl -s http://localhost:8000/docs >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ API is accessible ^(http://localhost:8000^)
    set /a TESTS_PASSED+=1
) else (
    echo ✗ API is not accessible ^(http://localhost:8000^)
    set /a TESTS_FAILED+=1
)

curl -s http://localhost:3000 >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ UI is accessible ^(http://localhost:3000^)
    set /a TESTS_PASSED+=1
) else (
    echo ✗ UI is not accessible ^(http://localhost:3000^)
    set /a TESTS_FAILED+=1
)

curl -s http://localhost:11434/api/tags >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Ollama is accessible ^(http://localhost:11434^)
    set /a TESTS_PASSED+=1
) else (
    echo ✗ Ollama is not accessible ^(http://localhost:11434^)
    set /a TESTS_FAILED+=1
)

echo.

REM Test 4: Check API endpoints
echo Test 4: API Endpoints
echo =====================

curl -s http://localhost:8000/health >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ API health endpoint is working
    set /a TESTS_PASSED+=1
) else (
    echo ✗ API health endpoint is not working
    set /a TESTS_FAILED+=1
)

curl -s http://localhost:8000/plugins | findstr /r "plugin_id" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ API plugins endpoint is working
    set /a TESTS_PASSED+=1
) else (
    echo ✗ API plugins endpoint is not working
    set /a TESTS_FAILED+=1
)

curl -s http://localhost:8000/docs | findstr /r "OpenAPI\|swagger" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ API docs are accessible
    set /a TESTS_PASSED+=1
) else (
    echo ✗ API docs are not accessible
    set /a TESTS_FAILED+=1
)

echo.

REM Test 5: Check volumes
echo Test 5: Docker Volumes
echo =======================

docker volume ls | findstr /r "docker_models" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Models volume exists
    set /a TESTS_PASSED+=1
) else (
    echo ✗ Models volume not found
    set /a TESTS_FAILED+=1
)

docker volume ls | findstr /r "docker_plugins" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Plugins volume exists
    set /a TESTS_PASSED+=1
) else (
    echo ✗ Plugins volume not found
    set /a TESTS_FAILED+=1
)

docker volume ls | findstr /r "docker_cache" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Cache volume exists
    set /a TESTS_PASSED+=1
) else (
    echo ✗ Cache volume not found
    set /a TESTS_FAILED+=1
)

docker volume ls | findstr /r "docker_ollama-models" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Ollama models volume exists
    set /a TESTS_PASSED+=1
) else (
    echo ✗ Ollama models volume not found
    set /a TESTS_FAILED+=1
)

echo.

REM Test 6: Check GPU support
echo Test 6: GPU Support
echo ===================

nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Host GPU support detected
    set /a TESTS_PASSED+=1
) else (
    echo ⓘ Host GPU not detected - CPU-only mode
)

echo.

REM Summary
echo ================================
echo Test Summary
echo ================================
echo Passed: !TESTS_PASSED!
echo Failed: !TESTS_FAILED!
echo.

if !TESTS_FAILED! equ 0 (
    echo 🎉 All tests passed!
) else (
    echo ⚠ Some tests failed. Review the output above.
)

echo.
pause
