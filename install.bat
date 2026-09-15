@echo off
REM ==========================================================================
REM   ATDRPS - one-time setup  (Windows)
REM
REM       install.bat              core install
REM       install.bat torch        also install PyTorch (CPU build)
REM       install.bat torch-cuda   also install PyTorch (CUDA 12.1 build)
REM       install.bat no-test      skip the test suite at the end
REM
REM   Creates .venv in this folder, installs the requirements, and proves the
REM   install by running the test suite. Nothing is installed system-wide.
REM ==========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "WANT_TORCH="
set "RUN_TESTS=1"
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="torch"       set "WANT_TORCH=cpu"
if /i "%~1"=="torch-cuda"  set "WANT_TORCH=cu121"
if /i "%~1"=="no-test"     set "RUN_TESTS=0"
if /i "%~1"=="--torch"      set "WANT_TORCH=cpu"
if /i "%~1"=="--torch-cuda" set "WANT_TORCH=cu121"
if /i "%~1"=="--no-test"    set "RUN_TESTS=0"
shift
goto parse
:parsed

echo.
echo   ATDRPS  Advanced Threat Detection and Reinforcement Protocol System
echo   SIH 2026 ^| PS 26153 ^| NTRO ^| Team Bell Labs
echo.

REM ------------------------------------------------------------------ python
echo ==^> looking for Python 3.9+
set "PY="
for %%C in ("py -3" "python" "python3") do (
  if not defined PY (
    %%~C -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY=%%~C"
  )
)
if not defined PY (
  echo   xx  no Python 3.9+ found.
  echo       Install it from https://www.python.org/downloads/
  echo       IMPORTANT: tick "Add Python to PATH" in the installer.
  pause
  exit /b 1
)
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do echo   ok  %%v

REM -------------------------------------------------------------------- venv
if exist ".venv\Scripts\python.exe" (
  echo ==^> reusing the existing virtual environment
  echo   ok  .venv
) else (
  echo ==^> creating a virtual environment in .venv
  %PY% -m venv .venv
  if errorlevel 1 (
    echo   xx  could not create .venv
    pause
    exit /b 1
  )
  echo   ok  .venv created
)

set "VPY=%CD%\.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo   xx  the virtual environment looks broken - delete .venv and re-run
  pause
  exit /b 1
)

REM ---------------------------------------------------------------- packages
echo ==^> upgrading pip
"%VPY%" -m pip install --quiet --upgrade pip >nul 2>&1
if errorlevel 1 (echo   !!  could not upgrade pip - continuing) else (echo   ok  pip upgraded)

echo ==^> installing requirements  (this takes a minute or two)
"%VPY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo   !!  pip could not install everything from requirements.txt
  echo   !!  if this machine is offline, see the note at the end
) else (
  echo   ok  core requirements installed
)

if defined WANT_TORCH (
  echo ==^> installing PyTorch ^(%WANT_TORCH% build^) - this is a large download
  "%VPY%" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/%WANT_TORCH%
  if errorlevel 1 (
    echo   !!  PyTorch install failed - ATDRPS still runs on the numpy backend
  ) else (
    echo   ok  PyTorch installed
  )
)

REM ------------------------------------------------------------- self-check
echo ==^> checking the install
"%VPY%" -c "import importlib,sys; m=[x for x in ('numpy','pandas','scipy','sklearn','yaml','flask','matplotlib') if not importlib.util.find_spec(x)]; print('  xx  missing: '+', '.join(m)) if m else print('  ok  core packages import cleanly'); sys.exit(1 if m else 0)"
if errorlevel 1 (
  echo   xx  the install is incomplete - see the messages above
  pause
  exit /b 1
)
"%VPY%" -c "import torch; print('  ok  PyTorch '+torch.__version__+(' (CUDA)' if torch.cuda.is_available() else ' (CPU)')+' - transformer backend available')" 2>nul
if errorlevel 1 (
  echo   !!  PyTorch not installed - the numpy backend will be used
  echo       to add it later:  install.bat torch
)

REM ------------------------------------------------------------------ tests
if "%RUN_TESTS%"=="1" (
  echo ==^> running the test suite
  "%VPY%" tests\run_tests.py
  if errorlevel 1 (
    echo   xx  tests failed - do not trust this install
    pause
    exit /b 1
  )
  echo   ok  tests passed
)

echo.
echo   Setup complete.
echo   Next:  run.bat              interactive menu
echo          run.bat all          full pipeline: corpus, benchmark, demo
echo          run.bat dashboard    offline dashboard
echo.
echo   Offline machine? Copy a wheelhouse across and run:
echo     .venv\Scripts\pip install --no-index --find-links=wheels -r requirements.txt
echo.
pause
endlocal
