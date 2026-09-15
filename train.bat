@echo off
REM ===================================================================
REM  ATDRPS -- train the world model on THIS machine.
REM
REM  Uses the CPU to build the training corpus (one capture per core)
REM  and the GPU to train the temporal transformer. If PyTorch or CUDA
REM  is missing it says so and still trains the linear-dynamics model,
REM  which is the shipped fallback -- a degraded result, never a broken
REM  one.
REM
REM  Usage:
REM    train.bat                 full run   (160 captures, ~1-2 h)
REM    train.bat --quick         fast run   (48 captures,  ~15-25 min)
REM    train.bat --skip-corpus   reuse the corpus already on disk
REM
REM  Anything else you pass goes straight to scripts\train_all.py,
REM  e.g.  train.bat --captures 320 --epochs 60
REM ===================================================================
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  echo   No .venv found -- creating one.
  python -m venv .venv
  if errorlevel 1 (
    echo   Could not create a virtual environment. Is Python 3.10+ installed?
    pause
    exit /b 1
  )
  set "PY=.venv\Scripts\python.exe"
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt
)

REM ---- translate --quick into the real arguments -------------------
set "ARGS=%*"
if /I "%~1"=="--quick" set "ARGS=--captures 48"

echo.
echo   Checking PyTorch...
"%PY%" -c "import torch, sys; print('   torch', torch.__version__, '| CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')" 2>nul
if errorlevel 1 (
  echo   PyTorch is not installed.
  echo   Run setup_torch.bat first to install the build that matches your GPU,
  echo   or continue now to train only the linear model.
  echo.
  pause
)

"%PY%" scripts\train_all.py %ARGS%
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
  echo   Training complete.
  echo     artifacts\           trained model^(s^)
  echo     docs\BENCHMARKS.md   the comparison table
  echo.
  echo   Next: run.bat dashboard
) else (
  echo   Training exited with code %RC% -- see the output above.
)
pause
endlocal
