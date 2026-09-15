@echo off
REM ===================================================================
REM  ATDRPS -- install the right PyTorch for THIS machine.
REM
REM  Picks the CUDA wheel your NVIDIA driver can actually run, or the
REM  CPU build if there is no NVIDIA GPU. Getting this wrong is the
REM  usual reason a GPU laptop silently trains on CPU: a plain
REM  "pip install torch" on Windows gives you the CPU build and says
REM  nothing about it.
REM
REM  Usage:  setup_torch.bat            install
REM          setup_torch.bat --dry-run  show the command, change nothing
REM          setup_torch.bat --cpu      force the CPU build
REM          setup_torch.bat --verify-only   report what is installed
REM ===================================================================
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

"%PY%" scripts\setup_torch.py %*
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
  echo   Done. Next: run.bat train-transformer
) else (
  echo   Exited with code %RC%.
)
pause
endlocal
