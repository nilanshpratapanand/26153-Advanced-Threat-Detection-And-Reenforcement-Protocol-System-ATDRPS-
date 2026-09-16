@echo off
REM ==========================================================================
REM   ATDRPS - run everything  (Windows)
REM
REM       run.bat                 interactive menu
REM       run.bat all             full pipeline: capture, corpus, benchmark
REM       run.bat demo            generate a labelled synthetic capture
REM       run.bat corpus          build the training corpus
REM       run.bat train           train the world model
REM       run.bat benchmark       train everything, write docs\BENCHMARKS.md
REM       run.bat predict [file]  forecast from a capture
REM       run.bat dashboard       start the offline dashboard
REM       run.bat test            run the test suite
REM       run.bat clean           remove generated data, models and reports
REM
REM   Everything runs locally. No cloud APIs, no telemetry, no downloads.
REM ==========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "CORPUS=data\corpus.npz"
set "DEMO=data\demo\capture.pcap"
set "MODEL_DIR=artifacts\model-linear"
if exist "artifacts\model-transformer\config.json" set "MODEL_DIR=artifacts\model-transformer"

REM ------------------------------------------------------------ interpreter
if exist ".venv\Scripts\python.exe" (
  set "PY=%CD%\.venv\Scripts\python.exe"
) else (
  echo   !!  .venv not found - falling back to the system Python
  echo   !!  run install.bat first for an isolated, reproducible setup
  set "PY=python"
)

"%PY%" -c "import numpy,pandas,sklearn,yaml,flask" >nul 2>&1
if errorlevel 1 (
  echo   xx  dependencies are missing. Run install.bat
  pause
  exit /b 1
)

if "%~1"=="" goto menu
if /i "%~1"=="menu"      goto menu
if /i "%~1"=="all"       goto all
if /i "%~1"=="demo"      goto demo
if /i "%~1"=="corpus"    goto corpus
if /i "%~1"=="train"     goto train
if /i "%~1"=="benchmark" goto benchmark
if /i "%~1"=="predict"   goto predict
if /i "%~1"=="dashboard" goto dashboard
if /i "%~1"=="serve"     goto dashboard
if /i "%~1"=="test"      goto test
if /i "%~1"=="clean"     goto clean
if /i "%~1"=="help"      goto help
if /i "%~1"=="--help"    goto help
if /i "%~1"=="-h"        goto help
echo   xx  unknown command: %~1   ^(try run.bat help^)
exit /b 1

:banner
echo.
echo   ATDRPS  world-model network attack forecasting
echo   SIH 2026 ^| PS 26153 ^| NTRO ^| Team Bell Labs
"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
  echo   backend: linear dynamics ^(PyTorch not installed^)
) else (
  echo   backend: temporal transformer ^(PyTorch available^)
)
goto :eof

:menu
call :banner
:menuloop
echo.
echo    1  Full pipeline          capture, corpus, benchmark, forecast
echo    2  Generate a capture     labelled synthetic traffic with a kill chain
echo    3  Build the corpus       48 captures, parallel across cores
echo    4  Train the model        world model only
echo    5  Run the benchmark      all models + docs\BENCHMARKS.md
echo    6  Forecast a capture     timeline, stage, drivers
echo    7  Offline dashboard      http://127.0.0.1:8501
echo    8  Run the tests          273 tests
echo    9  Clean generated files
echo    0  Quit
echo.
set "choice="
set /p "choice=  choose: "
if "%choice%"=="1" (call :do_all    & goto menuloop)
if "%choice%"=="2" (call :do_demo   & goto menuloop)
if "%choice%"=="3" (call :do_corpus & goto menuloop)
if "%choice%"=="4" (call :do_train  & goto menuloop)
if "%choice%"=="5" (call :do_bench  & goto menuloop)
if "%choice%"=="6" (call :do_pred   & goto menuloop)
if "%choice%"=="7" (call :do_dash   & goto menuloop)
if "%choice%"=="8" (call :do_test   & goto menuloop)
if "%choice%"=="9" (call :do_clean  & goto menuloop)
if "%choice%"=="0" (echo. & exit /b 0)
if /i "%choice%"=="q" (echo. & exit /b 0)
echo   !!  pick a number from the list
goto menuloop

REM ------------------------------------------------------------------ steps
:do_demo
echo.
echo ==^> generating a labelled synthetic capture
"%PY%" -m atdrps.cli synth --out data\demo --seed 4242 --duration 2400 --campaigns 2
if errorlevel 1 (echo   xx  capture generation failed & goto :eof)
echo   ok  %DEMO%   ^(ground truth in capture.timeline.json^)
goto :eof

:do_corpus
if exist "%CORPUS%" (
  set "rebuild="
  set /p "rebuild=  %CORPUS% already exists. Rebuild it? [y/N] "
  if /i not "!rebuild!"=="y" (echo   ok  keeping the existing corpus & goto :eof)
)
echo.
echo ==^> building the training corpus  ^(48 captures, parallel across all CPU cores^)
"%PY%" -m atdrps.cli corpus --captures 48 --out "%CORPUS%"
if errorlevel 1 (echo   xx  corpus build failed & goto :eof)
echo   ok  %CORPUS%
goto :eof

:do_train
if not exist "%CORPUS%" (echo   !!  no corpus yet & call :do_corpus)
echo.
echo ==^> training the world model
"%PY%" -m atdrps.cli train --corpus "%CORPUS%" --out artifacts\model
if errorlevel 1 (echo   xx  training failed & goto :eof)
echo   ok  artifacts\model
goto :eof

:do_bench
if not exist "%CORPUS%" (echo   !!  no corpus yet & call :do_corpus)
echo.
echo ==^> training every model and comparing them
echo     persistence, logistic regression ^(static + context^), world model
"%PY%" -m atdrps.cli benchmark --corpus "%CORPUS%" --split group
if errorlevel 1 (echo   xx  benchmark failed & goto :eof)
if exist "artifacts\model-transformer\config.json" set "MODEL_DIR=artifacts\model-transformer"
echo   ok  docs\BENCHMARKS.md
goto :eof

:do_pred
set "CAPTURE=%~2"
if "%CAPTURE%"=="" set "CAPTURE=%DEMO%"
if not exist "%CAPTURE%" (echo   !!  no capture at %CAPTURE% & call :do_demo & set "CAPTURE=%DEMO%")
if not exist "%MODEL_DIR%\config.json" (
  echo   xx  no trained model. Run: run.bat benchmark
  goto :eof
)
echo.
echo ==^> forecasting from %CAPTURE%
"%PY%" -m atdrps.cli predict "%CAPTURE%" --model "%MODEL_DIR%"
goto :eof

:do_dash
if not exist "%MODEL_DIR%\config.json" (
  echo   xx  no trained model. Run: run.bat benchmark
  goto :eof
)
if not exist "%DEMO%" call :do_demo
echo.
echo ==^> starting the offline dashboard
echo     open http://127.0.0.1:8501 and upload %DEMO%
echo     nothing leaves this machine. Ctrl-C to stop.
start "" http://127.0.0.1:8501
"%PY%" -m atdrps.cli serve --model "%MODEL_DIR%" --host 127.0.0.1 --port 8501
goto :eof

:do_test
echo.
echo ==^> running the test suite
"%PY%" tests\run_tests.py
goto :eof

:do_clean
echo.
echo ==^> removing generated files
set "confirm="
set /p "confirm=  this deletes %CORPUS%, artifacts\ and data\demo\. Continue? [y/N] "
if /i not "!confirm!"=="y" (echo   ok  nothing removed & goto :eof)
if exist "artifacts" rmdir /s /q "artifacts"
if exist "data\demo" rmdir /s /q "data\demo"
if exist "%CORPUS%" del /q "%CORPUS%"
mkdir "artifacts" 2>nul
type nul > "artifacts\.gitkeep"
echo   ok  cleaned
goto :eof

:do_all
call :do_demo
call :do_corpus
call :do_bench
call :do_pred
echo.
echo   Pipeline complete.
echo   results  docs\BENCHMARKS.md
echo   demo     run.bat dashboard
goto :eof

REM --------------------------------------------------------------- dispatch
:all
call :banner
call :do_all
pause
exit /b 0

:demo
call :banner
call :do_demo
exit /b 0

:corpus
call :banner
call :do_corpus
exit /b 0

:train
call :banner
call :do_train
exit /b 0

:benchmark
call :banner
call :do_bench
exit /b 0

:predict
call :banner
call :do_pred "" "%~2"
exit /b 0

:dashboard
call :banner
call :do_dash
exit /b 0

:test
call :banner
call :do_test
exit /b 0

:clean
call :banner
call :do_clean
exit /b 0

:help
echo.
echo   ATDRPS - run everything  ^(Windows^)
echo.
echo       run.bat                 interactive menu
echo       run.bat all             full pipeline: capture, corpus, benchmark
echo       run.bat demo            generate a labelled synthetic capture
echo       run.bat corpus          build the training corpus
echo       run.bat train           train the world model
echo       run.bat benchmark       train everything, write docs\BENCHMARKS.md
echo       run.bat predict [file]  forecast from a capture
echo       run.bat dashboard       start the offline dashboard
echo       run.bat test            run the test suite
echo       run.bat clean           remove generated data, models and reports
echo.
echo   Everything runs locally. No cloud APIs, no telemetry, no downloads.
echo.
exit /b 0
