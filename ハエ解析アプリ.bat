@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
if defined FLY_ACTIVITY_PYTHON goto custom
if exist "%~dp0.venv\Scripts\python.exe" goto local
py -3 --version >nul 2>&1
if not errorlevel 1 goto py
python --version >nul 2>&1
if not errorlevel 1 goto python
echo Python 3.10 or newer with Tcl/Tk is required.
set "fly_status=1"
goto finish
:custom
"%FLY_ACTIVITY_PYTHON%" "%~dp0launch.py" analysis %*
goto status
:local
"%~dp0.venv\Scripts\python.exe" "%~dp0launch.py" analysis %*
goto status
:py
py -3 "%~dp0launch.py" analysis %*
goto status
:python
python "%~dp0launch.py" analysis %*
:status
set "fly_status=%errorlevel%"
:finish
if not "%fly_status%"=="0" pause
popd
exit /b %fly_status%
