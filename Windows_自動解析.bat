@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\windows.ps1" -App analysis
if errorlevel 1 pause
exit /b %errorlevel%
