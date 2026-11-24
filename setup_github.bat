@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_github.ps1" %*
pause
