@echo off
setlocal
set "launch_args="
if not "%~2"=="" goto usage_error
if "%~1"=="" goto launch
if /i "%~1"=="--basic" (
    set "launch_args=-Basic"
    goto launch
)
if /i "%~1"=="--help" goto help
goto usage_error

:launch
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %launch_args%
set "launch_status=%errorlevel%"
if not "%launch_status%"=="0" if not "%launch_status%"=="130" (
    echo.
    echo Startup failed. Check the message above.
    echo For analysis without model downloads: start.cmd --basic
    pause
)
exit /b %launch_status%

:help
echo Usage: start.cmd [--basic]
echo Default: CLAP audio/text analysis. --basic: librosa only, no model downloads.
exit /b 0

:usage_error
echo Usage: start.cmd [--basic]
exit /b 2
