@echo off
setlocal
cd /d "%~dp0"
py -3 "%~dp0eval_finetuned_unified_run.py" %*
exit /b %ERRORLEVEL%
