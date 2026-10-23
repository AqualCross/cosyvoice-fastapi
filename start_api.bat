@echo off
rem =====================================================================
rem  cosyvoice-fastapi - one-click start (Windows)
rem  The wrapper stays outside the upstream repo; COSYVOICE_REPO defaults
rem  to the sibling ../CosyVoice folder.
rem  Edit GPU / PORT below, extra arguments are forwarded to run_api.py.
rem =====================================================================
setlocal
set GPU=0
set PORT=8001

set WRAPPER=%~dp0
cd /d "%WRAPPER%"
echo [start_api] wrapper: %WRAPPER%
echo [start_api] GPU=%GPU% PORT=%PORT% (fp32, warmup on)
uv run run_api.py --gpu %GPU% --host 0.0.0.0 --port %PORT% %*
endlocal
