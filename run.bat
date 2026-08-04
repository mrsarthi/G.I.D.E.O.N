@echo off
REM ============================================================
REM G.I.D.E.O.N — FastAPI Web Server
REM Port and tuning come from config.py / environment variables.
REM ============================================================

cd /d "%~dp0"

REM --- VRAM tuning for 4GB cards -------------------------------
REM IMPORTANT: these are read by the OLLAMA SERVICE, not by this
REM script's python process. Setting them here only takes effect if
REM Ollama is started afterwards from this same shell. To make them
REM stick, run once and then restart Ollama from the tray:
REM     setx OLLAMA_FLASH_ATTENTION 1
REM     setx OLLAMA_KV_CACHE_TYPE q8_0
REM Quantising the KV cache halves its footprint (~940MB -> ~470MB at
REM 8k context) — that is what leaves room for the 3B model and the
REM WebGL hologram to coexist on a 1650 Ti.
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0

if "%GIDEON_PORT%"=="" set GIDEON_PORT=7777
if "%GIDEON_HOST%"=="" set GIDEON_HOST=127.0.0.1

echo [G.I.D.E.O.N] Starting server on http://%GIDEON_HOST%:%GIDEON_PORT%/ ...
.venv\Scripts\python.exe -m uvicorn server:app --host %GIDEON_HOST% --port %GIDEON_PORT% --reload
