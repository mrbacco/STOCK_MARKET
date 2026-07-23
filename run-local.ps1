#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: run-local.ps1
#############################

<#
.SYNOPSIS
Runs the Streamlit app in lightweight laptop mode without Docker.

.DESCRIPTION
Forecasting, local SQLite persistence, provider retries, and last-known-good
market snapshots remain enabled. The memory-heavy continuous FinBERT collector
is disabled; previously collected sentiment can still be read by the app.
#>

param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $workspacePath ".venv-runtime\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Local runtime not found at $pythonPath. Install requirements into .venv-runtime first."
}

# Keep the laptop process self-contained. Empty shared-service variables select
# the existing SQLite and in-process cache fallbacks.
$env:DATABASE_URL = ""
$env:REDIS_URL = ""
$env:RUN_IN_PROCESS_SENTIMENT = "false"
$env:ANALYTICS_READ_ONLY = "false"

Write-Host "[BAC_LOG] local.start | mode=lightweight port=$Port sentiment_collector=false"
Write-Host "[BAC_LOG] local.start | url=http://127.0.0.1:$Port"

& $pythonPath -m streamlit run (Join-Path $workspacePath "app.py") `
    --server.port=$Port `
    --server.address=127.0.0.1 `
    --server.headless=true

