$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$executable = Join-Path $PSScriptRoot ".venv\Scripts\music-annotation-backend.exe"
if (-not (Test-Path -LiteralPath $executable)) {
    throw "Backend environment is missing. Run: uv venv --python 3.11 .venv; uv pip install --python .venv\Scripts\python.exe -e '.[dev,laion]'"
}
Write-Host "Starting CLAP backend. The model is loaded automatically from .env ..."
& $executable
