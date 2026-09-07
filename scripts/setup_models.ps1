param(
    [switch]$InstallLaion,
    [switch]$InstallMuQ,
    [switch]$CloneM2D
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Create the environment first: uv venv --python 3.11 .venv; uv sync --extra dev"
}

if ($InstallLaion) {
    & $python -m pip install laion-clap
}
if ($InstallMuQ) {
    & $python -m pip install muq
}
if ($CloneM2D) {
    $target = Join-Path $workspace "vendor\m2d"
    if (-not (Test-Path -LiteralPath $target)) {
        git clone --depth 1 https://github.com/nttcslab/m2d.git $target
    }
}

Write-Host "M2D checkpoint must be placed at models\m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025\checkpoint-30.pth."
