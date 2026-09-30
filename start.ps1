param([switch]$Basic)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

try {
    $uvCommand = Get-Command uv.exe -CommandType Application -ErrorAction SilentlyContinue
    $localUv = Join-Path $PSScriptRoot ".tools\uv\uv.exe"
    if ($uvCommand) {
        $uvExecutable = $uvCommand.Source
    } elseif (Test-Path -LiteralPath $localUv) {
        $uvExecutable = $localUv
    } else {
        Write-Host "Installing uv locally (no administrator password needed)..."
        $installDirectory = Join-Path $PSScriptRoot ".tools\uv"
        New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null
        $installer = Join-Path $installDirectory "install.ps1"
        $previousInstallDir = $env:UV_INSTALL_DIR
        $previousNoModifyPath = $env:UV_NO_MODIFY_PATH
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -UseBasicParsing -Uri "https://astral.sh/uv/install.ps1" -OutFile $installer
            $env:UV_INSTALL_DIR = $installDirectory
            $env:UV_NO_MODIFY_PATH = "1"
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $localUv)) {
                throw "uv installation failed. Check your internet connection and retry."
            }
        } finally {
            $env:UV_INSTALL_DIR = $previousInstallDir
            $env:UV_NO_MODIFY_PATH = $previousNoModifyPath
            if (Test-Path -LiteralPath $installer) {
                Remove-Item -LiteralPath $installer -Force
            }
        }
        $uvExecutable = $localUv
    }

    Write-Host "Preparing Python 3.11 and the backend environment..."
    & $uvExecutable sync --locked --python 3.11 --inexact
    if ($LASTEXITCODE -ne 0) {
        throw "Environment setup failed. Check your internet connection and retry."
    }
    $launchArguments = @("run", "--no-sync", "python", "scripts/launch.py", "--uv", $uvExecutable)
    if ($Basic) {
        $launchArguments += "--basic"
    }
    & $uvExecutable @launchArguments
    exit $LASTEXITCODE
} catch {
    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
