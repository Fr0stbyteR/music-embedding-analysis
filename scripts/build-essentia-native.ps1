param([switch]$DownloadModels, [string]$Generator = 'Visual Studio 18 2026')
$ErrorActionPreference = 'Stop'
$backendRoot = Split-Path $PSScriptRoot -Parent
$revision = '7320015a1cad3ac1dc038b52ef94803587d09986'
$downloads = Join-Path $backendRoot 'vendor/downloads'
New-Item -ItemType Directory -Force $downloads | Out-Null

function Get-VerifiedArchive([string]$Url, [string]$Name, [string]$Sha256) {
    $archive = Join-Path $downloads $Name
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest $Url -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Sha256) {
        throw "Checksum mismatch: $archive. Remove this archive and retry."
    }
    return $archive
}

$upstream = Join-Path $backendRoot 'vendor/essentia'
if (-not (Test-Path -LiteralPath (Join-Path $upstream '.git'))) {
    git clone --no-checkout https://github.com/MTG/essentia.git $upstream
    if ($LASTEXITCODE -ne 0) { throw 'Essentia source download failed' }
    git -C $upstream checkout --detach $revision
    if ($LASTEXITCODE -ne 0) { throw 'Essentia revision checkout failed' }
}
$actualRevision = git -c "safe.directory=$($upstream.Replace('\', '/'))" -C $upstream rev-parse HEAD
if ($actualRevision -ne $revision) { throw "Expected Essentia $revision, found $actualRevision. Use a separate vendor checkout of the pinned revision." }
$sourceChanges = git -c "safe.directory=$($upstream.Replace('\', '/'))" -C $upstream status --porcelain --untracked-files=no
if ($sourceChanges) { throw 'Essentia vendor sources have local changes; refusing an unreproducible build.' }
$eigenArchive = Get-VerifiedArchive 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz' 'eigen-3.4.0.tar.gz' '8586084F71F9BDE545EE7FA6D00288B264A2B7AC3607B974E54D13E7162C1C72'
if (-not (Test-Path -LiteralPath (Join-Path $backendRoot 'vendor/eigen-3.4.0/Eigen/Core'))) {
    tar -xzf $eigenArchive -C (Join-Path $backendRoot 'vendor')
    if ($LASTEXITCODE -ne 0) { throw 'Eigen extraction failed' }
}
$tensorflowArchive = Get-VerifiedArchive 'https://storage.googleapis.com/tensorflow/versions/2.18.1/libtensorflow-cpu-windows-x86_64.zip' 'libtensorflow-2.18.1.zip' '28ACDCEA6C6B34828CF0E95E67802B0F3577D51BC2E8915DE811B7AA0B04452D'
if (-not (Test-Path -LiteralPath (Join-Path $backendRoot 'vendor/libtensorflow/lib/tensorflow.dll'))) {
    Expand-Archive -LiteralPath $tensorflowArchive -DestinationPath (Join-Path $backendRoot 'vendor/libtensorflow')
}
$python = Join-Path $backendRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run the normal backend setup first to create .venv.' }
$build = Join-Path $backendRoot 'build/essentia'
cmake -S (Join-Path $backendRoot 'native/essentia') -B $build -G $Generator -A x64 "-DPython3_EXECUTABLE=$python"
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
cmake --build $build --config Release --parallel 6
if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
ctest --test-dir $build -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Native self-test failed' }
if ($DownloadModels) {
    Write-Host 'Downloading official DEAM/MusiCNN weights (CC BY-NC-SA 4.0; non-commercial).'
    $models = Join-Path $backendRoot 'models/essentia'
    New-Item -ItemType Directory -Force $models | Out-Null
    foreach ($entry in @(
        @('https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb', 'msd-musicnn-1.pb'),
        @('https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.pb', 'deam-msd-musicnn-2.pb')
    )) {
        $target = Join-Path $models $entry[1]
        if (-not (Test-Path -LiteralPath $target)) {
            $partial = "$target.download"
            Invoke-WebRequest $entry[0] -OutFile $partial
            Move-Item -LiteralPath $partial -Destination $target
        }
    }
}
Write-Host 'Native worker ready. Start the backend normally; no WSL or Essentia Python binding needed.'
