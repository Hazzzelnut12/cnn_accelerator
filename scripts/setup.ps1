$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$uvVersion = '0.13.0'
$uvDirectory = Join-Path $projectRoot ".tools\uv\$uvVersion"
$uvExecutable = Join-Path $uvDirectory 'uv.exe'

if (-not (Test-Path -LiteralPath $uvExecutable)) {
    $downloadDirectory = Join-Path $projectRoot '.cache\bootstrap'
    New-Item -ItemType Directory -Force -Path $downloadDirectory | Out-Null
    $archiveName = 'uv-x86_64-pc-windows-msvc.zip'
    $archivePath = Join-Path $downloadDirectory $archiveName
    $checksumPath = "$archivePath.sha256"
    $releaseUrl = "https://github.com/astral-sh/uv/releases/download/$uvVersion"
    Write-Host "Downloading uv $uvVersion..."
    Invoke-WebRequest -UseBasicParsing -Uri "$releaseUrl/$archiveName" -OutFile $archivePath
    Invoke-WebRequest -UseBasicParsing -Uri "$releaseUrl/$archiveName.sha256" -OutFile $checksumPath
    $expectedHash = [regex]::Match((Get-Content -LiteralPath $checksumPath -Raw), '\b[a-fA-F0-9]{64}\b').Value
    $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
    if (-not $expectedHash -or $actualHash -ne $expectedHash) {
        throw 'uv download checksum mismatch.'
    }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $uvDirectory -Force
}

$installedVersion = & $uvExecutable --version
if ($LASTEXITCODE -ne 0 -or $installedVersion -notmatch "^uv $([regex]::Escape($uvVersion))(\s|$)") {
    throw "Expected uv $uvVersion; got: $installedVersion"
}

# Keep all installations in this project, without changing the system Python or PATH.
$projectEnvironment = @{
    UV_CACHE_DIR = (Join-Path $projectRoot '.cache\uv')
    UV_PYTHON_INSTALL_DIR = (Join-Path $projectRoot '.tools\python')
    UV_PROJECT_ENVIRONMENT = (Join-Path $projectRoot '.venv')
}
$previousEnvironment = @{}
Push-Location $projectRoot
try {
    foreach ($variableName in $projectEnvironment.Keys) {
        $previousEnvironment[$variableName] = [Environment]::GetEnvironmentVariable($variableName, 'Process')
        [Environment]::SetEnvironmentVariable($variableName, $projectEnvironment[$variableName], 'Process')
    }
    if (-not (Test-Path -LiteralPath 'uv.lock')) {
        & $uvExecutable lock --managed-python
        if ($LASTEXITCODE -ne 0) { throw 'Dependency resolution failed.' }
    }
    & $uvExecutable sync --locked --managed-python
    if ($LASTEXITCODE -ne 0) { throw 'Environment setup failed.' }

    & (Join-Path $projectRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'check_env.py') --output (Join-Path $projectRoot 'docs\validation\environment.json')
    if ($LASTEXITCODE -ne 0) { throw 'CUDA validation failed.' }
}
finally {
    foreach ($variableName in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($variableName, $previousEnvironment[$variableName], 'Process')
    }
    Pop-Location
}
