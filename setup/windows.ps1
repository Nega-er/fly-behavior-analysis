param([ValidateSet('manual','analysis','analyze','annotate','check')][string]$App='manual',
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$AppArguments)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Runtime=Join-Path $Root '.runtime'
New-Item -ItemType Directory -Force -Path (Join-Path $Runtime 'logs'),(Join-Path $Runtime 'tools'),(Join-Path $Runtime 'cache') | Out-Null
$Log=Join-Path $Runtime 'logs/setup-windows.log'
Start-Transcript -Path $Log -Append | Out-Null
try {
    $Detected=($null -ne $env:CONDA_PREFIX -or $null -ne $env:CONDA_DEFAULT_ENV)
    Get-ChildItem Env: | Where-Object { $_.Name -match '^(CONDA|MAMBA|_CE_|PYTHON|VIRTUAL_ENV|TCL_LIBRARY|TK_LIBRARY|UV_)' } | ForEach-Object { Remove-Item ('Env:'+$_.Name) }
    $env:PATH="$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
    $env:UV_PYTHON_INSTALL_DIR=Join-Path $Runtime 'python'
    $env:UV_CACHE_DIR=Join-Path $Runtime 'cache'
    $env:UV_PYTHON_BIN_DIR=Join-Path $Runtime 'tools'
    $env:UV_NO_MODIFY_PATH='1';$env:UV_SYSTEM_CERTS='true';$env:PYTHONUTF8='1';$env:PYTHONNOUSERSITE='1'
    $env:FLY_CONDA_DETECTED=[string][int]$Detected
    $Lock=Get-Content (Join-Path $PSScriptRoot 'runtime-lock.json') -Raw | ConvertFrom-Json
    $Platform=$Lock.platforms.windows_x86_64
    $Uv=Join-Path $Runtime "tools/uv-$($Lock.uv_version).exe"
    if (!(Test-Path $Uv) -or ((& $Uv --version) -notmatch ('^uv '+[regex]::Escape($Lock.uv_version)+'(\s|$)'))) {
        $Temporary=Join-Path $Runtime ('cache/bootstrap-'+[Guid]::NewGuid())
        New-Item -ItemType Directory $Temporary | Out-Null
        try {
            [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
            $Archive=Join-Path $Temporary $Platform.archive
            Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$($Lock.uv_version)/$($Platform.archive)" -OutFile $Archive
            if ((Get-FileHash $Archive -Algorithm SHA256).Hash.ToLower() -ne $Platform.sha256) { throw 'uv checksum mismatch.' }
            Expand-Archive -Path $Archive -DestinationPath $Temporary
            $Binary=Get-ChildItem $Temporary -Filter uv.exe -Recurse | Select-Object -First 1
            if (!$Binary) { throw 'uv.exe not found in the official archive.' }
            Copy-Item $Binary.FullName $Uv
        } finally { Remove-Item $Temporary -Recurse -Force }
    }
    Write-Host "Conda inherited: $Detected; child runtime: isolated"
    & $Uv python install $Lock.python_version --no-bin
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
    $Python=(& $Uv python find --managed-python $Lock.python_version | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0 -or !(Test-Path $Python)) { throw 'Managed Python not found.' }
    & $Python (Join-Path $PSScriptRoot 'runtime_manager.py') --uv $Uv --app $App @AppArguments
    if ($LASTEXITCODE -ne 0) { throw "Application/setup failed with exit code $LASTEXITCODE. See $Log" }
} catch {
    Write-Host $_ -ForegroundColor Red
    if (!$env:CI) { Read-Host 'Press Enter to close' }
    exit 1
} finally { Stop-Transcript | Out-Null }
