# ---------------------------------------------------------------------------
# run-integration-tests.ps1
#
# Spins up truellm-server.exe with the requested config.toml, waits for the
# /health endpoint, runs the matching Python integration test script, then
# tears the server down. The server is always killed in the `finally` block
# even if the test script throws or the user hits Ctrl+C.
#
# Suites (mapping: config -> python script):
#   smoke        nemotron-cuda.toml           test_nemotron.py
#   batch        test-batching-stress.toml    test_batch_concurrent.py
#   compression  nemotron-cuda.toml           test_compression_rlm.py
#   megakernel   cuda-megakernel.toml         test_megakernel_compression.py
#                                              (filtered: cuda-only,megakernel-smoke,
#                                               compression-absent-shape)
#   kivi         cuda-kivi-compression.toml   test_megakernel_compression.py
#                                              (filtered: kivi-enabled,
#                                               presis-long-prompt,
#                                               compression-stress)
#   vq           cuda-vq-compression.toml     test_megakernel_compression.py
#                                              (filtered: vq-enabled, tome-enabled,
#                                               compression-stress)
#   stress       cuda-combined-stress.toml    test_megakernel_compression.py
#                                              (filtered: all)
#   all          runs every suite sequentially
#
# Usage:
#   pwsh .\scripts\windows\run-integration-tests.ps1 -Suite smoke
#   pwsh .\scripts\windows\run-integration-tests.ps1 -Suite all -Verbose
#   pwsh .\scripts\windows\run-integration-tests.ps1 `
#       -Suite compression `
#       -Config .\configs\scenario-quantized-kv.toml `
#       -ServerExe .\build\windows-cuda\bin\truellm-server.exe
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [ValidateSet('smoke', 'batch', 'compression', 'megakernel', 'kivi', 'vq', 'stress', 'all')]
    [string]$Suite = 'smoke',

    # Optional override: use this config instead of the suite default.
    [string]$Config = '',

    # Optional override: path to truellm-server.exe.
    [string]$ServerExe = '',

    # Optional override: python interpreter.
    [string]$Python = 'python',

    # Host/port used to health-check and passed to the Python scripts via
    # TRUELLM_HOST. Must match host/port in the chosen config.toml.
    [string]$ServerHost = '127.0.0.1',
    [int]$ServerPort = 9099,

    # Seconds to wait for /health to return 200 before giving up.
    [int]$StartupTimeoutSec = 90,

    # Optional bearer token (exported as TRUELLM_API_KEY to the Python runner).
    [string]$ApiKey = '',

    # Forward --verbose to the Python test scripts.
    [switch]$VerboseTests
)

$ErrorActionPreference = 'Stop'

# Resolve repository root as two levels up from this script (scripts/windows).
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')

function Resolve-ServerExe {
    if ($ServerExe) {
        return (Resolve-Path $ServerExe).Path
    }
    $candidates = @(
        (Join-Path $repoRoot 'build\windows-cuda\bin\truellm-server.exe'),
        (Join-Path $repoRoot 'build\windows-cuda\Release\truellm-server.exe'),
        (Join-Path $repoRoot 'build\windows-cuda\bin\Release\truellm-server.exe')
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return (Resolve-Path $c).Path }
    }
    throw "Could not locate truellm-server.exe. Pass -ServerExe or build windows-cuda first. Checked: $($candidates -join '; ')"
}

function Get-SuitePlan {
    param([string]$Name)
    switch ($Name) {
        'smoke'       { return @{ Config = 'configs\nemotron-cuda.toml';          Script = 'test_nemotron.py';                SubSuite = '' } }
        'batch'       { return @{ Config = 'configs\test-batching-stress.toml';   Script = 'test_batch_concurrent.py';        SubSuite = '' } }
        'compression' { return @{ Config = 'configs\nemotron-cuda.toml';          Script = 'test_compression_rlm.py';         SubSuite = '' } }
        'megakernel'  { return @{ Config = 'configs\cuda-megakernel.toml';        Script = 'test_megakernel_compression.py';
                                   SubSuite = 'cuda-only,megakernel-smoke,compression-absent-shape' } }
        'kivi'        { return @{ Config = 'configs\cuda-kivi-compression.toml';  Script = 'test_megakernel_compression.py';
                                   SubSuite = 'cuda-only,kivi-enabled,presis-long-prompt,compression-stress' } }
        'vq'          { return @{ Config = 'configs\cuda-vq-compression.toml';    Script = 'test_megakernel_compression.py';
                                   SubSuite = 'cuda-only,vq-enabled,tome-enabled,compression-stress' } }
        'stress'      { return @{ Config = 'configs\cuda-combined-stress.toml';   Script = 'test_megakernel_compression.py';
                                   SubSuite = 'all' } }
    }
    throw "Unknown suite: $Name"
}

function Wait-ForHealth {
    param([string]$Url, [int]$TimeoutSec)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) { return $true }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

function Invoke-Suite {
    param(
        [string]$Name,
        [string]$ExePath,
        [string]$ConfigOverride
    )

    $plan = Get-SuitePlan -Name $Name
    $configPath = if ($ConfigOverride) { (Resolve-Path $ConfigOverride).Path } else { (Resolve-Path (Join-Path $repoRoot $plan.Config)).Path }
    $scriptPath = (Resolve-Path (Join-Path $repoRoot "tests\integration\python\$($plan.Script)")).Path

    Write-Host "=========================================================" -ForegroundColor Cyan
    Write-Host "Suite : $Name"                   -ForegroundColor Cyan
    Write-Host "Config: $configPath"              -ForegroundColor Cyan
    Write-Host "Script: $scriptPath"              -ForegroundColor Cyan
    Write-Host "Server: $ExePath"                 -ForegroundColor Cyan
    Write-Host "=========================================================" -ForegroundColor Cyan

    $logDir = Join-Path $repoRoot 'build\integration-logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdoutLog = Join-Path $logDir "server-$Name-$stamp.out.log"
    $stderrLog = Join-Path $logDir "server-$Name-$stamp.err.log"

    $server = Start-Process -FilePath $ExePath `
        -ArgumentList @('-c', $configPath) `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError  $stderrLog `
        -PassThru -NoNewWindow

    try {
        $healthUrl = "http://${ServerHost}:${ServerPort}/health"
        Write-Host "Waiting for $healthUrl (timeout ${StartupTimeoutSec}s, pid=$($server.Id))..."
        if (-not (Wait-ForHealth -Url $healthUrl -TimeoutSec $StartupTimeoutSec)) {
            Write-Host "--- server stdout tail ---" -ForegroundColor Yellow
            if (Test-Path $stdoutLog) { Get-Content $stdoutLog -Tail 40 | Write-Host }
            Write-Host "--- server stderr tail ---" -ForegroundColor Yellow
            if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 40 | Write-Host }
            throw "Server did not become healthy within $StartupTimeoutSec s."
        }
        Write-Host "Server healthy." -ForegroundColor Green

        $env:TRUELLM_HOST    = "http://${ServerHost}:${ServerPort}"
        $env:TRUELLM_API_KEY = $ApiKey

        $pyArgs = @($scriptPath)
        if ($plan.SubSuite) { $pyArgs += @('--suite', $plan.SubSuite) }
        if ($VerboseTests) { $pyArgs += '--verbose' }

        & $Python @pyArgs
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            throw "Integration suite '$Name' failed (exit $code). Logs: $stdoutLog"
        }
        Write-Host "Suite '$Name' passed." -ForegroundColor Green
    }
    finally {
        if ($server -and -not $server.HasExited) {
            Write-Host "Stopping server (pid=$($server.Id))..."
            try {
                Stop-Process -Id $server.Id -Force -ErrorAction Stop
                $server.WaitForExit(10000) | Out-Null
            } catch {
                Write-Host "Failed to stop server cleanly: $_" -ForegroundColor Yellow
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
$exe = Resolve-ServerExe

$suitesToRun = if ($Suite -eq 'all') {
    @('smoke', 'batch', 'compression', 'megakernel', 'kivi', 'vq', 'stress')
} else { @($Suite) }

$failed = @()
foreach ($s in $suitesToRun) {
    try {
        Invoke-Suite -Name $s -ExePath $exe -ConfigOverride $Config
    } catch {
        Write-Host "FAILED: $s -> $_" -ForegroundColor Red
        $failed += $s
    }
}

if ($failed.Count -gt 0) {
    Write-Host "Failed suites: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "All suites passed." -ForegroundColor Green
