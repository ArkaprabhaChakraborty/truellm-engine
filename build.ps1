# ---------------------------------------------------------------------------
# build.ps1 - TrueLLM build script
#
# Usage:
#   .\build.ps1                          # CUDA Release build
#   .\build.ps1 -Backend CPU             # CPU-only build
#   .\build.ps1 -Config Debug            # Debug build
#   .\build.ps1 -Rebuild                 # wipe build dir, then configure + build
#   .\build.ps1 -Clean                   # wipe build dir and exit (no build)
#   .\build.ps1 -GenerateCert            # create dev code-signing cert, print thumbprint, exit
#   .\build.ps1 -Sign -CertSha1 <sha1>   # build + sign with existing cert
#   .\build.ps1 -SkipTests               # disable test targets
#   .\build.ps1 -Run                     # start server after build
#   .\build.ps1 -Test                    # build then run unit tests
#
# Cert workflow (two separate steps):
#   1.  .\build.ps1 -GenerateCert        # needs admin; prints thumbprint
#   2.  .\build.ps1 -Sign -CertSha1 <thumbprint>  # normal build + sign
# ---------------------------------------------------------------------------

param(
    [ValidateSet("CUDA","CPU")]
    [string]$Backend    = "CUDA",

    [ValidateSet("Release","Debug","RelWithDebInfo")]
    [string]$Config     = "Release",

    [string]$CudaArchs  = "75;80;86;89;90;120",

    # Thumbprint of the code-signing cert to use with -Sign.
    # Obtain one by running: .\build.ps1 -GenerateCert
    [string]$CertSha1   = "",

    [switch]$Sign,
    [switch]$Rebuild,       # clean build dir then build
    [switch]$Clean,         # clean build dir and exit
    [switch]$GenerateCert,  # create dev cert, print thumbprint, exit
    [switch]$SkipTests,
    [switch]$Benchmarks,
    [switch]$ClangTidy,
    [switch]$Cppcheck,
    [switch]$Run,
    [switch]$Test,

    [string]$ServerConfig = "configs\profiles\local-cpu.toml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root      = $PSScriptRoot
$BuildDir  = "$Root\build\windows-$(($Backend).ToLower())"
$ServerExe = "$BuildDir\bin\truellm-server.exe"
$TestExe   = "$BuildDir\bin\truellm_unit_tests.exe"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Step([string]$msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Die([string]$msg) {
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

function Require-Env([string]$name) {
    if (-not (Get-Item "env:$name" -ErrorAction SilentlyContinue)) {
        Die "$name environment variable is not set."
    }
}

function Is-Admin {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# ---------------------------------------------------------------------------
# -GenerateCert  --  create a self-signed dev code-signing certificate,
#                    install it into Trusted Root, print the thumbprint,
#                    then exit.  Run this ONCE; then use -Sign -CertSha1.
#
#                    Requires an Administrator PowerShell session.
# ---------------------------------------------------------------------------
if ($GenerateCert) {
    if (-not (Is-Admin)) {
        Die ("Certificate generation requires Administrator.`n" +
             "  Re-run from an elevated prompt:`n" +
             "    Start-Process powershell -Verb RunAs " +
             "-ArgumentList '-ExecutionPolicy Bypass -File $PSCommandPath -GenerateCert'")
    }

    Step "Generating TrueLLM dev code-signing certificate"

    $certSubject  = "CN=TrueLLM-Dev"
    $validityDays = 1825   # 5 years
    $exportPath   = "$Root\scripts\windows\truellm-dev-cert.cer"
    $notAfter     = (Get-Date).AddDays($validityDays)

    $cert = New-SelfSignedCertificate `
        -Type              CodeSigning `
        -Subject           $certSubject `
        -HashAlgorithm     SHA256 `
        -KeyLength         4096 `
        -KeyUsage          DigitalSignature `
        -NotAfter          $notAfter `
        -CertStoreLocation "Cert:\CurrentUser\My"

    # Export public cert so the Trusted Root import doesn't need the private key.
    $null = New-Item -ItemType Directory -Force -Path (Split-Path $exportPath)
    Export-Certificate -Cert $cert -FilePath $exportPath -Type CERT | Out-Null

    # Install into LocalMachine Trusted Root so signtool verification passes locally.
    $store = [System.Security.Cryptography.X509Certificates.X509Store]::new(
        [System.Security.Cryptography.X509Certificates.StoreName]::Root,
        [System.Security.Cryptography.X509Certificates.StoreLocation]::LocalMachine)
    $store.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadWrite)
    $store.Add([System.Security.Cryptography.X509Certificates.X509Certificate2]::new($exportPath))
    $store.Close()

    Write-Host ""
    Write-Host "  Thumbprint : " -NoNewline
    Write-Host $cert.Thumbprint -ForegroundColor Green
    Write-Host "  Subject    : $($cert.Subject)"
    Write-Host "  Expires    : $($cert.NotAfter)"
    Write-Host "  Trusted Root: installed (LocalMachine)"
    Write-Host "  Exported to : $exportPath"
    Write-Host ""
    Write-Host "  Build with signing:" -ForegroundColor Cyan
    Write-Host "    .\build.ps1 -Sign -CertSha1 $($cert.Thumbprint)" -ForegroundColor White
    Write-Host ""

    exit 0
}

# ---------------------------------------------------------------------------
# -Clean  --  wipe build directory and exit (no build).
# ---------------------------------------------------------------------------
if ($Clean) {
    if (Test-Path $BuildDir) {
        Step "Cleaning $BuildDir"
        Remove-Item -Recurse -Force $BuildDir
        Write-Host "Clean complete." -ForegroundColor Green
    } else {
        Write-Host "Nothing to clean ($BuildDir does not exist)." -ForegroundColor Yellow
    }
    exit 0
}

# ---------------------------------------------------------------------------
# -Rebuild  --  wipe build directory, then fall through to configure + build.
# ---------------------------------------------------------------------------
if ($Rebuild -and (Test-Path $BuildDir)) {
    Step "Cleaning $BuildDir (rebuild)"
    Remove-Item -Recurse -Force $BuildDir
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
Require-Env "VCPKG_ROOT"
$toolchain = "$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
if (-not (Test-Path $toolchain)) { Die "vcpkg toolchain not found at: $toolchain" }

if ($Sign -and -not $CertSha1) {
    Die ("-Sign requires -CertSha1 <thumbprint>.`n" +
         "  Generate a cert first: .\build.ps1 -GenerateCert")
}

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
Step "Configuring ($Backend / $Config)"

$cmake_args = @(
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DTRUELLM_BACKEND=$Backend",
    "-DTRUELLM_BUILD_TESTS=$(if ($SkipTests) { 'OFF' } else { 'ON' })",
    "-DTRUELLM_BUILD_BENCHMARKS=$(if ($Benchmarks) { 'ON' } else { 'OFF' })",
    "-DTRUELLM_BUILD_CUDA_ENGINE=$(if ($Backend -eq 'CUDA') { 'ON' } else { 'OFF' })",
    "-DTRUELLM_ENABLE_CLANG_TIDY=$(if ($ClangTidy) { 'ON' } else { 'OFF' })",
    "-DTRUELLM_ENABLE_CPPCHECK=$(if ($Cppcheck) { 'ON' } else { 'OFF' })"
)

if ($Backend -eq "CUDA") {
    $cmake_args += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchs"
}

if ($Sign) {
    $cmake_args += "-DTRUELLM_CODESIGN=ON"
    $cmake_args += "-DTRUELLM_CODESIGN_CERT_SHA1=$CertSha1"
}

& cmake @cmake_args
if ($LASTEXITCODE -ne 0) { Die "CMake configure failed." }

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
Step "Building"

& cmake --build $BuildDir --parallel
if ($LASTEXITCODE -ne 0) { Die "Build failed." }

Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "  Server : $ServerExe"
if (-not $SkipTests) {
    Write-Host "  Tests  : $TestExe"
}

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if ($Test) {
    if (-not (Test-Path $TestExe)) { Die "Test binary not found: $TestExe" }
    Step "Running unit tests"
    & ctest --test-dir $BuildDir --output-on-failure --parallel 4
    if ($LASTEXITCODE -ne 0) { Die "One or more tests failed." }
    Write-Host "All tests passed." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if ($Run) {
    $cfg = "$Root\$ServerConfig"
    if (-not (Test-Path $cfg)) { Die "Config not found: $cfg" }
    if (-not (Test-Path $ServerExe)) { Die "Server binary not found: $ServerExe" }
    Step "Starting server with config: $ServerConfig"
    & $ServerExe --config $cfg
}
