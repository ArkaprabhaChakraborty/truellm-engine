#Requires -RunAsAdministrator
# ---------------------------------------------------------------------------
# create-dev-cert.ps1
#
# Creates a local code-signing certificate for development builds of
# truellm-server, installs it into the Trusted Root store so Windows
# Smart App Control accepts self-signed binaries, and prints the thumbprint
# to copy into your CMake configure command.
#
# Run once from an Administrator PowerShell:
#   .\scripts\windows\create-dev-cert.ps1
#
# Then configure CMake with the printed thumbprint:
#   cmake -B build -DTRUELLM_CODESIGN=ON `
#         -DTRUELLM_CODESIGN_CERT_SHA1=<thumbprint printed below>
#
# NOTE: Self-signed certificates are trusted only on this machine.
#       For public distribution obtain a certificate from a trusted CA
#       (DigiCert, Sectigo, GlobalSign, etc.).
# ---------------------------------------------------------------------------

param(
    [string]$CertSubject  = "CN=TrueLLM-Dev",
    [string]$ExportPath   = "$PSScriptRoot\truellm-dev-cert.cer",
    [int]   $ValidityDays = 1825   # 5 years
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$notAfter = (Get-Date).AddDays($ValidityDays)

Write-Host "Creating code-signing certificate: $CertSubject"

# Create the certificate in the personal store.
$cert = New-SelfSignedCertificate `
    -Type          CodeSigning `
    -Subject       $CertSubject `
    -HashAlgorithm SHA256 `
    -KeyLength     4096 `
    -KeyUsage      DigitalSignature `
    -NotAfter      $notAfter `
    -CertStoreLocation "Cert:\CurrentUser\My"

Write-Host "Certificate created."
Write-Host "  Thumbprint : $($cert.Thumbprint)"
Write-Host "  Subject    : $($cert.Subject)"
Write-Host "  Expires    : $($cert.NotAfter)"

# Export the public cert so the Trusted Root import works without the private key.
Export-Certificate -Cert $cert -FilePath $ExportPath -Type CERT | Out-Null
Write-Host "  Exported to: $ExportPath"

# Import into Trusted Root so signtool verification passes locally.
$rootStore = [System.Security.Cryptography.X509Certificates.X509Store]::new(
    [System.Security.Cryptography.X509Certificates.StoreName]::Root,
    [System.Security.Cryptography.X509Certificates.StoreLocation]::LocalMachine
)
$rootStore.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadWrite)
$rootStore.Add([System.Security.Cryptography.X509Certificates.X509Certificate2]::new($ExportPath))
$rootStore.Close()
Write-Host "  Installed in Trusted Root (LocalMachine)."

Write-Host ""
Write-Host "---------------------------------------------------------------------"
Write-Host "CMake configure command:"
Write-Host ""
Write-Host "  cmake -B build\windows-cuda -DTRUELLM_CODESIGN=ON ``"
Write-Host "        -DTRUELLM_CODESIGN_CERT_SHA1=$($cert.Thumbprint)"
Write-Host ""
Write-Host "Every build will now call signtool automatically as a post-build step."
Write-Host "---------------------------------------------------------------------"
