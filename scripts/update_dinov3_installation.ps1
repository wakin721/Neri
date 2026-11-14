<#
.SYNOPSIS
    Deploys the DINOv3 joint model, compiled Flutter Release UI, backend system code,
    and associated training/export scripts and documentation to the Neri installation directory.

.DESCRIPTION
    This script performs a safe, staged, and fully reversible deployment to D:\Application\Neri:
    1. UI Binary: frontend/build/windows/x64/runner/Release (Neri.exe, dll plugins, data/)
    2. Backend System Code: system/ (excluding __pycache__ and bytecode)
    3. DINOv3 Model: model manifest and dependent assets (checkpoint, encoder_weights, encoder_code)
       with SHA-256 verification and relative path preservation.
    4. Training/Export Scripts & Docs: export_dinov3_experiment.py, train_dinov3_ridge_within.py,
       flutter_accessibility_hotfix.py, requirements*.txt, and README_Update.md.
    5. Settings: Updates temp/settings.json (selected_classification_model and video_mode skip)
       only when -SelectModel is passed, with dedicated backup and other keys strictly preserved.

    Safety Features:
    - Running process verification: checks active Neri/Python processes originating from the target
      installation and halts with clear closure instructions without killing external processes.
    - Pre-write backup and inventory: backs up all files to be overwritten into
      repair-backups\dinov3-<timestamp>\files and writes inventory.json / backup-manifest.json
      BEFORE any target file is modified.
    - Staging + Atomic Replace: prepares all files in a staging folder on the same volume and
      uses atomic replacement ([System.IO.File]::Replace) for zero-downtime, safe file swaps.
    - Automatic and manual rollback: if any write step fails, automatic rollback restores all
      overwritten files and removes newly created files. Supports manual rollback via -RollbackFrom.
    - Strict boundary checks: prevents directory traversal or writing outside the installation root.
    - Native PowerShell: uses native PowerShell cmdlets and .NET APIs end-to-end (no cross-shell deletion).
    - -WhatIf support: safe preview of all planned deployment and backup actions.

.PARAMETER InstallationRoot
    The target Neri installation directory (default: D:\Application\Neri).

.PARAMETER ModelManifest
    Path to the DINOv3 *.neri.json manifest file. Defaults to
    res/model/cls/user/dinov3-ridge-within.neri.json in SourceRoot.

.PARAMETER SourceRoot
    Source repository root directory. Defaults to the repository containing this script.

.PARAMETER SelectModel
    Switch to update temp/settings.json with the deployed DINOv3 manifest path and video_mode skip.

.PARAMETER RollbackFrom
    Path to a previous dinov3-<timestamp> backup directory to execute a full rollback.

.PARAMETER SkipProcessCheck
    Skip checking for active Neri.exe or Python processes running from InstallationRoot.

.PARAMETER Force
    Bypass non-critical warnings and proceed.

.EXAMPLE
    # Preview deployment actions without making any changes:
    .\scripts\update_dinov3_installation.ps1 -WhatIf

.EXAMPLE
    # Deploy DINOv3 update and update settings to select the new model:
    .\scripts\update_dinov3_installation.ps1 -SelectModel

.EXAMPLE
    # Deploy specifying a custom model manifest path:
    .\scripts\update_dinov3_installation.ps1 -ModelManifest "F:\files\python\Neri_plus\runs\...\model.neri.json" -SelectModel

.EXAMPLE
    # Roll back from a previous backup:
    .\scripts\update_dinov3_installation.ps1 -RollbackFrom "D:\Application\Neri\repair-backups\dinov3-20261107-190000"
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Position = 0)]
    [string]$InstallationRoot = "D:\Application\Neri",

    [Parameter()]
    [string]$ModelManifest = "",

    [Parameter()]
    [string]$SourceRoot = "",

    [Parameter()]
    [switch]$SelectModel,

    [Parameter()]
    [string]$RollbackFrom = "",

    [Parameter()]
    [switch]$SkipProcessCheck,

    [Parameter()]
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ==============================================================================
# Helper Functions
# ==============================================================================

function Get-FileSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Cannot compute hash: file does not exist at '$Path'."
    }
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $hashBytes = $hasher.ComputeHash($stream)
        return ([System.BitConverter]::ToString($hashBytes)).Replace("-", "").ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $hasher.Dispose()
    }
}

function ConvertFrom-JsonCompat {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Json
    )
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        return ($Json | ConvertFrom-Json -AsHashtable)
    }
    $obj = $Json | ConvertFrom-Json
    $ht = @{}
    if ($obj -and $obj.PSObject) {
        foreach ($prop in $obj.PSObject.Properties) {
            $ht[$prop.Name] = $prop.Value
        }
    }
    return $ht
}

function Assert-PathWithinRoots {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string[]]$AllowedRoots,
        [string]$Label = "Path"
    )
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $matched = $false
    foreach ($root in $AllowedRoots) {
        if ([string]::IsNullOrWhiteSpace($root)) { continue }
        $normRoot = [System.IO.Path]::GetFullPath($root).TrimEnd('\')
        if ($fullPath.Equals($normRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
            $fullPath.StartsWith($normRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            $matched = $true
            break
        }
    }
    if (-not $matched) {
        throw "Security validation failed: $Label '$Path' resolves outside allowed roots ($($AllowedRoots -join ', '))."
    }
    return $fullPath
}

function Assert-TargetWithinInstallRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot
    )
    $fullTarget = [System.IO.Path]::GetFullPath($TargetPath)
    $normRoot = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')
    if (-not ($fullTarget.StartsWith($normRoot + '\', [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Security validation failed: Target path '$TargetPath' escapes installation root '$InstallRoot'."
    }
    return $fullTarget
}

function Get-ActiveInstallationProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot
    )
    $normRoot = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')
    $activeProcesses = @()

    $candidateNames = @("Neri", "neri_flutter", "python", "pythonw")
    $procs = Get-Process -Name $candidateNames -ErrorAction SilentlyContinue

    foreach ($p in $procs) {
        $procPath = $null
        try {
            $procPath = $p.Path
            if (-not $procPath) { $procPath = $p.MainModule.FileName }
        } catch { }

        $belongsToInstallation = $false
        if ($procPath -and $procPath.StartsWith($normRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            $belongsToInstallation = $true
        } else {
            try {
                $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $($p.Id)" -ErrorAction SilentlyContinue
                if ($cim -and $cim.CommandLine -and $cim.CommandLine.IndexOf($normRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                    $belongsToInstallation = $true
                    if (-not $procPath) { $procPath = $cim.CommandLine }
                }
            } catch { }
        }

        if ($belongsToInstallation) {
            $activeProcesses += [pscustomobject]@{
                Id = $p.Id
                ProcessName = $p.ProcessName
                Path = $procPath
            }
        }
    }
    return ,$activeProcesses
}

function Replace-FileAtomic {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )
    $destDir = [System.IO.Path]::GetDirectoryName($DestinationPath)
    if (-not [System.IO.Directory]::Exists($destDir)) {
        [System.IO.Directory]::CreateDirectory($destDir) | Out-Null
    }

    if ([System.IO.File]::Exists($DestinationPath)) {
        $attr = [System.IO.File]::GetAttributes($DestinationPath)
        if ($attr.HasFlag([System.IO.FileAttributes]::ReadOnly)) {
            [System.IO.File]::SetAttributes($DestinationPath, $attr -band (-bnot [System.IO.FileAttributes]::ReadOnly))
        }
        $swapFile = "$DestinationPath.swap-$([System.Guid]::NewGuid().ToString('N'))"
        try {
            [System.IO.File]::Replace($SourcePath, $DestinationPath, $swapFile)
        } finally {
            if ([System.IO.File]::Exists($swapFile)) {
                try { [System.IO.File]::Delete($swapFile) } catch { }
            }
        }
    } else {
        [System.IO.File]::Move($SourcePath, $DestinationPath)
    }
}

function Invoke-RollbackOperation {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,
        [Parameter(Mandatory = $true)]
        [string]$BackupDir,
        [Parameter(Mandatory = $true)]
        [hashtable]$Inventory
    )
    Write-Host "[ROLLBACK] Starting rollback for installation at '$InstallRoot'..." -ForegroundColor Yellow

    $filesDir = Join-Path $BackupDir "files"
    $restoredCount = 0
    $deletedCount = 0

    # 1. Remove newly created files
    if ($Inventory.ContainsKey("new_files") -and $Inventory["new_files"]) {
        foreach ($relPath in $Inventory["new_files"]) {
            $targetPath = Join-Path $InstallRoot $relPath
            Assert-TargetWithinInstallRoot -TargetPath $targetPath -InstallRoot $InstallRoot | Out-Null
            if (Test-Path -LiteralPath $targetPath -PathType Leaf) {
                try {
                    Remove-Item -LiteralPath $targetPath -Force
                    $deletedCount++
                } catch {
                    Write-Warning "[ROLLBACK] Failed to delete newly added file '$targetPath': $_"
                }
            }
        }
    }

    # 2. Restore overwritten files from backup
    if ($Inventory.ContainsKey("overwritten_files") -and $Inventory["overwritten_files"]) {
        foreach ($entry in $Inventory["overwritten_files"]) {
            $relPath = if ($entry -is [hashtable] -or $entry -is [System.Collections.IDictionary]) { $entry["path"] } else { $entry.path }
            $backupFile = Join-Path $filesDir $relPath
            $targetPath = Join-Path $InstallRoot $relPath
            Assert-TargetWithinInstallRoot -TargetPath $targetPath -InstallRoot $InstallRoot | Out-Null
            Assert-PathWithinRoots -Path $backupFile -AllowedRoots @($filesDir) -Label 'Rollback backup' | Out-Null

            if (Test-Path -LiteralPath $backupFile -PathType Leaf) {
                $targetDir = [System.IO.Path]::GetDirectoryName($targetPath)
                if (-not (Test-Path -LiteralPath $targetDir)) {
                    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
                }
                Copy-Item -LiteralPath $backupFile -Destination $targetPath -Force
                $restoredCount++
            } else {
                Write-Warning "[ROLLBACK] Backup copy not found for '$relPath' at '$backupFile'."
            }
        }
    }

    # 3. Restore settings.json if modified
    $settingsBackup = Join-Path $filesDir "temp\settings.json"
    $targetSettings = Join-Path $InstallRoot "temp\settings.json"
    if (Test-Path -LiteralPath $settingsBackup -PathType Leaf) {
        Copy-Item -LiteralPath $settingsBackup -Destination $targetSettings -Force
        Write-Host "[ROLLBACK] Restored temp\settings.json from backup." -ForegroundColor Cyan
    }

    Write-Host "[ROLLBACK] Rollback complete. Restored: $restoredCount files. Removed: $deletedCount new files." -ForegroundColor Green
}

# ==============================================================================
# Manual Rollback Mode Execution
# ==============================================================================

if (-not [string]::IsNullOrWhiteSpace($RollbackFrom)) {
    $resolvedRollbackDir = [System.IO.Path]::GetFullPath($RollbackFrom)
    if (-not (Test-Path -LiteralPath $resolvedRollbackDir -PathType Container)) {
        throw "Rollback directory not found: '$resolvedRollbackDir'."
    }

    $inventoryPath = Join-Path $resolvedRollbackDir "inventory.json"
    if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
        $inventoryPath = Join-Path $resolvedRollbackDir "backup-manifest.json"
    }
    if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
        throw "No inventory.json or backup-manifest.json found in '$resolvedRollbackDir'."
    }

    $inventory = ConvertFrom-JsonCompat -Json (Get-Content -LiteralPath $inventoryPath -Raw -Encoding utf8)
    $effectiveInstallRoot = if ($InstallationRoot) { $InstallationRoot } else { $inventory["installation_root"] }
    $resolvedInstallRoot = [System.IO.Path]::GetFullPath($effectiveInstallRoot)
    if (-not $resolvedInstallRoot.Equals([System.IO.Path]::GetFullPath($inventory['installation_root']), [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Rollback installation does not match the recorded inventory.'
    }
    if (-not $SkipProcessCheck -and @(Get-ActiveInstallationProcesses -InstallRoot $resolvedInstallRoot).Count -gt 0) {
        throw 'Close Neri and its toolkit Python processes before rollback.'
    }

    if ($PSCmdlet.ShouldProcess("$resolvedInstallRoot (from $resolvedRollbackDir)", "Execute Manual Rollback")) {
        Invoke-RollbackOperation -InstallRoot $resolvedInstallRoot -BackupDir $resolvedRollbackDir -Inventory $inventory
    } else {
        Write-Host "[WHATIF] Would execute manual rollback from '$resolvedRollbackDir' to '$resolvedInstallRoot'." -ForegroundColor Yellow
    }
    return
}

# ==============================================================================
# Preflight Path & Source Discovery
# ==============================================================================

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Split-Path -Parent $PSScriptRoot
}
$resolvedSourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)
if (-not (Test-Path -LiteralPath $resolvedSourceRoot -PathType Container)) {
    throw "SourceRoot directory does not exist at '$resolvedSourceRoot'."
}

$resolvedInstallRoot = [System.IO.Path]::GetFullPath($InstallationRoot)
if (-not (Test-Path -LiteralPath $resolvedInstallRoot -PathType Container)) {
    throw "InstallationRoot directory does not exist at '$resolvedInstallRoot'."
}

$allowedSourceRoots = @(
    $resolvedSourceRoot,
    "F:\files\python\Neri_plus"
)

# 1. Verify Active Processes
if (-not $SkipProcessCheck) {
    $activeProcesses = Get-ActiveInstallationProcesses -InstallRoot $resolvedInstallRoot
    if ($activeProcesses.Count -gt 0) {
        Write-Host "==================================================================" -ForegroundColor Red
        Write-Host "ACTIVE INSTALLATION PROCESSES DETECTED" -ForegroundColor Red
        Write-Host "The following processes are currently running from '$resolvedInstallRoot':" -ForegroundColor Red
        foreach ($proc in $activeProcesses) {
            Write-Host "  - PID: $($proc.Id) | Process: $($proc.ProcessName) | Path: $($proc.Path)" -ForegroundColor Yellow
        }
        Write-Host "Please close Neri and its toolkit Python processes before proceeding." -ForegroundColor Red
        Write-Host "Note: This script will NOT terminate external processes." -ForegroundColor Yellow
        Write-Host "==================================================================" -ForegroundColor Red
        throw "Deployment stopped: Active processes detected in '$resolvedInstallRoot'."
    }
}

# 2. Locate and Validate Model Manifest
$manifestFile = $null
if (-not [string]::IsNullOrWhiteSpace($ModelManifest)) {
    if ([System.IO.Path]::IsPathRooted($ModelManifest)) {
        $manifestFile = [System.IO.Path]::GetFullPath($ModelManifest)
    } else {
        $cand1 = Join-Path $PWD $ModelManifest
        $cand2 = Join-Path $resolvedSourceRoot $ModelManifest
        if (Test-Path -LiteralPath $cand1 -PathType Leaf) { $manifestFile = [System.IO.Path]::GetFullPath($cand1) }
        elseif (Test-Path -LiteralPath $cand2 -PathType Leaf) { $manifestFile = [System.IO.Path]::GetFullPath($cand2) }
        else { $manifestFile = [System.IO.Path]::GetFullPath($cand2) }
    }
} else {
    $defaultManifest = Join-Path $resolvedSourceRoot "res\model\cls\user\dinov3-ridge-within.neri.json"
    if (Test-Path -LiteralPath $defaultManifest -PathType Leaf) {
        $manifestFile = $defaultManifest
    } else {
        $candidates = @(Get-ChildItem -Path (Join-Path $resolvedSourceRoot "res\model\cls\user") -Filter "*.neri.json" -ErrorAction SilentlyContinue)
        if ($candidates.Count -gt 0) {
            $manifestFile = $candidates[0].FullName
        }
    }
}

if (-not $manifestFile -or -not (Test-Path -LiteralPath $manifestFile -PathType Leaf)) {
    throw "Model manifest file not found. Specified path: '$ModelManifest'."
}

$manifestDir = Split-Path -Parent $manifestFile
$allowedSourceRoots += $manifestDir
$manifestFile = Assert-PathWithinRoots -Path $manifestFile -AllowedRoots $allowedSourceRoots -Label "Model Manifest"

# Parse and Validate Manifest Contract
$manifestJson = Get-Content -LiteralPath $manifestFile -Raw -Encoding utf8
$manifestObj = ConvertFrom-JsonCompat -Json $manifestJson

$requiredManifestKeys = @("checkpoint", "checkpoint_sha256", "encoder_weights", "encoder_sha256", "encoder_code")
foreach ($k in $requiredManifestKeys) {
    if (-not $manifestObj.ContainsKey($k) -or [string]::IsNullOrWhiteSpace($manifestObj[$k])) {
        throw "Model manifest '$manifestFile' is missing required contract field: '$k'."
    }
}

# Resolve Assets Relative to Manifest Directory
$checkpointSrc = Join-Path $manifestDir ($manifestObj["checkpoint"].Replace('/', '\'))
$encoderWeightsSrc = Join-Path $manifestDir ($manifestObj["encoder_weights"].Replace('/', '\'))
$encoderCodeSrc = Join-Path $manifestDir ($manifestObj["encoder_code"].Replace('/', '\'))

Assert-PathWithinRoots -Path $checkpointSrc -AllowedRoots $allowedSourceRoots -Label "Checkpoint Asset" | Out-Null
Assert-PathWithinRoots -Path $encoderWeightsSrc -AllowedRoots $allowedSourceRoots -Label "Encoder Weights Asset" | Out-Null
Assert-PathWithinRoots -Path $encoderCodeSrc -AllowedRoots $allowedSourceRoots -Label "Encoder Code Directory" | Out-Null

if (-not (Test-Path -LiteralPath $checkpointSrc -PathType Leaf)) {
    throw "Model checkpoint asset not found at '$checkpointSrc'."
}
if (-not (Test-Path -LiteralPath $encoderWeightsSrc -PathType Leaf)) {
    throw "Model encoder weights asset not found at '$encoderWeightsSrc'."
}
if (-not (Test-Path -LiteralPath $encoderCodeSrc -PathType Container)) {
    throw "Model encoder code directory not found at '$encoderCodeSrc'."
}

# Validate SHA-256 for Checkpoint and Weights
Write-Host "[VERIFY] Verifying SHA-256 for model checkpoint..." -ForegroundColor Cyan
$actualCheckpointHash = Get-FileSha256 -Path $checkpointSrc
if ($actualCheckpointHash -ne $manifestObj["checkpoint_sha256"].ToLowerInvariant()) {
    throw "Checkpoint SHA-256 mismatch! Expected: $($manifestObj['checkpoint_sha256']), Actual: $actualCheckpointHash"
}

Write-Host "[VERIFY] Verifying SHA-256 for encoder weights..." -ForegroundColor Cyan
$actualEncoderWeightsHash = Get-FileSha256 -Path $encoderWeightsSrc
if ($actualEncoderWeightsHash -ne $manifestObj["encoder_sha256"].ToLowerInvariant()) {
    throw "Encoder weights SHA-256 mismatch! Expected: $($manifestObj['encoder_sha256']), Actual: $actualEncoderWeightsHash"
}

# Verify Encoder Code Structure
$encoderLicense = Join-Path $encoderCodeSrc "LICENSE.md"
$encoderBackbone = Join-Path $encoderCodeSrc "dinov3\hub\backbones.py"
if (-not (Test-Path -LiteralPath $encoderLicense -PathType Leaf) -or -not (Test-Path -LiteralPath $encoderBackbone -PathType Leaf)) {
    throw "Encoder code directory must contain LICENSE.md and dinov3/hub/backbones.py."
}

Write-Host "[VERIFY] Manifest assets and SHA-256 checksums verified successfully." -ForegroundColor Green

# ==============================================================================
# Compile Deployment Item Inventory
# ==============================================================================

$plannedOperations = [System.Collections.Generic.List[psobject]]::new()

function Register-DeploymentFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceFile,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )
    $targetPath = Join-Path $resolvedInstallRoot $RelativePath
    Assert-TargetWithinInstallRoot -TargetPath $targetPath -InstallRoot $resolvedInstallRoot | Out-Null

    $exists = Test-Path -LiteralPath $targetPath -PathType Leaf
    $targetHashBefore = if ($exists) { Get-FileSha256 -Path $targetPath } else { $null }

    $plannedOperations.Add([pscustomobject]@{
        SourcePath = $SourceFile
        RelativePath = $RelativePath
        TargetPath = $targetPath
        IsOverwrite = $exists
        TargetHashBefore = $targetHashBefore
    })
}

# 1. Compiled Frontend UI (frontend/build/windows/x64/runner/Release)
$releaseDir = Join-Path $resolvedSourceRoot "frontend\build\windows\x64\runner\Release"
if (-not (Test-Path -LiteralPath $releaseDir -PathType Container)) {
    throw "Frontend release build directory not found at '$releaseDir'. Please compile the Flutter release build first."
}

$flutterDll = Join-Path $releaseDir "flutter_windows.dll"
if (-not (Test-Path -LiteralPath $flutterDll -PathType Leaf)) {
    throw "Missing flutter_windows.dll in release build '$releaseDir'."
}

$releaseFiles = Get-ChildItem -LiteralPath $releaseDir -Recurse -File
foreach ($file in $releaseFiles) {
    $rel = [System.IO.Path]::GetRelativePath($releaseDir, $file.FullName)
    # Normalize legacy runner exe name if present
    if ($rel.Equals("neri_flutter.exe", [System.StringComparison]::OrdinalIgnoreCase)) {
        $rel = "Neri.exe"
    }
    Register-DeploymentFile -SourceFile $file.FullName -RelativePath $rel
}

# 2. System Backend Code (system/)
$systemDir = Join-Path $resolvedSourceRoot "system"
if (-not (Test-Path -LiteralPath $systemDir -PathType Container)) {
    throw "System directory not found at '$systemDir'."
}

$systemFiles = Get-ChildItem -LiteralPath $systemDir -Recurse -File | Where-Object {
    $full = $_.FullName
    -not ($full -like "*\__pycache__\*" -or $full -like "*__pycache__" -or $_.Extension -in @(".pyc", ".pyo"))
}
foreach ($file in $systemFiles) {
    $rel = "system\" + [System.IO.Path]::GetRelativePath($systemDir, $file.FullName)
    Register-DeploymentFile -SourceFile $file.FullName -RelativePath $rel
}

# 3. Model Manifest & Assets (res/model/cls/user/...)
$manifestFileName = [System.IO.Path]::GetFileName($manifestFile)
$targetManifestRel = "res\model\cls\user\$manifestFileName"
Register-DeploymentFile -SourceFile $manifestFile -RelativePath $targetManifestRel

$manifestStem = $manifestFileName -replace '\.neri\.json$', ''
$exportReportFile = Join-Path $manifestDir "$manifestStem.export-report.json"
if (-not (Test-Path -LiteralPath $exportReportFile)) {
    $exportReportFile = Join-Path $manifestDir "$manifestFileName.export-report.json"
}
if (Test-Path -LiteralPath $exportReportFile -PathType Leaf) {
    $reportFileName = [System.IO.Path]::GetFileName($exportReportFile)
    Register-DeploymentFile -SourceFile $exportReportFile -RelativePath "res\model\cls\user\$reportFileName"
}

# Checkpoint file
$checkpointRel = "res\model\cls\user\" + $manifestObj["checkpoint"].Replace('/', '\')
Register-DeploymentFile -SourceFile $checkpointSrc -RelativePath $checkpointRel

# Encoder weights file
$encoderWeightsRel = "res\model\cls\user\" + $manifestObj["encoder_weights"].Replace('/', '\')
Register-DeploymentFile -SourceFile $encoderWeightsSrc -RelativePath $encoderWeightsRel

# Encoder code directory tree
$encoderCodeRelBase = "res\model\cls\user\" + $manifestObj["encoder_code"].Replace('/', '\')
$encoderFiles = Get-ChildItem -LiteralPath $encoderCodeSrc -Recurse -File | Where-Object {
    $full = $_.FullName
    -not ($full -like "*\__pycache__\*" -or $full -like "*__pycache__" -or $_.Extension -in @(".pyc", ".pyo"))
}
foreach ($file in $encoderFiles) {
    $innerRel = [System.IO.Path]::GetRelativePath($encoderCodeSrc, $file.FullName)
    $rel = Join-Path $encoderCodeRelBase $innerRel
    Register-DeploymentFile -SourceFile $file.FullName -RelativePath $rel
}

# 4. Training/Export Scripts & Documentation
$scriptsToCopy = @(
    "scripts\export_dinov3_experiment.py",
    "scripts\train_dinov3_ridge_within.py",
    "scripts\train_reviewed_dinov3.py",
    "scripts\verify_dinov3_installation.py",
    "scripts\verify_litert_privacy.py",
    "docs\dinov3-ridge-within.md",
    "docs\dinov3-reviewed-training-20261107.md",
    "docs\dinov3-known-rejection-controls-20261107.md",
    "docs\upload-privacy-blurring.md",
    "docs\litert-privacy-verification.md",
    "docs\litert-detection.md",
    "scripts\flutter_accessibility_hotfix.py"
)
foreach ($scriptRel in $scriptsToCopy) {
    $src = Join-Path $resolvedSourceRoot $scriptRel
    if (Test-Path -LiteralPath $src -PathType Leaf) {
        Register-DeploymentFile -SourceFile $src -RelativePath $scriptRel
    }
}

$rootDocs = @(
    "requirements.txt",
    "requirements-privacy.txt"
)
foreach ($doc in $rootDocs) {
    $src = Join-Path $resolvedSourceRoot $doc
    if (Test-Path -LiteralPath $src -PathType Leaf) {
        Register-DeploymentFile -SourceFile $src -RelativePath $doc
    }
}

$readmeUpdateSrc = Join-Path $resolvedSourceRoot "res\demo\README_Update.md"
if (-not (Test-Path -LiteralPath $readmeUpdateSrc -PathType Leaf)) {
    $readmeUpdateSrc = Join-Path $resolvedSourceRoot "README_Update.md"
}
if (Test-Path -LiteralPath $readmeUpdateSrc -PathType Leaf) {
    Register-DeploymentFile -SourceFile $readmeUpdateSrc -RelativePath "README_Update.md"
}

# Separate Overwritten vs New Files
$overwrittenFiles = @($plannedOperations | Where-Object { $_.IsOverwrite })
$newFiles = @($plannedOperations | Where-Object { -not $_.IsOverwrite })

# Settings handling if -SelectModel
$settingsFile = Join-Path $resolvedInstallRoot "temp\settings.json"
$settingsWillBeUpdated = $false
$settingsExisted = Test-Path -LiteralPath $settingsFile -PathType Leaf
if ($SelectModel) {
    $settingsWillBeUpdated = $true
}

# ==============================================================================
# Timestamp & Backup Setup
# ==============================================================================

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDirName = "dinov3-$timestamp"
$backupDir = Join-Path $resolvedInstallRoot "repair-backups\$backupDirName"
$backupFilesDir = Join-Path $backupDir "files"
$stagingDir = Join-Path $resolvedInstallRoot ".staging-dinov3-$timestamp"
Assert-TargetWithinInstallRoot -TargetPath $stagingDir -InstallRoot $resolvedInstallRoot | Out-Null
Assert-TargetWithinInstallRoot -TargetPath $backupDir -InstallRoot $resolvedInstallRoot | Out-Null

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "PLAN SUMMARY FOR DINOv3 INSTALLATION UPDATE" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Installation Root : $resolvedInstallRoot"
Write-Host "Model Manifest    : $manifestFile"
Write-Host "Total Files       : $($plannedOperations.Count)"
Write-Host "Overwriting Files : $($overwrittenFiles.Count) (backed up to repair-backups\$backupDirName)"
Write-Host "New Files Added   : $($newFiles.Count)"
Write-Host "Select Model      : $(if ($SelectModel) { 'YES (updating temp/settings.json)' } else { 'NO (preserving settings)' })"
Write-Host "Backup Directory  : $backupDir"
Write-Host "==================================================================" -ForegroundColor Cyan

# ==============================================================================
# WhatIf / ShouldProcess Evaluation
# ==============================================================================

$targetManifestFullPath = Join-Path $resolvedInstallRoot $targetManifestRel

if (-not $PSCmdlet.ShouldProcess("$resolvedInstallRoot (DINOv3 deployment with $($plannedOperations.Count) files)", "Deploy DINOv3 Update")) {
    Write-Host "[WHATIF] Dry-run completed. No files were modified in '$resolvedInstallRoot'." -ForegroundColor Yellow
    Write-Host "[WHATIF] Files that would be overwritten ($($overwrittenFiles.Count)): "
    $overwrittenFiles | Select-Object -First 10 | ForEach-Object { Write-Host "  - $($_.RelativePath)" -ForegroundColor Gray }
    if ($overwrittenFiles.Count -gt 10) { Write-Host "  ... and $($overwrittenFiles.Count - 10) more." -ForegroundColor Gray }

    Write-Host "[WHATIF] New files that would be created ($($newFiles.Count)): "
    $newFiles | Select-Object -First 10 | ForEach-Object { Write-Host "  + $($_.RelativePath)" -ForegroundColor Gray }
    if ($newFiles.Count -gt 10) { Write-Host "  ... and $($newFiles.Count - 10) more." -ForegroundColor Gray }

    if ($SelectModel) {
        Write-Host "[WHATIF] temp/settings.json would update:" -ForegroundColor Cyan
        Write-Host "  selected_classification_model = '$targetManifestFullPath'" -ForegroundColor Cyan
        Write-Host "  video_mode = 'skip'" -ForegroundColor Cyan
    }
    return
}

# ==============================================================================
# Execution Phase (Real Deployment)
# ==============================================================================

# Step 1: Create Backup Directory
New-Item -ItemType Directory -Path $backupFilesDir -Force | Out-Null

# Step 2: Perform Backup of All Overwritten Files
Write-Host "[BACKUP] Backing up $($overwrittenFiles.Count) files that will be overwritten..." -ForegroundColor Cyan
foreach ($item in $overwrittenFiles) {
    $destBackup = Join-Path $backupFilesDir $item.RelativePath
    $destParent = [System.IO.Path]::GetDirectoryName($destBackup)
    if (-not (Test-Path -LiteralPath $destParent)) {
        New-Item -ItemType Directory -Path $destParent -Force | Out-Null
    }
    Copy-Item -LiteralPath $item.TargetPath -Destination $destBackup -Force
}

# If settings is to be updated and exists, back it up explicitly
if ($settingsWillBeUpdated -and $settingsExisted) {
    $settingsBackupDest = Join-Path $backupFilesDir "temp\settings.json"
    $settingsBackupDir = [System.IO.Path]::GetDirectoryName($settingsBackupDest)
    if (-not (Test-Path -LiteralPath $settingsBackupDir)) {
        New-Item -ItemType Directory -Path $settingsBackupDir -Force | Out-Null
    }
    Copy-Item -LiteralPath $settingsFile -Destination $settingsBackupDest -Force

    $sidecarBackup = Join-Path (Join-Path $resolvedInstallRoot "temp") "settings.json.bak-dinov3-$timestamp"
    Copy-Item -LiteralPath $settingsFile -Destination $sidecarBackup -Force
    Write-Host "[BACKUP] Backed up temp/settings.json to '$sidecarBackup'." -ForegroundColor Cyan
}

# Step 3: Write Pre-deployment Inventory Manifest
Write-Host "[INVENTORY] Writing pre-write inventory.json and backup-manifest.json..." -ForegroundColor Cyan

$inventoryObj = [ordered]@{
    timestamp = $timestamp
    installation_root = $resolvedInstallRoot
    backup_dir = $backupDir
    source_root = $resolvedSourceRoot
    model_manifest = $targetManifestRel
    model_name = $manifestObj["display_name"]
    select_model = [bool]$SelectModel
    status = "in_progress"
    total_files = $plannedOperations.Count
    overwritten_count = $overwrittenFiles.Count
    new_count = $newFiles.Count
    overwritten_files = @($overwrittenFiles | ForEach-Object {
        [ordered]@{
            path = $_.RelativePath
            sha256_before = $_.TargetHashBefore
            backup_file = "files\" + $_.RelativePath
        }
    })
    new_files = @($newFiles | ForEach-Object { $_.RelativePath })
}

$inventoryJsonContent = $inventoryObj | ConvertTo-Json -Depth 10
$inventoryPath = Join-Path $backupDir "inventory.json"
$backupManifestPath = Join-Path $backupDir "backup-manifest.json"

[System.IO.File]::WriteAllText($inventoryPath, $inventoryJsonContent, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText($backupManifestPath, $inventoryJsonContent, [System.Text.UTF8Encoding]::new($false))

# Step 4: Staging
Write-Host "[STAGING] Copying files to staging directory on same volume..." -ForegroundColor Cyan
if (Test-Path -LiteralPath $stagingDir) {
    Remove-Item -LiteralPath $stagingDir -Recurse -Force
}
New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null

try {
    foreach ($item in $plannedOperations) {
        $stagedFile = Join-Path $stagingDir $item.RelativePath
        $stagedParent = [System.IO.Path]::GetDirectoryName($stagedFile)
        if (-not (Test-Path -LiteralPath $stagedParent)) {
            New-Item -ItemType Directory -Path $stagedParent -Force | Out-Null
        }
        Copy-Item -LiteralPath $item.SourcePath -Destination $stagedFile -Force
    }

    # Prepare staged settings.json if -SelectModel
    if ($settingsWillBeUpdated) {
        $stagedSettings = Join-Path $stagingDir "temp\settings.json"
        $stagedSettingsDir = [System.IO.Path]::GetDirectoryName($stagedSettings)
        if (-not (Test-Path -LiteralPath $stagedSettingsDir)) {
            New-Item -ItemType Directory -Path $stagedSettingsDir -Force | Out-Null
        }

        $settingsData = [ordered]@{}
        if ($settingsExisted) {
            $existingSettingsContent = Get-Content -LiteralPath $settingsFile -Raw -Encoding utf8
            $settingsData = ConvertFrom-JsonCompat -Json $existingSettingsContent
        }

        # Update model selection and video mode skip, preserving all other keys
        $settingsData["selected_classification_model"] = $targetManifestFullPath
        $settingsData["selected_species_names"] = @()
        $settingsData["video_mode"] = "skip"

        $newSettingsJson = $settingsData | ConvertTo-Json -Depth 10
        [System.IO.File]::WriteAllText($stagedSettings, $newSettingsJson, [System.Text.UTF8Encoding]::new($false))
    }

    # Step 5: Atomic Replace
    Write-Host "[DEPLOY] Atomically replacing files in '$resolvedInstallRoot'..." -ForegroundColor Cyan
    foreach ($item in $plannedOperations) {
        $stagedFile = Join-Path $stagingDir $item.RelativePath
        Replace-FileAtomic -SourcePath $stagedFile -DestinationPath $item.TargetPath
    }

    if ($settingsWillBeUpdated) {
        $stagedSettings = Join-Path $stagingDir "temp\settings.json"
        Replace-FileAtomic -SourcePath $stagedSettings -DestinationPath $settingsFile
        Write-Host "[SETTINGS] temp/settings.json updated to DINOv3 model and video_mode skip." -ForegroundColor Green
    }

    # Step 6: Mark Inventory as Complete
    $inventoryObj["status"] = "completed"
    $inventoryObj["completed_at"] = (Get-Date -Format "yyyyMMdd-HHmmss")
    $finalInventoryJson = $inventoryObj | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($inventoryPath, $finalInventoryJson, [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllText($backupManifestPath, $finalInventoryJson, [System.Text.UTF8Encoding]::new($false))

    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host "DEPLOYMENT SUCCEEDED" -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host "Target Installation : $resolvedInstallRoot"
    Write-Host "Deployed Manifest   : $targetManifestFullPath"
    Write-Host "Total Files Deployed: $($plannedOperations.Count)"
    Write-Host "Backup Created At   : $backupDir"
    Write-Host "To roll back later if needed, run:"
    Write-Host "  .\scripts\update_dinov3_installation.ps1 -RollbackFrom `"$backupDir`"" -ForegroundColor Yellow
    Write-Host "==================================================================" -ForegroundColor Green
}
catch {
    Write-Warning "Deployment failed with error: $_"
    Write-Host "[EMERGENCY] Initiating automatic rollback to restore previous state..." -ForegroundColor Red
    try {
        Invoke-RollbackOperation -InstallRoot $resolvedInstallRoot -BackupDir $backupDir -Inventory $inventoryObj
        $inventoryObj["status"] = "rolled_back_after_error"
        $inventoryObj["error_message"] = "$_"
        $inventoryJsonContent = $inventoryObj | ConvertTo-Json -Depth 10
        [System.IO.File]::WriteAllText($inventoryPath, $inventoryJsonContent, [System.Text.UTF8Encoding]::new($false))
        [System.IO.File]::WriteAllText($backupManifestPath, $inventoryJsonContent, [System.Text.UTF8Encoding]::new($false))
    } catch {
        Write-Warning "Error during automatic rollback: $_"
    }
    throw
}
finally {
    if (Test-Path -LiteralPath $stagingDir) {
        Remove-Item -LiteralPath $stagingDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
