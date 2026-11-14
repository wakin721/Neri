param(
    [string]$InstallationRoot = "D:\Application\Neri",
    [string]$SourceRoot = "E:\Files\Neri",
    [string]$ExtraFilesManifest = "",
    [switch]$SkipProcessCheck
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$ChildPath
    )
    $base = [System.IO.Path]::GetFullPath($BasePath).TrimEnd([char]92)
    $child = [System.IO.Path]::GetFullPath($ChildPath)
    if (-not $child.StartsWith($base + [char]92, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside base directory: $child"
    }
    return $child.Substring($base.Length + 1)
}

$install = [System.IO.Path]::GetFullPath($InstallationRoot).TrimEnd('\')
if (-not (Test-Path -LiteralPath $install -PathType Container)) { throw "Installation root not found: $install" }
$source = [System.IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$release = Join-Path $source "frontend\build\windows\x64\runner\Release"
$system = Join-Path $source "system"
if (-not (Test-Path -LiteralPath $release -PathType Container)) { throw "Release build missing: $release" }
if (-not (Test-Path -LiteralPath $system -PathType Container)) { throw "system dir missing: $system" }

if (-not $SkipProcessCheck) {
    $norm = $install
    $active = @()
    foreach ($p in (Get-Process -Name Neri, neri_flutter, python, pythonw -ErrorAction SilentlyContinue)) {
        $pp = $null
        try { $pp = $p.Path } catch { }
        if ($pp -and $pp.StartsWith($norm + '\', [System.StringComparison]::OrdinalIgnoreCase)) { $active += $p }
    }
    if ($active.Count -gt 0) { throw "Close Neri and its Python processes first." }
}

$planned = New-Object System.Collections.Generic.List[object]
foreach ($f in Get-ChildItem -LiteralPath $release -Recurse -File) {
    $rel = Get-RelativePathCompat -BasePath $release -ChildPath $f.FullName
    $planned.Add([pscustomobject]@{ Source = $f.FullName; Rel = $rel })
}
foreach ($f in Get-ChildItem -LiteralPath $system -Recurse -File) {
    if ($f.FullName -like "*\__pycache__\*" -or $f.Extension -in @(".pyc", ".pyo")) { continue }
    $rel = Join-Path "system" (Get-RelativePathCompat -BasePath $system -ChildPath $f.FullName)
    $planned.Add([pscustomobject]@{ Source = $f.FullName; Rel = $rel })
}

foreach ($rel in @('docs\dinov3-ridge-within.md', 'scripts\train_dinov3_ridge_within.py',
        'scripts\prepare_installed_learning_calibration.py', 'scripts\verify_dinov3_installation.py')) {
    $path = Join-Path $source $rel
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $planned.Add([pscustomobject]@{ Source = $path; Rel = $rel })
    }
}
if ($ExtraFilesManifest) {
    foreach ($item in (Get-Content -LiteralPath $ExtraFilesManifest -Raw | ConvertFrom-Json)) {
        $target = [System.IO.Path]::GetFullPath((Join-Path $install $item.Rel))
        if (-not $target.StartsWith($install + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Extra path escapes installation: $target" }
        $extraSource = [System.IO.Path]::GetFullPath($item.Source)
        if (-not $extraSource.StartsWith($source + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Extra source escapes workspace: $extraSource" }
        if ($item.PSObject.Properties.Name -contains 'ExpectedTargetSha256') {
            if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $item.ExpectedTargetSha256) {
                throw "Target changed after preparation: $target"
            }
        }
        $planned.Add([pscustomobject]@{ Source = $extraSource; Rel = $item.Rel })
    }
}

if ($planned.Count -eq 0) { throw "Nothing to deploy." }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backup = Join-Path $install "repair-backups\ui-update-$stamp"
$backupFiles = Join-Path $backup "files"
New-Item -ItemType Directory -Path $backupFiles -Force | Out-Null
$staging = Join-Path $backup 'staging'
$seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($item in $planned) {
    if (-not $seen.Add($item.Rel)) { throw "Duplicate deployment path: $($item.Rel)" }
    $staged = Join-Path $staging $item.Rel
    New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($staged)) -Force | Out-Null
    Copy-Item -LiteralPath $item.Source -Destination $staged -Force
    if ((Get-FileHash -LiteralPath $staged).Hash -ne (Get-FileHash -LiteralPath $item.Source).Hash) {
        throw "Staged file verification failed: $($item.Rel)"
    }
}

$overwritten = New-Object System.Collections.Generic.List[object]
$created = New-Object System.Collections.Generic.List[string]
foreach ($item in $planned) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $install $item.Rel))
    if (-not $target.StartsWith($install + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Path escapes install root: $target" }
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        $copy = Join-Path $backupFiles $item.Rel
        $parent = [System.IO.Path]::GetDirectoryName($copy)
        if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
        Copy-Item -LiteralPath $target -Destination $copy -Force
        $overwritten.Add([pscustomobject]@{ path = $item.Rel })
    } else {
        $created.Add($item.Rel)
    }
}

[ordered]@{
    timestamp = $stamp
    installation_root = $install
    source_root = $source
    total = $planned.Count
    overwritten = $overwritten
    created = $created
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $backup "inventory.json") -Encoding utf8

$applied = New-Object System.Collections.Generic.List[string]
try {
    foreach ($item in $planned) {
        $target = [System.IO.Path]::GetFullPath((Join-Path $install $item.Rel))
        if (-not $target.StartsWith($install + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Target escapes installation: $target" }
        $parent = [System.IO.Path]::GetDirectoryName($target)
        if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
        $staged = Join-Path $staging $item.Rel
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            $swapBackup = Join-Path $backup (Join-Path 'atomic-originals' $item.Rel)
            New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($swapBackup)) -Force | Out-Null
            [System.IO.File]::Replace($staged, $target, $swapBackup)
        } else {
            Move-Item -LiteralPath $staged -Destination $target
        }
        $applied.Add($item.Rel)
        if ((Get-FileHash -LiteralPath $target).Hash -ne (Get-FileHash -LiteralPath $item.Source).Hash) {
            throw "Installed file verification failed: $($item.Rel)"
        }
    }
} catch {
    $failure = $_
    foreach ($rel in $applied) {
        $target = [System.IO.Path]::GetFullPath((Join-Path $install $rel))
        if (-not $target.StartsWith($install + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Rollback path escapes installation" }
        $old = Join-Path $backupFiles $rel
        if (Test-Path -LiteralPath $old -PathType Leaf) {
            Copy-Item -LiteralPath $old -Destination $target -Force
        } elseif (Test-Path -LiteralPath $target -PathType Leaf) {
            Remove-Item -LiteralPath $target -Force
        }
    }
    throw $failure
}

Write-Host "[DEPLOY] files=$($planned.Count) overwritten=$($overwritten.Count) new=$($created.Count)"
Write-Host "[BACKUP] $backup"
