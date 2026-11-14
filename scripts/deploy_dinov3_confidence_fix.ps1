<# Deploy only the frame-confidence fix and changed UI files, with rollback. #>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$InstallationRoot = 'D:\Application\Neri',
    [string]$SourceRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PlanPath = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$install = [IO.Path]::GetFullPath($InstallationRoot).TrimEnd('\')
$source = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$release = Join-Path $source 'frontend\build\windows\x64\runner\Release'
if (-not (Test-Path -LiteralPath $install -PathType Container)) { throw 'Installation root does not exist.' }
if (-not (Test-Path -LiteralPath (Join-Path $release 'Neri.exe'))) { throw 'Build the release UI first.' }
function Target-Path([string]$Relative) {
    $resolved = [IO.Path]::GetFullPath((Join-Path $install $Relative))
    if (-not $resolved.StartsWith($install+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Target escapes installation.' }
    return $resolved
}
function File-Hash([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
$candidates = @(
    @{ Source = (Join-Path $source 'system\dinov3\classifier.py'); Relative = 'system\dinov3\classifier.py' },
    @{ Source = (Join-Path $source 'system\dinov3\processor.py'); Relative = 'system\dinov3\processor.py' },
    @{ Source = (Join-Path $source 'docs\dinov3-experiment-integration-20261114.md'); Relative = 'docs\dinov3-experiment-integration-20261114.md' }
)
foreach ($file in Get-ChildItem -LiteralPath $release -Recurse -File) {
    $relative = [IO.Path]::GetRelativePath($release, $file.FullName)
    $candidates += @{ Source = $file.FullName; Relative = $relative }
}
$operations = @()
foreach ($entry in $candidates) {
    $target = Target-Path $entry.Relative
    $before = if (Test-Path -LiteralPath $target -PathType Leaf) { File-Hash $target } else { $null }
    $after = File-Hash $entry.Source
    if ($before -ne $after) {
        $operations += [pscustomobject]@{ Source=$entry.Source; Relative=$entry.Relative; Target=$target; Before=$before; After=$after }
    }
}
$backup = Target-Path ('repair-backups\frame-confidence-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
$inventory = [ordered]@{ installation=$install; backup=$backup; status='prepared'; files=$operations; models_and_settings_modified=$false }
if ($PlanPath) {
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($PlanPath))) | Out-Null
    [IO.File]::WriteAllText([IO.Path]::GetFullPath($PlanPath), ($inventory | ConvertTo-Json -Depth 8))
}
Write-Host ('Changed files: '+$operations.Count+'; backup: '+$backup)
foreach ($op in $operations) { Write-Host ('  '+$op.Relative) }
if (-not $PSCmdlet.ShouldProcess($install, 'Install frame-confidence display fix with backup')) { return }
foreach ($process in Get-Process -Name Neri,neri_flutter,python,pythonw -ErrorAction SilentlyContinue) {
    $processPath = $null
    try { $processPath = $process.Path } catch { }
    if ($processPath -and $processPath.StartsWith($install+'\', [StringComparison]::OrdinalIgnoreCase)) {
        throw ('Close Neri and its backend before updating: '+$process.Id+' '+$processPath)
    }
}
if (-not $operations.Count) { return }
[IO.Directory]::CreateDirectory($backup) | Out-Null
foreach ($op in $operations) {
    if ($op.Before) {
        $saved = Join-Path $backup ('files\'+$op.Relative)
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($saved)) | Out-Null
        [IO.File]::Copy($op.Target, $saved, $false)
        if ((File-Hash $saved) -ne $op.Before) { throw 'Backup hash mismatch.' }
    }
}
$inventoryFile = Join-Path $backup 'inventory.json'
[IO.File]::WriteAllText($inventoryFile, ($inventory | ConvertTo-Json -Depth 8))
$written = [Collections.Generic.List[object]]::new()
try {
    foreach ($op in $operations) {
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($op.Target)) | Out-Null
        $stage = $op.Target+'.stage-'+[Guid]::NewGuid().ToString('N')
        $swap = $stage+'.old'
        try {
            [IO.File]::Copy($op.Source, $stage, $false)
            if ((File-Hash $stage) -ne $op.After) { throw 'Staged hash mismatch.' }
            if ($op.Before) { [IO.File]::Replace($stage, $op.Target, $swap) }
            else { [IO.File]::Move($stage, $op.Target) }
            $written.Add($op)
        } finally {
            if ([IO.File]::Exists($stage)) { [IO.File]::Delete($stage) }
            if ([IO.File]::Exists($swap)) { [IO.File]::Delete($swap) }
        }
    }
    $inventory.status = 'completed'
} catch {
    for ($i=$written.Count-1; $i -ge 0; $i--) {
        $op = $written[$i]
        if ($op.Before) { [IO.File]::Copy((Join-Path $backup ('files\'+$op.Relative)), $op.Target, $true) }
        else { [IO.File]::Delete($op.Target) }
    }
    $inventory.status = 'rolled_back'
    throw
} finally { [IO.File]::WriteAllText($inventoryFile, ($inventory | ConvertTo-Json -Depth 8)) }
if ($PlanPath) { [IO.File]::WriteAllText([IO.Path]::GetFullPath($PlanPath), ($inventory | ConvertTo-Json -Depth 8)) }
Write-Host 'Frame-confidence fix installed. Model manifests, checkpoints and settings were not changed.'
