param(
    [Parameter(Mandatory = $true)][string]$ToolkitPath
)

$ErrorActionPreference = "Stop"

$sitePackages = Join-Path $ToolkitPath "Lib\site-packages"
if (-not (Test-Path -LiteralPath $sitePackages -PathType Container)) {
    throw "Python site-packages directory is missing: $sitePackages"
}

$exactArtifacts = @(
    "torch",
    "torch.libs",
    "torchvision",
    "torchvision.libs",
    "torchaudio",
    "torchaudio.lib",
    "torchaudio.libs",
    "functorch",
    "torchgen"
)
$metadataPatterns = @(
    "torch-*.dist-info",
    "torchvision-*.dist-info",
    "torchaudio-*.dist-info"
)

foreach ($name in $exactArtifacts) {
    $path = Join-Path $sitePackages $name
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

foreach ($pattern in $metadataPatterns) {
    Get-ChildItem -LiteralPath $sitePackages -Force |
        Where-Object { $_.Name -like $pattern } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force
        }
}

$remaining = @()
foreach ($item in Get-ChildItem -LiteralPath $sitePackages -Force) {
    if ($exactArtifacts -contains $item.Name) {
        $remaining += $item
        continue
    }
    foreach ($pattern in $metadataPatterns) {
        if ($item.Name -like $pattern) {
            $remaining += $item
            break
        }
    }
}

if ($remaining.Count -gt 0) {
    $names = ($remaining | ForEach-Object { $_.Name }) -join ", "
    throw "PyTorch artifacts remain in lite toolkit: $names"
}
