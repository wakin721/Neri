$ErrorActionPreference = "Stop"

$versionMatch = Select-String `
  -Path "frontend\pubspec.yaml" `
  -Pattern '^version:\s*(\S+)' |
  Select-Object -First 1
if ($null -eq $versionMatch) {
  throw "Unable to read the application version from frontend/pubspec.yaml."
}

$pubspecVersion = $versionMatch.Matches[0].Groups[1].Value.Trim()
$baseVersion = ($pubspecVersion -split '\+', 2)[0]
$versionParts = $baseVersion -split '-', 2
$versionNumber = $versionParts[0]
$releaseLabel = if ($versionParts.Count -eq 2) {
  $versionParts[1]
} else {
  "release"
}

if ($versionNumber -notmatch '^\d+(?:\.\d+)+$') {
  throw "Invalid application version: $versionNumber"
}
if ($releaseLabel -notmatch '^[A-Za-z0-9.-]+$') {
  throw "Invalid release label: $releaseLabel"
}

$artifactName = "Neri_v${versionNumber}_${releaseLabel}_x64_lite"
"artifact_name=$artifactName" |
  Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
"Artifact: $artifactName.zip" |
  Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Encoding utf8 -Append

$packageRoot = Join-Path $env:GITHUB_WORKSPACE "package\$artifactName"
$package = Join-Path $packageRoot "Neri"
$release = Join-Path $env:GITHUB_WORKSPACE "frontend\build\windows\x64\runner\Release"
$pythonRoot = Split-Path -Parent (Get-Command python).Source

New-Item -ItemType Directory -Path $package -Force | Out-Null
Copy-Item "$release\*" $package -Recurse -Force
Copy-Item "system" $package -Recurse -Force
Copy-Item "res" $package -Recurse -Force
Copy-Item "requirements.txt" $package -Force
Copy-Item "res\demo\README_Update.md" (Join-Path $package "README_Update.md") -Force

$requiredModelPaths = @(
  "res\model\detect\user",
  "res\model\detect\sync",
  "res\model\cls\user",
  "res\model\cls\sync"
)
foreach ($relativePath in $requiredModelPaths) {
  if (-not (Test-Path (Join-Path $package $relativePath))) {
    throw "Missing canonical model path: $relativePath"
  }
}

$resDirectories = @(
  Get-ChildItem (Join-Path $package "res") -Directory |
    ForEach-Object { $_.Name }
)
if (-not ($resDirectories -ccontains "model")) {
  throw "Canonical lowercase res/model directory is missing."
}
if ($resDirectories -ccontains "Model") {
  throw "Uppercase res/Model directory is still packaged."
}
if ($resDirectories -ccontains "model_cls") {
  throw "Legacy res/model_cls directory is still packaged."
}

$toolkit = Join-Path $package "toolkit"
New-Item -ItemType Directory -Path $toolkit -Force | Out-Null
Copy-Item "$pythonRoot\*" $toolkit -Recurse -Force

$pruneScript = Join-Path $env:GITHUB_WORKSPACE "scripts\prune_lite_python.ps1"
if (-not (Test-Path -LiteralPath $pruneScript -PathType Leaf)) {
  throw "Missing lite Python pruning script: $pruneScript"
}
& $pruneScript -ToolkitPath $toolkit

$generatedExe = Join-Path $package "neri_flutter.exe"
$productExe = Join-Path $package "Neri.exe"
if (Test-Path $generatedExe) {
  Move-Item $generatedExe $productExe -Force
}
if (-not (Test-Path $productExe)) {
  throw "Neri.exe was not produced by the Flutter build."
}
if (Test-Path $generatedExe) {
  throw "The legacy neri_flutter.exe name is still present in the package."
}

& "$toolkit\python.exe" -c "import cv2, fastapi, lap, openpyxl, pandas, PIL, pydantic, pypinyin, uvicorn; import system.model_sync.manager"
if ($LASTEXITCODE -ne 0) { throw "Packaged Python import check failed." }
