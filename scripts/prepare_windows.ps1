$ErrorActionPreference = "Stop"

$backup = Join-Path $env:RUNNER_TEMP "neri-windows-runner"
New-Item -ItemType Directory -Path $backup -Force | Out-Null

$customFiles = @(
  "flutter_window.cpp",
  "flutter_window.h",
  "main.cpp",
  "win32_window.cpp",
  "win32_window.h"
)

foreach ($file in $customFiles) {
  Copy-Item "windows\runner\$file" $backup -Force
}

flutter config --enable-windows-desktop
if ($LASTEXITCODE -ne 0) { throw "Flutter configuration failed." }
flutter create --platforms=windows --project-name neri_flutter .
if ($LASTEXITCODE -ne 0) { throw "Windows project generation failed." }
Remove-Item "test\widget_test.dart" -Force -ErrorAction SilentlyContinue
& (Join-Path $env:GITHUB_WORKSPACE "scripts\configure_windows_product.ps1")

foreach ($file in $customFiles) {
  Copy-Item (Join-Path $backup $file) "windows\runner\$file" -Force
}

Copy-Item "..\res\ico.ico" "windows\runner\resources\app_icon.ico" -Force
