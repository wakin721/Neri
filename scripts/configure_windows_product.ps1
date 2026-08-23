param(
    [string]$WindowsRoot = (
        Join-Path $PSScriptRoot "..\frontend\windows"
    )
)

$ErrorActionPreference = "Stop"
$utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)

function Set-RequiredProductText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$GeneratedText,
        [Parameter(Mandatory = $true)]
        [string]$ProductText,
        [string[]]$AcceptedProductText = @()
    )

    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $content = [System.IO.File]::ReadAllText($resolvedPath)
    if ($content.Contains($GeneratedText)) {
        $content = $content.Replace($GeneratedText, $ProductText)
        [System.IO.File]::WriteAllText($resolvedPath, $content, $utf8WithoutBom)
        return
    }
    foreach ($acceptedText in @($ProductText) + @($AcceptedProductText)) {
        if ($content.Contains($acceptedText)) {
            return
        }
    }
    throw "Expected generated product text was not found in $resolvedPath"
}

$cmakePath = Join-Path $WindowsRoot "CMakeLists.txt"
$mainPath = Join-Path $WindowsRoot "runner\main.cpp"
$resourcePath = Join-Path $WindowsRoot "runner\Runner.rc"

Set-RequiredProductText $cmakePath `
    'project(neri_flutter LANGUAGES CXX)' `
    'project(Neri LANGUAGES CXX)'
Set-RequiredProductText $cmakePath `
    'set(BINARY_NAME "neri_flutter")' `
    'set(BINARY_NAME "Neri")'
Set-RequiredProductText `
    -Path $mainPath `
    -GeneratedText 'window.Create(L"neri_flutter", origin, size)' `
    -ProductText 'window.Create(L"Neri", origin, size)' `
    -AcceptedProductText @(
        'window.Create(kNeriWindowTitle, origin, size)'
    )
Set-RequiredProductText $resourcePath `
    'VALUE "FileDescription", "neri_flutter" "\0"' `
    'VALUE "FileDescription", "Neri" "\0"'
Set-RequiredProductText $resourcePath `
    'VALUE "InternalName", "neri_flutter" "\0"' `
    'VALUE "InternalName", "Neri" "\0"'
Set-RequiredProductText $resourcePath `
    'VALUE "OriginalFilename", "neri_flutter.exe" "\0"' `
    'VALUE "OriginalFilename", "Neri.exe" "\0"'
Set-RequiredProductText $resourcePath `
    'VALUE "ProductName", "neri_flutter" "\0"' `
    'VALUE "ProductName", "Neri" "\0"'

Write-Output "Windows product name configured as Neri."

