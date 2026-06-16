param(
    [Parameter(Mandatory = $true)]
    [int]$MainPid,

    [Parameter(Mandatory = $true)]
    [string]$Session,

    [Parameter(Mandatory = $true)]
    [string]$FrontendLog,

    [Parameter(Mandatory = $true)]
    [string]$BackendLog,

    [string]$DiagnosticLog = "",

    [string]$Title = "Neri"
)

$ErrorActionPreference = "SilentlyContinue"

function Write-Diagnostic {
    param([string]$Message)
    if ([string]::IsNullOrWhiteSpace($DiagnosticLog)) {
        return
    }
    try {
        $directory = Split-Path -Parent $DiagnosticLog
        if (-not [string]::IsNullOrWhiteSpace($directory)) {
            New-Item -ItemType Directory -Path $directory -Force | Out-Null
        }
        $line = "[{0}] {1}" -f [DateTime]::Now.ToString("o"), $Message
        Add-Content -LiteralPath $DiagnosticLog -Value $line -Encoding UTF8
    } catch {
    }
}

function Get-SessionData {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Get-SessionStatus {
    param($Data)
    if ($null -eq $Data) {
        return ""
    }
    try {
        return [string]$Data.status
    } catch {
        return ""
    }
}

function Get-CrashLogStartOffset {
    param($Data)
    if ($null -eq $Data) {
        return 0
    }
    try {
        $value = [int64]$Data.crash_log_start_offset
        if ($value -lt 0) {
            return 0
        }
        return $value
    } catch {
        return 0
    }
}

function Get-LogTail {
    param(
        [string]$Path,
        [int]$MaxBytes = 80000,
        [int64]$StartOffset = 0
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return ""
    }
    try {
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try {
            if ($StartOffset -ge $stream.Length) {
                return ""
            }
            $start = [Math]::Max($StartOffset, $stream.Length - $MaxBytes)
            $stream.Seek($start, [System.IO.SeekOrigin]::Begin) | Out-Null
            $buffer = New-Object byte[] ([int]($stream.Length - $stream.Position))
            $read = $stream.Read($buffer, 0, $buffer.Length)
            return [System.Text.Encoding]::UTF8.GetString($buffer, 0, $read)
        } finally {
            $stream.Dispose()
        }
    } catch {
        return ""
    }
}

function Test-ReasonLine {
    param([string]$Line)
    $lower = $Line.ToLowerInvariant()
    return $lower.Contains("error") `
        -or $lower.Contains("exception") `
        -or $lower.Contains("traceback") `
        -or $lower.Contains("failed") `
        -or $lower.Contains("crash")
}

function Get-LatestReason {
    param(
        [string]$Path,
        [int64]$StartOffset = 0
    )
    $content = Get-LogTail -Path $Path -StartOffset $StartOffset
    if ([string]::IsNullOrWhiteSpace($content)) {
        return $null
    }
    $lines = $content -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        if ($lines[$i].StartsWith("Message:")) {
            return $lines[$i].Substring("Message:".Length).Trim()
        }
    }
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        if (Test-ReasonLine -Line $lines[$i]) {
            return $lines[$i]
        }
    }
    return $null
}

function Add-FallbackCrashLog {
    param(
        [string]$Path,
        [string]$Reason,
        $ExitCode
    )
    try {
        $directory = Split-Path -Parent $Path
        if (-not [string]::IsNullOrWhiteSpace($directory)) {
            New-Item -ItemType Directory -Path $directory -Force | Out-Null
        }
        $exitCodeText = if ($null -eq $ExitCode) { "unknown" } else { [string]$ExitCode }
        $exceptionCode = ""
        if ($null -ne $ExitCode) {
            try {
                $unsignedCode = [int64]$ExitCode
                if ($unsignedCode -lt 0) {
                    $unsignedCode += 4294967296
                }
                $exceptionCode = "0x{0:X8}" -f ([uint32]$unsignedCode)
            } catch {
                $exceptionCode = ""
            }
        }
        $lines = @(
            "============================================================",
            "[Frontend process crash] $([DateTimeOffset]::Now.ToString('o'))",
            "Origin: Windows process watchdog",
            "Message: $Reason",
            "Exit code: $exitCodeText"
        )
        if (-not [string]::IsNullOrWhiteSpace($exceptionCode)) {
            $lines += "Windows exception code: $exceptionCode"
        }
        $lines += @(
            "",
            "No Dart/Flutter crash details were written before the process terminated, so the watchdog generated this fallback crash record.",
            ""
        )
        Add-Content -LiteralPath $Path -Value ($lines -join [Environment]::NewLine) -Encoding UTF8
    } catch {
    }
}

Write-Diagnostic "Watcher started. MainPid=$MainPid; Session=$Session; FrontendLog=$FrontendLog; BackendLog=$BackendLog"

try {
    $process = [System.Diagnostics.Process]::GetProcessById($MainPid)
    Write-Diagnostic "Attached to main process."
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    Write-Diagnostic "Main process exited. ExitCode=$exitCode"
} catch {
    Write-Diagnostic "Failed to attach to or wait for main process: $($_.Exception.Message)"
    $exitCode = $null
}

$sessionData = Get-SessionData -Path $Session
if ((Get-SessionStatus -Data $sessionData) -eq "normal") {
    Write-Diagnostic "Session marked normal. Exiting without crash popup."
    exit 0
}
if ($exitCode -eq 0) {
    Write-Diagnostic "Exit code is 0. Exiting without crash popup."
    exit 0
}

$logStartOffset = Get-CrashLogStartOffset -Data $sessionData
$reason = Get-LatestReason -Path $FrontendLog -StartOffset $logStartOffset
if (-not $reason) {
    $reason = Get-LatestReason -Path $BackendLog -StartOffset $logStartOffset
}
if (-not $reason) {
    $exitCodeText = if ($null -eq $exitCode) { "unknown" } else { [string]$exitCode }
    $reason = "Main process exited unexpectedly. Exit code: $exitCodeText."
    Add-FallbackCrashLog -Path $FrontendLog -Reason $reason -ExitCode $exitCode
}

$message = @(
    "Neri 已异常退出。",
    "",
    "原因：$reason",
    "",
    "崩溃日志：$FrontendLog"
) -join [Environment]::NewLine
$caption = $Title + " 崩溃提示"

function Show-CrashMessage {
    param(
        [string]$Text,
        [string]$WindowTitle
    )

    try {
        Write-Diagnostic "Attempting crash popup with WScript.Shell.Popup."
        $shell = New-Object -ComObject WScript.Shell -ErrorAction Stop
        # 4096 keeps the report visible above other windows, 16 is the error icon.
        $shell.Popup($Text, 0, $WindowTitle, 4112) | Out-Null
        Write-Diagnostic "Crash popup shown with WScript.Shell.Popup."
        return $true
    } catch {
        Write-Diagnostic "WScript.Shell.Popup failed: $($_.Exception.Message)"
    }

    try {
        Write-Diagnostic "Attempting crash popup with Windows Forms MessageBox."
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        [System.Windows.Forms.MessageBox]::Show(
            $Text,
            $WindowTitle,
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
        Write-Diagnostic "Crash popup shown with Windows Forms MessageBox."
        return $true
    } catch {
        Write-Diagnostic "Windows Forms MessageBox failed: $($_.Exception.Message)"
    }

    try {
        Write-Diagnostic "Attempting crash popup with WPF MessageBox."
        Add-Type -AssemblyName PresentationFramework -ErrorAction Stop
        [System.Windows.MessageBox]::Show($Text, $WindowTitle, "OK", "Error") | Out-Null
        Write-Diagnostic "Crash popup shown with WPF MessageBox."
        return $true
    } catch {
        Write-Diagnostic "WPF MessageBox failed: $($_.Exception.Message)"
    }

    return $false
}

try {
    if (-not (Show-CrashMessage -Text $message -WindowTitle $caption)) {
        Write-Diagnostic "All crash popup methods failed; writing to error stream."
        Write-Error $message
    }
} catch {
    Write-Diagnostic "Unexpected popup failure: $($_.Exception.Message)"
    try {
        Write-Error $message
    } catch {
    }
}
