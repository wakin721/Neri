<#
.SYNOPSIS
    AI 模型专用 SSH 远程安全连接与执行脚本 (PowerShell 原生版)
.DESCRIPTION
    专为 AI 模型 / 智能体 (Agent) 设计的非交互式安全 SSH 工具。
    彻底杜绝模型索要、接触或泄露服务器密码，内置命令安全审计、超时防护与 JSON 输出。

    【核心安全防线】
    1. 强制免密锁 (BatchMode): 彻底关闭交互式密码通道，认证失败立即退出报警，绝不弹窗或挂起。
    2. 零密码接触: 严禁在脚本、命令或配置文件中传递任何密码。
    3. 毁灭性命令防御拦截: 本地实时拦截 rm -rf /、磁盘格式化、篡改 authorized_keys 等操作。
    4. 结构化 JSON 输出 (-Json): 便于大模型准确解析执行结果与状态。
    5. 超时防护 (-Timeout): 默认 60s 超时，防止因挂起进程卡死模型执行流。

.EXAMPLE
    .\scripts\agent_ssh.ps1 -Action Test
    # 测试与目标服务器的免密连接连通性

.EXAMPLE
    .\scripts\agent_ssh.ps1 -Action Exec -Command "nvidia-smi" -Json
    # 执行远程命令并以结构化 JSON 输出结果

.EXAMPLE
    .\scripts\agent_ssh.ps1 -Action Env -Json
    # 探测远程系统、显卡与 PyTorch 深度学习运行时

.EXAMPLE
    .\scripts\agent_ssh.ps1 -Action Upload -Local "runs/best.pt" -Remote "/Neri_Data/Model/"
    # 安全上传模型权重文件
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("Test", "Exec", "Env", "Upload", "Download", "List")]
    [string]$Action = "Test",

    [Parameter(Position = 1)]
    [string]$Profile = "",

    [string]$Command = "",

    [string]$Local = "",

    [string]$Remote = "",

    [int]$Timeout = 60,

    [switch]$Force,

    [switch]$Json,

    [string]$HostOverride = "",

    [int]$PortOverride = 0,

    [string]$UserOverride = "",

    [string]$KeyPath = "",

    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
if ($Host.UI.RawUI) {
    try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
}

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    if ($PSScriptRoot) {
        $ConfigPath = Join-Path $PSScriptRoot "model_ssh_config.json"
    } else {
        $ConfigPath = Join-Path (Get-Location) "scripts\model_ssh_config.json"
    }
}

# 危险命令拦截正则表达式与原因
$CriticalDangerousRules = @(
    @{ Pattern = '\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(?:/\s*$|/\*|\s+/\s+)'; Desc = '尝试递归强制删除系统根目录 (rm -rf /)' },
    @{ Pattern = '\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+~\s*$'; Desc = '尝试递归强制删除用户主目录 (rm -rf ~)' },
    @{ Pattern = '\bmkfs(?:\.[a-zA-Z0-9]+)?\b'; Desc = '尝试格式化磁盘分区 (mkfs)' },
    @{ Pattern = '\bdd\s+if=/dev/(?:zero|urandom)\s+of=/dev/[a-z0-9]+'; Desc = '尝试使用 dd 覆盖磁盘设备' },
    @{ Pattern = '(?:>|>>)\s*(?:~|/home/[^/]+|/root)/\.ssh/authorized_keys'; Desc = '尝试篡改或覆盖远程 authorized_keys 文件' },
    @{ Pattern = '\brm\s+.*(?:~|/home/[^/]+|/root)/\.ssh'; Desc = '尝试删除远程 .ssh 密钥目录' },
    @{ Pattern = ':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:'; Desc = '检测到 Fork 炸弹代码' },
    @{ Pattern = '\b(?:shutdown|poweroff|halt|init\s+0)\b'; Desc = '尝试关闭远程服务器电源' }
)

$WarningDangerousRules = @(
    @{ Pattern = '\breboot\b'; Desc = '系统重启命令' },
    @{ Pattern = '\bkillall\s+-9\b'; Desc = '强制杀死进程' }
)

function Audit-RemoteCommand {
    param([string]$Cmd, [switch]$AllowWarning)
    foreach ($rule in $CriticalDangerousRules) {
        if ($Cmd -match $rule.Pattern) {
            return @{ Allowed = $false; Reason = "[安全拦截] 禁止执行毁灭性危险指令: $($rule.Desc)" }
        }
    }
    if (-not $AllowWarning) {
        foreach ($rule in $WarningDangerousRules) {
            if ($Cmd -match $rule.Pattern) {
                return @{ Allowed = $false; Reason = "[安全警告] 该指令存在运维风险: $($rule.Desc)。确认执行请追加 -Force 参数。" }
            }
        }
    }
    return @{ Allowed = $true; Reason = "" }
}

function Find-AgentKey {
    $candidates = @(
        (Join-Path $HOME ".ssh\id_ed25519_agent"),
        (Join-Path $HOME ".ssh\id_ed25519_neri_agent"),
        (Join-Path $HOME ".ssh\id_rsa_agent"),
        (Join-Path $PSScriptRoot "id_ed25519_agent")
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path -LiteralPath $c)) {
            return $c
        }
    }
    return ""
}

function Load-ProfileConfig {
    $cfg = $null
    if (Test-Path -LiteralPath $ConfigPath) {
        try {
            $raw = [System.IO.File]::ReadAllText($ConfigPath, [System.Text.Encoding]::UTF8)
            $cfg = ($raw | ConvertFrom-Json)
        } catch {}
    }
    if ($null -eq $cfg) {
        $cfg = [PSCustomObject]@{
            default_profile = "neri-cloud"
            profiles = [PSCustomObject]@{
                "neri-cloud" = [PSCustomObject]@{
                    name = "neri-cloud"
                    host = "47.108.14.199"
                    port = 22
                    user = "codex"
                    key_path = ""
                    description = "Neri 官方云端模型分发服务器"
                }
            }
        }
    }
    return $cfg
}

function Resolve-Target {
    param($Config)
    $targetName = if ($Profile) { $Profile } elseif ($Config.default_profile) { $Config.default_profile } else { "neri-cloud" }
    $prof = $null
    if ($Config.profiles -and $Config.profiles.$targetName) {
        $prof = $Config.profiles.$targetName
    } else {
        $prof = [PSCustomObject]@{
            name = $targetName
            host = $targetName
            port = 22
            user = ""
            key_path = ""
            description = "自定义配置"
        }
    }

    $resolved = [PSCustomObject]@{
        name = $targetName
        host = if ($HostOverride) { $HostOverride } else { $prof.host }
        port = if ($PortOverride -gt 0) { $PortOverride } elseif ($prof.port) { $prof.port } else { 22 }
        user = if ($UserOverride) { $UserOverride } else { $prof.user }
        key_path = if ($KeyPath) { $KeyPath } elseif ($prof.key_path) { $prof.key_path } else { Find-AgentKey }
        description = $prof.description
    }
    return $resolved
}

function Build-SafeSshArgs {
    param($P, [string[]]$ExtraCmd = @())
    $argsList = [System.Collections.Generic.List[string]]::new()
    $argsList.Add("-o")
    $argsList.Add("BatchMode=yes")
    $argsList.Add("-o")
    $argsList.Add("PasswordAuthentication=no")
    $argsList.Add("-o")
    $argsList.Add("KbdInteractiveAuthentication=no")
    $argsList.Add("-o")
    $argsList.Add("StrictHostKeyChecking=accept-new")
    $argsList.Add("-o")
    $argsList.Add("ConnectTimeout=10")
    $argsList.Add("-o")
    $argsList.Add("ServerAliveInterval=30")
    $argsList.Add("-o")
    $argsList.Add("ServerAliveCountMax=3")

    if ($P.port -and $P.port -ne 22) {
        $argsList.Add("-p")
        $argsList.Add("$($P.port)")
    }

    if ($P.key_path -and (Test-Path -LiteralPath $P.key_path)) {
        $argsList.Add("-i")
        $argsList.Add($P.key_path)
    }

    $target = if ($P.user) { "$($P.user)@$($P.host)" } else { $P.host }
    $argsList.Add($target)

    if ($ExtraCmd.Count -gt 0) {
        foreach ($c in $ExtraCmd) {
            $argsList.Add($c)
        }
    }

    return $argsList.ToArray()
}

function Execute-ProcessWithTimeout {
    param([string]$ExePath, [string[]]$Arguments, [int]$TimeoutSeconds = 60)
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = $ExePath
    $pinfo.Arguments = ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_.Replace('"', '\"')) + '"'
        } else {
            $_
        }
    }) -join " "
    $pinfo.RedirectStandardOutput = $true
    $pinfo.RedirectStandardError = $true
    $pinfo.UseShellExecute = $false
    $pinfo.CreateNoWindow = $true
    $pinfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $pinfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $pinfo

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    [void]$proc.Start()

    $stdOutTask = $proc.StandardOutput.ReadToEndAsync()
    $stdErrTask = $proc.StandardError.ReadToEndAsync()

    $timeoutMs = $TimeoutSeconds * 1000
    if ($proc.WaitForExit($timeoutMs)) {
        [System.Threading.Tasks.Task]::WaitAll(@($stdOutTask, $stdErrTask))
        $sw.Stop()
        return [PSCustomObject]@{
            ExitCode = $proc.ExitCode
            StdOut = $stdOutTask.Result
            StdErr = $stdErrTask.Result
            DurationSeconds = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
            TimedOut = $false
        }
    } else {
        try { $proc.Kill() } catch {}
        $sw.Stop()
        return [PSCustomObject]@{
            ExitCode = -1
            StdOut = ""
            StdErr = "[TIMEOUT] 执行超过限定时间 (${TimeoutSeconds}s)，进程已被强制终止。"
            DurationSeconds = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
            TimedOut = $true
        }
    }
}

$configData = Load-ProfileConfig

if ($Action -eq "List") {
    $listResult = [PSCustomObject]@{
        action = "list"
        default_profile = $configData.default_profile
        profiles = @()
    }
    if ($configData.profiles) {
        foreach ($prop in $configData.profiles.PSObject.Properties) {
            $pObj = $prop.Value
            $listResult.profiles += [PSCustomObject]@{
                name = $prop.Name
                host = $pObj.host
                port = $pObj.port
                user = $pObj.user
                has_key = [bool]$pObj.key_path
                key_path = $pObj.key_path
                description = $pObj.description
            }
        }
    }
    if ($Json) {
        $listResult | ConvertTo-Json -Depth 5
    } else {
        Write-Host "默认服务器: $($listResult.default_profile)" -ForegroundColor Cyan
        foreach ($item in $listResult.profiles) {
            Write-Host " - [$($item.name)] $($item.user)@$($item.host):$($item.port) : $($item.description)"
        }
    }
    exit 0
}

$targetProfile = Resolve-Target -Config $configData
$result = $null

if ($Action -eq "Test") {
    $testCmd = "echo '__NERI_SSH_TEST_OK__'"
    $sshArgs = Build-SafeSshArgs -P $targetProfile -ExtraCmd @($testCmd)
    $execRes = Execute-ProcessWithTimeout -ExePath "ssh.exe" -Arguments $sshArgs -TimeoutSeconds 15

    $isSuccess = ($execRes.ExitCode -eq 0 -and $execRes.StdOut -like "*__NERI_SSH_TEST_OK__*")
    $result = [PSCustomObject]@{
        action = "test"
        profile = $targetProfile.name
        target = "$($targetProfile.user)@$($targetProfile.host):$($targetProfile.port)"
        key_path = $targetProfile.key_path
        success = $isSuccess
        exit_code = $execRes.ExitCode
        duration_seconds = $execRes.DurationSeconds
    }

    if ($isSuccess) {
        $result | Add-Member -Name "message" -Value "SSH 免密连接成功！模型可在无密码暴露状态下安全执行运维命令。" -MemberType NoteProperty
    } else {
        $errMsg = "[安全锁定 - 免密认证失败]`n目标: $($targetProfile.user)@$($targetProfile.host):$($targetProfile.port)`n" +
                  "底层报错: $($execRes.StdErr.Trim())`n`n" +
                  "【核心防护说明】`n本脚本为 AI 模型专用，已强制开启 BatchMode 禁用密码交互，防止模型接触、索取或泄露密码。`n" +
                  "请勿在对话中提供密码！请通知人类管理员在本地终端运行以下脚本完成公钥授权：`n  PowerShell: .\scripts\setup_agent_ssh.ps1`n"
        $result | Add-Member -Name "error" -Value $errMsg -MemberType NoteProperty
    }
}
elseif ($Action -eq "Exec") {
    if ([string]::IsNullOrWhiteSpace($Command)) {
        $result = [PSCustomObject]@{
            action = "exec"
            success = $false
            error = "未提供要执行的远程命令 (-Command)。"
        }
    } else {
        $audit = Audit-RemoteCommand -Cmd $Command -AllowWarning:$Force
        if (-not $audit.Allowed) {
            $result = [PSCustomObject]@{
                action = "exec"
                command = $Command
                success = $false
                blocked = $true
                error = $audit.Reason
            }
        } else {
            $sshArgs = Build-SafeSshArgs -P $targetProfile -ExtraCmd @($Command)
            $execRes = Execute-ProcessWithTimeout -ExePath "ssh.exe" -Arguments $sshArgs -TimeoutSeconds $Timeout

            $result = [PSCustomObject]@{
                action = "exec"
                profile = $targetProfile.name
                command = $Command
                exit_code = $execRes.ExitCode
                success = ($execRes.ExitCode -eq 0)
                duration_seconds = $execRes.DurationSeconds
                stdout = $execRes.StdOut
                stderr = $execRes.StdErr
            }
            if ($execRes.ExitCode -eq 255 -and ($execRes.StdErr -like "*Permission denied*" -or $execRes.StdErr -like "*timed out*")) {
                $result | Add-Member -Name "auth_error" -Value "SSH 免密认证失败。出于安全防护，已禁止密码认证。请管理员运行 scripts/setup_agent_ssh.ps1。" -MemberType NoteProperty
            }
        }
    }
}
elseif ($Action -eq "Env") {
    $probeScript = "echo '=== HOST INFO ==='; hostname; uname -a; whoami; uptime; " +
                   "echo '=== GPU INFO ==='; if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader; else echo 'NO_NVIDIA_GPU'; fi; " +
                   "echo '=== PYTHON ==='; python3 -c ""import sys; print('Python:', sys.version.split()[0]);"" 2>&1"
    $sshArgs = Build-SafeSshArgs -P $targetProfile -ExtraCmd @($probeScript)
    $execRes = Execute-ProcessWithTimeout -ExePath "ssh.exe" -Arguments $sshArgs -TimeoutSeconds 30
    $result = [PSCustomObject]@{
        action = "env"
        profile = $targetProfile.name
        exit_code = $execRes.ExitCode
        success = ($execRes.ExitCode -eq 0)
        duration_seconds = $execRes.DurationSeconds
        stdout = $execRes.StdOut
        stderr = $execRes.StdErr
    }
}
elseif ($Action -eq "Upload") {
    if (-not (Test-Path -LiteralPath $Local)) {
        $result = [PSCustomObject]@{ action = "upload"; success = $false; error = "本地文件不存在: $Local" }
    } else {
        $scpArgs = [System.Collections.Generic.List[string]]::new()
        $scpArgs.Add("-r")
        $scpArgs.Add("-C")
        $scpArgs.Add("-o")
        $scpArgs.Add("BatchMode=yes")
        $scpArgs.Add("-o")
        $scpArgs.Add("PasswordAuthentication=no")
        $scpArgs.Add("-o")
        $scpArgs.Add("StrictHostKeyChecking=accept-new")
        if ($targetProfile.port -and $targetProfile.port -ne 22) {
            $scpArgs.Add("-P")
            $scpArgs.Add("$($targetProfile.port)")
        }
        if ($targetProfile.key_path -and (Test-Path -LiteralPath $targetProfile.key_path)) {
            $scpArgs.Add("-i")
            $scpArgs.Add($targetProfile.key_path)
        }
        $targetSpec = if ($targetProfile.user) { "$($targetProfile.user)@$($targetProfile.host):$Remote" } else { "$($targetProfile.host):$Remote" }
        $scpArgs.Add($Local)
        $scpArgs.Add($targetSpec)

        $execRes = Execute-ProcessWithTimeout -ExePath "scp.exe" -Arguments $scpArgs.ToArray() -TimeoutSeconds $Timeout
        $result = [PSCustomObject]@{
            action = "upload"
            profile = $targetProfile.name
            local = $Local
            remote = $Remote
            exit_code = $execRes.ExitCode
            success = ($execRes.ExitCode -eq 0)
            duration_seconds = $execRes.DurationSeconds
            stdout = $execRes.StdOut
            stderr = $execRes.StdErr
        }
    }
}
elseif ($Action -eq "Download") {
    $scpArgs = [System.Collections.Generic.List[string]]::new()
    $scpArgs.Add("-r")
    $scpArgs.Add("-C")
    $scpArgs.Add("-o")
    $scpArgs.Add("BatchMode=yes")
    $scpArgs.Add("-o")
    $scpArgs.Add("PasswordAuthentication=no")
    $scpArgs.Add("-o")
    $scpArgs.Add("StrictHostKeyChecking=accept-new")
    if ($targetProfile.port -and $targetProfile.port -ne 22) {
        $scpArgs.Add("-P")
        $scpArgs.Add("$($targetProfile.port)")
    }
    if ($targetProfile.key_path -and (Test-Path -LiteralPath $targetProfile.key_path)) {
        $scpArgs.Add("-i")
        $scpArgs.Add($targetProfile.key_path)
    }
    $srcSpec = if ($targetProfile.user) { "$($targetProfile.user)@$($targetProfile.host):$Remote" } else { "$($targetProfile.host):$Remote" }
    $scpArgs.Add($srcSpec)
    $scpArgs.Add($Local)

    $execRes = Execute-ProcessWithTimeout -ExePath "scp.exe" -Arguments $scpArgs.ToArray() -TimeoutSeconds $Timeout
    $result = [PSCustomObject]@{
        action = "download"
        profile = $targetProfile.name
        remote = $Remote
        local = $Local
        exit_code = $execRes.ExitCode
        success = ($execRes.ExitCode -eq 0)
        duration_seconds = $execRes.DurationSeconds
        stdout = $execRes.StdOut
        stderr = $execRes.StdErr
    }
}

if ($Json) {
    $result | ConvertTo-Json -Depth 5
} else {
    if ($result.success) {
        Write-Host "[SUCCESS] 操作 '$Action' 执行成功 (耗时: $($result.duration_seconds)s)" -ForegroundColor Green
        if ($result.message) { Write-Host $result.message -ForegroundColor Cyan }
        if ($result.stdout) { Write-Host "`n[STDOUT]:`n$($result.stdout.Trim())" }
    } else {
        Write-Host "[FAILED] 操作 '$Action' 失败 (退出码: $($result.exit_code))" -ForegroundColor Red
        if ($result.error) { Write-Host $result.error -ForegroundColor Yellow }
        if ($result.auth_error) { Write-Host $result.auth_error -ForegroundColor Yellow }
        if ($result.stderr) { Write-Host "`n[STDERR]:`n$($result.stderr.Trim())" -ForegroundColor DarkRed }
    }
}

if ($result.success) { exit 0 } else { exit 1 }
