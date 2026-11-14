<#
.SYNOPSIS
    AI 模型专用的 SSH 远程连接安全初始化向导 (人类管理员专用)
.DESCRIPTION
    本脚本由【人类管理员】在本地终端中运行一次。
    用于为 AI 模型/智能体生成独立的专用 SSH 密钥对，并将公钥安全部署至远程服务器。

    【核心安全目标】
    1. 模型永远不接触服务器密码：公钥部署过程仅由人类在真实控制台中输入一次密码，
       密码由 OpenSSH 原生控制台处理，绝不存入任何脚本、配置、环境变量或日志中。
    2. 独立密钥隔离：为模型生成专用的 Ed25519 密钥（如 id_ed25519_agent），
       不与管理员的个人私钥混用。
    3. 自动化配置同步：自动将密钥路径配置到 scripts/model_ssh_config.json 和 ~/.ssh/config。
    4. 连通性自检：初始化完成后自动以强制免密模式验证连接。

.EXAMPLE
    .\scripts\setup_agent_ssh.ps1
    # 启动交互式配置向导（默认针对 neri-cloud: 47.108.14.199）

.EXAMPLE
    .\scripts\setup_agent_ssh.ps1 -Profile neri-cloud -AutoTest
    # 非交互式针对 neri-cloud 初始化并在完成后测试
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Profile = "neri-cloud",

    [string]$RemoteHost = "",

    [int]$RemotePort = 0,

    [string]$RemoteUser = "",

    [string]$KeyName = "id_ed25519_agent",

    [string]$ConfigPath = "",

    [switch]$AutoTest,

    [switch]$SkipDeploy,

    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
if ($Host.UI.RawUI) {
    try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
}

function Write-Banner {
    param([string]$Title)
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Yellow
    Write-Host "==========================================================" -ForegroundColor Cyan
}

function Write-Step {
    param([int]$Step, [string]$Desc)
    Write-Host ""
    Write-Host "[$Step] $Desc" -ForegroundColor Green
}

# 1. 定位配置文件
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    if ($PSScriptRoot) {
        $ConfigPath = Join-Path $PSScriptRoot "model_ssh_config.json"
    } else {
        $ConfigPath = Join-Path (Get-Location) "scripts\model_ssh_config.json"
    }
}

Write-Banner "AI 模型 SSH 免密安全部署向导 (Zero-Password Architecture)"
Write-Host "【安全原则】通过 Ed25519 密钥对鉴权，防止 AI 模型接触、询问或泄露远程服务器密码。" -ForegroundColor Gray

# 2. 读取当前服务器配置
$configJson = $null
if (Test-Path -LiteralPath $ConfigPath) {
    try {
        $raw = [System.IO.File]::ReadAllText($ConfigPath, [System.Text.Encoding]::UTF8)
        $configJson = ($raw | ConvertFrom-Json)
    } catch {
        Write-Host "[WARN] 读取 $ConfigPath 失败: $_" -ForegroundColor Yellow
    }
}

$targetProfile = $null
if ($configJson -and $configJson.profiles) {
    $profileName = if ($Profile) { $Profile } else { $configJson.default_profile }
    $targetProfile = $configJson.profiles.$profileName
}

$finalHost = if ($RemoteHost) { $RemoteHost } elseif ($targetProfile -and $targetProfile.host) { $targetProfile.host } else { "47.108.14.199" }
$finalPort = if ($RemotePort -gt 0) { $RemotePort } elseif ($targetProfile -and $targetProfile.port) { $targetProfile.port } else { 22 }
$finalUser = if ($RemoteUser) { $RemoteUser } elseif ($targetProfile -and $targetProfile.user) { $targetProfile.user } else { "codex" }

Write-Host "当前目标配置: [$Profile]" -ForegroundColor Cyan
Write-Host "  主机地址: $finalHost" -ForegroundColor White
Write-Host "  SSH 端口: $finalPort" -ForegroundColor White
Write-Host "  远程用户: $finalUser" -ForegroundColor White

if (-not $Yes) {
    $confirm = Read-Host "`n是否确认使用上述目标服务器信息进行初始化? (Y/n)"
    if ($confirm -and $confirm.Trim().ToLower() -eq "n") {
        $finalHost = (Read-Host "请输入服务器主机名/IP [$finalHost]").Trim()
        if (-not $finalHost) { $finalHost = "47.108.14.199" }
        $inPort = (Read-Host "请输入 SSH 端口 [$finalPort]").Trim()
        if ($inPort) { $finalPort = [int]$inPort }
        $inUser = (Read-Host "请输入登录用户名 [$finalUser]").Trim()
        if ($inUser) { $finalUser = $inUser }
    }
}

# 3. 检查与生成模型专属 Ed25519 密钥
Write-Step 1 "准备模型专属 Ed25519 密钥对"
$sshDir = Join-Path $HOME ".ssh"
if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
}

$privateKeyPath = Join-Path $sshDir $KeyName
$publicKeyPath = "$privateKeyPath.pub"

if (Test-Path $privateKeyPath) {
    Write-Host "[OK] 检测到已存在专用密钥: $privateKeyPath" -ForegroundColor Green
    if (-not $Yes) {
        $recreate = Read-Host "是否重新生成该密钥? 重新生成需要重新向服务器授权公钥。(y/N)"
        if ($recreate -and $recreate.Trim().ToLower() -eq "y") {
            Remove-Item -Force $privateKeyPath -ErrorAction SilentlyContinue
            Remove-Item -Force $publicKeyPath -ErrorAction SilentlyContinue
        }
    }
}

if (-not (Test-Path $privateKeyPath)) {
    Write-Host "正在生成专用于 AI 模型的 Ed25519 无密码私钥..." -ForegroundColor Cyan
    $keygen = "ssh-keygen.exe"
    if (-not (Get-Command $keygen -ErrorAction SilentlyContinue)) {
        throw "未找到 ssh-keygen.exe，请确保系统已安装 OpenSSH。"
    }
    $comment = "neri-agent-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    & $keygen -t ed25519 -N '""' -f $privateKeyPath -C $comment
    if ($LASTEXITCODE -ne 0) {
        throw "ssh-keygen 生成密钥失败，退出码: $LASTEXITCODE"
    }
    Write-Host "[OK] 专属密钥已成功生成: $privateKeyPath" -ForegroundColor Green
}

if (-not (Test-Path $publicKeyPath)) {
    throw "未找到公钥文件: $publicKeyPath"
}

$pubKeyContent = (Get-Content -LiteralPath $publicKeyPath -Raw).Trim()
Write-Host "公钥指纹内容摘要: $($pubKeyContent.Substring(0, [Math]::Min(50, $pubKeyContent.Length)))..." -ForegroundColor Gray

# 4. 将公钥安全部署到远程服务器 (人类管理员输入密码)
if (-not $SkipDeploy) {
    Write-Step 2 "向远程服务器授权公钥 (~/.ssh/authorized_keys)"
    Write-Host "【重要提示】以下将调用 OpenSSH 将公钥写入远程主机的 authorized_keys。" -ForegroundColor Yellow
    Write-Host "如果是首次连接或尚未配置免密，终端将提示您输入一次远程服务器的登录密码。" -ForegroundColor Yellow
    Write-Host "此密码由 OpenSSH 原生控制台安全读取，绝不保存到任何文件，AI 模型完全无法接触！" -ForegroundColor Cyan
    Write-Host ""

    # 在 Linux 远程端执行安全的公钥追加命令
    $escapedKey = $pubKeyContent.Replace("'", "'\''")
    $remoteDeployCmd = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$escapedKey' >> ~/.ssh/authorized_keys && sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo '__DEPLOY_SUCCESS__'"

    $deployArgs = @(
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", "$finalPort",
        "$finalUser@$finalHost",
        $remoteDeployCmd
    )

    Write-Host "执行部署命令: ssh -p $finalPort $finalUser@$finalHost ..." -ForegroundColor DarkGray
    $deployOutput = & ssh.exe $deployArgs
    $deployExit = $LASTEXITCODE

    if ($deployExit -eq 0 -and ($deployOutput -join "`n") -like "*__DEPLOY_SUCCESS__*") {
        Write-Host "[OK] 远程公钥授权成功！" -ForegroundColor Green
    } else {
        Write-Host "[WARN] 自动公钥部署可能未完成 (退出码: $deployExit)。" -ForegroundColor Yellow
        Write-Host "您可以稍后手动将公钥内容追加到远程 ~/.ssh/authorized_keys 中：" -ForegroundColor Yellow
        Write-Host "公钥内容: $pubKeyContent" -ForegroundColor Cyan
    }
} else {
    Write-Host "[跳过] 已跳过远程公钥部署步骤。" -ForegroundColor DarkGray
}

# 5. 更新本地配置文件 model_ssh_config.json
Write-Step 3 "同步密钥路径至模型配置文件"
if ($configJson -and (Test-Path -LiteralPath $ConfigPath)) {
    $changed = $false
    if ($configJson.profiles.$Profile) {
        $configJson.profiles.$Profile.key_path = $privateKeyPath
        $changed = $true
    } else {
        $newProfileObj = [PSCustomObject]@{
            name = $Profile
            description = "AI 模型专用远程连接配置"
            host = $finalHost
            port = $finalPort
            user = $finalUser
            key_path = $privateKeyPath
            remote_model_dir = "/opt/neri-models"
            remote_weights_dir = "/Neri_Data/Model"
            local_weights_dir = "runs"
            forward_ports = @(8002)
            tmux_session = "neri-agent"
            service_name = ""
        }
        $configJson.profiles | Add-Member -Name $Profile -Value $newProfileObj -MemberType NoteProperty -Force
        $changed = $true
    }

    if ($changed) {
        $newJsonText = $configJson | ConvertTo-Json -Depth 10
        [System.IO.File]::WriteAllText($ConfigPath, $newJsonText, [System.Text.Encoding]::UTF8)
        Write-Host "[OK] 配置文件已更新 key_path: $privateKeyPath" -ForegroundColor Green
    }
}

# 6. 在 ~/.ssh/config 中追加专用安全别名
Write-Step 4 "写入 ~/.ssh/config 安全别名配置"
$userSshConfig = Join-Path $HOME ".ssh\config"
$aliasName = "$Profile-agent"
$aliasBlock = @"

# Added by Neri Agent SSH Setup ($Profile)
Host $aliasName
    HostName $finalHost
    Port $finalPort
    User $finalUser
    IdentityFile $privateKeyPath
    IdentitiesOnly yes
    BatchMode yes
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 3
"@

try {
    $existingContent = ""
    if (Test-Path -LiteralPath $userSshConfig) {
        $existingContent = [System.IO.File]::ReadAllText($userSshConfig, [System.Text.Encoding]::UTF8)
    }
    if ($existingContent -notmatch "Host\s+$aliasName\b") {
        [System.IO.File]::AppendAllText($userSshConfig, $aliasBlock, [System.Text.Encoding]::UTF8)
        Write-Host "[OK] 已将主机别名 '$aliasName' 写入 $userSshConfig" -ForegroundColor Green
    } else {
        Write-Host "[OK] ~/.ssh/config 中已存在 '$aliasName' 别名配置。" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARN] 无法自动写入 ~/.ssh/config: $_" -ForegroundColor Yellow
}

# 7. 强制免密连通性自检
Write-Step 5 "强制免密连通性自检 (BatchMode 验证)"
Write-Host "正在验证 AI 模型在【完全无密码】模式下是否可以正常执行命令..." -ForegroundColor Cyan

$testArgs = @(
    "-o", "BatchMode=yes",
    "-o", "PasswordAuthentication=no",
    "-o", "KbdInteractiveAuthentication=no",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=10",
    "-i", $privateKeyPath,
    "-p", "$finalPort",
    "$finalUser@$finalHost",
    "echo '__VERIFICATION_PASSED__'"
)

$testOutput = & ssh.exe $testArgs
$testExit = $LASTEXITCODE

if ($testExit -eq 0 -and ($testOutput -join "`n") -like "*__VERIFICATION_PASSED__*") {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "  [SUCCESS] 验证成功！模型免密安全连接已就绪！" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "远程服务器已信任专用密钥: $privateKeyPath" -ForegroundColor White
    Write-Host "AI 模型现在可以在完全无需密码的情况下执行运维、测试与文件传输：" -ForegroundColor White
    Write-Host ""
    Write-Host "模型调用命令示例:" -ForegroundColor Yellow
    Write-Host "  1. 连通性测试: python scripts/agent_ssh.py --action test" -ForegroundColor Cyan
    Write-Host "  2. 远程命令执行: python scripts/agent_ssh.py --action exec -c 'nvidia-smi' --json" -ForegroundColor Cyan
    Write-Host "  3. PowerShell执行: .\scripts\agent_ssh.ps1 -Action Exec -Command 'uname -a'" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Red
    Write-Host "  [FAILED] 免密自检未通过 (退出码: $testExit)" -ForegroundColor Red
    Write-Host "==========================================================" -ForegroundColor Red
    Write-Host "可能原因:" -ForegroundColor Yellow
    Write-Host "1. 公钥尚未正确写入远程 ~/.ssh/authorized_keys" -ForegroundColor Gray
    Write-Host "2. 远程 ~/.ssh 目录或 authorized_keys 文件权限不正确 (要求目录 700，文件 600)" -ForegroundColor Gray
    Write-Host "3. 远程 SSH 服务的 PubkeyAuthentication 被禁用" -ForegroundColor Gray
    Write-Host "请检查上述配置后重新运行本脚本。" -ForegroundColor Yellow
}
