<#
.SYNOPSIS
    Neri 模型专用 SSH 远程连接与训练管理脚本 (PowerShell 版)
.DESCRIPTION
    专为深度学习模型训练、微调与模型服务运维设计的 SSH 管理工具。
    支持模型服务端口映射 (TensorBoard / Jupyter / WebUI / FastAPI / Neri Service)、
    训练防掉线 (Tmux / ServerAliveKeepAlive)、GPU 与 PyTorch 环境检测、
    模型权重 (Weights) 双向同步，以及 VS Code Remote-SSH 配置导出。
.EXAMPLE
    .\scripts\connect_model_ssh.ps1
    # 打开交互式菜单
.EXAMPLE
    .\scripts\connect_model_ssh.ps1 -Profile neri-cloud -Action Connect
    # 直接连接 Neri 云端模型分发服务器并自动映射 8002 端口
.EXAMPLE
    .\scripts\connect_model_ssh.ps1 -Profile gpu-training -ForwardPorts 6006,7860,8888 -AutoTmux
    # 连接 GPU 训练节点，映射 TensorBoard(6006)、WebUI(7860)、Jupyter(8888) 并进入 tmux
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Profile = "",

    [Parameter(Position = 1)]
    [ValidateSet("Menu", "Connect", "GpuStatus", "EnvCheck", "Tmux", "DownloadModel", "UploadModel", "NeriService", "VsCodeConfig", "ListProfiles", "")]
    [string]$Action = "",

    [int[]]$ForwardPorts = @(),

    [switch]$NoForward,

    [string]$RemotePath = "",

    [string]$LocalPath = "",

    [string]$TmuxSession = "",

    [switch]$AutoTmux,

    [string]$HostOverride = "",

    [int]$PortOverride = 0,

    [string]$UserOverride = "",

    [string]$KeyPath = "",

    [string]$Command = "",

    [switch]$Yes,

    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    if ($PSScriptRoot) {
        $ConfigPath = Join-Path $PSScriptRoot "model_ssh_config.json"
    } else {
        $ConfigPath = Join-Path (Get-Location) "scripts\model_ssh_config.json"
        if (-not (Test-Path $ConfigPath)) {
            $ConfigPath = "model_ssh_config.json"
        }
    }
}
$OutputEncoding = [System.Text.Encoding]::UTF8
if ($Host.UI.RawUI) {
    try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Yellow
    Write-Host "==========================================================" -ForegroundColor Cyan
}

function Load-ModelSshConfig {
    if (-not (Test-Path -LiteralPath $ConfigPath)) {
        throw "配置文件未找到: $ConfigPath"
    }
    $rawJson = [System.IO.File]::ReadAllText($ConfigPath, [System.Text.Encoding]::UTF8)
    return ($rawJson | ConvertFrom-Json)
}

function Save-ModelSshConfig {
    param($ConfigObject)
    $jsonText = $ConfigObject | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($ConfigPath, $jsonText, [System.Text.Encoding]::UTF8)
    Write-Host "配置已保存至: $ConfigPath" -ForegroundColor Green
}

function Resolve-TargetProfile {
    param($Config, [string]$RequestedName)
    $name = if ([string]::IsNullOrWhiteSpace($RequestedName)) {
        if ($Config.default_profile) { $Config.default_profile } else { "neri-cloud" }
    } else {
        $RequestedName
    }

    $profileObj = $Config.profiles.$name
    if ($null -eq $profileObj) {
        throw "未找到名为 '$name' 的配置项。可用配置: $($Config.profiles.PSObject.Properties.Name -join ', ')"
    }

    $active = [PSCustomObject]@{
        name = $name
        description = $profileObj.description
        host = if ($HostOverride) { $HostOverride } else { $profileObj.host }
        port = if ($PortOverride -gt 0) { $PortOverride } elseif ($profileObj.port) { $profileObj.port } else { 22 }
        user = if ($UserOverride) { $UserOverride } else { $profileObj.user }
        key_path = if ($KeyPath) { $KeyPath } else { $profileObj.key_path }
        remote_model_dir = $profileObj.remote_model_dir
        remote_weights_dir = $profileObj.remote_weights_dir
        local_weights_dir = if ($profileObj.local_weights_dir) { $profileObj.local_weights_dir } else { "runs" }
        forward_ports = if ($NoForward) { @() } elseif ($ForwardPorts.Count -gt 0) { $ForwardPorts } elseif ($profileObj.forward_ports) { @($profileObj.forward_ports) } else { @() }
        tmux_session = if ($TmuxSession) { $TmuxSession } elseif ($profileObj.tmux_session) { $profileObj.tmux_session } else { "model-session" }
        service_name = $profileObj.service_name
    }
    return $active
}

function Build-SshArgs {
    param($P, [string[]]$ExtraCommands = @())
    $argsList = [System.Collections.Generic.List[string]]::new()

    # 保持会话活跃，防止模型训练断开
    $argsList.Add("-o")
    $argsList.Add("ServerAliveInterval=60")
    $argsList.Add("-o")
    $argsList.Add("ServerAliveCountMax=3")
    $argsList.Add("-o")
    $argsList.Add("TCPKeepAlive=yes")
    $argsList.Add("-o")
    $argsList.Add("StrictHostKeyChecking=accept-new")

    # 端口指定
    if ($P.port -and $P.port -ne 22) {
        $argsList.Add("-p")
        $argsList.Add("$($P.port)")
    }

    # 私钥指定
    if ($P.key_path -and (Test-Path -LiteralPath $P.key_path)) {
        $argsList.Add("-i")
        $argsList.Add($P.key_path)
    }

    # 端口映射隧道 (LocalForward: 本地端口 -> 远程回环端口)
    if (-not $NoForward -and $P.forward_ports) {
        foreach ($port in $P.forward_ports) {
            $argsList.Add("-L")
            $argsList.Add("${port}:127.0.0.1:${port}")
        }
    }

    # 登录目标 user@host
    $target = if ($P.user) { "$($P.user)@$($P.host)" } else { $P.host }
    $argsList.Add($target)

    if ($ExtraCommands.Count -gt 0) {
        foreach ($cmd in $ExtraCommands) {
            $argsList.Add($cmd)
        }
    }

    return $argsList.ToArray()
}

function Invoke-SshInteractive {
    param($P, [switch]$WithTmux)
    Write-Header "正在连接服务器: $($P.name) ($($P.user)@$($P.host):$($P.port))"

    if ($P.forward_ports.Count -gt 0) {
        Write-Host "已开启端口映射 (本地 -> 远程):" -ForegroundColor Cyan
        foreach ($p in $P.forward_ports) {
            Write-Host "  * http://localhost:$p" -ForegroundColor DarkCyan
        }
    } else {
        Write-Host "未启用端口映射" -ForegroundColor DarkGray
    }

    $sshExe = "ssh.exe"
    if (-not (Get-Command $sshExe -ErrorAction SilentlyContinue)) {
        throw "系统中未找到 OpenSSH 客户端 (ssh.exe)。请确保 Windows OpenSSH 已安装并在 PATH 中。"
    }

    if ($WithTmux -or $AutoTmux) {
        Write-Host "将自动附加/创建 Tmux 训练会话: '$($P.tmux_session)'" -ForegroundColor Green
        $tmuxCmd = "tmux new-session -A -s $($P.tmux_session)"
        $args = Build-SshArgs -P $P -ExtraCommands @("-t", $tmuxCmd)
    } else {
        $args = Build-SshArgs -P $P
    }

    Write-Host "执行命令: ssh $($args -join ' ')" -ForegroundColor DarkGray
    & $sshExe $args
}

function Show-GpuStatus {
    param($P)
    Write-Header "正在查询远程 GPU 状态: $($P.name)"
    $remoteCmd = "if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo '[提示] 未检测到 nvidia-smi 命令，该服务器可能无 NVIDIA GPU 或未安装显卡驱动。'; fi"
    $args = Build-SshArgs -P $P -ExtraCommands @("-t", $remoteCmd)
    & ssh.exe $args
}

function Show-PyTorchEnv {
    param($P)
    Write-Header "正在查询远程 Python 与 PyTorch/CUDA 深度学习环境: $($P.name)"
    $pyCmd = @'
python3 -c "
import sys
print('=== Python 运行时 ===')
print('Python 版本:', sys.version.split()[0])
try:
    import torch
    print('\n=== PyTorch 深度学习框架 ===')
    print('PyTorch 版本   :', torch.__version__)
    print('CUDA 是否可用  :', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA 编译版本  :', torch.version.cuda)
        print('cuDNN 版本     :', torch.backends.cudnn.version())
        print('GPU 数量       :', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = round(props.total_memory / (1024**3), 2)
            print(f'  [GPU {i}] {props.name} (显存: {vram_gb} GB)')
    else:
        print('当前仅支持 CPU 推理')
except ImportError:
    print('\n[提示] 未安装 torch 模块')
try:
    import ultralytics
    print('\n=== YOLO 框架 ===')
    print('Ultralytics 版本:', ultralytics.__version__)
except ImportError:
    pass
"
'@
    $args = Build-SshArgs -P $P -ExtraCommands @("-t", $pyCmd)
    & ssh.exe $args
}

function Invoke-ModelWeightsDownload {
    param($P, [string]$SourcePath, [string]$DestPath)
    Write-Header "从远程拉取模型权重/文件 (Download)"

    $remoteTarget = if ($SourcePath) { $SourcePath } else {
        $defaultDir = if ($P.remote_weights_dir) { $P.remote_weights_dir } else { "/root" }
        Read-Host "请输入远程文件/目录路径 (默认: $defaultDir)"
    }
    if ([string]::IsNullOrWhiteSpace($remoteTarget)) {
        $remoteTarget = if ($P.remote_weights_dir) { $P.remote_weights_dir } else { "/root" }
    }

    $localTarget = if ($DestPath) { $DestPath } else {
        $defaultLocal = if ($P.local_weights_dir) { $P.local_weights_dir } else { "runs" }
        Read-Host "请输入本地保存目录 (默认: $defaultLocal)"
    }
    if ([string]::IsNullOrWhiteSpace($localTarget)) {
        $localTarget = if ($P.local_weights_dir) { $P.local_weights_dir } else { "runs" }
    }

    if (-not (Test-Path -LiteralPath $localTarget)) {
        New-Item -ItemType Directory -Path $localTarget -Force | Out-Null
    }

    $scpArgs = [System.Collections.Generic.List[string]]::new()
    $scpArgs.Add("-r")
    $scpArgs.Add("-C") # 启用压缩加速传输
    if ($P.port -and $P.port -ne 22) {
        $scpArgs.Add("-P")
        $scpArgs.Add("$($P.port)")
    }
    if ($P.key_path -and (Test-Path -LiteralPath $P.key_path)) {
        $scpArgs.Add("-i")
        $scpArgs.Add($P.key_path)
    }

    $remoteSpec = "$($P.user)@$($P.host):$remoteTarget"
    $scpArgs.Add($remoteSpec)
    $scpArgs.Add($localTarget)

    Write-Host "执行下载: scp $($scpArgs -join ' ')" -ForegroundColor Cyan
    & scp.exe $scpArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "模型文件成功下载到: $localTarget" -ForegroundColor Green
    } else {
        Write-Host "下载失败，退出代码: $LASTEXITCODE" -ForegroundColor Red
    }
}

function Invoke-ModelWeightsUpload {
    param($P, [string]$SourcePath, [string]$DestPath)
    Write-Header "向远程上传模型权重/数据集/脚本 (Upload)"

    $localTarget = if ($SourcePath) { $SourcePath } else {
        Read-Host "请输入要上传的本地文件或目录路径"
    }
    if ([string]::IsNullOrWhiteSpace($localTarget) -or (-not (Test-Path -LiteralPath $localTarget))) {
        Write-Host "本地路径不存在: $localTarget" -ForegroundColor Red
        return
    }

    $remoteTarget = if ($DestPath) { $DestPath } else {
        $defaultDir = if ($P.remote_weights_dir) { $P.remote_weights_dir } else { "/root" }
        Read-Host "请输入远程目标目录 (默认: $defaultDir)"
    }
    if ([string]::IsNullOrWhiteSpace($remoteTarget)) {
        $remoteTarget = if ($P.remote_weights_dir) { $P.remote_weights_dir } else { "/root" }
    }

    $scpArgs = [System.Collections.Generic.List[string]]::new()
    $scpArgs.Add("-r")
    $scpArgs.Add("-C")
    if ($P.port -and $P.port -ne 22) {
        $scpArgs.Add("-P")
        $scpArgs.Add("$($P.port)")
    }
    if ($P.key_path -and (Test-Path -LiteralPath $P.key_path)) {
        $scpArgs.Add("-i")
        $scpArgs.Add($P.key_path)
    }

    $scpArgs.Add($localTarget)
    $remoteSpec = "$($P.user)@$($P.host):$remoteTarget"
    $scpArgs.Add($remoteSpec)

    Write-Host "执行上传: scp $($scpArgs -join ' ')" -ForegroundColor Cyan
    & scp.exe $scpArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "文件成功上传至远程: $remoteTarget" -ForegroundColor Green
    } else {
        Write-Host "上传失败，退出代码: $LASTEXITCODE" -ForegroundColor Red
    }
}

function Manage-NeriModelService {
    param($P)
    Write-Header "Neri 云端模型分发服务管理 ($($P.name))"
    $serviceName = if ($P.service_name) { $P.service_name } else { "neri-models" }

    Write-Host "请选择服务操作:" -ForegroundColor Yellow
    Write-Host "  [1] 查看服务运行状态 (systemctl status)"
    Write-Host "  [2] 重启服务 (sudo systemctl restart)"
    Write-Host "  [3] 实时查看最近日志 (journalctl -u $serviceName -f)"
    Write-Host "  [4] 测试公共健康接口 (curl https://myneri.top/api/models/health)"
    Write-Host "  [0] 返回上级菜单"
    Write-Host ""
    $choice = Read-Host "请输入选项 (0-4)"

    switch ($choice) {
        "1" {
            $cmd = "systemctl status $serviceName"
            $args = Build-SshArgs -P $P -ExtraCommands @("-t", $cmd)
            & ssh.exe $args
        }
        "2" {
            $cmd = "sudo systemctl restart $serviceName && systemctl status $serviceName"
            $args = Build-SshArgs -P $P -ExtraCommands @("-t", $cmd)
            & ssh.exe $args
        }
        "3" {
            $cmd = "journalctl -u $serviceName -n 50 -f"
            $args = Build-SshArgs -P $P -ExtraCommands @("-t", $cmd)
            & ssh.exe $args
        }
        "4" {
            Write-Host "请求 https://myneri.top/api/models/health ..." -ForegroundColor Cyan
            try {
                $resp = Invoke-RestMethod -Uri "https://myneri.top/api/models/health" -Method Get -TimeoutSec 10
                $resp | ConvertTo-Json | Write-Host -ForegroundColor Green
            } catch {
                Write-Host "接口请求失败: $_" -ForegroundColor Red
            }
        }
        default { return }
    }
}

function Export-VsCodeConfig {
    param($P)
    Write-Header "导出 VS Code Remote-SSH 配置"
    $sshUserDir = Join-Path $HOME ".ssh"
    $sshConfigFile = Join-Path $sshUserDir "config"

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("Host $($P.name)")
    $lines.Add("    HostName $($P.host)")
    $lines.Add("    User $($P.user)")
    if ($P.port -and $P.port -ne 22) {
        $lines.Add("    Port $($P.port)")
    }
    if ($P.key_path) {
        $lines.Add("    IdentityFile $($P.key_path)")
    }
    $lines.Add("    ServerAliveInterval 60")
    $lines.Add("    ServerAliveCountMax 3")
    if ($P.forward_ports) {
        foreach ($port in $P.forward_ports) {
            $lines.Add("    LocalForward ${port} 127.0.0.1:${port}")
        }
    }
    $block = $lines -join "`n"

    Write-Host "以下为适用于 ~/.ssh/config 的配置片段:`n" -ForegroundColor Yellow
    Write-Host $block -ForegroundColor Cyan
    Write-Host ""

    $shouldSave = $Yes
    if (-not $shouldSave) {
        $saveInput = Read-Host "是否自动追加到用户 SSH 配置文件 ($sshConfigFile)? [y/N]"
        $shouldSave = ($saveInput -match '^[yY]')
    }
    if ($shouldSave) {
        try {
            if (-not (Test-Path -LiteralPath $sshUserDir)) {
                New-Item -ItemType Directory -Path $sshUserDir -Force | Out-Null
            }
            $blockToAppend = "`n# Added by Neri Model SSH on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n$block`n"
            [System.IO.File]::AppendAllText($sshConfigFile, $blockToAppend, [System.Text.Encoding]::UTF8)
            Write-Host "已成功写入 $sshConfigFile !" -ForegroundColor Green
            Write-Host "现在可在 VS Code 中按 F1 -> Remote-SSH: Connect to Host -> 选择 '$($P.name)' 即可直连！" -ForegroundColor Green
        } catch {
            Write-Host "[提示] 无法自动写入 $sshConfigFile ($_)" -ForegroundColor Yellow
            Write-Host "请手动复制上方配置片段粘贴至该文件中。" -ForegroundColor Yellow
        }
    }
}

function Show-InteractiveMenu {
    param($Config)
    $currentProfileName = if ($Profile) { $Profile } else { $Config.default_profile }

    while ($true) {
        $active = Resolve-TargetProfile -Config $Config -RequestedName $currentProfileName
        Write-Header "Neri 深度学习模型 SSH 远程连接与训练管理"
        Write-Host "当前服务器配置: " -NoNewline
        Write-Host "[$($active.name)]" -ForegroundColor Green -NoNewline
        Write-Host " ($($active.description))" -ForegroundColor Gray
        Write-Host "连接信息      : " -NoNewline
        Write-Host "$($active.user)@$($active.host):$($active.port)" -ForegroundColor Cyan
        if ($active.forward_ports) {
            Write-Host "端口映射隧道  : " -NoNewline
            Write-Host "$($active.forward_ports -join ', ') (可在本地浏览器直连访问)" -ForegroundColor Magenta
        }
        Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray
        Write-Host " [1] 连接交互式终端 (带端口映射与会话保护)" -ForegroundColor White
        Write-Host " [2] 连接并附加 Tmux 训练会话 (训练防掉线)" -ForegroundColor White
        Write-Host " [3] 查看远程 GPU 状态 (nvidia-smi)" -ForegroundColor White
        Write-Host " [4] 检测远程 PyTorch / CUDA 深度学习环境" -ForegroundColor White
        Write-Host " [5] 从远程下载模型权重 (Pull Checkpoints)" -ForegroundColor White
        Write-Host " [6] 向远程上传模型权重/数据 (Push Files)" -ForegroundColor White
        if ($active.name -eq "neri-cloud" -or $active.service_name) {
            Write-Host " [7] 管理 Neri 模型分发服务 (Status/Restart/Logs)" -ForegroundColor White
        } else {
            Write-Host " [7] 管理远程模型服务 (自定义 Service)" -ForegroundColor DarkGray
        }
        Write-Host " [8] 导出 VS Code Remote-SSH 连接配置" -ForegroundColor White
        Write-Host " [9] 切换或编辑服务器配置" -ForegroundColor Yellow
        Write-Host " [0] 退出" -ForegroundColor Gray
        Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray
        $sel = Read-Host "请输入操作编号 (0-9)"

        switch ($sel) {
            "1" { Invoke-SshInteractive -P $active }
            "2" { Invoke-SshInteractive -P $active -WithTmux }
            "3" { Show-GpuStatus -P $active }
            "4" { Show-PyTorchEnv -P $active }
            "5" { Invoke-ModelWeightsDownload -P $active }
            "6" { Invoke-ModelWeightsUpload -P $active }
            "7" { Manage-NeriModelService -P $active }
            "8" { Export-VsCodeConfig -P $active }
            "9" {
                Write-Host "`n可用服务器配置项:" -ForegroundColor Yellow
                $allNames = @($Config.profiles.PSObject.Properties.Name)
                for ($i = 0; $i -lt $allNames.Count; $i++) {
                    $pName = $allNames[$i]
                    $pObj = $Config.profiles.$pName
                    Write-Host "  [$($i+1)] $pName - $($pObj.description) ($($pObj.user)@$($pObj.host))"
                }
                Write-Host "  [A] 添加新配置 (例如 AutoDL / 个人算力机)"
                $pChoice = Read-Host "请输入编号选择或输入 A 添加"
                if ($pChoice -match '^[aA]$') {
                    $newName = Read-Host "请输入新配置唯一名称 (如 autodl-4090)"
                    if (-not [string]::IsNullOrWhiteSpace($newName)) {
                        $newHost = Read-Host "请输入主机名或 IP"
                        $newPort = Read-Host "请输入 SSH 端口 (默认 22)"
                        $newPort = if ($newPort) { [int]$newPort } else { 22 }
                        $newUser = Read-Host "请输入 SSH 用户名 (如 root)"
                        $newDesc = Read-Host "请输入描述信息"
                        $newPorts = Read-Host "请输入要映射的端口 (逗号分隔，如 6006,7860,8888)"
                        $parsedPorts = if ($newPorts) { @($newPorts -split ',' | ForEach-Object { [int]$_.Trim() }) } else { @(6006, 7860, 8888) }

                        $newProfileObj = [PSCustomObject]@{
                            name = $newName
                            description = $newDesc
                            host = $newHost
                            port = $newPort
                            user = $newUser
                            key_path = ""
                            remote_model_dir = "/root"
                            remote_weights_dir = "/root"
                            local_weights_dir = "runs"
                            forward_ports = $parsedPorts
                            tmux_session = "model-train"
                            service_name = ""
                        }
                        $Config.profiles | Add-Member -MemberType NoteProperty -Name $newName -Value $newProfileObj -Force
                        Save-ModelSshConfig -ConfigObject $Config
                        $currentProfileName = $newName
                    }
                } elseif ($pChoice -match '^\d+$' -and [int]$pChoice -ge 1 -and [int]$pChoice -le $allNames.Count) {
                    $currentProfileName = $allNames[[int]$pChoice - 1]
                }
            }
            "0" { break }
            default { Write-Host "无效选项，请重新输入" -ForegroundColor Red }
        }
        Write-Host "`n按任意键继续..." -ForegroundColor DarkGray
        try { [Console]::ReadKey($true) | Out-Null } catch {}
    }
}

# 主入口逻辑
$config = Load-ModelSshConfig

if ([string]::IsNullOrWhiteSpace($Action) -and [string]::IsNullOrWhiteSpace($Profile) -and [string]::IsNullOrWhiteSpace($Command)) {
    Show-InteractiveMenu -Config $config
    exit 0
}

$targetProfile = Resolve-TargetProfile -Config $config -RequestedName $Profile

if ($Command) {
    $args = Build-SshArgs -P $targetProfile -ExtraCommands @("-t", $Command)
    & ssh.exe $args
    exit $LASTEXITCODE
}

switch ($Action) {
    "Connect" {
        Invoke-SshInteractive -P $targetProfile -WithTmux:$AutoTmux
    }
    "GpuStatus" {
        Show-GpuStatus -P $targetProfile
    }
    "EnvCheck" {
        Show-PyTorchEnv -P $targetProfile
    }
    "Tmux" {
        Invoke-SshInteractive -P $targetProfile -WithTmux
    }
    "DownloadModel" {
        Invoke-ModelWeightsDownload -P $targetProfile -SourcePath $RemotePath -DestPath $LocalPath
    }
    "UploadModel" {
        Invoke-ModelWeightsUpload -P $targetProfile -SourcePath $LocalPath -DestPath $RemotePath
    }
    "NeriService" {
        Manage-NeriModelService -P $targetProfile
    }
    "VsCodeConfig" {
        Export-VsCodeConfig -P $targetProfile
    }
    "ListProfiles" {
        Write-Header "已配置的服务器配置列表"
        $config.profiles.PSObject.Properties | ForEach-Object {
            $p = $_.Value
            Write-Host "  * $($p.name): $($p.description)" -ForegroundColor Cyan
            Write-Host "    目标: $($p.user)@$($p.host):$($p.port)" -ForegroundColor DarkCyan
            Write-Host "    映射端口: $($p.forward_ports -join ', ')" -ForegroundColor DarkGray
        }
    }
    default {
        Show-InteractiveMenu -Config $config
    }
}
