# Neri1.2preview 云端下载安装

## 当前状态

管理员已于 2026-11-14 执行发布命令。正式 `neri-models` 服务在 21:59:29 启动成功，工作目录已切换到 `/opt/neri-models/releases/preview-5d5debab54d7ac77-neri-preview-20261114-215917`。公网健康检查及 DINOv3 清单均返回 HTTP 200，新旧组件协议同时存在；客户端接入也已部署到 `D:\Application\Neri`。

用户入口：设置 → 环境维护 → Neri1.2preview → 安装/修复。成功安装后，分类模型列表通过 NeriCloud 发现 `Neri1.2preview`，按需选择使用。LiteRT-only 发行包继续只携带 LiteRT 模型，其他模型按需下载。

## 分发与完整性

沿用 `https://myneri.top/api/models/v1/dinov3/manifest`、`POST /dinov3/direct` 和 `/proxy/{token}`。通过 `NERI_DINOV3_ROOT` 将 DINOv3 指向独立的服务器发布目录，普通 OpenList 模型目录和旧 `/dinov2/*` 接口继续使用各自的原始来源。服务器本地路径不出现在公开下载能力响应中。

模型包含 45 个已知物种，配方为 `experiment_full`。完整下载包含 204 个文件，共 582,356,135 字节（203 个模型资产条目及 `package_inventory.json`）。编码器、源码、许可证、分类头及校准集都保留；没有改动模型参数或校准门限。

文件均逐一验证字节数和 SHA-256，下载失败不替换已有组件；组件健康检查额外检查校准集 SHA-256；安装目录切换失败会恢复旧组件。部署将请求额度设为每分钟 1200 次，以容纳完整源码树的数百次能力请求和代理请求，既有代理流量额度保持原配置。

公网整文件测试出现 Python/OpenSSL 的 `DECRYPTION_FAILED_OR_BAD_RECORD_MAC`、`RECORD_LAYER_FAILURE` 及连接重置；仅小范围成功不足以排除长传输问题。因此客户端对代理中的大文件改用 1 MiB 分段、失败最多尝试 5 次并逐步等待。能力令牌按使用次数续领，严格要求 HTTP 206 与匹配的 `Content-Range`，最终仍检查完整大小与 SHA-256。Windows 上 Python TLS 发生传输错误后，可转用系统 `curl.exe`；忽略用户 curl 配置、保留证书校验，不启用 `--insecure`。平台缺少该工具时继续有限重试。

本机修复已部署。公网实际连续下载曾达到约 30 MB，之后仍遇到传输中断；不能据此宣称已经通过公网整包验收。观察到的外网速度约 1 Mbps，完整组件约 582 MB。正式服务的完整 204 文件另经服务器回环 HTTP 下载全部通过哈希验证；这一结果与公网传输验证分别记录。

## 管理员发布

已上传到服务器：

```text
/home/codex/uploads/neri-preview-cloud-20261114/publish_dinov3_cloud.py
/home/codex/uploads/neri-preview-cloud-20261114/service-bundle.zip
```

服务包 SHA-256：`5d5debab54d7ac777c019f25aac655cab4428119069a6115873f467e3ca6c407`。

以下管理员发布命令已执行，留作复现：

```bash
sudo python3 /home/codex/uploads/neri-preview-cloud-20261114/publish_dinov3_cloud.py \
  --apply \
  --model /home/codex/uploads/Neri1.2preview_20261114/Neri1.2preview.zip \
  --bundle /home/codex/uploads/neri-preview-cloud-20261114/service-bundle.zip
```

该脚本再次验证原模型包和候选源码清单，在 `/opt/neri-models/releases/` 与 `/var/lib/neri-models/releases/` 创建独立版本目录，以 root 拥有、服务组只读的权限发布。通过 systemd `50-preview.conf` 切换工作目录，并加载专用预览环境文件。原 `/opt/neri-models/server/model_distribution`、原私密环境文件、OpenList、nginx 以及服务数据库不替换。

重启后，脚本验证本机清单中全部资产的大小和哈希，并下载公开模型清单文件核验；异常自动恢复原 drop-in 与预览环境文件，再重启原服务。部署结果写入 `/var/backups/neri-preview-时间/deployment.json`。需要手动回滚时：

```bash
sudo python3 /home/codex/uploads/neri-preview-cloud-20261114/publish_dinov3_cloud.py \
  --rollback /var/backups/neri-preview-实际时间
```

外网整包验证命令如下；请用软件嵌入式 Python 执行。报告只在完整下载及全部检查通过后生成：

```powershell
& 'D:/Application/Neri/toolkit/python.exe' scripts/verify_dinov3_cloud.py --root runs/cloud-preview-public-download --report runs/cloud-preview-public.json
```

## 已完成验证

- 相关后端 77 项测试、18 个子测试通过；覆盖完整下载、云端模型发现、旧协议隔离、范围请求、损坏下载保护、校准集损坏与安装切换回滚。
- 前端组件 API、状态加载与启动提示的 12 项测试通过，Windows Release 构建成功。
- 服务器隔离 HTTP 测试使用客户端实际的 `DinoV3DistributionClient` 下载全部文件，204 个文件大小和哈希一致。报告保存在 `runs/cloud_preview_20261114/preflight.json`。
- 本机更新了 9 个文件，保留模型选择和设置；备份在 `D:\Application\Neri\repair-backups\rules-update-20261114-215058-042439`。
- 正式服务完整下载 204 个文件通过，报告：`runs/cloud_preview_20261114/active-service-verification.json`。该验收通过服务器回环 HTTP，未计作外网整包下载。
- 下载器新增 TLS 错误、短流重试、原生传输切换、证书校验保留及范围响应校验的回归覆盖。
- 最终下载器及相关契约测试：44 项、13 个子测试通过。公网三个大文件的头尾范围均与原件一致，报告：`runs/cloud_preview_20261114/public-ranges-verification.json`；该报告明确标记 `full_public_download=false`。
- 下载传输修复备份：`D:\Application\Neri\repair-backups\cloud-transport-20261114-222358`。
- 最终含 5 次重试及等待的客户端版本已部署；备份：`D:\Application\Neri\repair-backups\rules-update-20261114-222856-158317`。安装文件 SHA-256：`c7b28fdc66dd0bf2706579ecb933a4e5dae69768e0803c17e5cecf64489dae58`，与工作区一致。

服务已发布，无需再执行管理员切换。公网整包下载与真实新安装的外网验收仍有上述网络限制，尚未将其标记为通过。
