# Windows 构建失败与 LiteRT-only 资源打包

## 两次失败的区别

第一次运行 `43590138251` 在依赖安装时引用已不存在的 `requirements-within-seq.txt`，已由提交 `e1d428d6` 改为 `requirements.txt`。

随后运行 `43590382305` 已通过依赖安装和后端测试，失败发生在 Flutter 测试阶段：283 项通过，1 项失败。失败测试为 `app_updater_test.dart` 中的 `starts and completes the Windows updater when SHA-256 is absent`。Windows 报错 `PathAccessException / errno=32`，临时目录仍由另一个进程占用。

测试原来以目标文件已复制作为完成条件，随即删除测试根目录。但更新器 PowerShell 子进程还会清理文件、记录日志，且工作目录也在该根目录内，复制完成不代表进程已经退出。现将启动更新器返回值改为可等待的 `Process`；产品仍在启动就绪后继续退出，测试明确等待阻塞父进程和更新器退出后再清理。保留原有目录释放重试，完整验证真实进程的退出与文件内容，不跳过更新器测试。

运行 `43596879406` 确认清理等待已进入真实进程退出阶段，但这项 Windows 集成测试仍受测试框架默认 30 秒总时限影响。工作流现先并行运行普通单元/界面测试，再单独串行执行 `windows-integration` 标签的真实更新器测试；集成测试总时限为 2 分钟，压缩、父进程退出、更新器退出仍各有 30/10/60 秒的独立上限。所有测试继续执行，不跳过更新器测试。

## 发行包只携带 LiteRT 模型

以前 `package_windows.ps1` 递归复制整个 `res`，干净 Git 检出时只有已跟踪的 LiteRT 模型，但从含本地实验资产的开发目录打包时会混入其他模型。

现在改为调用 `scripts/package_litert_resources.py`：

- 创建规范的 `model/detect/{user,sync}`、`model/cls/{user,sync}` 空目录。
- 模型只复制 `.tflite`、`.litert`，同时复制同名的 labels、LICENSE、SOURCE 配套文件。
- 不复制 DINOv3 组件、实验模型清单、编码器、分类头、校准集、旧版模型目录和其他格式权重/压缩包。
- 保留翻译、示例图片、说明、物种数据库和 tracker 配置等普通资源。
- 输出 `res/packaged-models.json`，记录实际携带的模型路径、大小与 SHA-256。打包后与 GitHub Artifact 下载后都重新核验，发现其他模型或哈希不符时让构建失败。
- 仅向空的构建资源目录写入；不删除或改动开发目录和用户已安装的模型。

使用当前包含本地实验模型的真实 `res` 进行隔离打包，最终模型仅有 `MDV6-yolov10-c.tflite`（4,717,072 字节，SHA-256 `d66f8fc9ca1f2c1b7c7ec322d4af67964e0ecf24d5a5e2f05ae6f8b76c7d23ef`），并带有它的标签、许可证、来源说明。

PyTorch 仍用于构建机上的可选 DINOv3 单元测试，原有 `prune_lite_python.ps1` 会在发行包组装时移除它及相关运行库。为保证轻量软件可启动，编码器将 PyTorch 和预处理模块延后到实际编码时导入；模型目录、清单检查和可选环境安装接口可在没有 PyTorch 时使用。包验证会实际导入发行包中的这些模块。

## “上传”与“线上发布”的含义

`Neri1.2preview` 完整模型包已经通过 SSH 上传并核验，保存在服务器 `/home/codex/uploads/Neri1.2preview_20261114/`，本地软件也已经完成显示名更改。

软件的云端下载依赖模型分发服务及其 OpenList 目录。当前远端活动服务仍提供旧 DINOv2 组件接口，而客户端使用新的 DINOv3 清单与下载协议；SSH 账号没有免密管理员权限来替换服务。因此普通用户目前不会通过“安装/修复组件”自动获得这个上传包。将包正式发布到云端下载目录、升级配套协议并验证下载，是后续单独的服务器部署操作。

本次 LiteRT-only 发行规则与可选模型独立上传并不冲突：发行包内只放 LiteRT，其他模型可以在需要时另外安装。

## 启动时“未找到有效清单”的本地原因

当前安装目录 `res/model/DINOv3` 仍有 9 月的 `multi_prototype.neri.json`，对应 22 物种旧多原型配方：`event_aggregation=mean_l2_normalized_crop_embeddings`。新联合模型需要不同的清单契约，因此组件检查将该目录报告为不兼容。独立的 `res/model/cls/user/dinov3-experiment-local.neri.json` 已验证有效，显示名为 Neri1.2preview，权重哈希也保持原值。

启动检查现在仅在用户选用受管理组件目录中的分类模型时弹出该组件的问题；LiteRT 或独立用户模型不会因闲置旧组件而弹错。环境维护页仍如实显示旧组件需要迁移，选中的组件若缺失、损坏或缺少依赖仍会提示。旧组件原文件不删除，不将旧模型自动标成新模型。

本轮本地完整验证：后端 558 项与 57 个子测试通过，普通前端测试 287 项、Windows 更新器集成测试 1 项通过，Windows Release 构建通过。
