# Neri1.2preview CPU / GPU 云端下载

环境维护的 Neri1.2preview 卡片提供“自动检测”“CPU（LiteRT FP32）”“GPU（PyTorch）”下载选择，默认自动检测。开始页和检测设置不再显示运行方式切换器。CPU 组件通过 NeriCloud 下载，安装时不要求 PyTorch；GPU 组件沿用原 PyTorch 版本。同源模型继续使用共享学习档案，补图、新物种及校验记录保留。

发行包保留 MDV6-yolov10-c·LiteRT 检测器。Neri1.2preview 权重、校准集、分类头和历史分片均不进入新的 GitHub 提交和便携包。本地已有的模型和开发分片保留在磁盘上并被 Git 忽略。

## 分发协议

- GPU：`GET /api/models/v1/dinov3/manifest`，保留原 `DINOv3/**` 逻辑路径。
- CPU：`GET /api/models/v1/dinov3/manifest?runtime=cpu`，使用独立 `DINOv3/cpu/**` 清单；安装时剥离 CPU 路径前缀。
- 两种版本沿用 `/dinov3/direct` 和 `/proxy/{token}`，分别校验清单身份、文件大小及 SHA-256，复用分段续传。
- 服务环境变量 `NERI_DINOV3_ROOT` 指向 GPU 发布目录，`NERI_DINOV3_CPU_ROOT` 指向 CPU 发布目录；CPU 清单不会替换 GPU 清单。

## 云端发布

2026-11-16 07:40:22（Asia/Taipei）管理员已激活正式服务。桌面实际下载器通过正式服务回环 HTTP 完整下载 6 个文件（581,089,419 字节），大小和 SHA-256 全部一致，分类头含 45 个物种。公网清单、3 个小文件及编码器、分类头、校准集的头尾范围均验证通过；未执行公网全量 581 MB 下载。

CPU 清单身份：`9c7a0c716823eb692c0c1ae062e1ff2f8a76eb99e9bd8d7fdf68c87a2a8f9e74`。GPU 清单仍为 `f7b389bd3381215a0c1d23fa9530c3e85752525a8c4fdb4f80d86664f3d256f8`，204 个文件，582,356,135 字节。

当前 SSH 账号不能免密 sudo，因此由管理员执行以下发布命令。脚本核对模型和服务包，保留原 GPU 模型，切换服务代码并检查两个清单；失败会回滚原配置：

```bash
sudo python3 /home/codex/uploads/neri-preview-litert-cloud-20261116/publish_dinov3_litert_cloud.py --apply --root /home/codex/uploads/neri-preview-litert-cloud-20261116/model --bundle /home/codex/uploads/neri-preview-litert-cloud-20261116/service-bundle.zip
```

已上传服务包 SHA-256：`0c74d6c5d07ce77f39d8a2b2d8e05dca2da88c771796bc2267adc70e8dc0ddac`。部署结果写入 `/var/backups/neri-litert-时间/deployment.json`。手动回滚使用同一脚本的 `--rollback /var/backups/neri-litert-实际时间`。

完整 CPU 下载验证命令：

```powershell
& ./toolkit/python.exe scripts/verify_dinov3_cloud.py --runtime cpu --root runs/cloud-cpu-download --report runs/cloud-cpu-download.json
```
