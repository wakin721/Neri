# Neri1.2preview 模型与 dev 构建

2026-11-14。将当前已核验的本地实验模型显示名改为 `Neri1.2preview`，并将实验配方接入、拒识 HDBSCAN 聚类、自动分组组合簇显示和按主内存/提交额度调整 batch 的更改汇入 `dev`。

更名保留 `dinov3-experiment-local.neri.json` 原路径，因此原来的模型选择和人工反馈身份继续有效。权重、校准集、冻结编码器和拒识门限没有因更名改变。离线联合校准与模拟未知物种实验脚本随代码保存，相关候选门限没有自动启用。

## 模型包

`scripts/package_dinov3_preview.py` 打包现有部署，并在写出之前校验每个资产文件。压缩包包含 `Neri1.2preview/Neri1.2preview.neri.json`、分类检查点、编码器权重、官方源码和许可证、校准集以及逐文件 SHA-256 清单。不得仅分发分类头而遗漏编码器及校准依赖。

- 模型配方：`experiment_full`，45 个已知物种。
- 模型检查点 SHA-256：`52f5b703b2b552fcdac6c8289bfac927ebdd548c95f8ca13bd777177665219ed`。
- 编码器 SHA-256：`73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`。
- 校准集 SHA-256：`65225e2862fc1f953605e84890ce5b2db9fa1b648b57802756763b2f8c9e62ab`。
- `Neri1.2preview.zip`：557,891,447 字节。
- 压缩包 SHA-256：`f7068b3a84ea32a42d3d89322eed79e750696086b8158fb028c46878bd95dbe1`。

复现打包：

```powershell
python scripts/package_dinov3_preview.py --manifest res/model/cls/user/dinov3-experiment-local.neri.json --output runs/Neri1.2preview.zip
```

现有模型和资产应先由本地导出脚本准备，权重不加入 Git。原始照片不在包内，校准特征和用于增量更新的参考特征随部署保存。

## 远端上传范围

使用仓库 `scripts/agent_ssh.py` 强制免密连接。服务器账号目录中的上传位置为：

`/home/codex/uploads/Neri1.2preview_20261114/Neri1.2preview.zip`

上传先采用 `.uploading` 临时名，远端核对完整 ZIP 和 203 个资产条目的大小、哈希后再改正式名称。相同目录保存 `.sha256`、包报告和远端校验结果。

当前服务器分发服务的活动代码仍提供旧 `/v1/dinov2/*` 接口，SSH 账号不能免密执行管理员命令。上传完成不代表已经替换线上默认组件或接入 `/v1/dinov3/*` 下载协议；后续服务迁移需要管理员部署配套的新协议与模型目录。

## 构建与验证

仓库 `.github/workflows/build-windows-x64.yml` 监听 `dev` 推送，合并后正常推送即可自动触发 Windows x64 构建，不重复分发同一提交。应用版本仍使用原来的 3.1.1-alpha1；`Neri1.2preview` 是分类模型名称。

提交前执行完整后端与前端测试。后端 `pytest.ini` 明确只收集 `tests/`，防止本地缓存和运行记录中的旧测试被误收集。包的可移植性、资产损坏拒绝和已有文件保护也有测试。
