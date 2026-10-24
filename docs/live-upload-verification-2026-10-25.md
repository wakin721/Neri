# 真实上传验证：2026-10-25

在用户明确授权升级线上上传服务及使用既有服务器凭据后完成。本次通过仓库中
`codex/upload-privacy-blurring` 分支的新后台代码进行测试；没有更改当前已安装客户端的
参与选择，也没有选取用户照片。测试使用独立临时训练状态目录和纯合成图片。

## 服务升级

- 公开地址：`https://myneri.top/api/training/health`。
- 升级前 `schema_version: 1`，升级后及测试结束时均为 `schema_version: 3`。
- 仅替换 `/opt/neri-training/training_broker/app.py` 并重启 `neri-training.service`。
- 升级前备份：`/var/backups/neri-training-upload-privacy-20261024T171536Z`，包含原代码、
  SQLite 一致性备份、服务单元副本和升级结果，目录权限为 0700。
- 数据库兼容迁移后，原有 19 条修订仍完整保留，`integrity_check` 为 `ok`。
- 本次未改动网站、Nginx、OpenList 配置或原有用户文件。

## 实际发现并修复的问题

第一轮普通样本成功，空照片失败。对同一空样本复现后，确认失败发生在向 OneDrive
直传 **0 字节** YOLO 标注的 PUT，请求返回 **HTTP 400**。

`system/training/media.py` 现将无目标标注编码为一个空白行（1 字节），继续使用正常
字节范围上传。空照片、无匹配检测框和因 EXIF 方向省略框的情况都采用此表示，不添加虚假目标。
[Ultralytics 的标签校验实现](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py)
会去除空白再解析。直接调用本机安装版本的 `verify_image_label`，确认结果为
`labels.shape == (0, 5)`、`empty == 1`、`corrupt == 0`。

补充回归覆盖无目标标注及上传链路不发送 0 字节 PUT。相关五个 Python 测试文件共
**69 项测试、33 个子场景通过**。本次修复仅涉及客户端生成标注的方式，无需再次替换线上服务。

## 修复后真实传输

使用新的独立测试设备 ID：`fbea36058133cd4b449ba3df47de2723`。
这只是本次已清理的测试 ID，不是用户已安装客户端的 ID。

| 样本 | 远端物种目录 | JPEG | YOLO .txt | .classes.txt |
| --- | --- | ---: | ---: | ---: |
| 普通合成图 | Neri上传自测 | 4399 B | 38 B | 26 B |
| 确认的空照片 | 空照片 | 4409 B | 1 B | 11 B |

全部文件实际保存在：

```text
/Neri_Data/Training/fbea36058133cd4b449ba3df47de2723/
├── Neri上传自测/<随机样本编号>-r1-<尝试编号>.jpg/.txt/.classes.txt
└── 空照片/<随机样本编号>-r1-<尝试编号>.jpg/.txt/.classes.txt
```

两个任务均为 `uploaded`，失败数为 0，共执行 6 次文件 PUT。随后分别再次请求已完成
修订的回执，服务均返回已完成，未重新发送文件。

经授权从远端逐个读取上述六个文件，确认实际目录、长度及 SHA-256 全部与本地待上传
字节一致。本地 JPEG 编码产物无 EXIF；合成源文件哈希未变化。本地真实隐私检测程序
已执行，合成图没有检测到隐私目标（打码区域数为 0）。这次实测验证传输及目录逻辑，
不用于证明人脸或车牌检测召回率。

## 清理与原有数据核验

- 使用各测试样本的私有所有权凭据执行删除。
- 确认测试修订记录为 0，并仅在检查为空后删除两个测试物种目录及测试设备目录。
- 远端总修订记录从测试期间的 21 恢复为原来的 **19**。
- 对照升级前数据库备份，逐条核对原有样本和修订的全部原有字段，结果完全一致；
  数据库完整性再次通过。
- 第一轮失败测试的数据及其独立设备目录也已清理。

运行证据保留在忽略目录 `temp/privacy-blurring-validation/`：
`deployment-result.txt`、`live-test-manifest.json`、`live-remote-verify.json`、
`live-remote-cleanup.json`。这些结果不包含上传会话 URL 或私有凭据；临时测试状态
SQLite 仍按私有本地状态对待，不应上传到仓库。

当前本机运行中的已安装后端仍为旧版（协议日期 2026-09-05，未提供设备 ID）；本次
没有替换该安装。要在现有桌面界面使用匿名 ID 和新上传功能，需更新客户端构建。
