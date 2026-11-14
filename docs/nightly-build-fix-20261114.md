# nightly 整合与 Windows Actions 修复

将当前 dev 的 Neri1.2preview、拒识 HDBSCAN 分组、自动 batch、LiteRT-only 打包与后续云端下载接入一起整合到 nightly。可选 DINOv3 模型继续在环境维护页下载安装，不随轻量安装包捆绑。

## 最近构建失败的证据

[dev 构建 43598449206](https://github.com/wakin721/Neri/actions/runs/43598449206) 的后端测试通过，286 项普通 Flutter 测试通过。单独的 Windows 更新器集成测试在等待更新器进程退出时触发 `TimeoutException after 0:01:00.000000`，后续编译、打包及上传都未执行。此次失败已不是先前缺失 requirements 文件、默认 30 秒测试时限或普通 Flutter 测试阶段的问题。

旧日志没有更新器自己的 update.log，因此最初只能定位到等待进程退出。加入非交互模式和日志后，[nightly 构建 43606204068](https://github.com/wakin721/Neri/actions/runs/43606204068) 确认实际错误发生在 ZIP 解压：`Cannot find an overload for "ExtractToDirectory" and the argument count: "3".`。`Expand-Archive` 选择的模块调用了当前 Windows PowerShell/.NET Framework 运行时不兼容的三参数重载。失败后旧脚本的模态 MessageBox 阻止退出，最终表现为 60 秒超时。

## 修复

- ZIP 目标目录由更新器预先清空并创建，直接调用 .NET Framework 兼容的两参数 `ZipFile.ExtractToDirectory(source, destination)`，消除解压时对 Archive 模块版本及三参数覆盖接口的依赖。
- 重启前检查可执行文件是否存在；缺失时只记录日志，避免触发系统 ShellExecute 对话框。
- 更新器支持调用方关闭失败对话框；正常软件使用维持默认提示，集成测试明确关闭对话框。
- 安装失败返回退出码 1，防止错误被当作成功退出。
- 测试失败或超时时输出实际更新日志，保留进程退出等待与目录释放检查。
- 增加损坏 ZIP 的真实进程测试，验证失败退出、旧安装内容保留及错误日志，成功安装测试继续验证无 SHA-256 时的原流程。
- 将此前被忽略的本地组件分发测试纳入 Git，并修正其向严格下载请求误传 `size` 字段的问题；服务端严格请求契约继续保留。

## 本地验证

成功安装与损坏 ZIP 两项 Windows 进程集成测试通过；287 项普通 Flutter 测试通过。完整后端初测中 581 项及 57 个子测试通过，另 1 项是上述测试请求字段错误；修正后该文件全部 11 项通过。nightly 云端构建结果在最终运行记录中确认。

nightly 推送会自动触发 `build-windows-x64.yml`，无需为同一提交重复触发。最终仍以 GitHub runner 的测试、Windows 编译、轻量打包及下载后验证全部完成为准。
