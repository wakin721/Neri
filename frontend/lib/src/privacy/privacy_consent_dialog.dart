import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'privacy_status.dart';

const privacyConsentDialogKey = Key('privacy-consent-dialog');
const privacyParticipationGroupKey = Key('privacy-participation-group');
const privacyParticipateOptionKey = Key('privacy-participate-option');
const privacyDeclineOptionKey = Key('privacy-decline-option');
const privacyAgreementCheckboxKey = Key('privacy-agreement-checkbox');
const privacySaveButtonKey = Key('privacy-save-button');
const privacyConsentCancelButtonKey = Key('privacy-consent-cancel-button');
const privacyAgreementMenuKey = Key('privacy-agreement-menu');

typedef PrivacySaveCallback =
    Future<PrivacyStatus> Function(bool trainingEnabled);
typedef PrivacyAgreementLoader = Future<String> Function();

class PrivacyConsentDialog extends StatefulWidget {
  const PrivacyConsentDialog({
    required this.onSave,
    required this.onSaved,
    required this.onCloseApp,
    this.agreementText,
    this.agreementLoader,
    this.showCloseApp = true,
    this.onCancel,
    super.key,
  });

  final PrivacySaveCallback onSave;
  final ValueChanged<PrivacyStatus> onSaved;
  final VoidCallback onCloseApp;
  final String? agreementText;
  final PrivacyAgreementLoader? agreementLoader;
  final bool showCloseApp;
  final VoidCallback? onCancel;

  @override
  State<PrivacyConsentDialog> createState() => _PrivacyConsentDialogState();
}

class _PrivacyConsentDialogState extends State<PrivacyConsentDialog> {
  bool _agreementAccepted = false;
  bool? _trainingEnabled;
  bool _saving = false;
  String? _error;
  String? _agreementText;
  String? _agreementLoadError;
  bool _agreementLoading = false;

  bool get _canSave =>
      _agreementAccepted &&
      _agreementText?.trim().isNotEmpty == true &&
      _trainingEnabled != null &&
      !_saving;

  @override
  void initState() {
    super.initState();
    _agreementText = widget.agreementText;
    if (_agreementText == null) unawaited(_loadAgreement());
  }

  Future<void> _loadAgreement() async {
    if (mounted) {
      setState(() {
        _agreementLoading = true;
        _agreementLoadError = null;
        _agreementAccepted = false;
      });
    }
    try {
      final text =
          await (widget.agreementLoader?.call() ??
              rootBundle.loadString(privacyAgreementAsset));
      if (!mounted) return;
      if (text.trim().isEmpty) throw const FormatException('协议内容为空');
      setState(() {
        _agreementText = text;
        _agreementLoading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _agreementText = null;
        _agreementLoading = false;
        _agreementLoadError = error.toString();
      });
    }
  }

  Future<void> _save() async {
    if (!_canSave) return;
    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      final status = await widget.onSave(_trainingEnabled!);
      if (!mounted) return;
      if (!status.isConsentComplete) {
        setState(() => _error = '本地服务尚未确认协议与参与选择，请重试。');
        return;
      }
      widget.onSaved(status);
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = '保存失败：$error');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return PopScope(
      canPop: false,
      child: Material(
        key: privacyConsentDialogKey,
        color: Colors.black54,
        child: SafeArea(
          minimum: const EdgeInsets.all(12),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 860, maxHeight: 760),
              child: Card(
                clipBehavior: Clip.antiAlias,
                elevation: 12,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Padding(
                      padding: const EdgeInsets.fromLTRB(24, 20, 24, 12),
                      child: Row(
                        children: [
                          Icon(
                            Icons.privacy_tip_rounded,
                            color: scheme.primary,
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              'Neri 用户协议与模型改进计划',
                              style: Theme.of(context).textTheme.titleLarge,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const Divider(height: 1),
                    Expanded(
                      child: SingleChildScrollView(
                        padding: const EdgeInsets.fromLTRB(24, 18, 24, 12),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Text(
                              '请先阅读协议，并单独决定是否参加模型改进计划。拒绝参加不会影响识别、校验或导出功能。',
                              style: Theme.of(context).textTheme.bodyLarge,
                            ),
                            const SizedBox(height: 14),
                            _DisclosureSummary(scheme: scheme),
                            const SizedBox(height: 14),
                            if (_agreementLoading)
                              const SizedBox(
                                height: 64,
                                child: Center(
                                  child: CircularProgressIndicator(),
                                ),
                              )
                            else if (_agreementLoadError case final error?)
                              Container(
                                padding: const EdgeInsets.all(14),
                                decoration: BoxDecoration(
                                  color: scheme.errorContainer,
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: Row(
                                  children: [
                                    Expanded(child: Text('协议加载失败：$error')),
                                    TextButton(
                                      onPressed: _loadAgreement,
                                      child: const Text('重试'),
                                    ),
                                  ],
                                ),
                              )
                            else
                              Material(
                                color: scheme.surfaceContainerLowest,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12),
                                  side: BorderSide(
                                    color: scheme.outlineVariant,
                                  ),
                                ),
                                clipBehavior: Clip.antiAlias,
                                child: ListTile(
                                  key: privacyAgreementMenuKey,
                                  leading: Icon(
                                    Icons.description_outlined,
                                    color: scheme.primary,
                                  ),
                                  title: const Text('用户协议与隐私政策'),
                                  subtitle: const Text('查看完整协议'),
                                  trailing: const Icon(
                                    Icons.chevron_right_rounded,
                                  ),
                                  onTap: _saving
                                      ? null
                                      : () => showPrivacyAgreementDialog(
                                          context,
                                          text: _agreementText,
                                        ),
                                ),
                              ),
                            const SizedBox(height: 12),
                            CheckboxListTile(
                              key: privacyAgreementCheckboxKey,
                              contentPadding: EdgeInsets.zero,
                              controlAffinity: ListTileControlAffinity.leading,
                              value: _agreementAccepted,
                              enabled:
                                  !_saving &&
                                  _agreementText?.trim().isNotEmpty == true,
                              onChanged: _saving
                                  ? null
                                  : (value) => setState(
                                      () => _agreementAccepted = value ?? false,
                                    ),
                              title: const Text('我已阅读并同意《Neri 用户协议与隐私政策》'),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              '是否参加模型改进计划（必须选择一项）',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                            RadioGroup<bool>(
                              key: privacyParticipationGroupKey,
                              groupValue: _trainingEnabled,
                              onChanged: (value) {
                                if (!_saving) {
                                  setState(() => _trainingEnabled = value);
                                }
                              },
                              child: Column(
                                children: [
                                  RadioListTile<bool>(
                                    key: privacyParticipateOptionKey,
                                    contentPadding: EdgeInsets.zero,
                                    value: true,
                                    enabled: !_saving,
                                    title: const Text('同意参加'),
                                    subtitle: const Text('按上述范围自动提交后续确认的照片与标注'),
                                  ),
                                  RadioListTile<bool>(
                                    key: privacyDeclineOptionKey,
                                    contentPadding: EdgeInsets.zero,
                                    value: false,
                                    enabled: !_saving,
                                    title: const Text('暂不参加'),
                                    subtitle: const Text(
                                      '继续使用 Neri 的全部核心功能，不重复提示',
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            if (_error case final error?) ...[
                              const SizedBox(height: 8),
                              MaterialBanner(
                                padding: const EdgeInsets.all(12),
                                leading: Icon(
                                  Icons.error_outline_rounded,
                                  color: scheme.error,
                                ),
                                content: Text(error),
                                actions: const [SizedBox.shrink()],
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                    const Divider(height: 1),
                    Padding(
                      padding: const EdgeInsets.fromLTRB(20, 12, 20, 16),
                      child: Wrap(
                        alignment: WrapAlignment.end,
                        crossAxisAlignment: WrapCrossAlignment.center,
                        spacing: 12,
                        runSpacing: 8,
                        children: [
                          if (widget.showCloseApp)
                            TextButton(
                              onPressed: _saving ? null : widget.onCloseApp,
                              child: const Text('关闭应用'),
                            ),
                          if (widget.onCancel != null)
                            TextButton(
                              key: privacyConsentCancelButtonKey,
                              onPressed: _saving ? null : widget.onCancel,
                              child: const Text('取消'),
                            ),
                          FilledButton(
                            key: privacySaveButtonKey,
                            onPressed: _canSave
                                ? () => unawaited(_save())
                                : null,
                            child: _saving
                                ? const SizedBox.square(
                                    dimension: 18,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                    ),
                                  )
                                : const Text('保存并继续'),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

Future<void> showPrivacyAgreementDialog(BuildContext context, {String? text}) {
  return showDialog<void>(
    context: context,
    builder: (dialogContext) => Dialog(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 860, maxHeight: 720),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                children: [
                  IconButton(
                    tooltip: '返回',
                    onPressed: () => Navigator.of(dialogContext).pop(),
                    icon: const Icon(Icons.arrow_back_rounded),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Neri 用户协议与隐私政策',
                      style: Theme.of(dialogContext).textTheme.titleLarge,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Expanded(
                child: PrivacyAgreementDocument(text: text, maxHeight: 640),
              ),
            ],
          ),
        ),
      ),
    ),
  );
}

class PrivacyAgreementDocument extends StatefulWidget {
  const PrivacyAgreementDocument({this.text, this.maxHeight = 300, super.key});

  final String? text;
  final double maxHeight;

  @override
  State<PrivacyAgreementDocument> createState() =>
      _PrivacyAgreementDocumentState();
}

class _PrivacyAgreementDocumentState extends State<PrivacyAgreementDocument> {
  late final Future<String>? _loadFuture = widget.text == null
      ? rootBundle.loadString(privacyAgreementAsset)
      : null;
  final _scrollController = ScrollController();

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (widget.text case final text?) return _documentFrame(context, text);
    return FutureBuilder<String>(
      future: _loadFuture,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return _documentFrame(context, '协议加载失败：${snapshot.error}');
        }
        if (!snapshot.hasData) {
          return const SizedBox(
            height: 120,
            child: Center(child: CircularProgressIndicator()),
          );
        }
        return _documentFrame(context, snapshot.data!);
      },
    );
  }

  Widget _documentFrame(BuildContext context, String value) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      constraints: BoxConstraints(minHeight: 140, maxHeight: widget.maxHeight),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerLowest,
        border: Border.all(color: scheme.outlineVariant),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Scrollbar(
        controller: _scrollController,
        child: SingleChildScrollView(
          controller: _scrollController,
          child: SelectableText(value),
        ),
      ),
    );
  }
}

class _DisclosureSummary extends StatelessWidget {
  const _DisclosureSummary({required this.scheme});

  final ColorScheme scheme;

  @override
  Widget build(BuildContext context) {
    const points = <String>[
      '仅提交之后人工确认的照片与标注，不上传视频、不扫描照片库；每个物理文件夹最多提交 3 张确认的空照片。',
      '上传前压缩照片并移除 GPS、EXIF 等元数据；已标记为人或车辆的照片会排除，但画面身份、文字和水印仍可能保留，请勿提交敏感照片。',
      '照片与标注存储在运营者的 OneDrive，可能跨境处理；已完成样本自修订创建起最长保存 365 天。',
      '您可随时停止参加，或联系 wakin721@outlook.com 申请删除。详细范围和清理规则见完整协议。',
    ];
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: scheme.primaryContainer.withValues(alpha: 0.38),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('参加前请确认', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          for (final point in points)
            Padding(
              padding: const EdgeInsets.only(bottom: 5),
              child: Text('• $point'),
            ),
        ],
      ),
    );
  }
}
