import 'package:flutter/material.dart';

/// A settings spinner with the same painted diameter in rows, buttons and dialogs.
class SettingsProgressIndicator extends StatelessWidget {
  const SettingsProgressIndicator({
    this.value,
    this.backgroundColor,
    this.semanticsLabel,
    super.key,
  });

  final double? value;
  final Color? backgroundColor;
  final String? semanticsLabel;

  @override
  Widget build(BuildContext context) => CircularProgressIndicator(
    value: value,
    backgroundColor: backgroundColor,
    semanticsLabel: semanticsLabel,
    constraints: const BoxConstraints.tightFor(width: 14, height: 14),
    padding: EdgeInsets.zero,
    strokeWidth: 2,
    strokeAlign: CircularProgressIndicator.strokeAlignInside,
  );
}
