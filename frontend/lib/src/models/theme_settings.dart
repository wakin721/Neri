import 'package:flutter/material.dart';

/// Persistent theme preferences held in a [ValueNotifier].
class ThemeSettings {
  const ThemeSettings({
    this.themeMode = ThemeMode.system,
    this.useDynamicColor = false,
    this.seedColor = _kDefaultSeed,
  });

  static const Color _kDefaultSeed = Color(0xFF386A20);

  final ThemeMode themeMode;
  final bool useDynamicColor;
  final Color seedColor;

  ThemeSettings copyWith({
    ThemeMode? themeMode,
    bool? useDynamicColor,
    Color? seedColor,
  }) {
    return ThemeSettings(
      themeMode: themeMode ?? this.themeMode,
      useDynamicColor: useDynamicColor ?? this.useDynamicColor,
      seedColor: seedColor ?? this.seedColor,
    );
  }
}

/// A named seed color shown in the palette picker.
class SeedColorOption {
  const SeedColorOption({required this.label, required this.color});

  final String label;
  final Color color;
}

/// Built-in color palette choices displayed in the Settings page.
const List<SeedColorOption> kSeedColorOptions = <SeedColorOption>[
  SeedColorOption(label: '青松绿', color: Color(0xFF386A20)),
  SeedColorOption(label: '山岚蓝', color: Color(0xFF1565C0)),
  SeedColorOption(label: '暮霞橙', color: Color(0xFFBF360C)),
  SeedColorOption(label: '山茶紫', color: Color(0xFF6750A4)),
  SeedColorOption(label: '珊瑚红', color: Color(0xFFC62828)),
  SeedColorOption(label: '墨玉青', color: Color(0xFF00695C)),
  SeedColorOption(label: '流沙金', color: Color(0xFFE65100)),
  SeedColorOption(label: '苍穹蓝', color: Color(0xFF0277BD)),
];
