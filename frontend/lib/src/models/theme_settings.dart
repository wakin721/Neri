// lib/src/models/theme_settings.dart

import 'package:flutter/material.dart';

const Color kDefaultSeedColor = Color(0xFFFA9D85);

/// Persistent theme preferences held in a [ValueNotifier].
class ThemeSettings {
  const ThemeSettings({
    this.themeMode = ThemeMode.system,
    this.useDynamicColor = false,
    this.seedColor = kDefaultSeedColor,
  });

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
  SeedColorOption(label: '抹茶绿', color: Color(0xFF94CC84)),
  SeedColorOption(label: '薄荷绿', color: Color(0xFF69D2B0)),
  SeedColorOption(label: '晴空蓝', color: Color(0xFF63D1E6)),
  SeedColorOption(label: '矢车菊蓝', color: Color(0xFF7CB5FB)),
  SeedColorOption(label: '长春花蓝', color: Color(0xFF9AA5FA)),
  SeedColorOption(label: '丁香紫', color: Color(0xFFB29AFB)),
  SeedColorOption(label: '紫藤花', color: Color(0xFFD6A1FB)),
  SeedColorOption(label: '樱花粉', color: Color(0xFFFA98D2)),
  SeedColorOption(label: '珊瑚红', color: Color(0xFFFA9D85)),
  SeedColorOption(label: '蜜橘橙', color: Color(0xFFFBB769)),
  SeedColorOption(label: '橄榄黄', color: Color(0xFFD3D667)),
];
