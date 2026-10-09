import 'package:flutter/material.dart';

/// Neri's Material 3 Expressive styling, shared by the app and crash window.
/// Keep the supplied scheme intact so platform colors and user seeds survive.
ThemeData buildNeriTheme(ColorScheme scheme) {
  final base = ThemeData(useMaterial3: true, colorScheme: scheme);
  final text = base.textTheme;
  final buttonStyle = ButtonStyle(
    minimumSize: const WidgetStatePropertyAll(Size(64, 40)),
    padding: const WidgetStatePropertyAll(
      EdgeInsets.symmetric(horizontal: 20, vertical: 10),
    ),
    textStyle: WidgetStatePropertyAll(
      text.labelLarge?.copyWith(fontWeight: FontWeight.w700),
    ),
    shape: WidgetStateProperty.resolveWith<OutlinedBorder>((states) {
      return RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(
          states.contains(WidgetState.pressed) ? 12 : 24,
        ),
      );
    }),
    animationDuration: const Duration(milliseconds: 200),
  );
  final fieldBorder = OutlineInputBorder(
    borderRadius: BorderRadius.circular(16),
    borderSide: BorderSide(color: scheme.outline),
  );

  return base.copyWith(
    scaffoldBackgroundColor: scheme.surface,
    textTheme: text.copyWith(
      headlineLarge: text.headlineLarge?.copyWith(fontWeight: FontWeight.w700),
      headlineMedium: text.headlineMedium?.copyWith(
        fontWeight: FontWeight.w700,
      ),
      headlineSmall: text.headlineSmall?.copyWith(fontWeight: FontWeight.w700),
      titleLarge: text.titleLarge?.copyWith(fontWeight: FontWeight.w600),
      titleMedium: text.titleMedium?.copyWith(fontWeight: FontWeight.w600),
      labelLarge: text.labelLarge?.copyWith(fontWeight: FontWeight.w700),
    ),
    appBarTheme: AppBarThemeData(
      backgroundColor: scheme.surface,
      foregroundColor: scheme.onSurface,
      elevation: 0,
      scrolledUnderElevation: 0,
      titleTextStyle: text.titleLarge?.copyWith(
        color: scheme.onSurface,
        fontWeight: FontWeight.w700,
      ),
    ),
    cardTheme: CardThemeData(
      color: scheme.surfaceContainerLow,
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
    ),
    filledButtonTheme: FilledButtonThemeData(style: buttonStyle),
    elevatedButtonTheme: ElevatedButtonThemeData(style: buttonStyle),
    outlinedButtonTheme: OutlinedButtonThemeData(style: buttonStyle),
    textButtonTheme: TextButtonThemeData(style: buttonStyle),
    iconButtonTheme: IconButtonThemeData(
      style: ButtonStyle(
        shape: WidgetStateProperty.resolveWith<OutlinedBorder>((states) {
          return RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(
              states.contains(WidgetState.pressed) ? 12 : 20,
            ),
          );
        }),
      ),
    ),
    inputDecorationTheme: InputDecorationThemeData(
      filled: true,
      fillColor: scheme.surfaceContainerHighest,
      border: fieldBorder,
      enabledBorder: fieldBorder.copyWith(
        borderSide: BorderSide(color: scheme.outlineVariant),
      ),
      focusedBorder: fieldBorder.copyWith(
        borderSide: BorderSide(color: scheme.primary, width: 2),
      ),
      errorBorder: fieldBorder.copyWith(
        borderSide: BorderSide(color: scheme.error),
      ),
      focusedErrorBorder: fieldBorder.copyWith(
        borderSide: BorderSide(color: scheme.error, width: 2),
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
    ),
    navigationRailTheme: NavigationRailThemeData(
      backgroundColor: scheme.surface,
      indicatorColor: scheme.secondaryContainer,
      indicatorShape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
      ),
      selectedIconTheme: IconThemeData(
        color: scheme.onSecondaryContainer,
        size: 26,
      ),
      unselectedIconTheme: IconThemeData(
        color: scheme.onSurfaceVariant,
        size: 26,
      ),
      selectedLabelTextStyle: text.labelLarge?.copyWith(
        color: scheme.onSurface,
        fontWeight: FontWeight.w700,
      ),
      unselectedLabelTextStyle: text.labelMedium?.copyWith(
        color: scheme.onSurfaceVariant,
      ),
    ),
    listTileTheme: ListTileThemeData(
      selectedColor: scheme.onSecondaryContainer,
      selectedTileColor: scheme.secondaryContainer,
      iconColor: scheme.onSurfaceVariant,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
    segmentedButtonTheme: SegmentedButtonThemeData(
      style: ButtonStyle(
        minimumSize: const WidgetStatePropertyAll(Size(48, 40)),
        textStyle: WidgetStatePropertyAll(text.labelLarge),
        shape: WidgetStatePropertyAll(
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
      ),
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: scheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
      titleTextStyle: text.headlineSmall?.copyWith(
        color: scheme.onSurface,
        fontWeight: FontWeight.w700,
      ),
    ),
    snackBarTheme: SnackBarThemeData(
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
    progressIndicatorTheme: ProgressIndicatorThemeData(
      // Flutter 3.44 still requires this opt-in for the updated geometry.
      // ignore: deprecated_member_use
      year2023: false,
      color: scheme.primary,
      linearTrackColor: scheme.secondaryContainer,
      circularTrackColor: scheme.secondaryContainer,
      borderRadius: BorderRadius.circular(8),
      trackGap: 4,
      stopIndicatorRadius: 2,
    ),
    sliderTheme: const SliderThemeData(
      // ignore: deprecated_member_use
      year2023: false,
    ),
  );
}
