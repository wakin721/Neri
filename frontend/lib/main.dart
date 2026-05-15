import 'package:dynamic_color/dynamic_color.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:window_manager/window_manager.dart'; // 確保 import 放在所有變數和宣告的最上方

import 'src/api_client.dart';
import 'src/models/theme_settings.dart';
import 'src/main_window.dart';

// SharedPreferences keys
const _kThemeModeKey = 'theme_mode';
const _kUseDynamicColorKey = 'use_dynamic_color';
const _kSeedColorKey = 'seed_color';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 初始化桌面視窗管理器 (取代預設標題列)
  await windowManager.ensureInitialized();
  WindowOptions windowOptions = const WindowOptions(
    size: Size(1280, 800),
    center: true,
    skipTaskbar: false,
    titleBarStyle: TitleBarStyle.hidden, // 隱藏系統原生標題列
  );
  windowManager.waitUntilReadyToShow(windowOptions, () async {
    await windowManager.show();
    await windowManager.focus();
  });

  // Restore persisted theme preferences before first frame.
  final prefs = await SharedPreferences.getInstance();
  final themeModeIndex = prefs.getInt(_kThemeModeKey) ?? ThemeMode.system.index;
  final useDynamic = prefs.getBool(_kUseDynamicColorKey) ?? false;
  final seedValue =
      prefs.getInt(_kSeedColorKey) ?? kSeedColorOptions.first.color.toARGB32();

  final themeNotifier = ValueNotifier<ThemeSettings>(
    ThemeSettings(
      themeMode: ThemeMode
          .values[themeModeIndex.clamp(0, ThemeMode.values.length - 1)],
      useDynamicColor: useDynamic,
      seedColor: Color(seedValue),
    ),
  );

  // 保持使用您原本的 NeriApp 類別，而不是 MyApp
  runApp(NeriApp(themeNotifier: themeNotifier));
}

class NeriApp extends StatelessWidget {
  const NeriApp({required this.themeNotifier, super.key});

  final ValueNotifier<ThemeSettings> themeNotifier;

  @override
  Widget build(BuildContext context) {
    // DynamicColorBuilder resolves the platform's dynamic color schemes (Android 12+).
    // On platforms that don't support it, both values are null and we fall back
    // to the user-selected seed color.
    return DynamicColorBuilder(
      builder: (ColorScheme? lightDynamic, ColorScheme? darkDynamic) {
        return ValueListenableBuilder<ThemeSettings>(
          valueListenable: themeNotifier,
          builder: (context, settings, _) {
            final ColorScheme lightScheme =
                (settings.useDynamicColor && lightDynamic != null)
                ? lightDynamic
                : ColorScheme.fromSeed(seedColor: settings.seedColor);

            final ColorScheme darkScheme =
                (settings.useDynamicColor && darkDynamic != null)
                ? darkDynamic
                : ColorScheme.fromSeed(
                    seedColor: settings.seedColor,
                    brightness: Brightness.dark,
                  );

            return MaterialApp(
              title: 'Neri',
              debugShowCheckedModeBanner: false,
              themeMode: settings.themeMode,
              theme: ThemeData(useMaterial3: true, colorScheme: lightScheme),
              darkTheme: ThemeData(useMaterial3: true, colorScheme: darkScheme),
              home: MainWindow(
                apiClient: NeriApiClient(),
                themeNotifier: themeNotifier,
              ),
            );
          },
        );
      },
    );
  }
}
