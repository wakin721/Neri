import 'dart:async';
import 'dart:io';
import 'dart:ui' show PlatformDispatcher;

import 'package:dynamic_color/dynamic_color.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:window_manager/window_manager.dart'; // 確保 import 放在所有變數和宣告的最上方

import 'src/api_client.dart';
import 'src/crash_reporter.dart';
import 'src/crash_watchdog.dart';
import 'src/models/theme_settings.dart';
import 'src/main_window.dart';

// SharedPreferences keys
const _kThemeModeKey = 'theme_mode';
const _kUseDynamicColorKey = 'use_dynamic_color';
const _kSeedColorKey = 'seed_color';

void main(List<String> args) {
  runZonedGuarded(
    () async {
      WidgetsFlutterBinding.ensureInitialized();
      CrashReporter.initialize();
      if (args.contains(CrashWatchdog.crashReportModeArg)) {
        await _runCrashReportMode(await _loadThemeSettings());
        return;
      }
      unawaited(CrashWatchdog.start());
      FlutterError.onError = (details) {
        FlutterError.presentError(details);
        CrashReporter.recordFlutterError(details);
      };
      PlatformDispatcher.instance.onError = (error, stackTrace) {
        CrashReporter.record(error, stackTrace, origin: 'Dart 未处理异常');
        return true;
      };

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

      final themeNotifier = ValueNotifier<ThemeSettings>(
        await _loadThemeSettings(),
      );

      // 保持使用您原本的 NeriApp 類別，而不是 MyApp
      runApp(NeriApp(themeNotifier: themeNotifier));
    },
    (error, stackTrace) {
      CrashReporter.record(error, stackTrace, origin: 'Dart Zone');
    },
  );
}

Future<ThemeSettings> _loadThemeSettings() async {
  final prefs = await SharedPreferences.getInstance();
  final themeModeIndex = prefs.getInt(_kThemeModeKey) ?? ThemeMode.system.index;
  final useDynamic = prefs.getBool(_kUseDynamicColorKey) ?? false;
  final seedValue =
      prefs.getInt(_kSeedColorKey) ?? kSeedColorOptions.first.color.toARGB32();

  return ThemeSettings(
    themeMode:
        ThemeMode.values[themeModeIndex.clamp(0, ThemeMode.values.length - 1)],
    useDynamicColor: useDynamic,
    seedColor: Color(seedValue),
  );
}

Future<void> _runCrashReportMode(ThemeSettings settings) async {
  await windowManager.ensureInitialized();
  WindowOptions windowOptions = const WindowOptions(
    size: Size(520, 300),
    center: true,
    skipTaskbar: false,
    titleBarStyle: TitleBarStyle.hidden,
  );
  windowManager.waitUntilReadyToShow(windowOptions, () async {
    await windowManager.show();
    await windowManager.focus();
  });

  runApp(
    CrashReportOnlyApp(
      settings: settings,
      report:
          CrashReporter.latestReport.value ??
          CrashReport(
            title: '程序崩溃提示',
            message: 'Neri 上次运行异常退出。',
            details: '',
            logPath: '',
            createdAt: DateTime.now(),
          ),
    ),
  );
}

class CrashReportOnlyApp extends StatelessWidget {
  const CrashReportOnlyApp({
    required this.settings,
    required this.report,
    super.key,
  });

  final ThemeSettings settings;
  final CrashReport report;

  @override
  Widget build(BuildContext context) {
    return DynamicColorBuilder(
      builder: (ColorScheme? lightDynamic, ColorScheme? darkDynamic) {
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
          home: _CrashReportOnlyScreen(report: report),
        );
      },
    );
  }
}

class _CrashReportOnlyScreen extends StatefulWidget {
  const _CrashReportOnlyScreen({required this.report});

  final CrashReport report;

  @override
  State<_CrashReportOnlyScreen> createState() => _CrashReportOnlyScreenState();
}

class _CrashReportOnlyScreenState extends State<_CrashReportOnlyScreen> {
  Future<void> _exitReportMode() async {
    try {
      await windowManager.destroy().timeout(const Duration(milliseconds: 300));
    } catch (_) {}
    exit(0);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Scaffold(
      backgroundColor: scheme.surface,
      body: Padding(
        padding: const EdgeInsets.fromLTRB(28, 26, 24, 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              widget.report.title,
              style: theme.textTheme.headlineSmall?.copyWith(
                color: scheme.onSurface,
              ),
            ),
            const SizedBox(height: 18),
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.report.message,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: scheme.onSurface,
                      ),
                    ),
                    if (widget.report.details.isNotEmpty) ...[
                      const SizedBox(height: 16),
                      SelectableText(
                        widget.report.details,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: scheme.onSurface,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 18),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton(
                onPressed: _exitReportMode,
                child: const Text('知道了'),
              ),
            ),
          ],
        ),
      ),
    );
  }
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
              navigatorKey: CrashReporter.navigatorKey,
              debugShowCheckedModeBanner: false,
              builder: (context, child) =>
                  CrashDialogListener(child: child ?? const SizedBox.shrink()),
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
