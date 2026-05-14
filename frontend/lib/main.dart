import 'package:flutter/material.dart';

import 'src/api_client.dart';
import 'src/screens/home_screen.dart';

void main() {
  runApp(const NeriApp());
}

class NeriApp extends StatelessWidget {
  const NeriApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Neri',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.system,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF386A20)),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF9CD67D),
          brightness: Brightness.dark,
        ),
      ),
      home: HomeScreen(apiClient: NeriApiClient()),
    );
  }
}
