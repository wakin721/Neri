import 'package:flutter/material.dart';

import '../models/settings.dart';
import '../widgets/section_card.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({required this.settings, super.key});

  final NeriSettings? settings;

  @override
  Widget build(BuildContext context) {
    return ListView(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.all(16),
      children: [
        SectionCard(
          title: '关于 Neri',
          subtitle: settings?.appVersion,
          icon: Icons.info_rounded,
          child: const Text(
            'Neri 是红外相机图像智能处理工具。当前 Flutter Material 3 客户端通过 Python 后端复用项目已有的 EXIF 提取、批量处理和 YOLO 识别能力。',
          ),
        ),
      ],
    );
  }
}
