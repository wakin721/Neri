import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/models/video_processing_mode.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';

void main() {
  testWidgets('DINOv3 disables full video mode and normalizes selection to fast', (
    tester,
  ) async {
    final input = TextEditingController();
    addTearDown(input.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StartScreen(
            settings: const NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: <String>['.jpg'],
              supportedVideoExtensions: <String>['.mp4'],
              modelDirectory: 'res/model/detect',
              classificationModelDirectory: 'res/model/cls',
              availableModels: <ModelInfo>[
                ModelInfo(name: 'det.pt', path: 'det.pt'),
              ],
              availableClassificationModels: <ModelInfo>[
                ModelInfo(
                  name: 'DINOv3',
                  path: 'dino.neri.json',
                  kind: 'cls',
                  backend: 'dinov3',
                  requiresDetector: true,
                  supportsVideoAll: false,
                ),
              ],
              speciesTypes: <String, String>{},
              settings: <String, dynamic>{},
              gpuAvailable: false,
              missingYoloDependencies: <String>[],
            ),
            inputController: input,
            selectedModelPath: 'det.pt',
            onModelChanged: (_) {},
            selectedClassificationModelPath: 'dino.neri.json',
            onClassificationModelChanged: (_) {},
            videoMode: videoProcessingModeAll,
            onVideoModeChanged: (_) {},
            vidStride: 3,
            onVidStrideChanged: (_) {},
            useFp16: false,
            onUseFp16Changed: (_) {},
            confidence: 0.25,
            onConfidenceChanged: (_) {},
            iou: 0.3,
            onIouChanged: (_) {},
            submitting: false,
            onCreateJob: () {},
            onCancelJob: (_) {},
            onResumeJob: (_) {},
            onDeleteJob: (_) {},
            onClearJobs: () {},
            pendingStartJobIds: const <String>{},
            pendingStopJobIds: const <String>{},
            jobs: const <ProcessingJob>[],
          ),
        ),
      ),
    );

    final dropdown = tester
        .widgetList<DropdownMenu<String>>(find.byType(DropdownMenu<String>))
        .singleWhere(
          (item) => item.dropdownMenuEntries.any(
            (entry) => entry.value == videoProcessingModeSkip,
          ),
        );
    expect(dropdown.initialSelection, videoProcessingModeFast);
    expect(
      dropdown.dropdownMenuEntries
          .singleWhere((entry) => entry.value == videoProcessingModeAll)
          .enabled,
      isFalse,
    );
    expect(find.textContaining('DINOv3'), findsWidgets);
  });
}
