import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/settings.dart';
import 'package:neri_flutter/src/screens/start_screen.dart';

void main() {
  testWidgets('探测模型可以选择不使用', (tester) async {
    final inputController = TextEditingController();
    addTearDown(inputController.dispose);
    String? selectedModelPath;

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StartScreen(
            settings: const NeriSettings(
              appTitle: 'Neri',
              appVersion: 'test',
              supportedImageExtensions: <String>['.jpg'],
              supportedVideoExtensions: <String>['.mp4'],
              modelDirectory: 'res/model',
              classificationModelDirectory: 'res/model_cls',
              availableModels: <ModelInfo>[
                ModelInfo(name: 'detector.pt', path: 'res/model/detector.pt'),
              ],
              availableClassificationModels: <ModelInfo>[
                ModelInfo(
                  name: 'classifier.pt',
                  path: 'res/model_cls/classifier.pt',
                ),
              ],
              speciesTypes: <String, String>{},
              settings: <String, dynamic>{},
              gpuAvailable: false,
              missingYoloDependencies: <String>[],
            ),
            inputController: inputController,
            selectedModelPath: '',
            onModelChanged: (value) => selectedModelPath = value,
            selectedClassificationModelPath: 'res/model_cls/classifier.pt',
            onClassificationModelChanged: (_) {},
            videoMode: 'all',
            onVideoModeChanged: (_) {},
            vidStride: 1,
            onVidStrideChanged: (_) {},
            useFp16: false,
            onUseFp16Changed: (_) {},
            confidence: 0.25,
            onConfidenceChanged: (_) {},
            iou: 0.30,
            onIouChanged: (_) {},
            submitting: false,
            onCreateJob: () {},
            onCancelJob: (_) {},
            onResumeJob: (_) {},
            onDeleteJob: (_) {},
            onClearJobs: () {},
            pendingStartJobIds: const <String>{},
            pendingStopJobIds: const <String>{},
            jobs: const [],
          ),
        ),
      ),
    );

    final dropdowns = tester
        .widgetList<DropdownMenu<String>>(find.byType(DropdownMenu<String>))
        .toList();
    final detectionModelDropdown = dropdowns.first;

    expect(detectionModelDropdown.initialSelection, '');
    expect(detectionModelDropdown.dropdownMenuEntries.first.value, '');
    expect(detectionModelDropdown.dropdownMenuEntries.first.label, '不使用');

    detectionModelDropdown.onSelected?.call('');
    expect(selectedModelPath, '');
  });
}
