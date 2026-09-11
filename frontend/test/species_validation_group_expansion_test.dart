import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neri_flutter/src/api_client.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/screens/species_validation_screen.dart';

void main() {
  late Directory tempDir;
  late NeriApiClient apiClient;
  late List<DetectionItem> items;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('neri_group_expansion_');
    final pngBytes = base64Decode(
      'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFUlEQVR4nGP8//8/A27AhEduBEsDAKXjAxF9kqZqAAAAAElFTkSuQmCC',
    );
    final paths = <String>[];
    for (var index = 0; index < 4; index++) {
      final file = File('${tempDir.path}/ECSP02${77 + index}.JPG');
      await file.writeAsBytes(pngBytes);
      paths.add(file.path);
    }

    apiClient = NeriApiClient(
      httpClient: MockClient(
        (_) async => http.Response(
          '{}',
          200,
          headers: const {'content-type': 'application/json; charset=utf-8'},
        ),
      ),
    );

    items = <DetectionItem>[
      for (var index = 0; index < paths.length; index++)
        DetectionItem(
          filename: 'ECSP02${77 + index}.JPG',
          path: paths[index],
          fileType: 'jpg',
          modifiedAt: '2026-09-11T12:00:0${index}Z',
          width: 10,
          height: 10,
          species: const <String>['豹猫'],
          confidence: 0.95,
          detectionData: const <String, dynamic>{'物种名称': '豹猫'},
        ),
    ];
  });

  tearDown(() async {
    apiClient.close();
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  testWidgets('expanded group stays open when metadata reorders its members', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      _GroupExpansionHarness(apiClient: apiClient, initialItems: items),
    );
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    final expandButton = find.byTooltip('展开分组');
    expect(expandButton, findsOneWidget);

    await tester.tap(expandButton);
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));

    // Expanding loads the representative item's full metadata. The harness
    // changes its timestamp so the same four files are reordered and the
    // automatic grouping cache is rebuilt. That rebuild must not discard the
    // user's explicit expansion state.
    expect(find.byTooltip('折叠分组'), findsOneWidget);
    expect(find.text('ECSP0277.JPG'), findsWidgets);
    expect(find.text('ECSP0278.JPG'), findsWidgets);
    expect(find.text('ECSP0279.JPG'), findsWidgets);
    expect(find.text('ECSP0280.JPG'), findsWidgets);
  });
}

class _GroupExpansionHarness extends StatefulWidget {
  const _GroupExpansionHarness({
    required this.apiClient,
    required this.initialItems,
  });

  final NeriApiClient apiClient;
  final List<DetectionItem> initialItems;

  @override
  State<_GroupExpansionHarness> createState() => _GroupExpansionHarnessState();
}

class _GroupExpansionHarnessState extends State<_GroupExpansionHarness> {
  late List<DetectionItem> _items = widget.initialItems;
  int _metadataRequests = 0;
  bool _metadataApplied = false;

  Future<void> _loadMetadata(DetectionItem item) async {
    _metadataRequests += 1;
    if (_metadataRequests < 2 || _metadataApplied) return;
    _metadataApplied = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      setState(() {
        _items = [
          for (final current in _items)
            if (current.path == item.path)
              DetectionItem(
                filename: current.filename,
                path: current.path,
                fileType: current.fileType,
                dateTaken: '2026-09-11T12:00:10Z',
                modifiedAt: current.modifiedAt,
                width: current.width,
                height: current.height,
                sizeBytes: current.sizeBytes,
                species: current.species,
                confidence: current.confidence,
                detectionBoxes: current.detectionBoxes,
                detectionData: current.detectionData,
                error: current.error,
                validated: current.validated,
              )
            else
              current,
        ];
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: SpeciesValidationScreen(
          apiClient: widget.apiClient,
          inputPath: File(widget.initialItems.first.path).parent.path,
          items: _items,
          loading: false,
          refreshVersion: 0,
          speciesTypes: const <String, String>{'豹猫': '兽类'},
          useCombinedConfidence: false,
          minFrameRatio: 0,
          autoGroup: true,
          collapseGroups: true,
          autoGroupDetectBurst: false,
          autoGroupBurstSize: 4,
          autoGroupGapSeconds: 1800,
          autoSortQuickMarks: false,
          undoSteps: 20,
          quickMarkSpecies: const <String>['豹猫'],
          quickMarkRecentHistory: const <String>[],
          quickMarkUsageCounts: const <String, int>{},
          quantityButtons: const <String>['1'],
          exportColumns: const <String>[],
          favoritePhotoPaths: const <String>[],
          favoritePhotoExportMode: favoritePhotoExportAsk,
          emptyPhotoDeleteMode: emptyPhotoDeleteAsk,
          onRefresh: () async {},
          onLoadMetadata: _loadMetadata,
          onOpenExternal: (_) {},
          onMarkItem:
              (
                candidate,
                action, {
                speciesName,
                speciesCount,
                speciesType,
                remark,
              }) async => candidate,
          onMarkItems:
              (
                candidates,
                action, {
                speciesName,
                speciesCount,
                speciesType,
                remark,
              }) async => candidates,
          onQuickMarkUsed: (_) async {},
          onQuickMarkReverted: (_) async {},
          onRedetectItems: (_, {required confidence}) async {},
          onFavoritePhotoPathsChanged: (_) async {},
          onFavoritePhotoExportModeChanged: (_) async {},
          onEmptyPhotoDeleteModeChanged: (_) async {},
          onAutoGroupInferredBurstSizeChanged: (_) {},
        ),
      ),
    );
  }
}
