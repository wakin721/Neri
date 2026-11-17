import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/models/job.dart';
import 'package:neri_flutter/src/utils/media_display_order.dart';

DetectionItem photo(
  String name, {
  String? date,
  String? modified,
  String camera = 'camera',
  String? time,
}) => DetectionItem(
  filename: name,
  path: '$camera/$name',
  fileType: 'jpg',
  dateTaken: date,
  modifiedAt: modified,
  detectionData: {if (time != null) '拍摄时间': time},
);

void main() {
  test('lazy video metadata keeps scan ordering and current recognition', () {
    final source = DetectionItem(
      filename: 'IMAG1136.MOV',
      path: 'camera/IMAG1136.MOV',
      fileType: 'mov',
      modifiedAt: '2025-11-22T16:32:32',
      species: const ['环颈雉'],
      validated: true,
      detectionData: const {'物种名称': '环颈雉', '最低置信度': '人工校验'},
    );
    final metadata = DetectionItem(
      filename: source.filename,
      path: source.path,
      fileType: source.fileType,
      dateTaken: '2025-11-22T16:32:22+00:00',
      width: 1920,
      height: 1080,
      species: const ['家羊'],
      detectionData: const {
        '物种名称': '家羊',
        'capture_time': {'timestamp_source': 'mp4:mvhd'},
      },
    );
    final hydrated = source.mergePreviewMetadata(metadata);
    expect(hydrated.dateTaken, metadata.dateTaken);
    expect(hydrated.width, 1920);
    expect(hydrated.height, 1080);
    expect(hydrated.species, ['环颈雉']);
    expect(hydrated.validated, isTrue);
    expect(hydrated.detectionData['最低置信度'], '人工校验');
    expect(
      hydrated.detectionData['capture_time'],
      metadata.detectionData['capture_time'],
    );
    expect(mediaSortTimestamp(hydrated), mediaSortTimestamp(source));
    expect(
      hydrated.mergePreviewMetadata(metadata).groupingTime,
      source.groupingTime,
    );
    final marked = hydrated.mergeValidationUpdate(metadata);
    expect(marked.species, ['家羊']);
    expect(mediaSortTimestamp(marked), mediaSortTimestamp(source));
    // A new directory snapshot is allowed to use new capture metadata.
    expect(metadata.previewGroupingTime, isNull);
    expect(
      mediaSortTimestamp(metadata),
      DateTime.utc(2025, 11, 22, 16, 32, 22),
    );
  });

  test(
    'unknown scan time remains unknown after lazy metadata and isolate sort',
    () async {
      final source = photo('2.jpg');
      final hydrated = source.mergePreviewMetadata(
        photo('2.jpg', date: '2020-01-01T12:00:00Z'),
      );
      expect(mediaSortTimestamp(hydrated), isNull);
      final other = photo('1.jpg');
      expect(
        await mediaDisplayIndicesAsync([hydrated, other], byDirectory: true),
        [1, 0],
      );
      expect(
        () => source.mergePreviewMetadata(photo('3.jpg')),
        throwsArgumentError,
      );
    },
  );

  test(
    'display order preserves timestamps and natural filename and path ties',
    () {
      final items = [
        photo('10.jpg'),
        photo('2.jpg', camera: 'camera10'),
        photo('2.jpg', camera: 'camera2'),
        photo('early.jpg', date: '2020-01-01', time: '10:00:00'),
        photo('later.jpg', modified: '2021-01-01T10:00:00'),
      ];
      expect(sortMediaItemsForDisplay(items).map((x) => x.path), [
        'camera/early.jpg',
        'camera/later.jpg',
        'camera2/2.jpg',
        'camera10/2.jpg',
        'camera/10.jpg',
      ]);
      expect(items.first.filename, '10.jpg');
    },
  );

  test('single metadata update moves earlier, later and to an equal key', () {
    final first = photo('1.jpg', modified: '2021-01-01T12:00:00');
    final middle = photo('2.jpg', modified: '2021-01-02T12:00:00');
    final last = photo('3.jpg', modified: '2021-01-03T12:00:00');
    final source = [first, middle, last];
    final earlier = photo('3.jpg', date: '2020-01-01T12:00:00');
    expect(replaceSortedMediaItem(source, 2, earlier), [
      earlier,
      first,
      middle,
    ]);
    final later = photo('1.jpg', date: '2022-01-01T12:00:00');
    expect(replaceSortedMediaItem(source, 0, later), [middle, last, later]);
    final equal = photo('2.jpg', date: '2021-01-02T12:00:00');
    expect(replaceSortedMediaItem(source, 1, equal), [first, equal, last]);
    expect(source, [first, middle, last]);
    expect(replaceSortedMediaItem([middle], 0, equal), [equal]);
  });

  test(
    'validation timestamp replacement remains ordered for metadata insertion',
    () {
      final first = photo('1.jpg', modified: '2021-01-01T12:00:00');
      final middle = photo('2.jpg', modified: '2021-01-02T12:00:00');
      final last = photo('3.jpg', modified: '2021-01-03T12:00:00');
      final changed = photo('1.jpg', modified: '2021-01-04T12:00:00');
      final validated = replaceSortedMediaItems(
        [first, middle, last],
        [changed],
      );
      expect(validated, [middle, last, changed]);
      final metadata = photo('2.jpg', date: '2021-01-03T18:00:00');
      expect(replaceSortedMediaItem(validated, 0, metadata), [
        last,
        metadata,
        changed,
      ]);
      expect(replaceSortedMediaItems(validated, const []), same(validated));
      final appended = photo('4.jpg', modified: '2020-01-01T12:00:00');
      expect(replaceSortedMediaItems(validated, [appended]), [
        appended,
        middle,
        last,
        changed,
      ]);
    },
  );

  test(
    'camera clock comparison ignores timezone suffixes without shifting dates',
    () {
      for (final suffix in ['', 'Z', '+00:00', '+08:00', '-05:00']) {
        expect(
          cameraClockTimestamp('2025-11-22T16:32:22$suffix'),
          DateTime.utc(2025, 11, 22, 16, 32, 22),
        );
      }
      final photoItem = photo('IMAG1133.JPG', date: '2025-11-22T16:32:19');
      final videoItem = photo(
        'IMAG1136.MOV',
        date: '2025-11-22T16:32:22+00:00',
      );
      expect(
        mediaSortTimestamp(
          videoItem,
        )!.difference(mediaSortTimestamp(photoItem)!),
        const Duration(seconds: 3),
      );
    },
  );

  test(
    'large snapshot async ordering preserves the same natural results',
    () async {
      final source = [for (var i = 3000; i > 0; i--) photo('$i.jpg')];
      final ordered = await sortMediaItemsForDisplayAsync(source);
      expect(ordered.map((x) => x.filename), [
        for (var i = 1; i <= 3000; i++) '$i.jpg',
      ]);
      expect(source.first.filename, '3000.jpg');
    },
  );
}
