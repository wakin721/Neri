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
