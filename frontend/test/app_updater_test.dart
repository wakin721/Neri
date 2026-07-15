import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/app_updater.dart';

void main() {
  group('compareNeriVersions', () {
    test('orders stable and preview releases', () {
      expect(compareNeriVersions('3.0.5', '3.0.5-alpha10'), greaterThan(0));
      expect(
        compareNeriVersions('3.0.5-alpha10', '3.0.5-alpha2'),
        greaterThan(0),
      );
    });

    test('ignores display build codes', () {
      expect(compareNeriVersions('3.0.5-alpha2(b5f6a6)', 'v3.0.5-alpha2'), 0);
      expect(compareNeriVersions('v3.0.4-release(dfc280)', '3.0.4'), 0);
    });

    test('compares numeric release components before prerelease labels', () {
      expect(compareNeriVersions('3.1.0-alpha1', '3.0.9'), greaterThan(0));
      expect(compareNeriVersions('3.0.6', '3.0.10'), lessThan(0));
    });
  });

  group('GitHub mirror templates', () {
    final official = Uri.parse(
      'https://github.com/wakin721/Neri/releases/download/v1/Neri.zip',
    );

    test('replaces the GitHub host for a mirror root', () {
      expect(
        resolveGithubMirrorUri(official, 'https://kkgithub.com').toString(),
        'https://kkgithub.com/wakin721/Neri/releases/download/v1/Neri.zip',
      );
    });

    test('supports URL and path placeholders', () {
      expect(
        resolveGithubMirrorUri(
          official,
          'https://proxy.example.com/{url}',
        ).toString(),
        'https://proxy.example.com/https://github.com/wakin721/Neri/releases/download/v1/Neri.zip',
      );
      expect(
        resolveGithubMirrorUri(
          official,
          'https://mirror.example.com/github{path}',
        ).toString(),
        'https://mirror.example.com/github/wakin721/Neri/releases/download/v1/Neri.zip',
      );
    });

    test('rejects unsupported or unsafe mirror addresses', () {
      expect(
        () => normalizeGithubMirrorTemplate('ftp://mirror.example.com'),
        throwsFormatException,
      );
      expect(
        () => normalizeGithubMirrorTemplate(
          'https://user:secret@mirror.example.com',
        ),
        throwsFormatException,
      );
    });

    test('keeps mirror order, removes duplicates, and uses official last', () {
      final downloadUris = buildGithubDownloadUris(
        officialUri: official,
        useMirrors: true,
        mirrorTemplates: const <String>[
          'https://first.example.com',
          'https://second.example.com/{url}',
          'https://first.example.com',
        ],
      );

      expect(downloadUris.map((uri) => uri.toString()), <String>[
        'https://first.example.com/wakin721/Neri/releases/download/v1/Neri.zip',
        'https://second.example.com/https://github.com/wakin721/Neri/releases/download/v1/Neri.zip',
        official.toString(),
      ]);
    });

    test('skips mirrors when the official source is selected', () {
      expect(
        buildGithubDownloadUris(
          officialUri: official,
          useMirrors: false,
          mirrorTemplates: const <String>['https://first.example.com'],
        ),
        <Uri>[official],
      );
    });
  });
}
