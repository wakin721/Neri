import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
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

  group('mainland China update channel', () {
    test('loads the validated download URL from channels.json', () async {
      if (!Platform.isWindows) return;

      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      final requestedPaths = <String>[];
      final subscription = server.listen((request) async {
        requestedPaths.add(request.uri.path);
        request.response
          ..statusCode = HttpStatus.ok
          ..headers.contentType = ContentType.json
          ..write(
            jsonEncode(
              request.uri.path == '/channels.json'
                  ? <String, dynamic>{
                      'release': <String, dynamic>{
                        'tag': 'v3.0.4-release(dfc280)',
                        'files': <String, dynamic>{
                          'Neri_v3.0.4_release_x64_lite.zip': <String, dynamic>{
                            'url':
                                'https://download.myneri.top/releases/release/id/Neri_v3.0.4_release_x64_lite.zip',
                            'size': 12345,
                            'sha256':
                                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                          },
                        },
                      },
                      'preview': <String, dynamic>{
                        'tag': 'v3.0.5-beta7(58b040)',
                        'files': <String, dynamic>{
                          'Neri_v3.0.5_beta7_x64_lite.zip': <String, dynamic>{
                            'url':
                                'https://download.myneri.top/releases/preview/id/Neri_v3.0.5_beta7_x64_lite.zip',
                            'size': 54321,
                            'sha256':
                                'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                          },
                        },
                      },
                    }
                  : <String, dynamic>{
                      'tag_name': request.uri.path.startsWith('/preview/')
                          ? 'v3.0.5-beta7(58b040)'
                          : 'v3.0.4-release(dfc280)',
                      'name': request.uri.path.startsWith('/preview/')
                          ? 'Neri 3.0.5 beta7'
                          : 'Neri 3.0.4',
                      'body': request.uri.path.startsWith('/preview/')
                          ? '国内预览版更新说明'
                          : '国内稳定版更新说明',
                      'html_url':
                          'https://github.com/wakin721/Neri/releases/tag/v3.0.4-release',
                      'prerelease': request.uri.path.startsWith('/preview/'),
                    },
            ),
          );
        await request.response.close();
      });
      final serverRoot = 'http://${server.address.address}:${server.port}';
      final updater = AppUpdater(
        channelsUri: Uri.parse('$serverRoot/channels.json'),
        mainlandReleaseUri: Uri.parse('$serverRoot/latest/release.json'),
        mainlandPreviewReleaseUri: Uri.parse(
          '$serverRoot/preview/release.json',
        ),
      );

      try {
        final release = await updater.checkForUpdate(
          currentVersion: '3.0.3',
          channel: 'Release',
          useMainlandChinaSource: true,
        );

        expect(release, isNotNull);
        expect(release!.tag, 'v3.0.4-release(dfc280)');
        expect(
          release.asset.downloadUri.toString(),
          'https://download.myneri.top/releases/release/id/Neri_v3.0.4_release_x64_lite.zip',
        );
        expect(release.asset.sizeBytes, 12345);
        expect(
          release.asset.sha256,
          'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        );
        expect(release.notes, '国内稳定版更新说明');
        final preview = await updater.checkForUpdate(
          currentVersion: '3.0.4',
          channel: 'Preview',
          useMainlandChinaSource: true,
        );
        expect(preview, isNotNull);
        expect(preview!.tag, 'v3.0.5-beta7(58b040)');
        expect(preview.notes, '国内预览版更新说明');
        expect(
          preview.asset.downloadUri.toString(),
          'https://download.myneri.top/releases/preview/id/Neri_v3.0.5_beta7_x64_lite.zip',
        );
        expect(requestedPaths, <String>[
          '/channels.json',
          '/latest/release.json',
          '/channels.json',
          '/preview/release.json',
        ]);
      } finally {
        updater.close();
        await subscription.cancel();
        await server.close(force: true);
      }
    });
  });

  group('resumable update downloads', () {
    test(
      'resumes a partial file and verifies SHA-256 without PowerShell',
      () async {
        final payload = List<int>.generate(32 * 1024, (index) => index % 251);
        final cutoff = 7000;
        final ranges = <String?>[];
        var requestCount = 0;
        final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
        final subscription = server.listen((request) async {
          requestCount++;
          ranges.add(request.headers.value(HttpHeaders.rangeHeader));
          final response = request.response;
          if (requestCount == 1) {
            response.contentLength = payload.length;
            final socket = await response.detachSocket();
            socket.add(payload.sublist(0, cutoff));
            await socket.flush();
            await socket.close();
            return;
          }

          final range = request.headers.value(HttpHeaders.rangeHeader);
          final start = int.parse(
            RegExp(r'^bytes=(\d+)-$').firstMatch(range!)!.group(1)!,
          );
          response.statusCode = HttpStatus.partialContent;
          response.headers.set(
            HttpHeaders.contentRangeHeader,
            'bytes $start-${payload.length - 1}/${payload.length}',
          );
          response.contentLength = payload.length - start;
          response.add(payload.sublist(start));
          await response.close();
        });
        final updater = AppUpdater();
        final uniqueTag =
            'v99.0.0-resume-${DateTime.now().microsecondsSinceEpoch}';
        final release = AppUpdateRelease(
          tag: uniqueTag,
          name: uniqueTag,
          notes: '',
          pageUri: Uri.parse('https://example.test/$uniqueTag'),
          prerelease: true,
          asset: AppUpdateAsset(
            name: 'Neri-test.zip',
            downloadUri: Uri.parse(
              'http://${server.address.address}:${server.port}/Neri-test.zip',
            ),
            sizeBytes: payload.length,
            sha256: sha256.convert(payload).toString(),
          ),
        );
        DownloadedAppUpdate? downloaded;

        try {
          await expectLater(
            updater.downloadUpdate(release),
            throwsA(isA<HttpException>()),
          );

          final progress = <UpdateDownloadProgress>[];
          downloaded = await updater.downloadUpdate(
            release,
            onProgress: progress.add,
          );

          expect(await downloaded.archive.readAsBytes(), payload);
          expect(requestCount, 2);
          expect(ranges, <String?>[null, 'bytes=$cutoff-']);
          expect(progress.any((item) => item.receivedBytes == cutoff), isTrue);

          final reused = await updater.downloadUpdate(release);
          expect(reused.archive.path, downloaded.archive.path);
          expect(requestCount, 2, reason: '完整更新包应直接复用，不应重新请求');
        } finally {
          updater.close();
          await subscription.cancel();
          await server.close(force: true);
          final updateDirectory = downloaded?.updateDirectory;
          if (updateDirectory != null && await updateDirectory.exists()) {
            await updateDirectory.delete(recursive: true);
          }
        }
      },
    );
  });

  test(
    'starts and completes the Windows updater when SHA-256 is absent',
    () async {
      if (!Platform.isWindows) return;

      final root = await Directory.systemTemp.createTemp('neri-updater-test-');
      final payloadDirectory = Directory('${root.path}\\payload');
      final updateDirectory = Directory('${root.path}\\update');
      final installDirectory = Directory('${root.path}\\install');
      final marker = File('${payloadDirectory.path}\\updated.txt');
      final archive = File('${updateDirectory.path}\\Neri-test.zip');
      final updater = AppUpdater();
      Process? blocker;

      try {
        await payloadDirectory.create(recursive: true);
        await updateDirectory.create(recursive: true);
        await installDirectory.create(recursive: true);
        await marker.writeAsString('updated');
        final zipResult = await Process.run('powershell.exe', <String>[
          '-NoLogo',
          '-NoProfile',
          '-NonInteractive',
          '-Command',
          r'''& { param([string]$Source, [string]$Destination) Compress-Archive -Path (Join-Path $Source '*') -DestinationPath $Destination -Force }''',
          payloadDirectory.path,
          archive.path,
        ]);
        expect(zipResult.exitCode, 0, reason: zipResult.stderr.toString());

        blocker = await Process.start('powershell.exe', const <String>[
          '-NoLogo',
          '-NoProfile',
          '-NonInteractive',
          '-Command',
          'Start-Sleep -Seconds 30',
        ]);
        final asset = AppUpdateAsset(
          name: archive.uri.pathSegments.last,
          downloadUri: Uri.parse('https://example.test/Neri-test.zip'),
          sizeBytes: await archive.length(),
        );
        final release = AppUpdateRelease(
          tag: 'v99.0.0-installer-test',
          name: 'installer test',
          notes: '',
          pageUri: Uri.parse('https://example.test/release'),
          prerelease: true,
          asset: asset,
        );

        await updater.launchInstaller(
          update: DownloadedAppUpdate(
            release: release,
            archive: archive,
            updateDirectory: updateDirectory,
          ),
          installDirectory: installDirectory,
          restartExecutable: File('${root.path}\\missing.exe'),
          parentProcessId: blocker.pid,
        );
        blocker.kill();

        final installedMarker = File('${installDirectory.path}\\updated.txt');
        final deadline = DateTime.now().add(const Duration(seconds: 10));
        while (!await installedMarker.exists() &&
            DateTime.now().isBefore(deadline)) {
          await Future<void>.delayed(const Duration(milliseconds: 100));
        }
        expect(await installedMarker.readAsString(), 'updated');
      } finally {
        blocker?.kill();
        updater.close();
        await Future<void>.delayed(const Duration(milliseconds: 300));
        if (await root.exists()) {
          await root.delete(recursive: true);
        }
      }
    },
  );
}
