import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';

typedef UpdateDownloadProgressCallback =
    void Function(UpdateDownloadProgress progress);

const defaultGithubMirrorTemplate = 'https://kkgithub.com';

String normalizeGithubMirrorTemplate(String value) {
  var template = value.trim();
  if (template.isEmpty) {
    throw const FormatException('镜像地址不能为空');
  }
  if (template.length > 500) {
    throw const FormatException('镜像地址不能超过 500 个字符');
  }
  final unknownPlaceholder = RegExp(
    r'\{(?!url\}|path\})[^}]+\}',
  ).firstMatch(template);
  if (unknownPlaceholder != null) {
    throw FormatException('不支持的占位符：${unknownPlaceholder.group(0)}');
  }
  if (!template.contains('{url}') && !template.contains('{path}')) {
    final base = Uri.tryParse(template);
    if (base == null ||
        (base.scheme != 'http' && base.scheme != 'https') ||
        base.host.isEmpty ||
        base.userInfo.isNotEmpty) {
      throw const FormatException('请输入有效且不包含账号密码的 HTTP 或 HTTPS 镜像地址');
    }
  }
  final probe = resolveGithubMirrorUri(
    Uri.parse(
      'https://github.com/wakin721/Neri/releases/download/v1.0.0/Neri.zip',
    ),
    template,
    normalize: false,
  );
  if (probe.scheme != 'http' && probe.scheme != 'https') {
    throw const FormatException('镜像地址仅支持 HTTP 或 HTTPS');
  }
  if (probe.host.isEmpty || probe.userInfo.isNotEmpty) {
    throw const FormatException('请输入有效且不包含账号密码的镜像地址');
  }
  if (!template.contains('{url}') && !template.contains('{path}')) {
    template = template.replaceFirst(RegExp(r'/+$'), '');
  }
  return template;
}

Uri resolveGithubMirrorUri(
  Uri officialUri,
  String template, {
  bool normalize = true,
}) {
  final normalized = normalize
      ? normalizeGithubMirrorTemplate(template)
      : template;
  final officialPath = officialUri.hasQuery
      ? '${officialUri.path}?${officialUri.query}'
      : officialUri.path;
  if (normalized.contains('{url}')) {
    return Uri.parse(normalized.replaceAll('{url}', officialUri.toString()));
  }
  if (normalized.contains('{path}')) {
    return Uri.parse(normalized.replaceAll('{path}', officialPath));
  }

  final base = Uri.parse(normalized);
  if (base.path.isEmpty || base.path == '/') {
    return officialUri.replace(
      scheme: base.scheme,
      host: base.host,
      port: base.hasPort ? base.port : null,
    );
  }
  final prefix = normalized.endsWith('/') ? normalized : '$normalized/';
  return Uri.parse('$prefix${officialUri.toString()}');
}

List<Uri> buildGithubDownloadUris({
  required Uri officialUri,
  required bool useMirrors,
  required Iterable<String> mirrorTemplates,
}) {
  if (!useMirrors || officialUri.host.toLowerCase() != 'github.com') {
    return <Uri>[officialUri];
  }
  final uris = <Uri>[];
  final seen = <String>{officialUri.toString()};
  for (final template in mirrorTemplates) {
    try {
      final mirrorUri = resolveGithubMirrorUri(officialUri, template);
      if (seen.add(mirrorUri.toString())) uris.add(mirrorUri);
    } on FormatException {
      // Ignore malformed entries that may come from manually edited settings.
    }
  }
  uris.add(officialUri);
  return uris;
}

class AppUpdateAsset {
  const AppUpdateAsset({
    required this.name,
    required this.downloadUri,
    required this.sizeBytes,
    this.sha256,
  });

  final String name;
  final Uri downloadUri;
  final int sizeBytes;
  final String? sha256;
}

class AppUpdateRelease {
  const AppUpdateRelease({
    required this.tag,
    required this.name,
    required this.notes,
    required this.pageUri,
    required this.prerelease,
    required this.asset,
  });

  final String tag;
  final String name;
  final String notes;
  final Uri pageUri;
  final bool prerelease;
  final AppUpdateAsset asset;

  String get displayName => name.trim().isEmpty ? tag : name.trim();
}

class UpdateDownloadProgress {
  const UpdateDownloadProgress({
    required this.sourceLabel,
    required this.receivedBytes,
    required this.totalBytes,
  });

  final String sourceLabel;
  final int receivedBytes;
  final int? totalBytes;

  double? get fraction {
    final total = totalBytes;
    if (total == null || total <= 0) return null;
    return (receivedBytes / total).clamp(0.0, 1.0).toDouble();
  }
}

class DownloadedAppUpdate {
  const DownloadedAppUpdate({
    required this.release,
    required this.archive,
    required this.updateDirectory,
  });

  final AppUpdateRelease release;
  final File archive;
  final Directory updateDirectory;
}

class AppUpdater {
  AppUpdater({HttpClient? httpClient})
    : _httpClient = httpClient ?? HttpClient() {
    _httpClient.connectionTimeout = const Duration(seconds: 12);
  }

  static final Uri _releasesUri = Uri.https(
    'api.github.com',
    '/repos/wakin721/Neri/releases',
    const <String, String>{'per_page': '30'},
  );
  static const _userAgent = 'Neri-Desktop-Updater';
  static const _requestTimeout = Duration(seconds: 20);
  static const _streamIdleTimeout = Duration(seconds: 45);

  final HttpClient _httpClient;

  bool get isSupported => Platform.isWindows;

  Future<AppUpdateRelease?> checkForUpdate({
    required String currentVersion,
    required String channel,
  }) async {
    if (!isSupported) return null;
    final installedVersion = _NeriVersion.tryParse(currentVersion);
    if (installedVersion == null) {
      throw FormatException('无法识别当前版本号：$currentVersion');
    }

    await cleanupStaleDownloads();
    final request = await _httpClient
        .getUrl(_releasesUri)
        .timeout(_requestTimeout);
    request.headers
      ..set(HttpHeaders.acceptHeader, 'application/vnd.github+json')
      ..set(HttpHeaders.userAgentHeader, _userAgent)
      ..set('X-GitHub-Api-Version', '2022-11-28');
    final response = await request.close().timeout(_requestTimeout);
    final responseText = await utf8.decoder
        .bind(response.timeout(_streamIdleTimeout))
        .join();
    if (response.statusCode != HttpStatus.ok) {
      final detail = responseText.trim();
      throw HttpException(
        'GitHub 版本服务返回 ${response.statusCode}'
        '${detail.isEmpty ? '' : '：${_shorten(detail, 240)}'}',
        uri: _releasesUri,
      );
    }

    final decoded = jsonDecode(responseText);
    if (decoded is! List) {
      throw const FormatException('GitHub 版本响应格式无效');
    }
    final includePrereleases = channel.trim().toLowerCase() != 'release';
    final candidates = <_ReleaseCandidate>[];
    for (final item in decoded) {
      if (item is! Map) continue;
      final release = Map<String, dynamic>.from(item);
      if (release['draft'] == true) continue;
      final prerelease = release['prerelease'] == true;
      if (!includePrereleases && prerelease) continue;
      final tag = release['tag_name']?.toString().trim() ?? '';
      final version = _NeriVersion.tryParse(tag);
      if (tag.isEmpty || version == null) continue;
      candidates.add(
        _ReleaseCandidate(
          data: release,
          version: version,
          prerelease: prerelease,
        ),
      );
    }
    candidates.sort((left, right) => right.version.compareTo(left.version));

    final newerCandidates = candidates
        .where((candidate) => candidate.version.compareTo(installedVersion) > 0)
        .toList(growable: false);
    if (newerCandidates.isEmpty) return null;

    for (final candidate in newerCandidates) {
      final asset = _selectWindowsAsset(candidate.data['assets']);
      if (asset == null) continue;
      final pageUri = Uri.tryParse(
        candidate.data['html_url']?.toString() ?? '',
      );
      return AppUpdateRelease(
        tag: candidate.data['tag_name']?.toString() ?? '',
        name: candidate.data['name']?.toString() ?? '',
        notes: candidate.data['body']?.toString() ?? '',
        pageUri:
            pageUri ??
            Uri.https(
              'github.com',
              '/wakin721/Neri/releases/tag/${candidate.data['tag_name']}',
            ),
        prerelease: candidate.prerelease,
        asset: asset,
      );
    }
    throw const FormatException(
      '发现新版本，但 Release 中没有可用的 Windows .zip 或 .7z 更新包',
    );
  }

  Future<DownloadedAppUpdate> downloadUpdate(
    AppUpdateRelease release, {
    required String mirror,
    List<String> mirrorTemplates = const <String>[defaultGithubMirrorTemplate],
    UpdateDownloadProgressCallback? onProgress,
  }) async {
    final updateRoot = _updatesRoot();
    await updateRoot.create(recursive: true);
    final safeTag = _safeFileName(release.tag);
    final updateDirectory = Directory(
      _joinPath(updateRoot.path, _stableDownloadDirectoryName(release)),
    );
    await updateDirectory.create(recursive: true);
    final archive = File(
      _joinPath(updateDirectory.path, _safeFileName(release.asset.name)),
    );
    final partialArchive = File('${archive.path}.part');
    await _adoptReusableDownload(
      updateRoot: updateRoot,
      updateDirectory: updateDirectory,
      safeTag: safeTag,
      asset: release.asset,
      archive: archive,
      partialArchive: partialArchive,
    );
    if (await _isCompleteDownload(archive, release.asset)) {
      return DownloadedAppUpdate(
        release: release,
        archive: archive,
        updateDirectory: updateDirectory,
      );
    }
    final sources = _downloadSources(
      release.asset.downloadUri,
      mirror,
      mirrorTemplates,
    );
    final errors = <String>[];

    for (final source in sources) {
      try {
        await _downloadFromSource(
          source,
          release.asset,
          partialArchive,
          onProgress,
        );
        if (await archive.exists()) await archive.delete();
        await partialArchive.rename(archive.path);
        await _verifyDigestIfAvailable(archive, release.asset.sha256);
        return DownloadedAppUpdate(
          release: release,
          archive: archive,
          updateDirectory: updateDirectory,
        );
      } catch (error) {
        errors.add('${source.label}：$error');
        if (await archive.exists()) {
          try {
            await archive.delete();
          } catch (_) {}
        }
      }
    }
    throw HttpException('所有下载源均失败：${errors.join('；')}。已保留下载进度，下次可继续下载。');
  }

  Future<void> launchInstaller({
    required DownloadedAppUpdate update,
    required Directory installDirectory,
    required File restartExecutable,
    int? parentProcessId,
  }) async {
    if (!isSupported) {
      throw UnsupportedError('自动安装更新目前仅支持 Windows');
    }
    if (!await update.archive.exists()) {
      throw StateError('下载的更新包不存在：${update.archive.path}');
    }
    final powershellExecutable = _resolveWindowsPowerShellExecutable();
    final script = File(
      _joinPath(
        Directory.systemTemp.path,
        'Neri-Updater-${DateTime.now().millisecondsSinceEpoch}-$pid.ps1',
      ),
    );
    final readyFile = File('${script.path}.ready');
    await script.writeAsBytes(<int>[
      0xef,
      0xbb,
      0xbf,
      ...utf8.encode(_windowsUpdaterScript),
    ], flush: true);

    try {
      final arguments = <String>[
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-WindowStyle',
        'Hidden',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        script.path,
        '-ParentProcessId',
        (parentProcessId ?? pid).toString(),
        '-ArchivePath',
        update.archive.path,
        '-InstallDirectory',
        installDirectory.absolute.path,
        '-RestartExecutable',
        restartExecutable.absolute.path,
        '-UpdateDirectory',
        update.updateDirectory.absolute.path,
      ];
      final expectedSha256 = update.release.asset.sha256?.trim() ?? '';
      if (expectedSha256.isNotEmpty) {
        arguments.addAll(<String>['-ExpectedSha256', expectedSha256]);
      }
      arguments.addAll(<String>['-ReadyPath', readyFile.path]);

      final process = await Process.start(
        powershellExecutable,
        arguments,
        workingDirectory: installDirectory.path,
        mode: ProcessStartMode.normal,
      );
      final stdoutFuture = process.stdout
          .transform(systemEncoding.decoder)
          .join();
      final stderrFuture = process.stderr
          .transform(systemEncoding.decoder)
          .join();
      final outcome = await Future.any<({bool ready, int? exitCode})>(<
        Future<({bool ready, int? exitCode})>
      >[
        _waitForFile(
          readyFile,
          const Duration(seconds: 8),
        ).then((ready) => (ready: ready, exitCode: null)),
        process.exitCode.then((exitCode) => (ready: false, exitCode: exitCode)),
      ]);
      if (!outcome.ready) {
        if (outcome.exitCode == null) {
          process.kill();
        }
        final output = <String>[
          await stderrFuture.timeout(
            const Duration(seconds: 2),
            onTimeout: () => '',
          ),
          await stdoutFuture.timeout(
            const Duration(seconds: 2),
            onTimeout: () => '',
          ),
        ].join('\n').trim();
        final reason = outcome.exitCode == null
            ? '更新脚本启动超时'
            : '更新脚本启动失败（退出码 ${outcome.exitCode}）';
        throw StateError(
          '$reason，程序将保持运行'
          '${output.isEmpty ? '' : '：${_shorten(output, 800)}'}',
        );
      }
    } catch (_) {
      if (await script.exists()) await script.delete();
      if (await readyFile.exists()) await readyFile.delete();
      rethrow;
    }
  }

  Future<void> cleanupStaleDownloads() async {
    final root = _updatesRoot();
    if (!await root.exists()) return;
    final cutoff = DateTime.now().subtract(const Duration(days: 3));
    await for (final entity in root.list(followLinks: false)) {
      try {
        final modified = await entity.stat().then((stat) => stat.modified);
        if (modified.isBefore(cutoff)) {
          await entity.delete(recursive: true);
        }
      } catch (_) {
        // A running updater may still own this directory; retry next startup.
      }
    }
  }

  void close() => _httpClient.close(force: true);

  Future<void> _downloadFromSource(
    _DownloadSource source,
    AppUpdateAsset asset,
    File destination,
    UpdateDownloadProgressCallback? onProgress,
  ) async {
    var existingBytes = await destination.exists()
        ? await destination.length()
        : 0;
    if (asset.sizeBytes > 0 && existingBytes > asset.sizeBytes) {
      await destination.writeAsBytes(const <int>[]);
      existingBytes = 0;
    }
    if (asset.sizeBytes > 0 && existingBytes == asset.sizeBytes) {
      onProgress?.call(
        UpdateDownloadProgress(
          sourceLabel: source.label,
          receivedBytes: existingBytes,
          totalBytes: asset.sizeBytes,
        ),
      );
      return;
    }

    for (var attempt = 0; attempt < 2; attempt++) {
      final request = await _httpClient
          .getUrl(source.uri)
          .timeout(_requestTimeout);
      request.followRedirects = true;
      request.maxRedirects = 8;
      request.headers
        ..set(HttpHeaders.userAgentHeader, _userAgent)
        ..set(HttpHeaders.acceptHeader, 'application/octet-stream');
      if (existingBytes > 0) {
        request.headers.set(HttpHeaders.rangeHeader, 'bytes=$existingBytes-');
      }
      if (source.referer != null) {
        request.headers.set(HttpHeaders.refererHeader, source.referer!);
      }
      final response = await request.close().timeout(_requestTimeout);

      if (response.statusCode == HttpStatus.requestedRangeNotSatisfiable &&
          existingBytes > 0) {
        await response.drain<void>();
        if (asset.sizeBytes > 0 && existingBytes == asset.sizeBytes) return;
        await destination.writeAsBytes(const <int>[]);
        existingBytes = 0;
        continue;
      }
      if (response.statusCode != HttpStatus.ok &&
          response.statusCode != HttpStatus.partialContent) {
        await response.drain<void>();
        throw HttpException('下载服务返回 ${response.statusCode}', uri: source.uri);
      }

      var append = response.statusCode == HttpStatus.partialContent;
      if (append) {
        final contentRange = response.headers.value(
          HttpHeaders.contentRangeHeader,
        );
        final rangeStart = _contentRangeStart(contentRange);
        if (rangeStart != existingBytes) {
          await response.drain<void>();
          await destination.writeAsBytes(const <int>[]);
          existingBytes = 0;
          continue;
        }
      } else {
        // The server ignored Range. Restart cleanly with its full response.
        existingBytes = 0;
      }

      final responseLength = response.contentLength;
      final totalBytes = asset.sizeBytes > 0
          ? asset.sizeBytes
          : responseLength > 0
          ? existingBytes + responseLength
          : null;
      var receivedBytes = existingBytes;
      final sink = destination.openWrite(
        mode: append ? FileMode.append : FileMode.write,
      );
      try {
        onProgress?.call(
          UpdateDownloadProgress(
            sourceLabel: source.label,
            receivedBytes: receivedBytes,
            totalBytes: totalBytes,
          ),
        );
        await for (final chunk in response.timeout(_streamIdleTimeout)) {
          sink.add(chunk);
          receivedBytes += chunk.length;
          onProgress?.call(
            UpdateDownloadProgress(
              sourceLabel: source.label,
              receivedBytes: receivedBytes,
              totalBytes: totalBytes,
            ),
          );
        }
        await sink.flush();
      } finally {
        await sink.close();
      }
      if (receivedBytes <= 0) {
        throw const HttpException('下载内容为空');
      }
      if (asset.sizeBytes > 0 && receivedBytes != asset.sizeBytes) {
        throw HttpException(
          '文件大小校验失败（应为 ${asset.sizeBytes} 字节，实际 $receivedBytes 字节）',
        );
      }
      return;
    }
    throw const HttpException('下载服务器返回了无效的断点续传响应');
  }

  Future<bool> _isCompleteDownload(File archive, AppUpdateAsset asset) async {
    if (!await archive.exists()) return false;
    if (asset.sizeBytes > 0 && await archive.length() != asset.sizeBytes) {
      final length = await archive.length();
      if (length > 0 && length < asset.sizeBytes) {
        final partial = File('${archive.path}.part');
        if (!await partial.exists() || await partial.length() < length) {
          if (await partial.exists()) await partial.delete();
          await archive.rename(partial.path);
        } else {
          await archive.delete();
        }
      } else {
        await archive.delete();
      }
      return false;
    }
    try {
      await _verifyDigestIfAvailable(archive, asset.sha256);
      return true;
    } catch (_) {
      await archive.delete();
      return false;
    }
  }

  Future<void> _adoptReusableDownload({
    required Directory updateRoot,
    required Directory updateDirectory,
    required String safeTag,
    required AppUpdateAsset asset,
    required File archive,
    required File partialArchive,
  }) async {
    final candidates = <File>[];
    await for (final entity in updateRoot.list(followLinks: false)) {
      if (entity is! Directory) continue;
      final name = _pathBaseName(entity.path);
      if (entity.path != updateDirectory.path &&
          !name.startsWith('${safeTag}_')) {
        continue;
      }
      for (final candidateName in <String>[
        _safeFileName(asset.name),
        '${_safeFileName(asset.name)}.part',
      ]) {
        final candidate = File(_joinPath(entity.path, candidateName));
        if (await candidate.exists()) candidates.add(candidate);
      }
    }
    if (candidates.isEmpty) return;
    candidates.sort((left, right) {
      final leftLength = left.lengthSync();
      final rightLength = right.lengthSync();
      return rightLength.compareTo(leftLength);
    });
    final candidate = candidates.first;
    final candidateLength = await candidate.length();
    if (asset.sizeBytes > 0 && candidateLength > asset.sizeBytes) return;
    final isComplete =
        asset.sizeBytes > 0 && candidateLength == asset.sizeBytes;
    final destination = isComplete ? archive : partialArchive;
    if (candidate.path == destination.path) return;
    if (await destination.exists() &&
        await destination.length() >= candidateLength) {
      return;
    }
    if (await destination.exists()) await destination.delete();
    try {
      await candidate.rename(destination.path);
    } on FileSystemException {
      await candidate.copy(destination.path);
    }
  }

  Future<void> _verifyDigestIfAvailable(File file, String? expected) async {
    if (expected == null || expected.isEmpty) return;
    final actual = (await sha256
            .bind(file.openRead())
            .first
            .timeout(const Duration(minutes: 3)))
        .toString()
        .toLowerCase();
    if (actual != expected.toLowerCase()) {
      throw StateError('更新包 SHA-256 校验失败');
    }
  }

  AppUpdateAsset? _selectWindowsAsset(dynamic rawAssets) {
    if (rawAssets is! List) return null;
    final assets = <({AppUpdateAsset asset, int score})>[];
    for (final item in rawAssets) {
      if (item is! Map) continue;
      final data = Map<String, dynamic>.from(item);
      final name = data['name']?.toString().trim() ?? '';
      final lowerName = name.toLowerCase();
      if (!lowerName.endsWith('.zip') && !lowerName.endsWith('.7z')) continue;
      if (lowerName.contains('source') ||
          lowerName.contains('symbols') ||
          lowerName.contains('debug')) {
        continue;
      }
      final uri = Uri.tryParse(data['browser_download_url']?.toString() ?? '');
      if (name.isEmpty || uri == null || !uri.hasScheme) continue;
      final digest = data['digest']?.toString().trim().toLowerCase() ?? '';
      final sha256 = RegExp(
        r'^sha256:([0-9a-f]{64})$',
      ).firstMatch(digest)?.group(1);
      var score = lowerName.endsWith('.zip') ? 5 : 0;
      if (lowerName.contains('neri')) score += 4;
      if (lowerName.contains('windows') || lowerName.contains('win')) {
        score += 3;
      }
      if (lowerName.contains('x64') || lowerName.contains('amd64')) score += 2;
      if (lowerName.contains('portable')) score += 1;
      assets.add((
        asset: AppUpdateAsset(
          name: name,
          downloadUri: uri,
          sizeBytes: _intValue(data['size']),
          sha256: sha256,
        ),
        score: score,
      ));
    }
    if (assets.isEmpty) return null;
    assets.sort((left, right) => right.score.compareTo(left.score));
    return assets.first.asset;
  }

  List<_DownloadSource> _downloadSources(
    Uri officialUri,
    String mirror,
    List<String> mirrorTemplates,
  ) {
    final useOfficial = mirror.trim().toLowerCase() == 'official';
    final uris = buildGithubDownloadUris(
      officialUri: officialUri,
      useMirrors: !useOfficial,
      mirrorTemplates: mirrorTemplates,
    );
    return <_DownloadSource>[
      for (var index = 0; index < uris.length; index++)
        if (uris[index] == officialUri)
          _DownloadSource(label: 'GitHub 官方源', uri: officialUri)
        else
          _DownloadSource(
            label: '国内镜像 ${index + 1}（${uris[index].host}）',
            uri: uris[index],
            referer: '${uris[index].scheme}://${uris[index].authority}/',
          ),
    ];
  }

  Directory _updatesRoot() => Directory(
    _joinPath(Directory.systemTemp.path, _joinPath('Neri', 'updates')),
  );
}

int compareNeriVersions(String left, String right) {
  final leftVersion = _NeriVersion.tryParse(left);
  final rightVersion = _NeriVersion.tryParse(right);
  if (leftVersion == null || rightVersion == null) {
    throw FormatException('无法比较版本号：$left / $right');
  }
  return leftVersion.compareTo(rightVersion);
}

class _ReleaseCandidate {
  const _ReleaseCandidate({
    required this.data,
    required this.version,
    required this.prerelease,
  });

  final Map<String, dynamic> data;
  final _NeriVersion version;
  final bool prerelease;
}

class _DownloadSource {
  const _DownloadSource({required this.label, required this.uri, this.referer});

  final String label;
  final Uri uri;
  final String? referer;
}

class _NeriVersion implements Comparable<_NeriVersion> {
  const _NeriVersion(this.major, this.minor, this.patch, this.prereleaseTokens);

  static final _versionPattern = RegExp(
    r'(\d+)\.(\d+)(?:\.(\d+))?(?:[-_]?([0-9A-Za-z][0-9A-Za-z.-]*))?',
  );
  static final _tokenPattern = RegExp(r'[A-Za-z]+|\d+');

  final int major;
  final int minor;
  final int patch;
  final List<Object> prereleaseTokens;

  static _NeriVersion? tryParse(String value) {
    final match = _versionPattern.firstMatch(value.trim());
    if (match == null) return null;
    final rawPrerelease = match.group(4)?.split('+').first.trim() ?? '';
    final prerelease =
        const <String>{
          'release',
          'stable',
          'final',
        }.contains(rawPrerelease.toLowerCase())
        ? ''
        : rawPrerelease;
    final tokens = <Object>[
      for (final tokenMatch in _tokenPattern.allMatches(prerelease))
        int.tryParse(tokenMatch.group(0)!) ??
            tokenMatch.group(0)!.toLowerCase(),
    ];
    return _NeriVersion(
      int.parse(match.group(1)!),
      int.parse(match.group(2)!),
      int.tryParse(match.group(3) ?? '') ?? 0,
      tokens,
    );
  }

  @override
  int compareTo(_NeriVersion other) {
    for (final pair in <(int, int)>[
      (major, other.major),
      (minor, other.minor),
      (patch, other.patch),
    ]) {
      final comparison = pair.$1.compareTo(pair.$2);
      if (comparison != 0) return comparison;
    }
    final stable = prereleaseTokens.isEmpty;
    final otherStable = other.prereleaseTokens.isEmpty;
    if (stable != otherStable) return stable ? 1 : -1;
    for (
      var index = 0;
      index < prereleaseTokens.length && index < other.prereleaseTokens.length;
      index++
    ) {
      final left = prereleaseTokens[index];
      final right = other.prereleaseTokens[index];
      if (left == right) continue;
      if (left is int && right is int) return left.compareTo(right);
      if (left is int) return -1;
      if (right is int) return 1;
      return left.toString().compareTo(right.toString());
    }
    return prereleaseTokens.length.compareTo(other.prereleaseTokens.length);
  }
}

int _intValue(dynamic value) {
  if (value is int) return value;
  if (value is num) return value.toInt();
  return int.tryParse(value?.toString() ?? '') ?? 0;
}

String _shorten(String value, int maxLength) {
  if (value.length <= maxLength) return value;
  return '${value.substring(0, maxLength)}…';
}

String _safeFileName(String value) {
  final normalized = value.trim().replaceAll(RegExp(r'[^0-9A-Za-z._-]+'), '_');
  return normalized.isEmpty ? 'update.zip' : normalized;
}

String _stableDownloadDirectoryName(AppUpdateRelease release) {
  final safeTag = _safeFileName(release.tag);
  final displayTag = safeTag.length > 60 ? safeTag.substring(0, 60) : safeTag;
  final identity =
      '${release.tag}\n${release.asset.name}\n${release.asset.downloadUri}';
  var hash = 0x811c9dc5;
  for (final codeUnit in identity.codeUnits) {
    hash ^= codeUnit;
    hash = (hash * 0x01000193) & 0xffffffff;
  }
  return '${displayTag}_${hash.toRadixString(16).padLeft(8, '0')}';
}

int? _contentRangeStart(String? value) {
  if (value == null) return null;
  final match = RegExp(
    r'^bytes\s+(\d+)-\d+/(?:\d+|\*)$',
    caseSensitive: false,
  ).firstMatch(value.trim());
  return match == null ? null : int.tryParse(match.group(1)!);
}

String _pathBaseName(String path) {
  final normalized = path.replaceAll(RegExp(r'[\\/]+$'), '');
  final separator = normalized.lastIndexOf(RegExp(r'[\\/]'));
  return separator < 0 ? normalized : normalized.substring(separator + 1);
}

Future<bool> _waitForFile(File file, Duration timeout) async {
  final deadline = DateTime.now().add(timeout);
  while (DateTime.now().isBefore(deadline)) {
    if (await file.exists()) return true;
    await Future<void>.delayed(const Duration(milliseconds: 100));
  }
  return file.exists();
}

String _joinPath(String first, String second) =>
    '$first${Platform.pathSeparator}$second';

String _resolveWindowsPowerShellExecutable() {
  final environment = Platform.environment;
  final windowsRoots = <String>{
    environment['SystemRoot']?.trim() ?? '',
    environment['WINDIR']?.trim() ?? '',
    r'C:\Windows',
  }..removeWhere((value) => value.isEmpty);
  final candidates = <File>[
    for (final root in windowsRoots)
      for (final relativePath in const <String>[
        r'Sysnative\WindowsPowerShell\v1.0\powershell.exe',
        r'System32\WindowsPowerShell\v1.0\powershell.exe',
      ])
        File('${root.replaceFirst(RegExp(r'[\\/]+$'), '')}\\$relativePath'),
    for (final programFiles in <String?>[
      environment['ProgramW6432'],
      environment['ProgramFiles'],
      environment['ProgramFiles(x86)'],
    ])
      if (programFiles != null && programFiles.trim().isNotEmpty)
        File(
          '${programFiles.trim().replaceFirst(RegExp(r'[\\/]+$'), '')}'
          r'\PowerShell\7\pwsh.exe',
        ),
  ];
  for (final candidate in candidates) {
    if (candidate.existsSync()) return candidate.absolute.path;
  }
  throw StateError(
    '找不到 Windows PowerShell，无法启动自动更新安装程序。'
    r'请确认 C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe 存在。',
  );
}

const _windowsUpdaterScript = r'''param(
    [Parameter(Mandatory = $true)][int]$ParentProcessId,
    [Parameter(Mandatory = $true)][string]$ArchivePath,
    [Parameter(Mandatory = $true)][string]$InstallDirectory,
    [Parameter(Mandatory = $true)][string]$RestartExecutable,
    [Parameter(Mandatory = $true)][string]$UpdateDirectory,
    [string]$ExpectedSha256 = '',
    [Parameter(Mandatory = $true)][string]$ReadyPath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$LogDirectory = Join-Path $InstallDirectory 'logs'
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LogPath = Join-Path $LogDirectory 'update.log'
$Succeeded = $false

function Write-UpdateLog([string]$Message) {
    try {
        Add-Content -LiteralPath $LogPath -Encoding UTF8 -Value ('{0:yyyy-MM-dd HH:mm:ss} {1}' -f (Get-Date), $Message)
    } catch {}
}

try {
    Set-Content -LiteralPath $ReadyPath -Encoding ASCII -Force -Value $PID
    Write-UpdateLog "更新脚本已启动（PID $PID）"
} catch {
    Write-UpdateLog "无法创建更新脚本启动标记：$($_.Exception.Message)"
    throw
}

function Expand-UpdateArchive([string]$Source, [string]$Destination) {
    $Extension = [System.IO.Path]::GetExtension($Source).ToLowerInvariant()
    if ($Extension -eq '.zip') {
        Expand-Archive -LiteralPath $Source -DestinationPath $Destination -Force
        return
    }
    if ($Extension -ne '.7z') {
        throw "不支持的更新包格式：$Extension"
    }

    $Candidates = @(
        (Join-Path $InstallDirectory 'toolkit\7z.exe'),
        (Join-Path ([Environment]::GetFolderPath('ProgramFiles')) '7-Zip\7z.exe'),
        (Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) '7-Zip\7z.exe')
    )
    $SevenZipCommand = Get-Command '7z.exe' -ErrorAction SilentlyContinue
    if ($null -ne $SevenZipCommand) {
        $Candidates += $SevenZipCommand.Source
    }
    $SevenZip = $Candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
    if ($SevenZip) {
        & $SevenZip x '-y' "-o$Destination" $Source | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip 解压失败，退出码：$LASTEXITCODE"
        }
        return
    }

    $TarCommand = Get-Command 'tar.exe' -ErrorAction SilentlyContinue
    if ($null -eq $TarCommand) {
        throw '未找到可解压 .7z 的 7-Zip 或 Windows tar.exe'
    }
    & $TarCommand.Source '-xf' $Source '-C' $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe 解压失败，退出码：$LASTEXITCODE"
    }
}

try {
    Write-UpdateLog "等待 Neri 进程退出（PID $ParentProcessId）"
    $Deadline = [DateTime]::UtcNow.AddSeconds(30)
    while (Get-Process -Id $ParentProcessId -ErrorAction SilentlyContinue) {
        if ([DateTime]::UtcNow -ge $Deadline) {
            Write-UpdateLog '正常退出超时，将结束主程序进程'
            Stop-Process -Id $ParentProcessId -Force -ErrorAction SilentlyContinue
            break
        }
        Start-Sleep -Milliseconds 250
    }
    Start-Sleep -Milliseconds 600

    if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        throw "更新包不存在：$ArchivePath"
    }
    if ($ExpectedSha256) {
        $ActualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath).Hash.ToLowerInvariant()
        if ($ActualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
            throw '更新包 SHA-256 校验失败'
        }
    }

    $ExtractDirectory = Join-Path $UpdateDirectory 'extracted'
    if (Test-Path -LiteralPath $ExtractDirectory) {
        Remove-Item -LiteralPath $ExtractDirectory -Recurse -Force
    }
    New-Item -ItemType Directory -Path $ExtractDirectory -Force | Out-Null
    Write-UpdateLog "正在解压 $ArchivePath"
    Expand-UpdateArchive -Source $ArchivePath -Destination $ExtractDirectory

    $TopLevelItems = @(Get-ChildItem -LiteralPath $ExtractDirectory -Force)
    if ($TopLevelItems.Count -eq 0) {
        throw '更新包解压后为空'
    }
    $TopLevelFiles = @($TopLevelItems | Where-Object { -not $_.PSIsContainer })
    $TopLevelDirectories = @($TopLevelItems | Where-Object { $_.PSIsContainer })
    $PayloadDirectory = $ExtractDirectory
    if ($TopLevelFiles.Count -eq 0 -and $TopLevelDirectories.Count -eq 1) {
        $PayloadDirectory = $TopLevelDirectories[0].FullName
    }

    Write-UpdateLog "正在覆盖安装目录 $InstallDirectory"
    $PreservedNames = @('temp', 'logs')
    foreach ($Item in Get-ChildItem -LiteralPath $PayloadDirectory -Force) {
        if ($PreservedNames -contains $Item.Name.ToLowerInvariant()) {
            continue
        }
        Copy-Item -LiteralPath $Item.FullName -Destination $InstallDirectory -Recurse -Force
    }
    $Succeeded = $true
    Write-UpdateLog '更新安装完成'
} catch {
    $Message = $_.Exception.Message
    Write-UpdateLog "更新失败：$Message"
    try {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show(
            "Neri 更新失败：`n`n$Message`n`n详情请查看：$LogPath",
            'Neri 更新失败',
            'OK',
            'Error'
        ) | Out-Null
    } catch {}
} finally {
    try {
        Remove-Item -LiteralPath $ReadyPath -Force -ErrorAction SilentlyContinue
    } catch {}
    try {
        if (Test-Path -LiteralPath $UpdateDirectory) {
            Remove-Item -LiteralPath $UpdateDirectory -Recurse -Force
        }
    } catch {
        Write-UpdateLog "清理临时更新文件失败：$($_.Exception.Message)"
    }
}

if ($Succeeded) {
    try {
        Start-Process -FilePath $RestartExecutable -WorkingDirectory $InstallDirectory
    } catch {
        Write-UpdateLog "更新完成，但重新启动失败：$($_.Exception.Message)"
    }
}

try {
    Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue
} catch {}
''';
