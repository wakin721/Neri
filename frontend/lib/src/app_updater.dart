import 'dart:async';
import 'dart:convert';
import 'dart:io';

typedef UpdateDownloadProgressCallback =
    void Function(UpdateDownloadProgress progress);

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
    UpdateDownloadProgressCallback? onProgress,
  }) async {
    final updateRoot = _updatesRoot();
    await updateRoot.create(recursive: true);
    final safeTag = _safeFileName(release.tag);
    final updateDirectory = Directory(
      _joinPath(
        updateRoot.path,
        '${safeTag}_${DateTime.now().millisecondsSinceEpoch}_$pid',
      ),
    );
    await updateDirectory.create(recursive: true);
    final archive = File(
      _joinPath(updateDirectory.path, _safeFileName(release.asset.name)),
    );
    final partialArchive = File('${archive.path}.part');
    final sources = _downloadSources(release.asset.downloadUri, mirror);
    final errors = <String>[];

    try {
      for (final source in sources) {
        try {
          if (await partialArchive.exists()) await partialArchive.delete();
          if (await archive.exists()) await archive.delete();
          await _downloadFromSource(
            source,
            release.asset,
            partialArchive,
            onProgress,
          );
          await partialArchive.rename(archive.path);
          await _verifyDigestIfAvailable(archive, release.asset.sha256);
          return DownloadedAppUpdate(
            release: release,
            archive: archive,
            updateDirectory: updateDirectory,
          );
        } catch (error) {
          errors.add('${source.label}：$error');
        }
      }
      throw HttpException('所有下载源均失败：${errors.join('；')}');
    } catch (_) {
      if (await updateDirectory.exists()) {
        await updateDirectory.delete(recursive: true);
      }
      rethrow;
    }
  }

  Future<void> launchInstaller({
    required DownloadedAppUpdate update,
    required Directory installDirectory,
    required File restartExecutable,
  }) async {
    if (!isSupported) {
      throw UnsupportedError('自动安装更新目前仅支持 Windows');
    }
    if (!await update.archive.exists()) {
      throw StateError('下载的更新包不存在：${update.archive.path}');
    }
    final script = File(
      _joinPath(
        Directory.systemTemp.path,
        'Neri-Updater-${DateTime.now().millisecondsSinceEpoch}-$pid.ps1',
      ),
    );
    await script.writeAsString(_windowsUpdaterScript, flush: true);

    try {
      await Process.start(
        'powershell.exe',
        <String>[
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
          pid.toString(),
          '-ArchivePath',
          update.archive.path,
          '-InstallDirectory',
          installDirectory.absolute.path,
          '-RestartExecutable',
          restartExecutable.absolute.path,
          '-UpdateDirectory',
          update.updateDirectory.absolute.path,
          '-ExpectedSha256',
          update.release.asset.sha256 ?? '',
        ],
        workingDirectory: installDirectory.path,
        mode: ProcessStartMode.detached,
      );
    } catch (_) {
      if (await script.exists()) await script.delete();
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
    final request = await _httpClient
        .getUrl(source.uri)
        .timeout(_requestTimeout);
    request.followRedirects = true;
    request.maxRedirects = 8;
    request.headers
      ..set(HttpHeaders.userAgentHeader, _userAgent)
      ..set(HttpHeaders.acceptHeader, 'application/octet-stream');
    if (source.referer != null) {
      request.headers.set(HttpHeaders.refererHeader, source.referer!);
    }
    final response = await request.close().timeout(_requestTimeout);
    if (response.statusCode != HttpStatus.ok) {
      await response.drain<void>();
      throw HttpException('下载服务返回 ${response.statusCode}', uri: source.uri);
    }

    final responseLength = response.contentLength;
    final totalBytes = responseLength > 0
        ? responseLength
        : asset.sizeBytes > 0
        ? asset.sizeBytes
        : null;
    var receivedBytes = 0;
    final sink = destination.openWrite();
    try {
      onProgress?.call(
        UpdateDownloadProgress(
          sourceLabel: source.label,
          receivedBytes: 0,
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
  }

  Future<void> _verifyDigestIfAvailable(File file, String? expected) async {
    if (expected == null || expected.isEmpty) return;
    final result = await Process.run('powershell.exe', <String>[
      '-NoLogo',
      '-NoProfile',
      '-NonInteractive',
      '-Command',
      r'& { param([string]$Path) (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash }',
      file.path,
    ]).timeout(const Duration(minutes: 3));
    if (result.exitCode != 0) {
      throw StateError('无法校验更新包：${result.stderr}');
    }
    final actual = result.stdout.toString().trim().toLowerCase();
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

  List<_DownloadSource> _downloadSources(Uri officialUri, String mirror) {
    final useOfficial = mirror.trim().toLowerCase() == 'official';
    if (useOfficial || officialUri.host.toLowerCase() != 'github.com') {
      return <_DownloadSource>[
        _DownloadSource(label: 'GitHub 官方源', uri: officialUri),
      ];
    }
    return <_DownloadSource>[
      _DownloadSource(
        label: '国内源',
        uri: officialUri.replace(scheme: 'https', host: 'kkgithub.com'),
        referer: 'https://kkgithub.com/',
      ),
      _DownloadSource(label: 'GitHub 官方源', uri: officialUri),
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

String _joinPath(String first, String second) =>
    '$first${Platform.pathSeparator}$second';

const _windowsUpdaterScript = r'''param(
    [Parameter(Mandatory = $true)][int]$ParentProcessId,
    [Parameter(Mandatory = $true)][string]$ArchivePath,
    [Parameter(Mandatory = $true)][string]$InstallDirectory,
    [Parameter(Mandatory = $true)][string]$RestartExecutable,
    [Parameter(Mandatory = $true)][string]$UpdateDirectory,
    [string]$ExpectedSha256 = ''
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
