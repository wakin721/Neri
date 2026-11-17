import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';

import '../models/job.dart';
import '../utils/detection_box_label.dart';
import '../utils/preview_decode_size.dart';
import '../utils/validation_video_session.dart';
import '../utils/video_detection_index.dart';

typedef ValidationVideoBackendFactory = ValidationVideoBackend Function();

const _viewerImageTypes = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'};
const _viewerVideoTypes = {
  'mp4',
  'avi',
  'mov',
  'mkv',
  'wmv',
  'flv',
  'webm',
  'm4v',
  'ts',
};

class DetectionMediaViewer extends StatelessWidget {
  const DetectionMediaViewer({
    required this.item,
    required this.visibleBoxes,
    required this.showDetections,
    required this.onOpenExternal,
    this.isFavorite = false,
    this.onToggleFavorite,
    this.selectedObservationId,
    this.onDetectionBoxSelected,
    @visibleForTesting this.videoBackendFactory,
    @visibleForTesting this.videoLogger,
    super.key,
  });

  final DetectionItem item;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final VoidCallback onOpenExternal;
  final bool isFavorite;
  final VoidCallback? onToggleFavorite;
  final String? selectedObservationId;
  final ValueChanged<DetectionBox?>? onDetectionBoxSelected;
  final ValidationVideoBackendFactory? videoBackendFactory;
  final ValidationVideoLogger? videoLogger;

  bool get _isImage {
    return _viewerImageTypes.contains(item.fileType.toLowerCase());
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final showFavoriteButton = onToggleFavorite != null;
    final favoriteLabel = _isImage ? '照片' : '视频';
    // 彻底移除 Semantics(container: true)，因为它会强行抓取内部高速变换的进度条和 YOLO 框，导致 AXTree 崩溃
    return ExcludeSemantics(
      child: ClipRRect(
        borderRadius: BorderRadius.circular(6),
        child: ColoredBox(
          color: Theme.of(context).colorScheme.surfaceContainerHighest,
          child: Stack(
            fit: StackFit.expand,
            children: [
              _MediaContent(
                item: item,
                visibleBoxes: visibleBoxes,
                showDetections: showDetections,
                onOpenExternal: onOpenExternal,
                selectedObservationId: selectedObservationId,
                onDetectionBoxSelected: onDetectionBoxSelected,
                videoBackendFactory: videoBackendFactory,
                videoLogger: videoLogger,
              ),
              Positioned(
                top: 12,
                right: 12,
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (showFavoriteButton) ...[
                      IconButton.filledTonal(
                        tooltip: isFavorite
                            ? '取消收藏$favoriteLabel'
                            : '收藏$favoriteLabel',
                        icon: Icon(
                          isFavorite
                              ? Icons.star_rounded
                              : Icons.star_border_rounded,
                        ),
                        style: isFavorite
                            ? IconButton.styleFrom(
                                backgroundColor: scheme.primaryContainer,
                                foregroundColor: scheme.onPrimaryContainer,
                              )
                            : null,
                        onPressed: onToggleFavorite,
                      ),
                      const SizedBox(width: 8),
                    ],
                    IconButton.filledTonal(
                      tooltip: '使用系统应用打开',
                      icon: const Icon(Icons.open_in_new_rounded),
                      onPressed: onOpenExternal,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _MediaContent extends StatelessWidget {
  const _MediaContent({
    required this.item,
    required this.visibleBoxes,
    required this.showDetections,
    required this.onOpenExternal,
    this.selectedObservationId,
    this.onDetectionBoxSelected,
    this.videoBackendFactory,
    this.videoLogger,
  });

  final DetectionItem item;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final VoidCallback onOpenExternal;
  final String? selectedObservationId;
  final ValueChanged<DetectionBox?>? onDetectionBoxSelected;
  final ValidationVideoBackendFactory? videoBackendFactory;
  final ValidationVideoLogger? videoLogger;

  bool get isVideo {
    return _viewerVideoTypes.contains(item.fileType.toLowerCase());
  }

  @override
  Widget build(BuildContext context) {
    if (isVideo) {
      return ValidationVideoPlayer(
        path: item.path,
        visibleBoxes: visibleBoxes,
        showDetections: showDetections,
        detectionData: item.detectionData,
        onOpenExternal: onOpenExternal,
        selectedObservationId: selectedObservationId,
        onDetectionBoxSelected: onDetectionBoxSelected,
        backendFactory: videoBackendFactory,
        logger: videoLogger,
      );
    } else {
      return _ImageMediaViewer(
        path: item.path,
        visibleBoxes: visibleBoxes,
        showDetections: showDetections,
        selectedObservationId: selectedObservationId,
        onDetectionBoxSelected: onDetectionBoxSelected,
      );
    }
  }
}

class _ImageMediaViewer extends StatefulWidget {
  const _ImageMediaViewer({
    required this.path,
    required this.visibleBoxes,
    required this.showDetections,
    this.selectedObservationId,
    this.onDetectionBoxSelected,
  });

  final String path;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final String? selectedObservationId;
  final ValueChanged<DetectionBox?>? onDetectionBoxSelected;

  @override
  State<_ImageMediaViewer> createState() => _ImageMediaViewerState();
}

class _ImageMediaViewerState extends State<_ImageMediaViewer> {
  Size? _imageSize;
  Object? _imageError;
  int _sizeRequestId = 0;

  @override
  void initState() {
    super.initState();
    _resolveImageSize();
  }

  @override
  void didUpdateWidget(covariant _ImageMediaViewer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.path != widget.path) {
      _resolveImageSize();
    }
  }

  @override
  void dispose() {
    _sizeRequestId++;
    super.dispose();
  }

  void _resolveImageSize() {
    final requestId = ++_sizeRequestId;
    // Backend dimensions may describe the unrotated EXIF image. Read the
    // display dimensions from the same codec used by Flutter's Image widget.
    _imageSize = null;
    _imageError = null;
    unawaited(_readImageSize(widget.path, requestId));
  }

  Future<void> _readImageSize(String path, int requestId) async {
    ui.ImmutableBuffer? buffer;
    ui.ImageDescriptor? descriptor;
    try {
      // Read original dimensions without decoding a full-resolution bitmap.
      buffer = await ui.ImmutableBuffer.fromFilePath(path);
      descriptor = await ui.ImageDescriptor.encoded(buffer);
      final size = Size(
        descriptor.width.toDouble(),
        descriptor.height.toDouble(),
      );
      if (!mounted || requestId != _sizeRequestId) return;
      setState(() => _imageSize = size);
    } catch (error) {
      if (!mounted || requestId != _sizeRequestId) return;
      setState(() => _imageError = error);
    } finally {
      descriptor?.dispose();
      buffer?.dispose();
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_imageError != null) {
      return _MissingImagePlaceholder(path: widget.path);
    }
    final imageSize = _imageSize;
    if (imageSize == null) {
      return const Center(child: CircularProgressIndicator());
    }
    return LayoutBuilder(
      builder: (context, constraints) {
        final viewport = Size(constraints.maxWidth, constraints.maxHeight);
        final imageProvider = ResizeImage(
          FileImage(File(widget.path)),
          width: previewDecodeWidth(
            imageSize,
            viewport,
            MediaQuery.devicePixelRatioOf(context),
          ),
          policy: ResizeImagePolicy.fit,
        );
        return GestureDetector(
          behavior: HitTestBehavior.translucent,
          onTapUp: widget.onDetectionBoxSelected == null || _imageSize == null
              ? null
              : (details) {
                  final selected = widget.showDetections
                      ? _hitTestDinoBox(
                          details.localPosition,
                          viewport,
                          _imageSize!,
                          widget.visibleBoxes,
                        )
                      : null;
                  widget.onDetectionBoxSelected!(selected);
                },
          child: Stack(
            fit: StackFit.expand,
            children: [
              Image(
                image: imageProvider,
                key: ValueKey(widget.path),
                fit: BoxFit.contain,
                errorBuilder: (context, error, stackTrace) =>
                    _MissingImagePlaceholder(path: widget.path),
              ),
              if (widget.showDetections)
                RepaintBoundary(
                  child: CustomPaint(
                    painter: _DetectionOverlayPainter(
                      boxes: widget.visibleBoxes,
                      mediaSize: imageSize,
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }
}

class _MissingImagePlaceholder extends StatelessWidget {
  const _MissingImagePlaceholder({required this.path});

  final String path;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final pathSegments = File(path).uri.pathSegments;
    final filename = pathSegments.isEmpty ? path : pathSegments.last;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.image_not_supported_outlined,
              size: 48,
              color: scheme.onSurfaceVariant,
            ),
            const SizedBox(height: 12),
            Text('图片已被删除或移动', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            Text(
              filename,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
            ),
          ],
        ),
      ),
    );
  }
}

List<DetectionBox> currentVideoDetectionBoxes({
  required List<DetectionBox> boxes,
  required Duration position,
  required Duration duration,
  required Map<String, dynamic> detectionData,
  VideoDetectionIndex? index,
}) {
  final boxIndex = index ?? VideoDetectionIndex(boxes);
  int? intData(String key) {
    final value = detectionData[key];
    if (value is int) return value;
    if (value is num) return value.round();
    return int.tryParse(value?.toString() ?? '');
  }

  int? totalFrameHint() {
    final processedFrames = intData('total_frames_processed');
    final stride = math.max(1, intData('vid_stride') ?? 1);
    final maxBoxFrame = boxIndex.maxFrame;
    final processedHint = processedFrames == null
        ? null
        : math.max(1, processedFrames * stride);
    if (processedHint == null) {
      return maxBoxFrame == null ? null : maxBoxFrame + 1;
    }
    if (maxBoxFrame == null) return processedHint;
    return math.max(processedHint, maxBoxFrame + 1);
  }

  int? currentFrameIndexEstimate() {
    if (duration <= Duration.zero || position < Duration.zero) {
      return null;
    }
    final totalFrames = totalFrameHint();
    if (totalFrames == null || totalFrames <= 0) return null;
    final progress = (position.inMilliseconds / duration.inMilliseconds).clamp(
      0.0,
      1.0,
    );
    return (progress * totalFrames).round();
  }

  int frameTolerance() {
    final stride = math.max(1, intData('vid_stride') ?? 1);
    final totalFrames = totalFrameHint();
    if (duration > Duration.zero && totalFrames != null && totalFrames > 0) {
      final fps = totalFrames / duration.inMilliseconds * 1000;
      return math.max(stride, (fps * 0.25).ceil());
    }
    return math.max(stride, 6);
  }

  int trackSortIndex(DetectionBox box, int fallback) {
    final trackId = box.trackId;
    if (trackId == null || trackId.isEmpty) return fallback;
    return int.tryParse(trackId) ?? fallback;
  }

  _TimedBoxMatch? matchForBox(
    DetectionBox box,
    int index,
    int? currentFrame,
    int currentFrame25,
    int currentFrame30,
    int currentFrame60,
    double currentSeconds,
    int toleranceFrames,
    double toleranceSeconds,
  ) {
    if (box.frameIndex != null) {
      final distance = currentFrame == null
          ? [
              (box.frameIndex! - currentFrame25).abs(),
              (box.frameIndex! - currentFrame30).abs(),
              (box.frameIndex! - currentFrame60).abs(),
            ].reduce((a, b) => a < b ? a : b)
          : (box.frameIndex! - currentFrame).abs();
      if (distance <= toleranceFrames) {
        return _TimedBoxMatch(
          box: box,
          distance: distance.toDouble(),
          sortIndex: trackSortIndex(box, index),
        );
      }
      return null;
    }
    if (box.timestamp != null) {
      final distance = (box.timestamp! - currentSeconds).abs();
      if (distance <= toleranceSeconds) {
        return _TimedBoxMatch(
          box: box,
          distance: distance,
          sortIndex: trackSortIndex(box, index),
        );
      }
    }
    return null;
  }

  final currentMs = position.inMilliseconds;
  final currentSeconds = currentMs / 1000;
  final currentFrame = currentFrameIndexEstimate();
  final toleranceFrames = frameTolerance();
  const toleranceSeconds = 0.25;

  final currentFrame25 = (currentMs / 1000 * 25).round();
  final currentFrame30 = (currentMs / 1000 * 30).round();
  final currentFrame60 = (currentMs / 1000 * 60).round();

  if (!boxIndex.hasTime) return boxes;

  final selectedByTrack = <String, _TimedBoxMatch>{};
  final untrackedBoxes = <_TimedBoxMatch>[];
  final candidates = boxIndex.candidates(
    frames: currentFrame == null
        ? [currentFrame25, currentFrame30, currentFrame60]
        : [currentFrame],
    frameTolerance: toleranceFrames,
    seconds: currentSeconds,
    timeTolerance: toleranceSeconds,
  );
  for (final entry in candidates) {
    final box = entry.box;
    final match = matchForBox(
      box,
      entry.originalIndex,
      currentFrame,
      currentFrame25,
      currentFrame30,
      currentFrame60,
      currentSeconds,
      toleranceFrames,
      toleranceSeconds,
    );
    if (match == null) continue;
    final trackId = box.trackId?.trim();
    if (trackId == null || trackId.isEmpty) {
      untrackedBoxes.add(match);
      continue;
    }
    final previous = selectedByTrack[trackId];
    if (previous == null || match.distance < previous.distance) {
      selectedByTrack[trackId] = match;
    }
  }

  final matches = <_TimedBoxMatch>[...selectedByTrack.values, ...untrackedBoxes]
    ..sort((a, b) => a.sortIndex.compareTo(b.sortIndex));
  return matches.map((match) => match.box).toList();
}

class DinoVideoDetectionOverlay extends StatelessWidget {
  const DinoVideoDetectionOverlay({
    required this.boxes,
    required this.mediaSize,
    required this.position,
    required this.duration,
    required this.detectionData,
    this.selectedObservationId,
    this.onDetectionBoxSelected,
    this.onBackgroundTap,
    this.index,
    super.key,
  });

  final List<DetectionBox> boxes;
  final Size mediaSize;
  final Duration position;
  final Duration duration;
  final Map<String, dynamic> detectionData;
  final String? selectedObservationId;
  final ValueChanged<DetectionBox?>? onDetectionBoxSelected;
  final VoidCallback? onBackgroundTap;
  final VideoDetectionIndex? index;

  @override
  Widget build(BuildContext context) {
    final currentBoxes = currentVideoDetectionBoxes(
      boxes: boxes,
      position: position,
      duration: duration,
      detectionData: detectionData,
      index: index,
    );
    return LayoutBuilder(
      builder: (context, constraints) {
        final viewport = Size(constraints.maxWidth, constraints.maxHeight);
        return GestureDetector(
          behavior: HitTestBehavior.translucent,
          onTapUp: onDetectionBoxSelected == null && onBackgroundTap == null
              ? null
              : (details) {
                  final selected = _hitTestDinoBox(
                    details.localPosition,
                    viewport,
                    mediaSize,
                    currentBoxes,
                  );
                  onDetectionBoxSelected?.call(selected);
                  if (selected == null) onBackgroundTap?.call();
                },
          child: CustomPaint(
            painter: _DetectionOverlayPainter(
              boxes: currentBoxes,
              mediaSize: mediaSize,
            ),
            child: const SizedBox.expand(),
          ),
        );
      },
    );
  }
}

class _MediaKitVideoBackend implements ValidationVideoBackend {
  _MediaKitVideoBackend() {
    _player = Player();
    _controller = VideoController(_player);
  }

  late final Player _player;
  late final VideoController _controller;

  @override
  bool get isPlaying => _player.state.playing;

  @override
  bool get isCompleted => _player.state.completed;

  @override
  Duration get position => _player.state.position;

  @override
  Duration get duration => _player.state.duration;

  @override
  Size? get videoSize {
    final vp = _player.state.videoParams;
    final w = vp.w;
    final h = vp.h;
    if (w == null || h == null || w <= 0 || h <= 0) return null;
    return Size(w.toDouble(), h.toDouble());
  }

  @override
  String? get currentMediaId {
    final playlist = _player.state.playlist;
    if (playlist.medias.isEmpty) return null;
    final index = playlist.index;
    if (index < 0 || index >= playlist.medias.length) return null;
    return playlist.medias[index].uri;
  }

  @override
  Stream<bool> get playingStream => _player.stream.playing;

  @override
  Stream<Duration> get positionStream => _player.stream.position;

  @override
  Stream<Duration> get durationStream => _player.stream.duration;

  @override
  Stream<Size?> get videoSizeStream => _player.stream.videoParams.map((vp) {
    final w = vp.w;
    final h = vp.h;
    if (w == null || h == null || w <= 0 || h <= 0) return null;
    return Size(w.toDouble(), h.toDouble());
  });

  @override
  Stream<bool> get completedStream => _player.stream.completed;

  @override
  Stream<String> get errorStream => _player.stream.error;

  @override
  String normalizeMediaId(String path) => Media.normalizeURI(path);

  @override
  Future<void> open(String path) async {
    await _player.open(Media(path), play: false);
    await _player.setPlaylistMode(PlaylistMode.none);
  }

  @override
  Future<void> play() => _player.play();

  @override
  Future<void> pause() => _player.pause();

  @override
  Future<void> seek(Duration position) => _player.seek(position);

  @override
  Future<void> dispose() => _player.dispose();

  @override
  Widget buildVideoSurface(BuildContext context) {
    return ExcludeSemantics(
      child: Video(controller: _controller, controls: NoVideoControls),
    );
  }
}

class ValidationVideoPlayer extends StatefulWidget {
  const ValidationVideoPlayer({
    required this.path,
    required this.visibleBoxes,
    required this.showDetections,
    required this.detectionData,
    required this.onOpenExternal,
    this.selectedObservationId,
    this.onDetectionBoxSelected,
    @visibleForTesting this.backendFactory,
    @visibleForTesting this.logger,
    super.key,
  });

  /// Test-only default backend factory used when widget tests mount
  /// [DetectionMediaViewer] through higher-level screens without native
  /// media_kit initialization.
  @visibleForTesting
  static ValidationVideoBackendFactory? debugBackendFactoryOverride;

  final String path;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final Map<String, dynamic> detectionData;
  final VoidCallback onOpenExternal;
  final String? selectedObservationId;
  final ValueChanged<DetectionBox?>? onDetectionBoxSelected;
  final ValidationVideoBackendFactory? backendFactory;
  final ValidationVideoLogger? logger;

  @override
  State<ValidationVideoPlayer> createState() => _ValidationVideoPlayerState();
}

class _ValidationVideoPlayerState extends State<ValidationVideoPlayer> {
  late final ValidationVideoSessionController _session;
  bool _didInitSession = false;
  VideoDetectionIndex? _boxIndex;

  VideoDetectionIndex get _videoBoxIndex {
    if (_boxIndex == null ||
        !identical(_boxIndex!.boxes, widget.visibleBoxes)) {
      _boxIndex = VideoDetectionIndex(widget.visibleBoxes);
    }
    return _boxIndex!;
  }

  @override
  void initState() {
    super.initState();
    final backend =
        widget.backendFactory?.call() ??
        ValidationVideoPlayer.debugBackendFactoryOverride?.call() ??
        _MediaKitVideoBackend();
    _session = ValidationVideoSessionController(
      backend: backend,
      logger: widget.logger,
    )..addListener(_onSessionChanged);
  }

  void _onSessionChanged() {
    if (!mounted) return;
    setState(() {});
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final visible = TickerMode.valuesOf(context).enabled;
    if (!_didInitSession) {
      _didInitSession = true;
      _session.setVisible(visible);
      unawaited(_session.open(widget.path));
    } else {
      _session.setVisible(visible);
    }
  }

  @override
  void didUpdateWidget(ValidationVideoPlayer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.path != widget.path) {
      unawaited(_session.open(widget.path));
    }
  }

  @override
  void dispose() {
    _session.removeListener(_onSessionChanged);
    _session.dispose();
    super.dispose();
  }

  String _formatDuration(Duration d) {
    final hours = d.inHours;
    final mins = (d.inMinutes % 60).toString().padLeft(2, '0');
    final secs = (d.inSeconds % 60).toString().padLeft(2, '0');
    return hours > 0 ? '$hours:$mins:$secs' : '$mins:$secs';
  }

  @override
  Widget build(BuildContext context) {
    if (_session.isError) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline_rounded, size: 56),
            const SizedBox(height: 12),
            const Text('媒体加载失败', textAlign: TextAlign.center),
            const SizedBox(height: 16),
            Wrap(
              spacing: 12,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: [
                FilledButton.icon(
                  key: const ValueKey('validation-video-retry-button'),
                  onPressed: () => unawaited(_session.retry()),
                  icon: const Icon(Icons.refresh_rounded),
                  label: const Text('重试'),
                ),
                FilledButton.tonalIcon(
                  onPressed: widget.onOpenExternal,
                  icon: const Icon(Icons.open_in_new_rounded),
                  label: const Text('使用系统应用打开'),
                ),
              ],
            ),
          ],
        ),
      );
    }

    final videoSize = _session.videoSize;
    return MouseRegion(
      onHover: (_) => _session.wakeUpControls(),
      onEnter: (_) => _session.wakeUpControls(),
      onExit: (_) {
        if (_session.isPlaying && !_session.isDragging) {
          _session.hideControlsImmediately();
        }
      },
      child: Stack(
        fit: StackFit.expand,
        children: [
          // 1. 底层静态视频播放器，避免频繁重绘
          GestureDetector(
            onTap: () => unawaited(_session.togglePlayPause()),
            behavior: HitTestBehavior.opaque,
            child: _session.backend.buildVideoSurface(context),
          ),

          if (_session.isOpening)
            const IgnorePointer(
              child: Center(
                child: CircularProgressIndicator(
                  key: ValueKey('validation-video-loading'),
                ),
              ),
            ),

          // 2. 随时间进度局部刷新检测框绘制层，通过 RepaintBoundary 隔绝重绘污染
          if (widget.showDetections && videoSize != null && !_session.isOpening)
            ValueListenableBuilder<Duration>(
              valueListenable: _session.positionNotifier,
              builder: (context, position, child) {
                return RepaintBoundary(
                  child: DinoVideoDetectionOverlay(
                    boxes: widget.visibleBoxes,
                    mediaSize: videoSize,
                    position: position,
                    duration: _session.duration,
                    detectionData: widget.detectionData,
                    index: _videoBoxIndex,
                    selectedObservationId: widget.selectedObservationId,
                    onDetectionBoxSelected: widget.onDetectionBoxSelected,
                    onBackgroundTap: () =>
                        unawaited(_session.togglePlayPause()),
                  ),
                );
              },
            ),

          // 3. 随时间进度局部刷新底部控制层
          Align(
            alignment: Alignment.bottomCenter,
            child: ValueListenableBuilder<Duration>(
              valueListenable: _session.positionNotifier,
              builder: (context, position, child) {
                return RepaintBoundary(child: _buildControls(position));
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildControls(Duration position) {
    final colorScheme = Theme.of(context).colorScheme;
    final duration = _session.duration;
    final showControls = _session.showControls;
    final progress = duration.inMilliseconds == 0
        ? 0.0
        : (position.inMilliseconds / duration.inMilliseconds).clamp(0.0, 1.0);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeOutCubic,
      height: showControls ? 72 : 4,
      decoration: BoxDecoration(
        gradient: showControls
            ? const LinearGradient(
                begin: Alignment.bottomCenter,
                end: Alignment.topCenter,
                colors: [Colors.black87, Colors.black45, Colors.transparent],
              )
            : null,
      ),
      child: Stack(
        alignment: Alignment.bottomLeft,
        children: [
          AnimatedOpacity(
            duration: const Duration(milliseconds: 300),
            opacity: showControls ? 0.0 : 1.0,
            child: Container(
              height: 4,
              width: double.infinity,
              color: Colors.white24,
              alignment: Alignment.centerLeft,
              child: FractionallySizedBox(
                widthFactor: progress,
                child: Container(
                  decoration: BoxDecoration(
                    color: colorScheme.primary,
                    borderRadius: const BorderRadius.horizontal(
                      right: Radius.circular(2),
                    ),
                  ),
                ),
              ),
            ),
          ),

          AnimatedOpacity(
            duration: const Duration(milliseconds: 300),
            opacity: showControls ? 1.0 : 0.0,
            child: IgnorePointer(
              ignoring: !showControls,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 8, left: 8, right: 16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    IconButton(
                      key: const ValueKey('validation-video-play-pause'),
                      icon: Icon(
                        _session.displayPlaying
                            ? Icons.pause_rounded
                            : Icons.play_arrow_rounded,
                        color: Colors.white,
                        size: 32,
                      ),
                      onPressed: () => unawaited(_session.togglePlayPause()),
                    ),
                    Expanded(
                      child: SliderTheme(
                        data: SliderTheme.of(context).copyWith(
                          trackHeight: 4,
                          thumbShape: const RoundSliderThumbShape(
                            enabledThumbRadius: 6,
                            pressedElevation: 8,
                          ),
                          overlayShape: const RoundSliderOverlayShape(
                            overlayRadius: 16,
                          ),
                          activeTrackColor: colorScheme.primary,
                          inactiveTrackColor: Colors.white30,
                          thumbColor: colorScheme.primary,
                          overlayColor: colorScheme.primary.withValues(
                            alpha: 0.12,
                          ),
                        ),
                        child: Slider(
                          value: position.inMilliseconds.toDouble().clamp(
                            0,
                            math.max(0, duration.inMilliseconds.toDouble()),
                          ),
                          max: math.max(1, duration.inMilliseconds.toDouble()),
                          onChangeStart: (_) {
                            _session.setDragging(true);
                          },
                          onChanged: (v) {
                            unawaited(
                              _session.seek(Duration(milliseconds: v.toInt())),
                            );
                          },
                          onChangeEnd: (_) {
                            _session.setDragging(false);
                          },
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '${_formatDuration(position)} / ${_formatDuration(duration)}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

Rect _fittedMediaRect(Size viewport, Size media) {
  if (viewport.isEmpty || media.isEmpty) return Rect.zero;
  final scale = math.min(
    viewport.width / media.width,
    viewport.height / media.height,
  );
  final fittedSize = Size(media.width * scale, media.height * scale);
  return Rect.fromLTWH(
    (viewport.width - fittedSize.width) / 2,
    (viewport.height - fittedSize.height) / 2,
    fittedSize.width,
    fittedSize.height,
  );
}

Rect? _renderRectForBox(DetectionBox box, Size viewport, Size mediaSize) {
  if (box.bbox.length < 4 || viewport.isEmpty || mediaSize.isEmpty) {
    return null;
  }
  final fitted = _fittedMediaRect(viewport, mediaSize);
  if (fitted.isEmpty) return null;

  final xmin = box.bbox[0];
  final ymin = box.bbox[1];
  final xmax = box.bbox[2];
  final ymax = box.bbox[3];
  final isNormalized = xmax <= 1.5 && ymax <= 1.5;
  final scaleX = fitted.width / mediaSize.width;
  final scaleY = fitted.height / mediaSize.height;

  return Rect.fromLTRB(
    fitted.left + (isNormalized ? xmin * fitted.width : xmin * scaleX),
    fitted.top + (isNormalized ? ymin * fitted.height : ymin * scaleY),
    fitted.left + (isNormalized ? xmax * fitted.width : xmax * scaleX),
    fitted.top + (isNormalized ? ymax * fitted.height : ymax * scaleY),
  );
}

DetectionBox? _hitTestDinoBox(
  Offset localPosition,
  Size viewport,
  Size mediaSize,
  List<DetectionBox> boxes,
) {
  for (final box in boxes.reversed) {
    final rect = _renderRectForBox(box, viewport, mediaSize);
    if (rect != null && rect.contains(localPosition)) return box;
  }
  return null;
}

class _TimedBoxMatch {
  const _TimedBoxMatch({
    required this.box,
    required this.distance,
    required this.sortIndex,
  });

  final DetectionBox box;
  final double distance;
  final int sortIndex;
}

class _DetectionOverlayPainter extends CustomPainter {
  _DetectionOverlayPainter({required this.boxes, required this.mediaSize});

  final List<DetectionBox> boxes;
  final Size mediaSize;

  @override
  void paint(Canvas canvas, Size size) {
    if (mediaSize.isEmpty || boxes.isEmpty) return;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
      maxLines: 1,
      ellipsis: '…',
    );

    for (var index = 0; index < boxes.length; index++) {
      final box = boxes[index];
      if (box.bbox.length < 4) continue;

      final rect = _renderRectForBox(box, size, mediaSize);
      if (rect == null) continue;
      final fitted = _fittedMediaRect(size, mediaSize);
      final color = detectionBoxColor(box);
      paint.color = color;

      canvas.drawRect(rect, paint);

      final label = detectionBoxLabel(box, index);

      textPainter.text = TextSpan(
        text: ' $label ',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          height: 1.2,
        ),
      );
      textPainter.layout(maxWidth: fitted.width);

      final textY = math.max(fitted.top, rect.top - textPainter.height - 2);
      final textX = rect.left
          .clamp(fitted.left, fitted.right - textPainter.width)
          .toDouble();

      final bgRect = Rect.fromLTWH(
        textX,
        textY,
        textPainter.width,
        textPainter.height,
      );

      final bgPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = color.withValues(alpha: 0.85);

      canvas.drawRect(bgRect, bgPaint);
      textPainter.paint(canvas, Offset(textX, textY));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionOverlayPainter oldDelegate) {
    return oldDelegate.boxes != boxes || oldDelegate.mediaSize != mediaSize;
  }
}
