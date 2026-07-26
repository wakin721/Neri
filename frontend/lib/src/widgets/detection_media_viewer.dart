import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';

import '../models/job.dart';

class DetectionMediaViewer extends StatelessWidget {
  const DetectionMediaViewer({
    required this.item,
    required this.visibleBoxes,
    required this.showDetections,
    required this.onOpenExternal,
    this.isFavorite = false,
    this.onToggleFavorite,
    super.key,
  });

  final DetectionItem item;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final VoidCallback onOpenExternal;
  final bool isFavorite;
  final VoidCallback? onToggleFavorite;

  bool get _isImage {
    const imageTypes = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'};
    return imageTypes.contains(item.fileType.toLowerCase());
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final showFavoriteButton = _isImage && onToggleFavorite != null;
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
              ),
              Positioned(
                top: 12,
                right: 12,
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (showFavoriteButton) ...[
                      IconButton.filledTonal(
                        tooltip: isFavorite ? '取消收藏照片' : '收藏照片',
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
  });

  final DetectionItem item;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final VoidCallback onOpenExternal;

  bool get isVideo {
    const videoTypes = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'};
    return videoTypes.contains(item.fileType.toLowerCase());
  }

  @override
  Widget build(BuildContext context) {
    if (isVideo) {
      return _ValidationVideoPlayer(
        path: item.path,
        visibleBoxes: visibleBoxes,
        showDetections: showDetections,
        detectionData: item.detectionData,
        onOpenExternal: onOpenExternal,
      );
    } else {
      return _ImageMediaViewer(
        path: item.path,
        visibleBoxes: visibleBoxes,
        showDetections: showDetections,
      );
    }
  }
}

class _ImageMediaViewer extends StatefulWidget {
  const _ImageMediaViewer({
    required this.path,
    required this.visibleBoxes,
    required this.showDetections,
  });

  final String path;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;

  @override
  State<_ImageMediaViewer> createState() => _ImageMediaViewerState();
}

class _ImageMediaViewerState extends State<_ImageMediaViewer> {
  Size? _imageSize;
  Object? _imageError;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;

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
    _removeImageStreamListener();
    super.dispose();
  }

  void _removeImageStreamListener() {
    final stream = _imageStream;
    final listener = _imageStreamListener;
    if (stream != null && listener != null) {
      stream.removeListener(listener);
    }
    _imageStream = null;
    _imageStreamListener = null;
  }

  void _resolveImageSize() {
    _removeImageStreamListener();
    _imageSize = null;
    _imageError = null;
    final file = File(widget.path);
    if (!file.existsSync()) {
      _imageError = FileSystemException('文件不存在', widget.path);
      return;
    }

    final imageProvider = FileImage(file);
    final imageStream = imageProvider.resolve(const ImageConfiguration());
    final listener = ImageStreamListener(
      (info, _) {
        if (mounted) {
          setState(() {
            _imageSize = Size(
              info.image.width.toDouble(),
              info.image.height.toDouble(),
            );
          });
        }
      },
      onError: (error, _) {
        if (mounted) {
          setState(() {
            _imageError = error;
            _imageSize = null;
          });
        }
      },
    );
    _imageStream = imageStream;
    _imageStreamListener = listener;
    imageStream.addListener(listener);
  }

  @override
  Widget build(BuildContext context) {
    if (_imageError != null) {
      return _MissingImagePlaceholder(path: widget.path);
    }
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.file(
          File(widget.path),
          key: ValueKey(widget.path),
          fit: BoxFit.contain,
          errorBuilder: (context, error, stackTrace) =>
              _MissingImagePlaceholder(path: widget.path),
        ),
        if (widget.showDetections && _imageSize != null)
          CustomPaint(
            painter: _DetectionOverlayPainter(
              boxes: widget.visibleBoxes,
              mediaSize: _imageSize!,
            ),
          ),
      ],
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

class _ValidationVideoPlayer extends StatefulWidget {
  const _ValidationVideoPlayer({
    required this.path,
    required this.visibleBoxes,
    required this.showDetections,
    required this.detectionData,
    required this.onOpenExternal,
  });

  final String path;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final Map<String, dynamic> detectionData;
  final VoidCallback onOpenExternal;

  @override
  State<_ValidationVideoPlayer> createState() => _ValidationVideoPlayerState();
}

class _ValidationVideoPlayerState extends State<_ValidationVideoPlayer> {
  late final Player _player;
  late final VideoController _controller;
  late final List<StreamSubscription> _subs;

  bool _isError = false;
  bool _isPlaying = false;
  bool _showControls = true;
  bool _isDragging = false;
  Timer? _hideTimer;
  Timer? _positionTimer;
  Duration _duration = Duration.zero;
  Size? _videoSize;

  // 使用 Notifier 代替原先的 _position 进行局部刷新，彻底避免全局频繁 setState
  final ValueNotifier<Duration> _positionNotifier = ValueNotifier(
    Duration.zero,
  );

  @override
  void initState() {
    super.initState();
    _player = Player();
    _controller = VideoController(_player);

    _subs = [
      _player.stream.playing.listen((p) {
        if (mounted) {
          setState(() => _isPlaying = p);
          if (p) {
            _startPositionTimer();
            _startHideTimer();
          } else {
            _syncPlayerState();
            _positionTimer?.cancel();
            _positionTimer = null;
            _hideTimer?.cancel();
            setState(() => _showControls = true);
          }
        }
      }),
      _player.stream.position.listen((p) {
        if (!mounted || _positionTimer != null) return;
        final deltaMs = (p - _positionNotifier.value).inMilliseconds.abs();
        if (deltaMs < 200) return;
        // 直接更新 Notifier 的值，不会触发整个 Video 树的重绘
        _positionNotifier.value = p;
      }),
      _player.stream.duration.listen((d) {
        if (mounted) setState(() => _duration = d);
      }),
      _player.stream.videoParams.listen((vp) {
        if (mounted && vp.w != null && vp.h != null) {
          setState(() {
            _videoSize = Size(vp.w!.toDouble(), vp.h!.toDouble());
          });
        }
      }),
      _player.stream.completed.listen((completed) {
        if (mounted && completed) {
          setState(() {
            _isPlaying = false;
            _showControls = true;
          });
          _positionTimer?.cancel();
          _positionTimer = null;
          _player.pause();
          _hideTimer?.cancel();
        }
      }),
    ];

    _initPlayer(widget.path);
  }

  Future<void> _initPlayer(String path) async {
    setState(() => _isError = false);
    try {
      await _player.open(Media(path));
      await _player.setPlaylistMode(PlaylistMode.none);
      await _player.play();
    } catch (_) {
      if (mounted) setState(() => _isError = true);
    }
  }

  void _startHideTimer() {
    _hideTimer?.cancel();
    _hideTimer = Timer(const Duration(seconds: 2), () {
      if (mounted && _isPlaying && !_isDragging) {
        setState(() => _showControls = false);
      }
    });
  }

  void _startPositionTimer() {
    _positionTimer ??= Timer.periodic(
      const Duration(milliseconds: 250),
      (_) => _syncPlayerState(),
    );
  }

  void _syncPlayerState() {
    if (!mounted) return;
    final state = _player.state;
    final position = state.position;
    final duration = state.duration;

    if (position != _positionNotifier.value) {
      _positionNotifier.value = position;
    }
    if (duration != _duration) {
      setState(() => _duration = duration);
    }
  }

  void _wakeUpControls() {
    if (mounted && !_showControls) {
      setState(() => _showControls = true);
    }
    _startHideTimer();
  }

  void _togglePlayPause() {
    if (_isPlaying) {
      _player.pause();
    } else {
      if (_positionNotifier.value >= _duration && _duration > Duration.zero) {
        _player.seek(Duration.zero);
      }
      _player.play();
    }
    _wakeUpControls();
  }

  @override
  void didUpdateWidget(_ValidationVideoPlayer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.path != widget.path) {
      _initPlayer(widget.path);
    }
  }

  @override
  void dispose() {
    for (final s in _subs) {
      s.cancel();
    }
    _positionTimer?.cancel();
    _hideTimer?.cancel();
    _positionNotifier.dispose();
    _player.dispose();
    super.dispose();
  }

  List<DetectionBox> _currentBoxes(Duration position) {
    final currentMs = position.inMilliseconds;
    final currentSeconds = currentMs / 1000;
    final currentFrame = _currentFrameIndexEstimate(position);
    final toleranceFrames = _frameTolerance();
    final toleranceSeconds = _timeTolerance();

    final currentFrame25 = (currentMs / 1000 * 25).round();
    final currentFrame30 = (currentMs / 1000 * 30).round();
    final currentFrame60 = (currentMs / 1000 * 60).round();

    final anyBoxHasTime = widget.visibleBoxes.any(
      (b) => b.frameIndex != null || b.timestamp != null,
    );

    if (!anyBoxHasTime) {
      return widget.visibleBoxes;
    }

    final selectedByTrack = <String, _TimedBoxMatch>{};
    final untrackedBoxes = <_TimedBoxMatch>[];
    for (var index = 0; index < widget.visibleBoxes.length; index++) {
      final box = widget.visibleBoxes[index];
      final match = _matchForBox(
        box,
        index,
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

    final matches = <_TimedBoxMatch>[
      ...selectedByTrack.values,
      ...untrackedBoxes,
    ]..sort((a, b) => a.sortIndex.compareTo(b.sortIndex));
    return matches.map((match) => match.box).toList();
  }

  _TimedBoxMatch? _matchForBox(
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
          sortIndex: _trackSortIndex(box, index),
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
          sortIndex: _trackSortIndex(box, index),
        );
      }
    }
    return null;
  }

  int? _currentFrameIndexEstimate(Duration position) {
    if (_duration <= Duration.zero || position < Duration.zero) {
      return null;
    }
    final totalFrames = _totalFrameHint();
    if (totalFrames == null || totalFrames <= 0) return null;
    final progress = (position.inMilliseconds / _duration.inMilliseconds).clamp(
      0.0,
      1.0,
    );
    return (progress * totalFrames).round();
  }

  int _frameTolerance() {
    final stride = math.max(1, _intData('vid_stride') ?? 1);
    final totalFrames = _totalFrameHint();
    if (_duration > Duration.zero && totalFrames != null && totalFrames > 0) {
      final fps = totalFrames / _duration.inMilliseconds * 1000;
      return math.max(stride, (fps * 0.25).ceil());
    }
    return math.max(stride, 6);
  }

  double _timeTolerance() => 0.25;

  int? _totalFrameHint() {
    final processedFrames = _intData('total_frames_processed');
    final stride = math.max(1, _intData('vid_stride') ?? 1);
    final maxBoxFrame = widget.visibleBoxes
        .map((box) => box.frameIndex)
        .whereType<int>()
        .fold<int?>(null, (maxFrame, frame) {
          if (maxFrame == null || frame > maxFrame) return frame;
          return maxFrame;
        });
    final processedHint = processedFrames == null
        ? null
        : math.max(1, processedFrames * stride);
    if (processedHint == null) {
      return maxBoxFrame == null ? null : maxBoxFrame + 1;
    }
    if (maxBoxFrame == null) return processedHint;
    return math.max(processedHint, maxBoxFrame + 1);
  }

  int? _intData(String key) {
    final value = widget.detectionData[key];
    if (value is int) return value;
    if (value is num) return value.round();
    return int.tryParse(value?.toString() ?? '');
  }

  int _trackSortIndex(DetectionBox box, int fallback) {
    final trackId = box.trackId;
    if (trackId == null || trackId.isEmpty) return fallback;
    return int.tryParse(trackId) ?? fallback;
  }

  String _formatDuration(Duration d) {
    final hours = d.inHours;
    final mins = (d.inMinutes % 60).toString().padLeft(2, '0');
    final secs = (d.inSeconds % 60).toString().padLeft(2, '0');
    return hours > 0 ? '$hours:$mins:$secs' : '$mins:$secs';
  }

  @override
  Widget build(BuildContext context) {
    if (_isError) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline_rounded, size: 56),
            const SizedBox(height: 12),
            const Text('媒体加载失败', textAlign: TextAlign.center),
            const SizedBox(height: 16),
            FilledButton.tonalIcon(
              onPressed: widget.onOpenExternal,
              icon: const Icon(Icons.open_in_new_rounded),
              label: const Text('使用系统应用打开'),
            ),
          ],
        ),
      );
    }

    return MouseRegion(
      onHover: (_) => _wakeUpControls(),
      onEnter: (_) => _wakeUpControls(),
      onExit: (_) {
        if (_isPlaying && !_isDragging) {
          _hideTimer?.cancel();
          setState(() => _showControls = false);
        }
      },
      child: Stack(
        fit: StackFit.expand,
        children: [
          // 1. 底层静态视频播放器，避免频繁重绘
          GestureDetector(
            onTap: _togglePlayPause,
            behavior: HitTestBehavior.opaque,
            child: ExcludeSemantics(
              child: Video(controller: _controller, controls: NoVideoControls),
            ),
          ),

          // 2. 随时间进度局部刷新检测框绘制层，通过 RepaintBoundary 隔绝重绘污染
          if (widget.showDetections && _videoSize != null)
            ValueListenableBuilder<Duration>(
              valueListenable: _positionNotifier,
              builder: (context, position, child) {
                return RepaintBoundary(
                  child: IgnorePointer(
                    child: CustomPaint(
                      painter: _DetectionOverlayPainter(
                        boxes: _currentBoxes(position),
                        mediaSize: _videoSize!,
                      ),
                    ),
                  ),
                );
              },
            ),

          // 3. 随时间进度局部刷新底部控制层
          Align(
            alignment: Alignment.bottomCenter,
            child: ValueListenableBuilder<Duration>(
              valueListenable: _positionNotifier,
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
    final progress = _duration.inMilliseconds == 0
        ? 0.0
        : (position.inMilliseconds / _duration.inMilliseconds).clamp(0.0, 1.0);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeOutCubic,
      height: _showControls ? 72 : 4,
      decoration: BoxDecoration(
        gradient: _showControls
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
            opacity: _showControls ? 0.0 : 1.0,
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
            opacity: _showControls ? 1.0 : 0.0,
            child: IgnorePointer(
              ignoring: !_showControls,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 8, left: 8, right: 16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    IconButton(
                      icon: Icon(
                        _isPlaying
                            ? Icons.pause_rounded
                            : Icons.play_arrow_rounded,
                        color: Colors.white,
                        size: 32,
                      ),
                      onPressed: _togglePlayPause,
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
                            math.max(0, _duration.inMilliseconds.toDouble()),
                          ),
                          max: math.max(1, _duration.inMilliseconds.toDouble()),
                          onChangeStart: (_) {
                            _isDragging = true;
                            _hideTimer?.cancel();
                          },
                          onChanged: (v) {
                            _player.seek(Duration(milliseconds: v.toInt()));
                          },
                          onChangeEnd: (_) {
                            _isDragging = false;
                            _startHideTimer();
                          },
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '${_formatDuration(position)} / ${_formatDuration(_duration)}',
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

  Color _getColorForSpecies(String species) {
    final hash = species.hashCode;
    final hue = (hash % 360).toDouble();
    return HSVColor.fromAHSV(1.0, hue, 0.85, 0.95).toColor();
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (mediaSize.isEmpty || boxes.isEmpty) return;

    final mediaRatio = mediaSize.width / mediaSize.height;
    final canvasRatio = size.width / size.height;
    double w, h, dx, dy;

    if (mediaRatio > canvasRatio) {
      w = size.width;
      h = w / mediaRatio;
      dx = 0;
      dy = (size.height - h) / 2;
    } else {
      h = size.height;
      w = h * mediaRatio;
      dx = (size.width - w) / 2;
      dy = 0;
    }

    final scaleX = w / mediaSize.width;
    final scaleY = h / mediaSize.height;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;

    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    for (var index = 0; index < boxes.length; index++) {
      final box = boxes[index];
      if (box.bbox.length < 4) continue;

      final xmin = box.bbox[0];
      final ymin = box.bbox[1];
      final xmax = box.bbox[2];
      final ymax = box.bbox[3];

      final isNormalized = xmax <= 1.5 && ymax <= 1.5;

      double renderXMin, renderYMin, renderXMax, renderYMax;
      if (isNormalized) {
        renderXMin = dx + xmin * w;
        renderYMin = dy + ymin * h;
        renderXMax = dx + xmax * w;
        renderYMax = dy + ymax * h;
      } else {
        renderXMin = dx + xmin * scaleX;
        renderYMin = dy + ymin * scaleY;
        renderXMax = dx + xmax * scaleX;
        renderYMax = dy + ymax * scaleY;
      }

      final rect = Rect.fromLTRB(
        renderXMin,
        renderYMin,
        renderXMax,
        renderYMax,
      );
      final color = _getColorForSpecies(box.species);
      paint.color = color;

      canvas.drawRect(rect, paint);

      final confidenceStr = (box.confidence ?? 0).toStringAsFixed(2);
      final objectId = _objectIdForBox(box, index);
      final label = '#$objectId ${box.species} $confidenceStr';

      textPainter.text = TextSpan(
        text: ' $label ',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          height: 1.2,
        ),
      );
      textPainter.layout();

      final textY = math.max(dy, rect.top - textPainter.height - 2);

      final bgRect = Rect.fromLTWH(
        rect.left,
        textY,
        textPainter.width,
        textPainter.height,
      );

      final bgPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = color.withValues(alpha: 0.85);

      canvas.drawRect(bgRect, bgPaint);
      textPainter.paint(canvas, Offset(rect.left, textY));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionOverlayPainter oldDelegate) {
    return oldDelegate.boxes != boxes || oldDelegate.mediaSize != mediaSize;
  }

  String _objectIdForBox(DetectionBox box, int index) {
    final trackId = box.trackId?.trim();
    if (trackId != null && trackId.isNotEmpty) return trackId;
    return '${index + 1}';
  }
}
