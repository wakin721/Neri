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
    super.key,
  });

  final DetectionItem item;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
  final VoidCallback onOpenExternal;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
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
              child: IconButton.filledTonal(
                tooltip: '使用系统应用打开',
                icon: const Icon(Icons.open_in_new_rounded),
                onPressed: onOpenExternal,
              ),
            ),
          ],
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

  void _resolveImageSize() {
    final imageProvider = FileImage(File(widget.path));
    final imageStream = imageProvider.resolve(const ImageConfiguration());
    final listener = ImageStreamListener((info, _) {
      if (mounted) {
        setState(() {
          _imageSize = Size(
            info.image.width.toDouble(),
            info.image.height.toDouble(),
          );
        });
      }
    });
    imageStream.addListener(listener);
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.file(File(widget.path), fit: BoxFit.contain),
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

class _ValidationVideoPlayer extends StatefulWidget {
  const _ValidationVideoPlayer({
    required this.path,
    required this.visibleBoxes,
    required this.showDetections,
    required this.onOpenExternal,
  });

  final String path;
  final List<DetectionBox> visibleBoxes;
  final bool showDetections;
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
  Duration _position = Duration.zero;
  Duration _duration = Duration.zero;
  Size? _videoSize;

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
            _startHideTimer();
          } else {
            // 如果处于暂停状态，保持控制条显示
            _hideTimer?.cancel();
            setState(() => _showControls = true);
          }
        }
      }),
      _player.stream.position.listen((p) {
        if (mounted) setState(() => _position = p);
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
        // 播放完成时触发暂停并唤醒控件
        if (mounted && completed) {
          setState(() {
            _isPlaying = false;
            _showControls = true;
          });
          _player.pause(); // 强制执行暂停
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
      // 改为不循环，播放完一遍自动停止
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
      // 如果视频已播放到结尾，点击播放时重头开始
      if (_position >= _duration && _duration > Duration.zero) {
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
    _hideTimer?.cancel();
    _player.dispose();
    super.dispose();
  }

  /// 实时过滤当前帧应当显示的检测框
  List<DetectionBox> get _currentBoxes {
    final currentMs = _position.inMilliseconds;
    
    // 假设常用帧率为 25, 30 或 60 fps
    final currentFrame25 = (currentMs / 1000 * 25).round();
    final currentFrame30 = (currentMs / 1000 * 30).round();
    final currentFrame60 = (currentMs / 1000 * 60).round();

    bool anyBoxHasTime = widget.visibleBoxes.any(
      (b) => b.frameIndex != null || b.timestamp != null
    );

    // 如果没有任何框包含帧/时间信息，则视为静态结果或未定义结果，全部显示
    if (!anyBoxHasTime) {
      return widget.visibleBoxes;
    }

    return widget.visibleBoxes.where((box) {
      if (box.frameIndex != null) {
        // 允许 +- 5帧 的容差，以防帧率对齐产生闪烁
        return (box.frameIndex! - currentFrame25).abs() <= 5 ||
               (box.frameIndex! - currentFrame30).abs() <= 5 ||
               (box.frameIndex! - currentFrame60).abs() <= 5;
      }
      if (box.timestamp != null) {
        return (box.timestamp! - currentMs / 1000).abs() <= 0.25;
      }
      return false; // 当前有带帧信息的框，这个框没有则过滤掉
    }).toList();
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
          // 1. 底层视频播放器
          GestureDetector(
            onTap: _togglePlayPause,
            behavior: HitTestBehavior.opaque,
            child: Video(
              controller: _controller,
              controls: NoVideoControls,
            ),
          ),
          
          // 2. 随时间进度实时更新的检测框绘制层
          if (widget.showDetections && _videoSize != null)
            IgnorePointer(
              child: CustomPaint(
                painter: _DetectionOverlayPainter(
                  boxes: _currentBoxes,
                  mediaSize: _videoSize!,
                ),
              ),
            ),
            
          // 3. 底部 MD3 风格自动简化进度条与控制层
          Align(
            alignment: Alignment.bottomCenter,
            child: _buildControls(),
          ),
        ],
      ),
    );
  }

  Widget _buildControls() {
    final colorScheme = Theme.of(context).colorScheme;
    final progress = _duration.inMilliseconds == 0
        ? 0.0
        : (_position.inMilliseconds / _duration.inMilliseconds).clamp(0.0, 1.0);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeOutCubic,
      // 展开时高度给足，简化时只留出 4 像素位于最底部
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
          // A: 简化版沉浸式底边细条进度 (隐藏控制条时显示)
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

          // B: 完整版 MD3 控制面板 (唤醒时显示)
          AnimatedOpacity(
            duration: const Duration(milliseconds: 300),
            opacity: _showControls ? 1.0 : 0.0,
            child: IgnorePointer(
              ignoring: !_showControls, // 隐藏时不阻挡手势
              child: Padding(
                padding: const EdgeInsets.only(bottom: 8, left: 8, right: 16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    IconButton(
                      icon: Icon(
                        _isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded,
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
                          overlayColor: colorScheme.primary.withOpacity(0.12),
                        ),
                        child: Slider(
                          value: _position.inMilliseconds.toDouble().clamp(
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
                      '${_formatDuration(_position)} / ${_formatDuration(_duration)}',
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

class _DetectionOverlayPainter extends CustomPainter {
  _DetectionOverlayPainter({
    required this.boxes,
    required this.mediaSize,
  });

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

    for (final box in boxes) {
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

      final rect = Rect.fromLTRB(renderXMin, renderYMin, renderXMax, renderYMax);
      final color = _getColorForSpecies(box.species);
      paint.color = color;
      
      canvas.drawRect(rect, paint);

      final confidenceStr = (box.confidence ?? 0).toStringAsFixed(2);
      final label = '${box.species} $confidenceStr';
      
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
        ..color = color.withOpacity(0.85);
        
      canvas.drawRect(bgRect, bgPaint);
      textPainter.paint(canvas, Offset(rect.left, textY));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionOverlayPainter oldDelegate) {
    return oldDelegate.boxes != boxes || oldDelegate.mediaSize != mediaSize;
  }
}