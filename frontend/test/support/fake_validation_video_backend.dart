import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neri_flutter/src/utils/validation_video_session.dart';
import 'package:neri_flutter/src/widgets/detection_media_viewer_impl.dart';

/// Deterministic in-memory [ValidationVideoBackend] for widget and session
/// tests that should not initialize native media_kit / libmpv.
class FakeValidationVideoBackend implements ValidationVideoBackend {
  FakeValidationVideoBackend({this.onOpen, this.onPlay, this.onSeek});

  Future<void> Function(String path, FakeValidationVideoBackend backend)?
  onOpen;
  Future<void> Function(FakeValidationVideoBackend backend)? onPlay;
  Future<void> Function(
    Duration position,
    FakeValidationVideoBackend backend,
  )?
  onSeek;

  final StreamController<bool> _playingController =
      StreamController<bool>.broadcast();
  final StreamController<Duration> _positionController =
      StreamController<Duration>.broadcast();
  final StreamController<Duration> _durationController =
      StreamController<Duration>.broadcast();
  final StreamController<Size?> _videoSizeController =
      StreamController<Size?>.broadcast();
  final StreamController<bool> _completedController =
      StreamController<bool>.broadcast();
  final StreamController<String> _errorController =
      StreamController<String>.broadcast();

  final List<String> operations = <String>[];
  int inFlightOperations = 0;
  int maxConcurrentOperations = 0;
  bool disposed = false;

  bool _isPlaying = false;
  bool _isCompleted = false;
  Duration _position = Duration.zero;
  Duration _duration = Duration.zero;
  Size? _videoSize;
  String? _currentMediaId;

  @override
  bool get isPlaying => _isPlaying;

  @override
  bool get isCompleted => _isCompleted;

  @override
  Duration get position => _position;

  @override
  Duration get duration => _duration;

  @override
  Size? get videoSize => _videoSize;

  @override
  String? get currentMediaId => _currentMediaId;

  set currentMediaId(String? value) => _currentMediaId = value;

  @override
  Stream<bool> get playingStream => _playingController.stream;

  @override
  Stream<Duration> get positionStream => _positionController.stream;

  @override
  Stream<Duration> get durationStream => _durationController.stream;

  @override
  Stream<Size?> get videoSizeStream => _videoSizeController.stream;

  @override
  Stream<bool> get completedStream => _completedController.stream;

  @override
  Stream<String> get errorStream => _errorController.stream;

  @override
  String normalizeMediaId(String path) => path;

  Future<T> _track<T>(String label, Future<T> Function() body) async {
    operations.add(label);
    inFlightOperations++;
    if (inFlightOperations > maxConcurrentOperations) {
      maxConcurrentOperations = inFlightOperations;
    }
    try {
      return await body();
    } finally {
      inFlightOperations--;
    }
  }

  @override
  Future<void> open(String path) {
    return _track('open:$path', () async {
      _isPlaying = false;
      _isCompleted = false;
      _position = Duration.zero;
      _duration = Duration.zero;
      _videoSize = null;
      if (onOpen != null) {
        await onOpen!(path, this);
      } else {
        _currentMediaId = path;
      }
    });
  }

  @override
  Future<void> play() {
    return _track('play', () async {
      _isCompleted = false;
      if (onPlay != null) {
        await onPlay!(this);
      } else {
        emitPlaying(true);
      }
    });
  }

  @override
  Future<void> pause() {
    return _track('pause', () async {
      emitPlaying(false);
    });
  }

  @override
  Future<void> seek(Duration position) {
    return _track('seek:${position.inMilliseconds}', () async {
      _position = position;
      if (onSeek != null) {
        await onSeek!(position, this);
      }
      _isCompleted = false;
      if (!_positionController.isClosed) {
        _positionController.add(position);
      }
    });
  }

  @override
  Future<void> dispose() async {
    disposed = true;
    await _playingController.close();
    await _positionController.close();
    await _durationController.close();
    await _videoSizeController.close();
    await _completedController.close();
    await _errorController.close();
  }

  @override
  Widget buildVideoSurface(BuildContext context) {
    return const ColoredBox(
      key: ValueKey('fake-video-surface'),
      color: Colors.black,
    );
  }

  void emitPlaying(bool playing) {
    _isPlaying = playing;
    if (!_playingController.isClosed) {
      _playingController.add(playing);
    }
  }

  void emitMetadata({
    Duration? duration,
    Duration? position,
    Size? videoSize,
  }) {
    if (duration != null) {
      _duration = duration;
      if (!_durationController.isClosed) {
        _durationController.add(duration);
      }
    }
    if (position != null) {
      _position = position;
      if (!_positionController.isClosed) {
        _positionController.add(position);
      }
    }
    if (videoSize != null) {
      _videoSize = videoSize;
      if (!_videoSizeController.isClosed) {
        _videoSizeController.add(videoSize);
      }
    }
  }

  void emitCompleted(bool completed) {
    _isCompleted = completed;
    if (completed) {
      _isPlaying = false;
    }
    if (!_completedController.isClosed) {
      _completedController.add(completed);
    }
  }

  void emitRawCompletedEvent(bool completed) {
    if (!_completedController.isClosed) {
      _completedController.add(completed);
    }
  }

  void emitError(String message) {
    if (!_errorController.isClosed) {
      _errorController.add(message);
    }
  }
}

/// Installs a [FakeValidationVideoBackend] override for tests that mount
/// [DetectionMediaViewer] indirectly and restores any previous override on
/// teardown.
void installFakeValidationVideoBackend([
  ValidationVideoBackendFactory? factory,
]) {
  final previous = ValidationVideoPlayer.debugBackendFactoryOverride;
  ValidationVideoPlayer.debugBackendFactoryOverride =
      factory ?? () => FakeValidationVideoBackend();
  addTearDown(() {
    ValidationVideoPlayer.debugBackendFactoryOverride = previous;
  });
}
