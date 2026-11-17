import 'dart:async';

import 'package:flutter/widgets.dart';

typedef ValidationVideoLogger =
    void Function(String message, {Object? error, StackTrace? stackTrace});

/// Minimal player backend interface used by [ValidationVideoSessionController]
/// so the real widget can run against either media_kit or a deterministic fake.
abstract interface class ValidationVideoBackend {
  bool get isPlaying;
  bool get isCompleted;
  Duration get position;
  Duration get duration;
  Size? get videoSize;
  String? get currentMediaId;

  Stream<bool> get playingStream;
  Stream<Duration> get positionStream;
  Stream<Duration> get durationStream;
  Stream<Size?> get videoSizeStream;
  Stream<bool> get completedStream;
  Stream<String> get errorStream;

  String normalizeMediaId(String path);

  Future<void> open(String path);
  Future<void> play();
  Future<void> pause();
  Future<void> seek(Duration position);
  Future<void> dispose();

  Widget buildVideoSurface(BuildContext context);
}

final RegExp _logUriPattern = RegExp(
  r'[a-zA-Z][a-zA-Z0-9+.-]{1,15}://[^\s<>")\]]+',
);

/// Strips credentials, queries, and fragments from non-file URIs in log text.
String redactValidationVideoLogText(String text) {
  return text.replaceAllMapped(_logUriPattern, (match) {
    final raw = match.group(0)!;
    final uri = Uri.tryParse(raw);
    if (uri == null || !uri.hasScheme || uri.scheme.length <= 1 || uri.scheme == 'file') {
      return raw;
    }
    return uri.replace(userInfo: '', query: '', fragment: '').toString();
  });
}

void _defaultValidationVideoLogger(
  String message, {
  Object? error,
  StackTrace? stackTrace,
}) {
  final buffer = StringBuffer(
    '[ValidationVideoPlayer] ${redactValidationVideoLogText(message)}',
  );
  if (error != null) {
    buffer.write(' | error=${redactValidationVideoLogText(error.toString())}');
  }
  if (stackTrace != null) {
    buffer.write('\n${redactValidationVideoLogText(stackTrace.toString())}');
  }
  debugPrint(buffer.toString());
}

/// Coordinates race-safe video opening, playback intent, visibility restoration,
/// serialized player commands, and error recovery across media transitions.
class ValidationVideoSessionController extends ChangeNotifier {
  ValidationVideoSessionController({
    required this.backend,
    ValidationVideoLogger? logger,
  }) : _log = logger ?? _defaultValidationVideoLogger {
    _subs = [
      backend.playingStream.listen(_handlePlayingEvent),
      backend.positionStream.listen(_handlePositionEvent),
      backend.durationStream.listen(_handleDurationEvent),
      backend.videoSizeStream.listen(_handleVideoSizeEvent),
      backend.completedStream.listen(_handleCompletedEvent),
      backend.errorStream.listen(_handleErrorEvent),
    ];
  }

  final ValidationVideoBackend backend;
  final ValidationVideoLogger _log;

  /// Localized position notifier to avoid rebuilding the whole player subtree.
  final ValueNotifier<Duration> positionNotifier = ValueNotifier(Duration.zero);

  late final List<StreamSubscription<dynamic>> _subs;
  Timer? _positionTimer;
  Timer? _hideTimer;

  String? _currentPath;
  String? _expectedMediaId;
  int _openRequestId = 0;
  int? _activeOpenRequestId;
  Future<void>? _openTask;
  Future<void> _controlChain = Future<void>.value();
  int _pendingControlCount = 0;
  bool _disposed = false;

  bool _isVisible = true;
  bool _isOpening = false;
  bool _isOpened = false;
  bool _isError = false;
  String? _errorMessage;
  bool _playIntent = true;
  bool _isPlaying = false;
  bool _completed = false;
  bool _needsReplaySeek = false;
  bool _replayInProgress = false;
  int _replayToken = 0;
  bool _showControls = true;
  bool _isDragging = false;
  Duration _duration = Duration.zero;
  Size? _videoSize;

  String? get currentPath => _currentPath;
  int get openRequestId => _openRequestId;
  bool get isVisible => _isVisible;
  bool get isOpening => _isOpening;
  bool get isOpened => _isOpened;
  bool get isError => _isError;
  String? get errorMessage => _errorMessage;
  bool get playIntent => _playIntent;
  bool get isPlaying => _isPlaying;
  bool get completed => _completed;
  bool get showControls => _showControls;
  bool get isDragging => _isDragging;
  Duration get duration => _duration;
  Duration get position => positionNotifier.value;
  Size? get videoSize => _videoSize;

  /// Effective play/pause icon state reflecting user intent during open and
  /// pending control transitions.
  bool get displayPlaying {
    if (_isError) return false;
    if (_isOpening || _pendingControlCount > 0) return _playIntent;
    return _isPlaying;
  }

  /// Opens [path] in a fresh media session, resetting stale playback metadata.
  Future<void> open(String path) async {
    if (_disposed) return;
    final requestId = ++_openRequestId;
    _activeOpenRequestId = null;
    _currentPath = path;
    _expectedMediaId = backend.normalizeMediaId(path);
    _isOpening = true;
    _isOpened = false;
    _isError = false;
    _errorMessage = null;
    _playIntent = true;
    _isPlaying = false;
    _completed = false;
    _needsReplaySeek = false;
    _replayInProgress = false;
    _replayToken++;
    _duration = Duration.zero;
    _videoSize = null;
    _showControls = true;
    _stopPositionTimer();
    _hideTimer?.cancel();
    _hideTimer = null;
    if (positionNotifier.value != Duration.zero) {
      positionNotifier.value = Duration.zero;
    }
    notifyListeners();

    final previous = _openTask;
    if (previous != null) {
      await previous.catchError((_) {});
    }
    if (_disposed || requestId != _openRequestId) return;

    final task = _openMedia(path, requestId);
    _openTask = task;
    await task;
  }

  /// Retries loading the current media path after a failure.
  Future<void> retry() {
    final path = _currentPath;
    if (path == null || _disposed) return Future<void>.value();
    return open(path);
  }

  /// Updates tab visibility without losing pending play intent.
  void setVisible(bool visible) {
    if (_disposed || visible == _isVisible) return;
    _isVisible = visible;
    if (!visible) {
      _isPlaying = false;
      _stopPositionTimer();
      _hideTimer?.cancel();
      _hideTimer = null;
      notifyListeners();
      if (_isOpened && !_isError) {
        final requestId = _openRequestId;
        unawaited(
          _enqueueControl(requestId, 'pause-hidden', () async {
            if (_disposed ||
                requestId != _openRequestId ||
                _isVisible ||
                !backend.isPlaying) {
              return;
            }
            await backend.pause();
          }),
        );
      }
    } else {
      if (_isOpened && !_isError && _playIntent && !_completed) {
        unawaited(_applyPlaybackIntent(_openRequestId));
      }
    }
  }

  /// Toggles the current session's play intent and applies it once opened.
  Future<void> togglePlayPause() async {
    if (_disposed || !_isVisible || _isError) return;
    _playIntent = !_playIntent;
    _wakeUpControls(notify: false);
    if (_isOpening || !_isOpened) {
      notifyListeners();
      return;
    }
    if (_playIntent) {
      if (_shouldSeekReplay()) {
        _replayToken++;
        _replayInProgress = true;
      }
      _completed = false;
    } else {
      _replayToken++;
      _replayInProgress = false;
      _isPlaying = false;
      _stopPositionTimer();
      _hideTimer?.cancel();
      _hideTimer = null;
      _showControls = true;
    }
    notifyListeners();
    await _applyPlaybackIntent(_openRequestId);
  }

  /// Seeks the opened media in serialized order after prior player commands.
  Future<void> seek(Duration target) async {
    if (_disposed || _isError || _isOpening || !_isOpened) return;
    var clamped = target;
    if (clamped < Duration.zero) {
      clamped = Duration.zero;
    } else if (_duration > Duration.zero && clamped > _duration) {
      clamped = _duration;
    }
    if (_duration > Duration.zero && clamped < _duration) {
      _completed = false;
    }
    _needsReplaySeek = false;
    _replayToken++;
    _replayInProgress = false;
    positionNotifier.value = clamped;
    final requestId = _openRequestId;
    await _enqueueControl(requestId, 'seek', () async {
      if (_disposed || requestId != _openRequestId || !_isOpened || _isError) {
        return;
      }
      await backend.seek(clamped);
    });
  }

  void setDragging(bool dragging) {
    if (_disposed || _isDragging == dragging) return;
    _isDragging = dragging;
    if (dragging) {
      _hideTimer?.cancel();
      _hideTimer = null;
    } else {
      _startHideTimer();
    }
  }

  void wakeUpControls() {
    _wakeUpControls(notify: true);
  }

  void hideControlsImmediately() {
    if (_disposed) return;
    _hideTimer?.cancel();
    _hideTimer = null;
    if (_showControls) {
      _showControls = false;
      notifyListeners();
    }
  }

  Future<void> _openMedia(String path, int requestId) async {
    await _controlChain.catchError((_) {});
    if (_disposed || requestId != _openRequestId || _isError) return;

    _activeOpenRequestId = requestId;
    try {
      await backend.open(path);
      if (_disposed || requestId != _openRequestId || _isError) return;

      _isOpened = true;
      _isOpening = false;
      _syncInitialStateAfterOpen();
      notifyListeners();

      await _applyPlaybackIntent(requestId);
    } catch (error, stackTrace) {
      _log(
        'Failed to open validation video '
        '(requestId=$requestId, path=${_sanitizePathForLog(path)})',
        error: error,
        stackTrace: stackTrace,
      );
      if (_disposed || requestId != _openRequestId) return;
      _enterErrorState(
        requestId: requestId,
        reason: error.toString(),
        alreadyLogged: true,
      );
    }
  }

  void _syncInitialStateAfterOpen() {
    if (!_matchesCurrentMediaIdentity()) return;
    final initialDuration = backend.duration;
    if (initialDuration > Duration.zero) {
      _duration = initialDuration;
    }
    final initialPosition = backend.position;
    if (initialPosition >= Duration.zero) {
      positionNotifier.value = initialPosition;
    }
    final initialSize = backend.videoSize;
    if (initialSize != null &&
        initialSize.width > 0 &&
        initialSize.height > 0) {
      _videoSize = initialSize;
    }
  }

  Future<void> _applyPlaybackIntent(int requestId) {
    return _enqueueControl(requestId, 'apply-intent', () async {
      if (_disposed || requestId != _openRequestId || !_isOpened || _isError) {
        return;
      }
      if (_isVisible && _playIntent) {
        final int? replayToken;
        if (_shouldSeekReplay()) {
          replayToken = _replayInProgress ? _replayToken : ++_replayToken;
          _replayInProgress = true;
        } else if (_replayInProgress) {
          replayToken = _replayToken;
        } else {
          replayToken = null;
        }
        try {
          if (_shouldSeekReplay()) {
            _completed = false;
            _needsReplaySeek = false;
            positionNotifier.value = Duration.zero;
            await backend.seek(Duration.zero);
            if (_disposed ||
                requestId != _openRequestId ||
                !_isOpened ||
                _isError ||
                !_isVisible ||
                !_playIntent) {
              return;
            }
          }
          await backend.play();
          if (_disposed || requestId != _openRequestId || _isError) return;
          if (!_isVisible || !_playIntent) {
            if (backend.isPlaying) {
              await backend.pause();
            }
            if (_isPlaying) {
              _isPlaying = false;
              _stopPositionTimer();
              _hideTimer?.cancel();
              _hideTimer = null;
              _showControls = true;
              notifyListeners();
            }
            return;
          }
          if (backend.isPlaying && !_isPlaying) {
            _isPlaying = true;
            _startPositionTimer();
            _startHideTimer();
            notifyListeners();
          }
        } finally {
          if (replayToken != null && _replayToken == replayToken) {
            _replayInProgress = false;
          }
        }
      } else {
        if (backend.isPlaying) {
          await backend.pause();
        }
        if (_disposed || requestId != _openRequestId) return;
        if (_isPlaying) {
          _isPlaying = false;
          _stopPositionTimer();
          _hideTimer?.cancel();
          _hideTimer = null;
          _showControls = true;
          notifyListeners();
        }
      }
    });
  }

  Future<void> _enqueuePauseIfHidden(int requestId) {
    return _enqueueControl(requestId, 'pause-hidden', () async {
      if (_disposed ||
          requestId != _openRequestId ||
          _isVisible ||
          !backend.isPlaying) {
        return;
      }
      await backend.pause();
    });
  }

  Future<void> _enqueueControl(
    int requestId,
    String operation,
    Future<void> Function() action,
  ) {
    final completer = Completer<void>();
    _pendingControlCount++;
    _controlChain = _controlChain.catchError((_) {}).then((_) async {
      try {
        if (_disposed || requestId != _openRequestId) return;
        await action();
      } catch (error, stackTrace) {
        _log(
          'Playback control "$operation" failed '
          '(requestId=$requestId, path=${_sanitizePathForLog(_currentPath)})',
          error: error,
          stackTrace: stackTrace,
        );
        if (!_disposed &&
            requestId == _openRequestId &&
            operation != 'pause-hidden' &&
            operation != 'pause-completed' &&
            operation != 'seek') {
          _enterErrorState(
            requestId: requestId,
            reason: error.toString(),
            alreadyLogged: true,
          );
        }
      } finally {
        _pendingControlCount--;
        if (!completer.isCompleted) {
          completer.complete();
        }
      }
    });
    return completer.future;
  }

  void _handlePlayingEvent(bool playing) {
    if (_disposed || _isError || _isOpening || !_isOpened) return;
    if (!_matchesCurrentMediaIdentity()) return;
    if (!_isVisible) {
      if (playing) {
        unawaited(_enqueuePauseIfHidden(_openRequestId));
      }
      return;
    }
    var changed = false;
    if (_isPlaying != playing) {
      _isPlaying = playing;
      changed = true;
    }
    if (playing) {
      _startPositionTimer();
      _startHideTimer();
    } else {
      _syncPlayerState();
      _stopPositionTimer();
      _hideTimer?.cancel();
      _hideTimer = null;
      if (!_showControls) {
        _showControls = true;
        changed = true;
      }
    }
    if (changed) notifyListeners();
  }

  void _handlePositionEvent(Duration position) {
    if (_disposed ||
        _isError ||
        _isOpening ||
        !_isOpened ||
        !_isVisible ||
        _isDragging ||
        _positionTimer != null) {
      return;
    }
    if (!_matchesCurrentMediaIdentity()) return;
    final deltaMs = (position - positionNotifier.value).inMilliseconds.abs();
    if (_isPlaying && deltaMs < 200) return;
    if (position != positionNotifier.value) {
      positionNotifier.value = position;
    }
  }

  void _handleDurationEvent(Duration duration) {
    if (_disposed || _isError || _isOpening || !_isOpened) return;
    if (!_matchesCurrentMediaIdentity()) return;
    if (duration != _duration) {
      _duration = duration;
      notifyListeners();
    }
  }

  void _handleVideoSizeEvent(Size? size) {
    if (_disposed || _isError || _isOpening || !_isOpened) return;
    if (!_matchesCurrentMediaIdentity()) return;
    final normalized = (size != null && size.width > 0 && size.height > 0)
        ? size
        : null;
    if (normalized != null && normalized != _videoSize) {
      _videoSize = normalized;
      notifyListeners();
    }
  }

  void _handleCompletedEvent(bool completed) {
    if (_disposed || _isError || _isOpening || !_isOpened) return;
    if (!_matchesCurrentMediaIdentity()) return;
    if (!completed) {
      _completed = false;
      _needsReplaySeek = false;
      return;
    }
    if (_replayInProgress && _playIntent) return;
    // Reject stale queued completions when the current session is still
    // actively playing or the native backend has not reached EOF.
    if (backend.isPlaying || !backend.isCompleted) return;
    if (_completed) return;
    _completed = true;
    _needsReplaySeek = true;
    _playIntent = false;
    _isPlaying = false;
    _showControls = true;
    _stopPositionTimer();
    _hideTimer?.cancel();
    _hideTimer = null;
    notifyListeners();
    final requestId = _openRequestId;
    unawaited(
      _enqueueControl(requestId, 'pause-completed', () async {
        if (_disposed || requestId != _openRequestId || !_completed) return;
        await backend.pause();
      }),
    );
  }

  void _handleErrorEvent(String message) {
    final trimmed = message.trim();
    if (_disposed || trimmed.isEmpty) return;
    final safeMessage = redactValidationVideoLogText(trimmed);
    if (_activeOpenRequestId != _openRequestId) {
      _log(
        'Ignoring stale native video error '
        '(activeOpenRequestId=$_activeOpenRequestId, '
        'currentRequestId=$_openRequestId, '
        'path=${_sanitizePathForLog(_currentPath)}): $safeMessage',
      );
      return;
    }
    if (!_matchesCurrentMediaIdentity()) {
      _log(
        'Ignoring mismatched native video error '
        '(requestId=$_openRequestId, '
        'expectedMedia=${_sanitizePathForLog(_expectedMediaId)}, '
        'currentMedia=${_sanitizePathForLog(backend.currentMediaId)}): '
        '$safeMessage',
      );
      return;
    }
    _enterErrorState(requestId: _openRequestId, reason: safeMessage);
  }

  void _enterErrorState({
    required int requestId,
    required String reason,
    bool alreadyLogged = false,
  }) {
    if (_disposed || requestId != _openRequestId) return;
    final safeReason = redactValidationVideoLogText(reason);
    if (!alreadyLogged) {
      _log(
        'Validation video playback error '
        '(requestId=$requestId, path=${_sanitizePathForLog(_currentPath)}): '
        '$safeReason',
      );
    }
    _isError = true;
    _errorMessage = safeReason;
    _isOpening = false;
    _isOpened = false;
    _isPlaying = false;
    _playIntent = false;
    _replayToken++;
    _replayInProgress = false;
    _showControls = true;
    _stopPositionTimer();
    _hideTimer?.cancel();
    _hideTimer = null;
    notifyListeners();
  }

  bool _matchesCurrentMediaIdentity() {
    final expected = _expectedMediaId;
    final current = backend.currentMediaId;
    if (expected == null || current == null || current.isEmpty) {
      return true;
    }
    return current == expected;
  }

  bool _shouldSeekReplay() {
    return _completed ||
        _needsReplaySeek ||
        (_duration > Duration.zero && positionNotifier.value >= _duration);
  }

  void _wakeUpControls({required bool notify}) {
    if (_disposed) return;
    var changed = false;
    if (!_showControls) {
      _showControls = true;
      changed = true;
    }
    _startHideTimer();
    if (notify && changed) {
      notifyListeners();
    }
  }

  void _startHideTimer() {
    _hideTimer?.cancel();
    _hideTimer = Timer(const Duration(seconds: 2), () {
      if (_disposed || !_isPlaying || _isDragging) return;
      if (_showControls) {
        _showControls = false;
        notifyListeners();
      }
    });
  }

  void _startPositionTimer() {
    if (!_isVisible || _disposed) return;
    _positionTimer ??= Timer.periodic(
      const Duration(milliseconds: 250),
      (_) => _syncPlayerState(),
    );
  }

  void _stopPositionTimer() {
    _positionTimer?.cancel();
    _positionTimer = null;
  }

  void _syncPlayerState() {
    if (_disposed ||
        !_isVisible ||
        !_isOpened ||
        _isOpening ||
        _isError ||
        _isDragging) {
      return;
    }
    if (!_matchesCurrentMediaIdentity()) return;
    final position = backend.position;
    final duration = backend.duration;
    if (position != positionNotifier.value) {
      positionNotifier.value = position;
    }
    if (duration != _duration) {
      _duration = duration;
      notifyListeners();
    }
  }

  String _sanitizePathForLog(String? path) {
    if (path == null || path.isEmpty) return '<none>';
    return redactValidationVideoLogText(path);
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _openRequestId++;
    _activeOpenRequestId = null;
    _stopPositionTimer();
    _hideTimer?.cancel();
    _hideTimer = null;
    for (final sub in _subs) {
      unawaited(sub.cancel());
    }
    _subs.clear();
    positionNotifier.dispose();
    unawaited(
      backend.dispose().catchError((Object error, StackTrace stackTrace) {
        _log(
          'Failed to dispose validation video backend '
          '(requestId=$_openRequestId, path=${_sanitizePathForLog(_currentPath)})',
          error: error,
          stackTrace: stackTrace,
        );
      }),
    );
    super.dispose();
  }
}
