import 'dart:async';

import 'package:flutter/foundation.dart';

import 'api_client.dart';
import 'models/model_sync_status.dart';

typedef ModelCatalogChangedCallback = Future<void> Function();

class ModelSyncController extends ChangeNotifier {
  ModelSyncController(
    this._client, {
    this.pollInterval = const Duration(milliseconds: 500),
    required ModelCatalogChangedCallback onCatalogChanged,
  }) : _onCatalogChanged = onCatalogChanged;

  final NeriApiClient _client;
  final Duration pollInterval;
  final ModelCatalogChangedCallback _onCatalogChanged;

  Timer? _pollTimer;
  ModelSyncStatus? _status;
  bool _pollInFlight = false;
  bool _disposed = false;
  String? _catalogRefreshKey;

  ModelSyncStatus? get status => _status;

  Future<ModelSyncStatus> runNow() async {
    final started = await _client.runModelSync();
    _setStatus(started);
    if (started.isActive) {
      _startPolling();
    } else {
      await _refreshCatalogOnce(started);
    }
    return started;
  }

  Future<ModelSyncStatus> refreshStatus() async {
    final previousWasActive = _status?.isActive == true;
    final next = await _client.fetchModelSyncStatus();
    _setStatus(next);
    if (next.isActive) {
      _startPolling();
    } else if (previousWasActive) {
      _stopPolling();
      await _refreshCatalogOnce(next);
    }
    return next;
  }

  void _startPolling() {
    if (_disposed || _pollTimer != null) return;
    _pollTimer = Timer.periodic(pollInterval, (_) {
      unawaited(_pollOnce());
    });
  }

  void _stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  Future<void> _pollOnce() async {
    if (_disposed || _pollInFlight) return;
    _pollInFlight = true;
    try {
      final previousWasActive = _status?.isActive == true;
      final next = await _client.fetchModelSyncStatus();
      if (_disposed) return;
      _setStatus(next);
      if (!next.isActive) {
        _stopPolling();
        if (previousWasActive) {
          await _refreshCatalogOnce(next);
        }
      }
    } finally {
      _pollInFlight = false;
    }
  }

  void _setStatus(ModelSyncStatus value) {
    _status = value;
    if (!_disposed) notifyListeners();
  }

  Future<void> _refreshCatalogOnce(ModelSyncStatus terminalStatus) async {
    final key = terminalStatus.runId ??
        terminalStatus.manifestId ??
        terminalStatus.lastSuccessfulSync ??
        terminalStatus.state;
    if (_catalogRefreshKey == key) return;
    _catalogRefreshKey = key;
    await _onCatalogChanged();
  }

  @override
  void dispose() {
    _disposed = true;
    _stopPolling();
    super.dispose();
  }
}
