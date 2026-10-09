import 'package:flutter/material.dart';

/// Keeps tab state without rebuilding hidden content on unrelated updates.
class RetainedTab extends StatefulWidget {
  const RetainedTab({
    required this.active,
    required this.builder,
    this.preload = false,
    this.refreshKey,
    super.key,
  });

  final bool active;
  final WidgetBuilder builder;
  // Service-hosting tabs still receive relevant changes while hidden.
  final bool preload;
  final Object? refreshKey;

  @override
  State<RetainedTab> createState() => _RetainedTabState();
}

class _RetainedTabState extends State<RetainedTab> {
  Widget? _content;
  Object? _lastRefreshKey;
  bool? _lastActive;

  @override
  Widget build(BuildContext context) {
    if (widget.active ||
        (widget.preload &&
            (_content == null ||
                _lastRefreshKey != widget.refreshKey ||
                _lastActive != widget.active))) {
      _content = widget.builder(context);
    }
    _lastRefreshKey = widget.refreshKey;
    _lastActive = widget.active;
    return TickerMode(
      enabled: widget.active,
      child: ExcludeSemantics(
        excluding: !widget.active,
        child: FocusScope(
          canRequestFocus: widget.active,
          child: _content ?? const SizedBox.shrink(),
        ),
      ),
    );
  }
}
