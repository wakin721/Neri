import 'package:flutter/material.dart';

/// Registers a card in the root overlay's shared, scrollable notification lane.
class NotificationCard extends StatefulWidget {
  const NotificationCard({required this.child, super.key});

  final Widget child;

  @override
  State<NotificationCard> createState() => _NotificationCardState();
}

class _NotificationCardState extends State<NotificationCard> {
  static final _lanes = Expando<_NotificationLane>();
  _NotificationLane? _lane;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _scheduleUpdate();
  }

  @override
  void didUpdateWidget(covariant NotificationCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    _scheduleUpdate();
  }

  void _scheduleUpdate() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final overlay = Overlay.of(context, rootOverlay: true);
      final lane = _lanes[overlay] ??= _NotificationLane(overlay);
      if (_lane != lane) _lane?.remove(this);
      _lane = lane;
      lane.cards[this] = InheritedTheme.captureAll(context, widget.child);
      lane.refresh();
    });
  }

  @override
  void dispose() {
    _lane?.remove(this);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => const SizedBox.shrink();
}

class _NotificationLane {
  _NotificationLane(this.overlay);

  final OverlayState overlay;
  final cards = <_NotificationCardState, Widget>{};
  OverlayEntry? _entry;

  void remove(_NotificationCardState card) {
    cards.remove(card);
    // Disposal can run while the overlay itself is building.
    WidgetsBinding.instance.addPostFrameCallback((_) => refresh());
  }

  void refresh() {
    if (cards.isEmpty || !overlay.mounted) {
      _entry?.remove();
      _entry?.dispose();
      _entry = null;
      return;
    }
    if (_entry != null) {
      _entry!.markNeedsBuild();
      return;
    }
    _entry = OverlayEntry(
      builder: (context) => Positioned.fill(
        child: SafeArea(
          minimum: const EdgeInsets.all(20),
          child: LayoutBuilder(
            builder: (context, constraints) => Align(
              alignment: Alignment.bottomRight,
              child: ConstrainedBox(
                constraints: BoxConstraints(
                  maxWidth: 408,
                  maxHeight: constraints.maxHeight,
                ),
                child: SingleChildScrollView(
                  reverse: true,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      for (final entry in cards.entries) ...[
                        if (entry.key != cards.keys.first)
                          const SizedBox(height: 12),
                        KeyedSubtree(
                          key: ObjectKey(entry.key),
                          child: entry.value,
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
    overlay.insert(_entry!);
  }
}
