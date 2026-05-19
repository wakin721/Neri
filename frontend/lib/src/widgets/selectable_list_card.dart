import 'package:flutter/material.dart';

class SelectableListCard<T> extends StatefulWidget {
  const SelectableListCard({
    required this.items,
    required this.selectedIndex,
    required this.titleBuilder,
    required this.onSelected,
    this.subtitleBuilder,
    this.leadingBuilder,
    this.trailingBuilder,
    this.tileColorBuilder,
    this.isSelected,
    this.menuChildrenBuilder,
    this.menuStyle,
    this.onMenuOpening,
    this.onSecondaryTapDown,
    this.onLongPress,
    this.padding = EdgeInsets.zero,
    super.key,
  });

  final List<T> items;
  final int selectedIndex;
  final String Function(T item) titleBuilder;
  final String? Function(T item)? subtitleBuilder;
  final Widget Function(T item)? leadingBuilder;
  final Widget? Function(T item)? trailingBuilder;
  final Color? Function(int index, T item)? tileColorBuilder;
  final bool Function(int index, T item)? isSelected;
  final List<Widget> Function(BuildContext context, int index, T item)?
  menuChildrenBuilder;
  final MenuStyle? menuStyle;
  final void Function(int index, T item)? onMenuOpening;
  final void Function(TapDownDetails details, int index, T item)?
  onSecondaryTapDown;
  final void Function(int index, T item)? onLongPress;
  final void Function(int index, T item) onSelected;
  final EdgeInsetsGeometry padding;

  @override
  State<SelectableListCard<T>> createState() => _SelectableListCardState<T>();
}

class _SelectableListCardState<T> extends State<SelectableListCard<T>> {
  final _selectedTileKey = GlobalKey();
  int? _lastSelectedIndex;

  @override
  void initState() {
    super.initState();
    _lastSelectedIndex = widget.selectedIndex;
    _scheduleSelectedTileScroll();
  }

  @override
  void didUpdateWidget(covariant SelectableListCard<T> oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.selectedIndex != widget.selectedIndex) {
      _scheduleSelectedTileScroll(previousIndex: oldWidget.selectedIndex);
    }
  }

  void _scheduleSelectedTileScroll({int? previousIndex}) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || widget.selectedIndex < 0) return;
      final selectedContext = _selectedTileKey.currentContext;
      if (selectedContext == null) return;
      final oldIndex = previousIndex ?? _lastSelectedIndex;
      final policy = oldIndex != null && widget.selectedIndex < oldIndex
          ? ScrollPositionAlignmentPolicy.keepVisibleAtStart
          : ScrollPositionAlignmentPolicy.keepVisibleAtEnd;
      _lastSelectedIndex = widget.selectedIndex;
      Scrollable.ensureVisible(
        selectedContext,
        duration: Duration.zero,
        alignmentPolicy: policy,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Card.filled(
      margin: EdgeInsets.zero,
      clipBehavior: Clip.antiAlias,
      child: ListView.separated(
        padding: widget.padding,
        itemCount: widget.items.length,
        separatorBuilder: (_, __) => const Divider(height: 1),
        itemBuilder: (context, index) {
          final item = widget.items[index];
          final subtitle = widget.subtitleBuilder?.call(item);
          final selected =
              widget.isSelected?.call(index, item) ??
              index == widget.selectedIndex;
          final scrollTarget = index == widget.selectedIndex;
          Widget buildTile({
            void Function(TapDownDetails details)? onMenuTapDown,
          }) {
            final hasMenu = onMenuTapDown != null;
            final tile = GestureDetector(
              behavior: HitTestBehavior.opaque,
              onSecondaryTapDown: hasMenu || widget.onSecondaryTapDown != null
                  ? (details) {
                      if (onMenuTapDown != null) {
                        onMenuTapDown(details);
                        return;
                      }
                      widget.onSecondaryTapDown?.call(details, index, item);
                    }
                  : null,
              onLongPress: widget.onLongPress == null
                  ? null
                  : () => widget.onLongPress!(index, item),
              child: ListTile(
                selected: selected,
                tileColor: widget.tileColorBuilder?.call(index, item),
                leading: widget.leadingBuilder?.call(item),
                title: Text(
                  widget.titleBuilder(item),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                subtitle: subtitle == null
                    ? null
                    : Text(
                        subtitle,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                trailing: widget.trailingBuilder?.call(item),
                onTap: () => widget.onSelected(index, item),
              ),
            );
            if (!scrollTarget) return tile;
            return KeyedSubtree(key: _selectedTileKey, child: tile);
          }

          final menuBuilder = widget.menuChildrenBuilder;
          if (menuBuilder != null) {
            // 已移除有问题的 ExcludeSemantics，现在你的右键菜单可以正常展开且不报错了
            return _LazyMenuAnchor<T>(
              item: item,
              index: index,
              menuStyle: widget.menuStyle,
              menuChildrenBuilder: menuBuilder,
              onMenuOpening: widget.onMenuOpening,
              tileBuilder: (context, openMenu) {
                return buildTile(onMenuTapDown: openMenu);
              },
            );
          }

          return buildTile();
        },
      ),
    );
  }
}

class _LazyMenuAnchor<T> extends StatefulWidget {
  const _LazyMenuAnchor({
    required this.item,
    required this.index,
    required this.menuChildrenBuilder,
    required this.tileBuilder,
    this.menuStyle,
    this.onMenuOpening,
  });

  final T item;
  final int index;
  final MenuStyle? menuStyle;
  final List<Widget> Function(BuildContext context, int index, T item)
  menuChildrenBuilder;
  final void Function(int index, T item)? onMenuOpening;
  final Widget Function(
    BuildContext context,
    void Function(TapDownDetails details) openMenu,
  )
  tileBuilder;

  @override
  State<_LazyMenuAnchor<T>> createState() => _LazyMenuAnchorState<T>();
}

class _LazyMenuAnchorState<T> extends State<_LazyMenuAnchor<T>> {
  List<Widget> _menuChildren = const <Widget>[];

  @override
  void didUpdateWidget(covariant _LazyMenuAnchor<T> oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.index != widget.index) {
      _menuChildren = const <Widget>[];
    }
  }

  void _openMenu(
    BuildContext context,
    MenuController controller,
    TapDownDetails details,
  ) {
    widget.onMenuOpening?.call(widget.index, widget.item);
    setState(() {
      _menuChildren = widget.menuChildrenBuilder(
        context,
        widget.index,
        widget.item,
      );
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      if (controller.isOpen) {
        controller.close();
      }
      controller.open(position: details.localPosition);
    });
  }

  @override
  Widget build(BuildContext context) {
    return MenuAnchor(
      style: widget.menuStyle,
      clipBehavior: Clip.antiAlias,
      menuChildren: _menuChildren,
      builder: (context, controller, child) {
        return widget.tileBuilder(
          context,
          (details) => _openMenu(context, controller, details),
        );
      },
    );
  }
}
