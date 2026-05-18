import 'package:flutter/material.dart';

class SelectableListCard<T> extends StatelessWidget {
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
  Widget build(BuildContext context) {
    return Card.filled(
      margin: EdgeInsets.zero,
      clipBehavior: Clip.antiAlias,
      child: ListView.separated(
        padding: padding,
        itemCount: items.length,
        separatorBuilder: (_, __) => const Divider(height: 1),
        itemBuilder: (context, index) {
          final item = items[index];
          final subtitle = subtitleBuilder?.call(item);
          final selected =
              isSelected?.call(index, item) ?? index == selectedIndex;
          Widget buildTile([MenuController? menuController]) {
            final hasMenu =
                menuController != null && menuChildrenBuilder != null;
            return GestureDetector(
              behavior: HitTestBehavior.opaque,
              onSecondaryTapDown: hasMenu || onSecondaryTapDown != null
                  ? (details) {
                      if (menuController != null &&
                          menuChildrenBuilder != null) {
                        onMenuOpening?.call(index, item);
                        if (menuController.isOpen) {
                          menuController.close();
                        }
                        menuController.open(position: details.localPosition);
                        return;
                      }
                      onSecondaryTapDown?.call(details, index, item);
                    }
                  : null,
              onLongPress: onLongPress == null
                  ? null
                  : () => onLongPress!(index, item),
              child: ListTile(
                selected: selected,
                tileColor: tileColorBuilder?.call(index, item),
                leading: leadingBuilder?.call(item),
                title: Text(
                  titleBuilder(item),
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
                trailing: trailingBuilder?.call(item),
                onTap: () => onSelected(index, item),
              ),
            );
          }

          final menuBuilder = menuChildrenBuilder;
          if (menuBuilder != null) {
            // 已移除有问题的 ExcludeSemantics，现在你的右键菜单可以正常展开且不报错了
            return MenuAnchor(
              style: menuStyle,
              clipBehavior: Clip.antiAlias,
              menuChildren: menuBuilder(context, index, item),
              builder: (context, controller, child) {
                return buildTile(controller);
              },
            );
          }

          return buildTile();
        },
      ),
    );
  }
}
