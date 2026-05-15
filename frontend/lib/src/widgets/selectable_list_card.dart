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
    this.padding = EdgeInsets.zero,
    super.key,
  });

  final List<T> items;
  final int selectedIndex;
  final String Function(T item) titleBuilder;
  final String? Function(T item)? subtitleBuilder;
  final Widget Function(T item)? leadingBuilder;
  final Widget? Function(T item)? trailingBuilder;
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
          return ListTile(
            selected: index == selectedIndex,
            leading: leadingBuilder?.call(item),
            title: Text(
              titleBuilder(item),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
            subtitle: subtitle == null
                ? null
                : Text(subtitle, maxLines: 1, overflow: TextOverflow.ellipsis),
            trailing: trailingBuilder?.call(item),
            onTap: () => onSelected(index, item),
          );
        },
      ),
    );
  }
}
