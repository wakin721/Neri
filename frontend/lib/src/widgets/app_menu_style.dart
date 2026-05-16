import 'package:flutter/material.dart';

class AppMenuOption<T> {
  const AppMenuOption({required this.value, required this.label});

  final T value;
  final String label;
}

class AppMenuButton<T> extends StatelessWidget {
  const AppMenuButton({
    required this.value,
    required this.options,
    required this.onChanged,
    this.placeholder = '请选择',
    this.minMenuWidth = 180,
    this.maxMenuWidth = 360,
    super.key,
  });

  final T? value;
  final List<AppMenuOption<T>> options;
  final ValueChanged<T> onChanged;
  final String placeholder;
  final double minMenuWidth;
  final double maxMenuWidth;

  @override
  Widget build(BuildContext context) {
    final selected = _selectedOption;
    final label = selected?.label ?? placeholder;
    final enabled = options.isNotEmpty;

    return MenuAnchor(
      style: appMenuStyle(context, minWidth: minMenuWidth),
      menuChildren: options.map((option) {
        final selected = option.value == value;
        return MenuItemButton(
          style: appMenuItemStyle(
            context,
            selected: selected,
            minWidth: minMenuWidth,
          ),
          leadingIcon: selected
              ? const Icon(Icons.check_rounded)
              : const SizedBox(width: 24),
          onPressed: () => onChanged(option.value),
          child: ConstrainedBox(
            constraints: BoxConstraints(
              minWidth: minMenuWidth,
              maxWidth: maxMenuWidth,
            ),
            child: Text(option.label, overflow: TextOverflow.ellipsis),
          ),
        );
      }).toList(),
      builder: (context, controller, child) {
        return OutlinedButton(
          onPressed: enabled
              ? () {
                  if (controller.isOpen) {
                    controller.close();
                  } else {
                    controller.open();
                  }
                }
              : null,
          style: OutlinedButton.styleFrom(
            minimumSize: const Size.fromHeight(52),
            padding: const EdgeInsets.symmetric(horizontal: 14),
            alignment: Alignment.centerLeft,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  label,
                  overflow: TextOverflow.ellipsis,
                  textAlign: TextAlign.left,
                ),
              ),
              const SizedBox(width: 8),
              const Icon(Icons.arrow_drop_down_rounded),
            ],
          ),
        );
      },
    );
  }

  AppMenuOption<T>? get _selectedOption {
    for (final option in options) {
      if (option.value == value) return option;
    }
    return null;
  }
}

MenuStyle appMenuStyle(
  BuildContext context, {
  double minWidth = 180,
  EdgeInsetsGeometry padding = const EdgeInsets.symmetric(
    horizontal: 6,
    vertical: 8,
  ),
}) {
  final scheme = Theme.of(context).colorScheme;
  return MenuStyle(
    elevation: const WidgetStatePropertyAll(8),
    backgroundColor: WidgetStatePropertyAll(scheme.surfaceContainerHighest),
    surfaceTintColor: const WidgetStatePropertyAll(Colors.transparent),
    shadowColor: WidgetStatePropertyAll(Colors.black.withValues(alpha: 0.20)),
    minimumSize: WidgetStatePropertyAll(Size(minWidth, 0)),
    padding: WidgetStatePropertyAll(padding),
    shape: WidgetStatePropertyAll(
      RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
    ),
  );
}

MenuStyle appDropdownMenuStyle(BuildContext context, {double minWidth = 180}) {
  final scheme = Theme.of(context).colorScheme;
  return MenuStyle(
    elevation: const WidgetStatePropertyAll(8),
    backgroundColor: WidgetStatePropertyAll(scheme.surfaceContainerHighest),
    surfaceTintColor: const WidgetStatePropertyAll(Colors.transparent),
    shadowColor: WidgetStatePropertyAll(Colors.black.withValues(alpha: 0.20)),
    minimumSize: WidgetStatePropertyAll(Size(minWidth, 0)),
    padding: const WidgetStatePropertyAll(
      EdgeInsets.symmetric(horizontal: 6, vertical: 8),
    ),
    shape: WidgetStatePropertyAll(
      RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
  );
}

ButtonStyle appMenuItemStyle(
  BuildContext context, {
  bool selected = false,
  double minWidth = 180,
}) {
  final scheme = Theme.of(context).colorScheme;
  return MenuItemButton.styleFrom(
    minimumSize: Size(minWidth, 52),
    padding: const EdgeInsets.symmetric(horizontal: 12),
    foregroundColor: selected ? scheme.onSecondaryContainer : scheme.onSurface,
    backgroundColor: selected ? scheme.secondaryContainer : Colors.transparent,
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
    textStyle: Theme.of(context).textTheme.bodyLarge,
  );
}
