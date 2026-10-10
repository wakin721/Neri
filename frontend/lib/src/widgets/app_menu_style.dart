import 'dart:math' as math;

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
              borderRadius: BorderRadius.circular(16),
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

class AppFormMenu<T> extends StatelessWidget {
  const AppFormMenu({
    required this.value,
    required this.label,
    required this.helperText,
    required this.leadingIcon,
    required this.options,
    required this.onSelected,
    this.enabled = true,
    super.key,
  });

  final T? value;
  final String label;
  final String helperText;
  final Widget leadingIcon;
  final List<DropdownMenuEntry<T>> options;
  final ValueChanged<T?> onSelected;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    final selected = options.where((option) => option.value == value);
    final selectedLabel = selected.isEmpty ? '请选择' : selected.first.label;
    final menuWidth = MediaQuery.sizeOf(
      context,
    ).width.clamp(180.0, 360.0).toDouble();
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth.clamp(180.0, menuWidth).toDouble();
        return MenuAnchor(
          style: appMenuStyle(context, minWidth: width),
          menuChildren: [
            ConstrainedBox(
              constraints: BoxConstraints(
                maxHeight: appMenuMaxHeight(context),
                minWidth: width,
              ),
              child: SingleChildScrollView(
                primary: false,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    for (final option in options)
                      MenuItemButton(
                        style: appMenuItemStyle(
                          context,
                          selected: option.value == value,
                          minWidth: width - 12,
                        ),
                        leadingIcon: option.value == value
                            ? const Icon(Icons.check_rounded)
                            : const SizedBox(width: 24),
                        onPressed: enabled && option.enabled
                            ? () => onSelected(option.value)
                            : null,
                        child: SizedBox(
                          width: width - 60,
                          child: Text(
                            option.label,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ],
          builder: (context, controller, child) => Semantics(
            button: true,
            enabled: enabled,
            child: InkWell(
              borderRadius: BorderRadius.circular(16),
              onTap: enabled
                  ? () => controller.isOpen
                        ? controller.close()
                        : controller.open()
                  : null,
              child: InputDecorator(
                isFocused: controller.isOpen,
                isEmpty: false,
                decoration: InputDecoration(
                  labelText: label,
                  helperText: helperText,
                  enabled: enabled,
                  prefixIcon: leadingIcon,
                  suffixIcon: const Icon(Icons.arrow_drop_down_rounded),
                ),
                child: Text(selectedLabel, overflow: TextOverflow.ellipsis),
              ),
            ),
          ),
        );
      },
    );
  }
}

double appMenuMaxHeight(BuildContext context) {
  final media = MediaQuery.of(context);
  final availableHeight =
      media.size.height - media.padding.vertical - media.viewInsets.vertical;
  return math.min(320, math.max(64, availableHeight / 2 - 80));
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
    backgroundColor: WidgetStatePropertyAll(scheme.surfaceContainerHigh),
    surfaceTintColor: const WidgetStatePropertyAll(Colors.transparent),
    shadowColor: WidgetStatePropertyAll(scheme.shadow.withValues(alpha: 0.20)),
    minimumSize: WidgetStatePropertyAll(Size(minWidth, 0)),
    padding: WidgetStatePropertyAll(padding),
    shape: WidgetStatePropertyAll(
      RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
    ),
  );
}

MenuStyle appDropdownMenuStyle(BuildContext context, {double minWidth = 180}) {
  final scheme = Theme.of(context).colorScheme;
  return MenuStyle(
    elevation: const WidgetStatePropertyAll(8),
    backgroundColor: WidgetStatePropertyAll(scheme.surfaceContainerHigh),
    surfaceTintColor: const WidgetStatePropertyAll(Colors.transparent),
    shadowColor: WidgetStatePropertyAll(scheme.shadow.withValues(alpha: 0.20)),
    minimumSize: WidgetStatePropertyAll(Size(minWidth, 0)),
    padding: const WidgetStatePropertyAll(
      EdgeInsets.symmetric(horizontal: 6, vertical: 8),
    ),
    shape: WidgetStatePropertyAll(
      RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
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
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    textStyle: Theme.of(context).textTheme.bodyLarge,
  );
}
