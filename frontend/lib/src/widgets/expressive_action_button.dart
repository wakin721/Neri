import 'package:flutter/material.dart';
import 'package:flutter/physics.dart';

/// A medium, prominent action with M3 Expressive spring press feedback.
/// Flutter's FilledButton owns focus, keyboard activation and semantics.
class ExpressiveActionButton extends StatefulWidget {
  const ExpressiveActionButton({
    required this.onPressed,
    required this.icon,
    required this.label,
    super.key,
  });

  final VoidCallback? onPressed;
  final Widget icon;
  final Widget label;

  @override
  State<ExpressiveActionButton> createState() => _ExpressiveActionButtonState();
}

class _ExpressiveActionButtonState extends State<ExpressiveActionButton>
    with SingleTickerProviderStateMixin {
  final _states = WidgetStatesController();
  late final AnimationController _scale = AnimationController.unbounded(
    vsync: this,
    value: 1,
  );
  bool _disableAnimations = false;

  @override
  void initState() {
    super.initState();
    _states.addListener(_updatePress);
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _disableAnimations = MediaQuery.disableAnimationsOf(context);
    if (_disableAnimations) {
      _scale.stop();
      _scale.value = 1;
    }
  }

  void _updatePress() {
    if (_disableAnimations) return;
    final pressed =
        _states.value.contains(WidgetState.pressed) &&
        !_states.value.contains(WidgetState.disabled);
    _scale.animateWith(
      SpringSimulation(
        const SpringDescription(mass: 1, stiffness: 700, damping: 40),
        _scale.value,
        pressed ? 0.96 : 1,
        _scale.velocity,
      ),
    );
  }

  @override
  void dispose() {
    _states.removeListener(_updatePress);
    _states.dispose();
    _scale.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ScaleTransition(
      scale: _scale,
      child: FilledButton.icon(
        statesController: _states,
        onPressed: widget.onPressed,
        style: FilledButton.styleFrom(
          minimumSize: const Size(160, 56),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          textStyle: Theme.of(context).textTheme.titleMedium,
        ),
        icon: widget.icon,
        label: widget.label,
      ),
    );
  }
}
