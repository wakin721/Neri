import 'package:flutter/services.dart';

class ValidationSelectionModifiers {
  final Set<LogicalKeyboardKey> _pressed = <LogicalKeyboardKey>{};

  static final Set<LogicalKeyboardKey> _rangeKeys = <LogicalKeyboardKey>{
    LogicalKeyboardKey.shiftLeft,
    LogicalKeyboardKey.shiftRight,
  };
  static final Set<LogicalKeyboardKey> _toggleKeys = <LogicalKeyboardKey>{
    LogicalKeyboardKey.controlLeft,
    LogicalKeyboardKey.controlRight,
    LogicalKeyboardKey.metaLeft,
    LogicalKeyboardKey.metaRight,
  };

  bool get rangeSelection => _pressed.any(_rangeKeys.contains);
  bool get toggleSelection => _pressed.any(_toggleKeys.contains);

  bool press(LogicalKeyboardKey key) {
    if (!_isSelectionModifier(key)) return false;
    _pressed.add(key);
    return true;
  }

  bool release(LogicalKeyboardKey key) {
    if (!_isSelectionModifier(key)) return false;
    _pressed.remove(key);
    return true;
  }

  void clear() => _pressed.clear();

  bool _isSelectionModifier(LogicalKeyboardKey key) =>
      _rangeKeys.contains(key) || _toggleKeys.contains(key);
}
