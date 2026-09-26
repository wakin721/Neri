import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class InputFolderField extends StatefulWidget {
  const InputFolderField({required this.controller, super.key});

  final TextEditingController controller;

  @override
  State<InputFolderField> createState() => _InputFolderFieldState();
}

class _InputFolderFieldState extends State<InputFolderField> {
  static const _dialogsChannel = MethodChannel('neri/dialogs');

  bool _selecting = false;

  Future<void> _selectInputFolder() async {
    if (_selecting) return;
    setState(() => _selecting = true);
    try {
      final selected = await _dialogsChannel.invokeMethod<String>(
        'chooseDirectory',
        <String, String>{'initialDirectory': widget.controller.text.trim()},
      );
      if (!mounted || selected == null || selected.isEmpty) return;
      widget.controller
        ..text = selected
        ..selection = TextSelection.collapsed(offset: selected.length);
    } on PlatformException catch (error) {
      if (!mounted) return;
      final message = error.message?.isNotEmpty == true
          ? error.message!
          : '无法打开目录选择器';
      ScaffoldMessenger.maybeOf(
        context,
      )?.showSnackBar(SnackBar(content: Text(message)));
    } finally {
      if (mounted) setState(() => _selecting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: widget.controller,
      decoration: InputDecoration(
        labelText: '输入文件夹',
        hintText: '/path/to/camera-trap-folder',
        border: const OutlineInputBorder(),
        suffixIcon: IconButton(
          tooltip: '选择文件夹',
          onPressed: _selecting ? null : _selectInputFolder,
          icon: _selecting
              ? const SizedBox.square(
                  dimension: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.folder_open_rounded),
        ),
      ),
    );
  }
}
