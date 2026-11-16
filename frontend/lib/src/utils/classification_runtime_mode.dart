import '../models/settings.dart';

enum ClassificationRuntimeMode {
  cpu('litert', 'CPU（LiteRT FP32）'),
  gpu('pytorch', 'GPU（PyTorch）');

  const ClassificationRuntimeMode(this.encoderBackend, this.label);

  final String encoderBackend;
  final String label;
}

class ClassificationRuntimeResolution {
  const ClassificationRuntimeResolution({
    required this.selectedModel,
    required this.familyId,
    required this.activeMode,
    required this.cpuModel,
    required this.gpuModel,
    required this.pytorchAvailable,
    required this.hardwareGpuAvailable,
  });

  final ModelInfo selectedModel;
  final String familyId;
  final ClassificationRuntimeMode? activeMode;
  final ModelInfo? cpuModel;
  final ModelInfo? gpuModel;
  final bool pytorchAvailable;
  final bool hardwareGpuAvailable;

  bool get cpuAvailable => cpuModel != null;
  bool get gpuAvailable =>
      gpuModel != null && pytorchAvailable && hardwareGpuAvailable;

  String? pathForMode(ClassificationRuntimeMode mode) {
    return switch (mode) {
      ClassificationRuntimeMode.cpu => cpuModel?.path,
      ClassificationRuntimeMode.gpu => gpuModel?.path,
    };
  }

  bool isModeAvailable(ClassificationRuntimeMode mode) {
    return switch (mode) {
      ClassificationRuntimeMode.cpu => cpuAvailable,
      ClassificationRuntimeMode.gpu => gpuAvailable,
    };
  }
}

/// Resolves CPU (LiteRT) / GPU (PyTorch) runtime peers sharing [learningFamilyId].
ClassificationRuntimeResolution? resolveClassificationRuntimeMode(
  NeriSettings? settings,
  String? selectedClassificationModelPath,
) {
  if (settings == null) return null;
  final target = selectedClassificationModelPath?.trim() ?? '';
  if (target.isEmpty) return null;

  final models = settings.availableClassificationModels;
  ModelInfo? selectedModel;
  for (final model in models) {
    if (model.path == target || model.checkpointPath == target) {
      selectedModel = model;
      break;
    }
  }
  if (selectedModel == null) return null;

  final familyId = selectedModel.learningFamilyId?.trim() ?? '';
  if (familyId.isEmpty) return null;

  final cpuModel = _findFamilyBackendModel(
    models,
    selectedModel: selectedModel,
    familyId: familyId,
    encoderBackend: ClassificationRuntimeMode.cpu.encoderBackend,
  );
  final gpuModel = _findFamilyBackendModel(
    models,
    selectedModel: selectedModel,
    familyId: familyId,
    encoderBackend: ClassificationRuntimeMode.gpu.encoderBackend,
  );

  final activeBackend = _normalizedEncoderBackend(selectedModel);
  final activeMode = switch (activeBackend) {
    'litert' => ClassificationRuntimeMode.cpu,
    'pytorch' => ClassificationRuntimeMode.gpu,
    _ => null,
  };

  return ClassificationRuntimeResolution(
    selectedModel: selectedModel,
    familyId: familyId,
    activeMode: activeMode,
    cpuModel: cpuModel,
    gpuModel: gpuModel,
    pytorchAvailable: settings.pytorchAvailable,
    hardwareGpuAvailable: settings.gpuAvailable,
  );
}

ModelInfo? _findFamilyBackendModel(
  List<ModelInfo> models, {
  required ModelInfo selectedModel,
  required String familyId,
  required String encoderBackend,
}) {
  if ((selectedModel.learningFamilyId?.trim() ?? '') == familyId &&
      _normalizedEncoderBackend(selectedModel) == encoderBackend) {
    return selectedModel;
  }
  for (final model in models) {
    if ((model.learningFamilyId?.trim() ?? '') == familyId &&
        _normalizedEncoderBackend(model) == encoderBackend) {
      return model;
    }
  }
  return null;
}

String _normalizedEncoderBackend(ModelInfo model) {
  return model.encoderBackend?.trim().toLowerCase() ?? '';
}
