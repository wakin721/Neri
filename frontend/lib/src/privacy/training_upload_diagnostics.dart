import 'privacy_status.dart';

/// Display and clipboard share the same explicit allowlist of diagnostic facts.
class TrainingUploadDiagnostics {
  TrainingUploadDiagnostics._(Map<String, String> facts)
    : facts = Map.unmodifiable(facts);

  factory TrainingUploadDiagnostics.fromJson(Map<String, dynamic> json) {
    final status = PrivacyStatus.fromJson(
      json['status'] as Map<String, dynamic>,
    );
    String number(String key) => json[key] is num ? '${json[key]}' : '未知';
    String mebibytes(String key) => json[key] is num
        ? '${((json[key] as num) / (1024 * 1024)).toStringAsFixed(1)} MiB'
        : '未知';
    final suffixes = (json['image_suffixes'] as List<dynamic>?)
        ?.whereType<String>()
        .join('、');
    final privacy = json['privacy_blurring'];
    final blurring = privacy is Map<String, dynamic> ? privacy : null;
    return TrainingUploadDiagnostics._({
      '参与状态': !status.isConsentComplete
          ? '待选择'
          : status.trainingEnabled
          ? '已参加'
          : '未参加',
      '协议版本':
          '${status.agreementVersion} · ${status.hasCurrentAgreement ? '已同意' : '待确认'}',
      '设备匿名 ID': status.hasDeviceId ? status.deviceId : '当前后端未提供',
      '任务数量':
          '待处理 ${status.stats.pending} · 上传中 ${status.stats.uploading} · '
          '已上传 ${status.stats.uploaded} · 失败 ${status.stats.failed} · 已跳过 ${status.stats.skipped}',
      '后台线程': json['worker_running'] == true ? '运行中（仅在参加后上传）' : '未运行',
      '提交范围': '仅提交参加后人工确认的照片及 YOLO 标注，不扫描照片库、不提交视频或动图；MPO 相机照片取主图。',
      '照片格式': suffixes == null || suffixes.isEmpty ? '未知' : suffixes,
      '照片压缩':
          'JPEG · 最长边 ${number('max_image_edge')} 像素 · 质量 ${number('jpeg_quality')}/100',
      '压缩后大小上限': mebibytes('max_image_bytes'),
      '元数据处理': '移除 GPS、EXIF、XMP、ICC 等元数据，保留本地原图。',
      '上传前打码': blurring?['required'] != true
          ? '服务未确认强制打码状态，请检查后端版本。'
          : blurring?['dependencies_available'] != true
          ? 'LiteRT 隐私运行库不可用，请更新或修复程序；完成前不会上传照片，无需安装 PyTorch。'
          : '本机检测并整体打码人物、车辆、屏幕和书本；打码失败不上传，自动检测仍可能漏检。',
      '打码模型': blurring?['model'] is String
          ? '${blurring!['model']} · ${blurring['model_cached'] == true ? '已缓存（使用前校验）' : '首次处理时从 Ultralytics 官方 GitHub 下载'}'
          : '未知',
      '打码运行库': blurring?['runtime'] == 'LiteRT'
          ? 'LiteRT CPU · 无需安装 PyTorch'
          : '当前后端未提供运行库信息',
      '排除规则': '排除已标记为人或车辆、未确认、未知物种及无效标签的照片。',
      '空照片配额': '每个物理文件夹最多 ${number('max_empty_per_folder')} 张，跨重启保留。',
      '分类与标注': '每个物种分别保存照片、YOLO 检测框 .txt 和类别 .classes.txt；模型检测框未经人工验证。',
      '提交等待': '最后一次修改后等待 ${number('debounce_seconds')} 秒，合并期间的重复修改。',
      '任务检查间隔': '${number('poll_seconds')} 秒（新任务会唤醒后台线程）',
      '请求超时': '${number('request_timeout_seconds')} 秒',
      '失败重试':
          '从 ${number('retry_initial_seconds')} 秒开始指数退避，最长间隔 ${number('retry_max_seconds')} 秒。',
      '默认上传分块': '${mebibytes('default_chunk_bytes')}（实际大小由服务端返回）',
    });
  }

  final Map<String, String> facts;

  String get copyText => [
    'Neri 上传详细设置',
    for (final entry in facts.entries) '${entry.key}：${entry.value}',
  ].join('\n');
}
