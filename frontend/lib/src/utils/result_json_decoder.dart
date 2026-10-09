import 'dart:convert';

import 'package:flutter/foundation.dart';

import '../models/job.dart';

const _backgroundDecodeThreshold = 64 * 1024;

Future<List<DetectionItem>> decodeDetectionItems(String body) async {
  return body.length < _backgroundDecodeThreshold
      ? _parseItems(body)
      : compute(_parseItems, body);
}

Future<DetectionItem> decodeDetectionItem(String body) async {
  return body.length < _backgroundDecodeThreshold
      ? _parseItem(body)
      : compute(_parseItem, body);
}

Future<ProcessingJob> decodeProcessingJob(String body) async {
  return body.length < _backgroundDecodeThreshold
      ? _parseJob(body)
      : compute(_parseJob, body);
}

List<DetectionItem> _parseItems(String body) =>
    (jsonDecode(body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DetectionItem.fromJson)
        .toList();

DetectionItem _parseItem(String body) =>
    DetectionItem.fromJson(jsonDecode(body) as Map<String, dynamic>);

ProcessingJob _parseJob(String body) =>
    ProcessingJob.fromJson(jsonDecode(body) as Map<String, dynamic>);
