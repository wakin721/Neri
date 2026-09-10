from pathlib import Path

path = Path('frontend/lib/src/api_client_core.dart')
text = path.read_text(encoding='utf-8')
old = '''    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .whereType<Map<String, dynamic>>()
        .map(DinoV3RegistryEntry.fromJson)
        .toList();
  }

  Future<DinoV3RegistryEntry> fetchDinoV3RegistryEntry(
'''
new = '''    final response = await _httpClient.get(uri);
    _ensureSuccess(response);
    final decoded = jsonDecode(response.body);
    if (decoded is! List<dynamic>) {
      // Compatibility with older backends and existing mocked clients that do
      // not expose the unified catalog endpoint yet.
      return fetchDinoV3Registry(classificationModelPath);
    }
    return decoded
        .whereType<Map<String, dynamic>>()
        .map(DinoV3RegistryEntry.fromJson)
        .toList();
  }

  Future<DinoV3RegistryEntry> fetchDinoV3RegistryEntry(
'''
if old not in text:
    raise RuntimeError('catalog compatibility anchor missing')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
