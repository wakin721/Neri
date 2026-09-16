def test_server_manifest_root_is_dinov2():
    from server.model_distribution.manifest import DINO_ROOT, DinoV2ManifestBuilder

    assert DINO_ROOT.endswith("/DINOv2")
    assert DinoV2ManifestBuilder.__name__ == "DinoV2ManifestBuilder"
