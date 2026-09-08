import json
from pathlib import Path
from system.model_sync.catalog import discover_models,resolve_saved_model_path
from system.model_sync.layout import get_model_layout
def test_manifest_exposes_dinov3_capabilities(tmp_path):
    layout=get_model_layout(tmp_path);head=layout.cls_user/'head.pt';head.write_bytes(b'x')
    manifest=layout.cls_user/'head.neri.json';manifest.write_text(json.dumps({'backend':'dinov3','display_name':'DINOv3 reviewed','checkpoint':'head.pt','architecture':'dinov3_vitb16','feature_dim':768,'requires_detector':True,'supports_video_fast':True,'supports_video_all':False}),encoding='utf-8')
    models=discover_models(layout,'cls');assert len(models)==1;model=models[0]
    assert model.backend=='dinov3';assert model.feature_dim==768;assert model.requires_detector;assert not model.supports_video_all
    assert resolve_saved_model_path(str(head.resolve()),models)==str(manifest.resolve())
