"""Build a portable local deployment, keeping manifest publication atomic."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import uuid
from contextlib import nullcontext

import numpy as np

from .encoder import PREPROCESSING, sha256
from .runtime import ARCHITECTURE
from .experiment import PRODUCTION_DISPLAY_NAME, head_type


def publish_model(model, manifest_path, *, encoder_weights=None, encoder_code=None,
                  display_name=PRODUCTION_DISPLAY_NAME, existing_manifest=None, calibration_data=None,
                  publication_guard=None):
    """Publish a new generation; an unsuccessful build keeps the old manifest usable."""
    if model.calibration is None:
        raise ValueError("Calibrate the joint model before publication")
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.name.endswith(".neri.json"):
        raise ValueError("Deployment manifest must end with .neri.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if existing_manifest is None:
        if manifest_path.exists():
            raise FileExistsError("A model already exists; use the update command")
        weights = Path(encoder_weights).resolve()
        code = Path(encoder_code).resolve()
        if not (code / "LICENSE.md").is_file() or not (code / "dinov3/hub/backbones.py").is_file():
            raise ValueError("Official DINOv3 source and LICENSE.md are required")
        digest = sha256(weights)
        if model.provenance.get("encoder_sha256") != digest:
            raise ValueError("Feature provenance and supplied encoder weights differ")
        encoder_backend = "pytorch"
    else:
        digest = existing_manifest["encoder_sha256"]
        if model.provenance.get("encoder_sha256") != digest:
            raise ValueError("Updated model encoder provenance differs")
        encoder_backend = existing_manifest.get("encoder_backend", "pytorch")
        if encoder_backend not in {"pytorch", "litert"}:
            raise ValueError(f"Unsupported DINOv3 encoder_backend: {encoder_backend!r}")
        if encoder_backend == "litert":
            litert_digest = existing_manifest.get("encoder_litert_sha256")
            if not isinstance(litert_digest, str) or len(litert_digest) != 64 or any(
                c not in "0123456789abcdef" for c in litert_digest
            ):
                raise ValueError("Invalid DINOv3 encoder_litert_sha256")
    stem = manifest_path.name.removesuffix(".neri.json")
    generation = manifest_path.parent / f"{stem}-assets-{uuid.uuid4().hex[:12]}"
    generation.mkdir()
    published = False
    try:
        if existing_manifest is None:
            shutil.copy2(weights, generation / "encoder.pth")
            shutil.copytree(code / "dinov3", generation / "encoder-code/dinov3",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copy2(code / "LICENSE.md", generation / "encoder-code/LICENSE.md")
            weights_relative = f"{generation.name}/encoder.pth"
            code_relative = f"{generation.name}/encoder-code"
        else:
            weights_relative = existing_manifest["encoder_weights"]
            code_relative = existing_manifest["encoder_code"] if encoder_backend == "pytorch" else None
        checkpoint = generation / "model.neri-model"
        model.save(checkpoint)
        # Verify the archive before making it discoverable by the software.
        type(model).load(checkpoint)
        payload = dict(schema_version=1, backend="dinov3", architecture=ARCHITECTURE,
                       head_type=head_type(model.inference_recipe), inference_recipe=model.inference_recipe,
                       feature_dim=768, preprocessing=PREPROCESSING,
                       event_aggregation="complete_sequence", requires_detector=True,
                       supports_video_fast=True, supports_video_all=False,
                       display_name=display_name, checkpoint=f"{generation.name}/{checkpoint.name}",
                       checkpoint_sha256=sha256(checkpoint), encoder_sha256=digest,
                       encoder_weights=weights_relative)
        if encoder_backend == "litert":
            payload["encoder_backend"] = "litert"
            payload["encoder_litert_sha256"] = existing_manifest["encoder_litert_sha256"]
        else:
            payload["encoder_code"] = code_relative
            if existing_manifest is not None and "encoder_backend" in existing_manifest:
                payload["encoder_backend"] = "pytorch"
        if existing_manifest is not None:
            for key in ('shared_learning','learning_family_id','learning_base','learning_base_sha256',
                        'learning_base_calibration_id','learning_encoder_hashes','learning_encoder_variants'):
                if key in existing_manifest:
                    payload[key]=existing_manifest[key]
        if calibration_data is not None:
            fields = {"features", "labels", "cameras", "sequences"}
            optional = {"sites", "contents", "source_contents", "adapted_features", "datetimes"}
            if not fields <= calibration_data.keys():
                raise ValueError("Complete calibration inputs are required")
            data = {k: np.asarray(calibration_data[k]) for k in fields | optional if k in calibration_data}
            if any(value.dtype.hasobject for value in data.values()):
                raise ValueError("Calibration inputs must not contain pickle/object arrays")
            claimed = str(calibration_data.get("encoder_sha256", digest))
            if claimed != digest:
                raise ValueError("Calibration encoder provenance differs")
            data["encoder_sha256"] = np.asarray(digest)
            calibration = generation / "calibration.npz"
            np.savez_compressed(calibration, **data)
            payload.update(calibration=f"{generation.name}/{calibration.name}",
                           calibration_sha256=sha256(calibration))
        elif existing_manifest is not None and existing_manifest.get("calibration"):
            payload.update(calibration=existing_manifest["calibration"],
                           calibration_sha256=existing_manifest["calibration_sha256"])
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=manifest_path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        try:
            with (publication_guard() if publication_guard else nullcontext()):
                if existing_manifest is not None and existing_manifest.get('learning_family_id'):
                    latest=json.loads(manifest_path.read_text('utf8'))
                    # Preserve a newly registered peer and the user's current
                    # backend choice while publishing the common learned head.
                    payload['learning_encoder_variants']={
                        **payload.get('learning_encoder_variants',{}),
                        **latest.get('learning_encoder_variants',{})}
                    payload['learning_encoder_hashes']=sorted(set(payload.get('learning_encoder_hashes',[]))
                                                              |set(latest.get('learning_encoder_hashes',[])))
                    for key in ('encoder_backend','encoder_weights','encoder_code','encoder_litert_sha256','display_name'):
                        if key in latest:
                            payload[key]=latest[key]
                        else:
                            payload.pop(key,None)
                    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
                temp.replace(manifest_path)
                published = True
        finally:
            temp.unlink(missing_ok=True)
    except Exception:
        # A bookkeeping error after the atomic manifest swap must never delete
        # the now-active generation. Its receipts allow a durable retry.
        if not published:
            shutil.rmtree(generation)
        raise
    return manifest_path
