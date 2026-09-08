# DINOv3 classifier assets

Neri bundles the reviewed 17-species classifier head and its `.neri.json` manifest. The frozen Meta DINOv3 ViT-B/16 encoder weights and DINOv3 source code are not bundled. Place the encoder at the path declared by `encoder_weights`, and set `NERI_DINOV3_SOURCE` to the directory containing the native `dinov3` package. The encoder file must match the checkpoint SHA-256.
