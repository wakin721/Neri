# Joint photo/video sequences

Extend the existing DINOv3 fast-video pipeline so trusted capture times allow sampled video frames and photographs from the same camera to share the existing 30-minute event grouping and strictly causal past-event context. Unknown/sentinel times retain independent per-video behavior. Filesystem creation/modification times never become capture times. Preserve photo-only output and existing CPU/GPU/LiteRT FP32 encoding.

Photo EXIF is commonly a camera-local clock; embedded video timestamps may be UTC. Normalize both consistently under an explicit camera-clock policy, preserving source and assumption metadata. Missing trustworthy timestamps must not borrow neighboring filenames or decoder completion order.

The user confirmed UTC+8 for camera-local timestamps. Interpret naive capture dates with a fixed +08:00 offset; preserve explicit offsets and normalize the common timeline to UTC. Record this policy in joint results and cache signatures.

Encode each crop once. Mixed jobs defer final photo/video classification until all observations are available. Reuse photo event/track grouping, history adaptation, rejection handling and clustering through a shared finalization adapter. Keep distinct same-frame objects separate and persist each file in its existing layout with video frame indices, timestamps and globally valid observation IDs.

Each photo crop has evidence weight 1; each video's target trajectory has total weight 1 split across its frames. Weighted classification and rejection pooling prevent repeated video frames overwhelming photos. No-weights inference remains unchanged; encoder/checkpoints/thresholds remain unchanged. Validate both production and experiment recipes; weighted mixed scores are a new inference policy and must not be claimed calibrated without separate mixed-event validation.

Persist complete video frame features transactionally with source identity and model/settings signature, so cancellation/resume can rebuild the joint result without recoding completed videos. Only reuse final decisions after whole mixed-event finalization; adding/recovering photos must re-finalize cached video features. Error/cancellation leaves partial status rather than falsely complete sequences.

Deploy only verified changed backend files to the detected local installation, staging and backing up every overwritten file with SHA-256 and rollback instructions. Preserve models, calibration, annotations, settings and unrelated working-tree changes. Validate with installed Python and a real mixed-media smoke test before reporting completion.

Local installation inspection found `video_mode=skip`. Enable `fast` as part of the requested video feature deployment, explicitly backing up the setting and preserving every other settings key.
