# Conversion record: MDV6-yolov10-c-tflite

This package retains the supplied runtime bytes under the recorded upstream terms. Sparrow did not train or reconvert those bytes in this remediation.

- Model ID: `MDV6-yolov10-c-tflite`
- Model title: MDV6-yolov10-c-tflite
- Creator/developer: Microsoft AI for Good Lab (AI4G)
- Rights holder (recorded): Microsoft AI for Good Lab (AI4G)
- Rights status: `verified`
- Rights record: `model-zoo-rights-adjudication-v1#3`
- Final hosting status: `hosted`
- Catalog hosting status: `hosted`
- Original source: https://raw.githubusercontent.com/microsoft/MegaDetector/main/README.md
- Terms evidence source: https://raw.githubusercontent.com/microsoft/MegaDetector/main/README.md
- Catalog weight terms: `AGPL-3.0`
- Framework terms (separate from weight authority): none recorded
- Commercial use: `true`; status `allowed`
- Restrictions (recorded): copyleft, source_offer
- Conversion permission: `verified`
- Source version (catalog): v6
- Required attribution/reference (catalog): MegaDetector V6 yolov10-c TFLite conversion; inherits MDV6-yolov10-c; https://raw.githubusercontent.com/microsoft/MegaDetector/main/docs/model_zoo.md

Frozen audit records (historical evidence, not a new grant or conversion run):

- Recorded weight terms: AGPL-3.0
- Terms/conflicts: CITATION REPAIRED IN ROUND 3, on subarea-camera-trap-detectors-r3's finding. My first pass auto-derived this row's source URL from the catalog reference field and landed on doi.org/10.5281/zenodo.15398270 - a record whose metadata declares cc-by-4.0, i.e. a source that does not state the grant recorded here. That is the same 'cited source does not publish the claim' defect I enforced on contact rows and had not enforced on rights rows. The source that DOES state it, which the peer supplied and I verified, is microsoft/MegaDetector README l.423: 'The MegaDetector CODE in this repository is released under the MIT License. The MODEL WEIGHTS carry per-variant licenses, MIT, Apache-2.0, or AGPL-3.0 depending on the variant, so check the license of the specific model you deploy.' That sentence is first-party and expressly WEIGHTS-SCOPED, which is exactly what PASS-OPEN-SOURCE requires and what the peer's own suggested citation supplies. UNRESOLVED CONFLICT CARRIED: the rights holder's own client code fetches all nine variants from Zenodo 15398270, whose record-level field says cc-by-4.0. I read the per-variant weights-scoped statement as governing because that record is disproved as a per-file grant - it and 13357337 both blanket-license ULiege HerdNet checkpoints their depositor cannot relicense - but the conflict is real and a promotion consumer must see it.

This generator changes packaging/compliance text only; it performs no model conversion.

Frozen audit conversion assessment (not a reproduced recipe):

The shipped artifact is a GRAPH-MODIFIED conversion (quantisation, embedded NMS or a stripped output layer), not a straight format export. CC 4.0 s2(a)(4)'s 'changing format never produces Adapted Material' safe harbour is therefore NOT clearly available, so under a CC grant this artifact may be Adapted Material and any ShareAlike or NoDerivatives clause may attach. Under an OSI grant modification is permitted outright and only the notice and change-statement duties follow.

## Recorded tool, version, graph and precision details

```json
{}
```

Unrecorded quantization, embedded non-maximum suppression, removed heads, source-checkpoint hashes and converter identity are not inferred from filenames.

## Explicit unknowns

- `conversion.command`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `conversion.converter`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `conversion.graph_nms`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `conversion.quantization`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `conversion.removed_heads`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `conversion.tool`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `conversion.tool_version`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `copyleft_version_choice`: The recorded identifier does not resolve only versus or-later; the generic licence's application example is not a model-specific grant.
- `corresponding_source_route`: No independently verified complete corresponding-source bundle or written source offer was supplied. An upstream URL is not a source offer.
- `frozen_evidence.evidence_gaps_r3`: PROVENANCE NOT PINNED (subarea-camera-trap-detectors-r3's D-R3-2, verified by me at source and unrebutted). Upstream declares the MDv6 weights refreshed in place, and Zenodo 18177050 - same concept DOI 10.5281/zenodo.10023413 - carries THREE distinct artifacts whose names all map to the single catalog id MDV6-yolov10-e: MDV6-yolov10-e.pt (64,097,586 B, md5 37ec2a0c...), MDV6-yolov10-e-1280.pt (127,821,713 B, md5 5dba53d4...) and MDV6-yolov10-e-1280-v2.pt (64,247,993 B, md5 c704e98d...). No Sparrow package records an upstream checksum or an acquisition date. So the shipped artifact cannot be shown to be the one the per-variant licence table describes TODAY, and a table that is edited in place carries no history. This does not move the verdict - every yolov10-e variant is granted the same licence by that table - but it means the grant is attached to a NAME, not to a payload, and it is the single cheapest thing Sparrow could fix about its own provenance record.
- `frozen_evidence.weights_grant_note_r3`: CITATION REPAIRED IN ROUND 3, on subarea-camera-trap-detectors-r3's finding. My first pass auto-derived this row's source URL from the catalog reference field and landed on doi.org/10.5281/zenodo.15398270 - a record whose metadata declares cc-by-4.0, i.e. a source that does not state the grant recorded here. That is the same 'cited source does not publish the claim' defect I enforced on contact rows and had not enforced on rights rows. The source that DOES state it, which the peer supplied and I verified, is microsoft/MegaDetector README l.423: 'The MegaDetector CODE in this repository is released under the MIT License. The MODEL WEIGHTS carry per-variant licenses, MIT, Apache-2.0, or AGPL-3.0 depending on the variant, so check the license of the specific model you deploy.' That sentence is first-party and expressly WEIGHTS-SCOPED, which is exactly what PASS-OPEN-SOURCE requires and what the peer's own suggested citation supplies. UNRESOLVED CONFLICT CARRIED: the rights holder's own client code fetches all nine variants from Zenodo 15398270, whose record-level field says cc-by-4.0. I read the per-variant weights-scoped statement as governing because that record is disproved as a per-file grant - it and 13357337 both blanket-license ULiege HerdNet checkpoints their depositor cannot relicense - but the conflict is real and a promotion consumer must see it.
- `original_copyright_notice`: A complete model-specific original licence/copyright notice was not supplied. Catalog attribution is reproduced without inventing a copyright year or grant.
- `source_artifact.acquisition_date`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `source_artifact.file`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `source_artifact.revision`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `source_artifact.sha256`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.
- `source_artifact.url`: Unavailable in the supplied structured provenance; no checkpoint identity or conversion detail has been inferred.

## Retained runtime-resource SHA-256s

| Retained resource | Bytes | SHA-256 |
| --- | ---: | --- |
| `MDV6-yolov10-c.tflite` | 4717072 | `d66f8fc9ca1f2c1b7c7ec322d4af67964e0ecf24d5a5e2f05ae6f8b76c7d23ef` |
| `labels.txt` | 24 | `ddf6be5b8d9a976a6b0418ef0acb0a1e23575baae5c02ef853a8dfbb962cc335` |
| `manifest.toml` | 1713 | `622174c0d6af4ac5982c9d8a49204cf43b101b3289c32710506eba0339495a18` |
