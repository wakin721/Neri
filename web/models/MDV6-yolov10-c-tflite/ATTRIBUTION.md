# Attribution: MDV6-yolov10-c-tflite

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

Attribution does not imply endorsement. Retain the rights holder's notices and the terms of any separately acquired model.
