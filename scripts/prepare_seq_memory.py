"""Freeze current reviewed-folder inputs against the existing sequence catalog."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

SOURCE = Path(r"F:\files\python\Neri_plus")
REVIEWED = SOURCE / "data/by_species/00已校验"
CATALOG = SOURCE / "data/sequence_recovered_20261006"
OUT = Path(r"E:\Files\Neri\runs\seq_memory_reviewed_20261009")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "classification_bank.jsonl").exists():
        raise FileExistsError("Frozen manifest already exists; preserve this run")
    sys.path.insert(0, str(SOURCE))
    from pipeline.sequence_rules import select_verified_bank

    files = sorted(p for p in REVIEWED.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    by_id = {p.stem: p for p in files}
    if len(by_id) != len(files):
        raise ValueError("Duplicate reviewed crop IDs")
    verified = read_rows(CATALOG / "verified_crops.jsonl")
    if set(by_id) != {r["id"] for r in verified}:
        raise ValueError("Reviewed folder changed; rebuild the sequence catalog before training")
    print(f"Audit {len(files)} reviewed image contents", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(zip(by_id, pool.map(digest, by_id.values())))
    for row in verified:
        file = by_id[row["id"]]
        if row["species"] != file.relative_to(REVIEWED).parts[0] or row["sha256"] != hashes[row["id"]]:
            raise ValueError(f"Reviewed label/content changed: {row['id']}")
        row["file"] = str(file)
        row["review_file"] = str(file)
    bank = select_verified_bank(verified)
    prior = read_rows(CATALOG / "classification_bank.jsonl")
    if {r["id"] for r in bank} != {r["id"] for r in prior}:
        raise ValueError("Sequence bank selection does not replay")
    manifest = OUT / "classification_bank.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bank), encoding="utf-8")
    counts = Counter(r["species"] for r in bank)
    audit = dict(status="passed", reviewed_root=str(REVIEWED), reviewed_images=len(files),
                 verified_hashes=len(hashes), classes=len(counts), sequence_bank_images=len(bank),
                 sequence_catalog=str(CATALOG), sequence_manifest_sha256=digest(manifest),
                 selection="one reviewed crop per species/camera/acquisition; sharpness, then area; content deduplication",
                 initial_bank_policy="All verified folder species retained as trusted initial classes; single-acquisition species do not meet incremental new-class admission requirements.",
                 single_acquisition_species=sorted(c for c, n in counts.items() if n < 2),
                 class_counts=dict(sorted(counts.items())))
    (OUT / "input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "class_counts"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
