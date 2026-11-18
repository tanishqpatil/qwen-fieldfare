"""Verify and reuse the local S3-00b arrays without replaying any controls."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("captures", type=Path)
parser.add_argument("evidence_b", type=Path)
parser.add_argument("working", type=Path)
args = parser.parse_args()
controls = json.loads((args.evidence_b / "comparison.json").read_text())
for name in ("histories.json", "selection.json", "mlx-device.json"):
    data = (args.captures / name).read_bytes()
    if data != (args.evidence_b / name).read_bytes():
        raise ValueError(f"Control provenance changed: {name}")
    (args.working / name).write_bytes(data)
count = 0
for history, control in controls.items():
    dest = args.working / history
    dest.mkdir(exist_ok=True)
    for position, sample in control["samples"].items():
        for mode, expected in sample["modes"].items():
            for array in ("logits", "hidden"):
                path = args.captures / history / mode / f"{position}-{array}.f32"
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected[f"{array}_sha256"]:
                    raise ValueError(f"Unmatched control: {history}/{mode}/{position}/{array}")
                count += 1
    for mode in control["records"]:
        (dest / mode).symlink_to((args.captures / history / mode).resolve(), target_is_directory=True)
    (dest / "residency.json").symlink_to((args.captures / history / "residency.json").resolve())
print(f"Verified {count} original raw array digests against S3-00b evidence")
