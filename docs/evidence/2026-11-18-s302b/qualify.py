#!/usr/bin/env python3
"""Run frozen Set and bitmap binaries serially. Publish only bounded reviewed metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import os
from statistics import fmean, stdev
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-11-18-s302a"))
from run_profile import OWNERS, checkpoint_metadata, memory_snapshot

IDENTICAL = (
    "label", "prompt_sha256", "prompt_count", "output_sha256", "raw_output_ids",
    "output_count", "finish", "metal_bytes", "hits", "misses", "fetched_bytes",
    "reused", "fresh", "reason", "hidden_logits_sha256",
)
OPTIONAL = ("expert_sha256_by_layer", "validated_tool", "cancelled")


def binary(checkout, name):
    product = checkout / ".build" / "release" / name
    if name.endswith(".xctest"):
        product /= "Contents/MacOS/" + name.removesuffix(".xctest")
    return product


def preflight(model, checkpoint):
    if checkpoint_metadata(model) != checkpoint:
        raise RuntimeError("Checkpoint identity changed")
    processes = subprocess.check_output(["ps", "-axo", "comm"], text=True).splitlines()
    if any(OWNERS.search(proc.strip()) or "QwenFieldfareServerTests" in proc for proc in processes):
        raise RuntimeError("Another model owner is running")
    value = memory_snapshot()
    if value["pressureLevel"] != 1:
        raise RuntimeError("Memory pressure is not normal")
    return value


def run(checkout, model, checkpoint, phase, length, destination):
    before = preflight(model, checkpoint)
    with tempfile.TemporaryDirectory(prefix="qwen-bitmap-") as tmp:
        path = Path(tmp) / "trace.json"
        env = dict(os.environ, QWEN_BITMAP_CHECKPOINT=str(model), QWEN_BITMAP_OUTPUT=str(path),
                   QWEN_BITMAP_PHASE=phase, QWEN_BITMAP_LIMIT=str(length))
        result = subprocess.run(
            ["swift", "test", "-c", "release", "--skip-build", "--no-parallel", "--filter", "BitmapModelQualification"],
            cwd=checkout, env=env, text=True, capture_output=True, timeout=300,
        )
        if result.returncode:
            destination.write_text(result.stdout + result.stderr)
            raise RuntimeError("Model qualification failed, see local diagnostics")
        value = json.loads(path.read_text())
    value.update(memory_before=before, memory_after=preflight(model, checkpoint))
    return value


def compare(base, bitmap):
    if base["metadata"] != bitmap["metadata"]:
        raise RuntimeError("Model configuration or input IDs differ")
    if len(base["records"]) != len(bitmap["records"]):
        raise RuntimeError("Model records differ in count")
    for control, candidate in zip(base["records"], bitmap["records"]):
        for field in IDENTICAL + OPTIONAL:
            if control.get(field) != candidate.get(field):
                raise RuntimeError(f"First difference in {control['label']} field {field}")
        control.pop("raw_output_ids")
        candidate.pop("raw_output_ids")


def report(pairs):
    values = []
    for item in pairs:
        control, candidate = [sum(item[variant]["records"][0]["intervals_ns"])
                              for variant in ("set", "bitmap")]
        values.append(100 * (control - candidate) / control)
    result = {"latency_reduction_percent": values, "mean_percent": fmean(values)}
    if len(values) == 5:
        half_width = 2.7764451051977987 * stdev(values) / (len(values) ** .5)
        result["ci95_percent"] = [fmean(values) - half_width, fmean(values) + half_width]
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--bitmap", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("correctness", "timing"), required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    roots = {"set": args.set.resolve(), "bitmap": args.bitmap.resolve()}
    model = args.model.resolve()
    checkpoint = checkpoint_metadata(model)
    files = ("QwenFieldfareBench", "QwenFieldfareServerTests.xctest")
    provenance = {variant: {name: hashlib.sha256(binary(root, name).read_bytes()).hexdigest()
                            for name in files} for variant, root in roots.items()}
    write_json(args.out / f"{args.phase}-binaries.json", provenance)

    counts = [(0, 0)] if args.phase == "correctness" else (
        [(128, index) for index in range(5)] + [(512, index) for index in range(2)]
    )
    results = []
    for length, index in counts:
        order = ["set", "bitmap"] if index % 2 == 0 else ["bitmap", "set"]
        values = {}
        for variant in order:
            values[variant] = run(roots[variant], model, checkpoint, args.phase, length,
                                  args.out / f"failure-{variant}.log")
            print(args.phase, length, index + 1, variant, "passed", flush=True)
        compare(values["set"], values["bitmap"])
        results.append(dict(length=length, pair=index + 1, order=order, **values))
        write_json(args.out / f"{args.phase}.json", results)
    if args.phase == "timing":
        write_json(args.out / "effect.json", {str(n): report([r for r in results if r["length"] == n])
                                             for n in (128, 512)})


if __name__ == "__main__":
    main()
