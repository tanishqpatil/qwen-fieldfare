#!/usr/bin/env python3
"""Run one checkpoint-verified strict replay and a separate bucket diagnostic pair."""

import argparse
import json
from pathlib import Path
import subprocess

from qualify import binary, checkpoint_metadata, preflight

PREFIXES = (
    "Active-streaming budget:", "Expert cache slots:", "Loaded footprint:", "Prompt tokens:",
    "Generated tokens:", "Finish reason:", "Prefill/TTFT:", "Decode:", "Inter-token latency:",
    "CPU token selection:", "Decode-only cache:", "Final footprint:", "Final Metal allocation:",
    "At ", "exact GPU:", "Cache including prefill:",
)
TEXT = (
    "Explain HashMap in Java with a short example. Then write a detailed tutorial "
    "of at least 1500 words, with complete runnable examples for frequency counting, "
    "grouping, and composite keys, followed by implementation details, failure modes, "
    "and exercises with full solutions."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--bitmap", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    model = options.model.resolve()
    checkpoint = checkpoint_metadata(model)
    fixture = (options.bitmap / "docs/evidence/2026-11-17-parity/reference.json").resolve()
    candidate = binary(options.bitmap.resolve(), "QwenFieldfareBench")
    common = ["--model", str(model), "--context", "8192", "--memory-gib", "5", "--metrics-only"]
    before = preflight(model, checkpoint)
    result = subprocess.run([str(candidate), *common, "--teacher-reference", str(fixture)],
                            capture_output=True, text=True, timeout=240)
    after = preflight(model, checkpoint)
    log = options.out / "strict.txt"
    log.write_text("\n".join(line for line in (result.stdout + result.stderr).splitlines()
        if line.startswith(("Reference checkpoint SHA-256 verified", "Mismatch at reference token ",
                            "Teacher-forced parity:", "Error: Teacher-forced parity failed"))) + "\n")
    if result.returncode != 1:
        raise RuntimeError(f"Unexpected strict exit {result.returncode}")
    subprocess.run(["python3", str(options.bitmap / "Scripts/check_numerical_non_regression.py"),
                    "--log", str(log)], check=True)
    records = {"strict": {"exit": result.returncode, "memory_before": before, "memory_after": after}}
    for mode in ("set", "bitmap"):
        before = preflight(model, checkpoint)
        result = subprocess.run([str(binary(getattr(options, mode).resolve(), "QwenFieldfareBench")),
            *common, "--prompt", TEXT, "--sampler", "app-seeded", "--seed", "42",
            "--max-new", "512", "--gpu-profile"], capture_output=True, text=True, timeout=180, check=True)
        after = preflight(model, checkpoint)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith(PREFIXES)]
        if "Generated tokens: 512" not in lines:
            raise RuntimeError("Diagnostic continuation ended early")
        records[mode] = {"memory_before": before, "memory_after": after, "footer": lines}
    (options.out / "diagnostics.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
