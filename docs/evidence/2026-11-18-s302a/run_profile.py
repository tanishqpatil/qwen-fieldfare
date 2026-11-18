#!/usr/bin/env python3
"""Run the opt-in bench probe serially and save numeric metrics without model content."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

PROMPT = (
    "Explain HashMap in Java with a short example. Then write a detailed tutorial "
    "of at least 1500 words, with complete runnable examples for frequency counting, "
    "grouping, and composite keys, followed by implementation details, failure modes, "
    "and exercises with full solutions."
)
OWNERS = re.compile(
    r"(?:^|/)(?:QwenFieldfareMac|QwenFieldfareBench|QwenFieldfareServer|"
    r"mlx_lm|mlx-lm|ollama|LM Studio)(?:[/. ]|$)", re.IGNORECASE
)
PROFILE_PREFIX = "SELECTOR_PROFILE_JSON "


def command(*args):
    return subprocess.check_output(args, text=True)


def memory_snapshot():
    pressure_level = int(command("sysctl", "-n", "kern.memorystatus_vm_pressure_level"))
    pressure = command("memory_pressure", "-Q")
    swap = command("sysctl", "-n", "vm.swapusage")
    return {
        "pressureLevel": pressure_level,
        "freePercent": int(re.search(r"free percentage: (\d+)%", pressure)[1]),
        "swapUsedMiB": float(re.search(r"used = ([\d.]+)M", swap)[1]),
    }


def checkpoint_metadata(model):
    shards = sorted(model.glob("model-*.safetensors"))
    if len(shards) != 4 or not all(path.is_file() for path in shards):
        raise RuntimeError("Expected the four compatible model shards")
    return [
        (path.stat().st_dev, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
        for path in shards
    ]


def run_one(executable, model, output, repeat, sampler, tokens, mode, checkpoint):
    if checkpoint_metadata(model) != checkpoint:
        raise RuntimeError("Checkpoint metadata changed")
    if any(OWNERS.search(proc.strip()) for proc in command("ps", "-axo", "comm").splitlines()):
        raise RuntimeError("Another known model owner is running")
    before = memory_snapshot()
    if before["pressureLevel"] != 1:
        raise RuntimeError("Memory pressure is not normal")
    env = dict(os.environ, QWEN_SELECTOR_DIAGNOSTIC=mode)
    args = [
        str(executable), "--model", str(model), "--prompt", PROMPT,
        "--context", "8192", "--memory-gib", "5", "--max-new", str(tokens),
        "--metrics-only", "--sampler", sampler, "--seed", "42",
    ]
    started = time.monotonic_ns()
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=180)
    elapsed = time.monotonic_ns() - started
    after = memory_snapshot()
    lines = [line.removeprefix(PROFILE_PREFIX) for line in proc.stdout.splitlines()
             if line.startswith(PROFILE_PREFIX)]
    if proc.returncode or len(lines) != 1:
        raise RuntimeError(
            f"Run failed with exit {proc.returncode} and {len(lines)} measurement records"
        )
    record = json.loads(lines[0])
    if checkpoint_metadata(model) != checkpoint:
        raise RuntimeError("Checkpoint metadata changed")
    identity = (
        record["mode"], record["sampler"], record["targetTokens"], record["seed"],
        record["cacheCapacity"],
    )
    if identity != (mode, sampler, tokens, 42, 2183):
        raise RuntimeError("Unexpected profile configuration")
    record.update(repeat=repeat, memoryBefore=before, memoryAfter=after, processWallNs=elapsed)
    with output.open("a") as stream:
        print(json.dumps(record, separators=(",", ":"), sort_keys=True), file=stream)
    print(f"{repeat} {sampler:10} {tokens} {mode:3} {record['emittedTokens']} output "
          f"{sum(record['transitionsNs']) / 1e9:.3f}s decode "
          f"{before['pressureLevel']}→{after['pressureLevel']} pressure "
          f"{before['swapUsedMiB']}→{after['swapUsedMiB']} MiB swap", flush=True)
    if after["pressureLevel"] != 1 or record["emittedTokens"] != tokens:
        raise RuntimeError("Workload ended early or encountered elevated memory pressure")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-sustained-only", action="store_true",
                        help="Run the two reversed-order 512-token control pairs")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    executable = args.executable.resolve()
    model = args.model.resolve()
    checkpoint = checkpoint_metadata(model)
    print(f"Executable SHA256 {hashlib.sha256(executable.read_bytes()).hexdigest()}", flush=True)
    print(f"Ordered shard bytes {[item[2] for item in checkpoint]}", flush=True)
    if args.reverse_sustained_only:
        for sampler, modes in (("app-seeded", ("on", "off")), ("greedy", ("off", "on"))):
            for mode in modes:
                run_one(executable, model, args.output, 4, sampler, 512, mode, checkpoint)
        return
    # Rotate the workload order. Start with a sustained control so a
    # premature EOS is visible before running the shorter cases.
    # The independent reversal pass flips activation order for each sampler.
    workloads = [("app-seeded", 512), ("greedy", 512), ("app-seeded", 128), ("greedy", 128)]
    for repeat in (1, 2, 3):
        order = workloads if repeat != 2 else list(reversed(workloads))
        for index, (sampler, tokens) in enumerate(order):
            modes = ("off", "on") if (repeat + index) % 2 == 1 else ("on", "off")
            for mode in modes:
                run_one(executable, model, args.output, repeat, sampler, tokens, mode, checkpoint)


if __name__ == "__main__":
    main()
