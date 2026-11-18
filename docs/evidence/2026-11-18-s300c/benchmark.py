"""Compare combine groupings with a single model owner using the archived diagnostic build."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

model = Path(sys.argv[1]).resolve()
exe = Path(sys.argv[2]).resolve()
out = Path(sys.argv[3]).resolve()
out.mkdir(parents=True, exist_ok=True)
prompt = (
    "Write a complete Python implementation of an LRU cache with type annotations, "
    "detailed docstrings, and at least 200 lines of example-based tests. "
    "Do not summarize. Provide the complete module."
)
owner = re.compile(r"QwenFieldfare(?:Mac|Bench|Server)|mlx_lm|mlx-lm|ollama|LM Studio", re.I)
rows = []
identity = {"executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "prompt": prompt,
            "pairs_128": 2, "pairs_512": 1, "diagnostic_gpu_pair": False}
(out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")


def system_state():
    return subprocess.check_output(
        ["sysctl", "kern.memorystatus_vm_pressure_level", "vm.swapusage"], text=True
    )


def drive_totals():
    # System-wide totals include unrelated activity and OS cache effects.
    cmd = subprocess.run(["iostat", "-Id", "disk0"], capture_output=True, text=True)
    return cmd.stdout.strip() if cmd.returncode == 0 else None


def execute(tokens, pair, mode):
    tag = f"{tokens}-{pair}-{mode}"
    pre = system_state()
    if "kern.memorystatus_vm_pressure_level: 1" not in pre:
        raise RuntimeError(f"Non-normal memory pressure before {tag}")
    active = subprocess.check_output(["ps", "-axo", "ucomm"], text=True)
    if owner.search(active):
        raise RuntimeError(f"Another model owner exists before {tag}")
    env = os.environ.copy()
    env["QWENFIELDFARE_PROBE_CANONICAL_REDUCTION"] = "1"
    env["QWENFIELDFARE_PROBE_GROUPED_COMBINE"] = "1" if mode == "grouped" else "0"
    cmd = [str(exe), "--model", str(model), "--prompt", prompt, "--context", "8192",
           "--memory-gib", "5", "--max-new", str(tokens), "--sampler", "greedy", "--seed", "42",
           "--warmup-runs", "1", "--runs", "1", "--metrics-only"]
    disk_before = drive_totals()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with (out / (tag + ".txt")).open("w") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    text = (out / (tag + ".txt")).read_text()
    measured = text.rsplit("=== Measured 1/1 ===", 1)[-1]
    count = re.search(r"Generated tokens: (\d+)", measured)
    row = {"workload": tokens, "pair": pair, "mode": mode, "profile": False,
           "arguments": ["/absolute/model-directory" if arg == str(model) else arg for arg in cmd[1:]], "exit": result.returncode, "whole_process_seconds": elapsed,
           "process_cpu_user_seconds": after.ru_utime - before.ru_utime,
           "process_cpu_system_seconds": after.ru_stime - before.ru_stime,
           "generated_tokens": int(count[1]) if count else None, "system_pre": pre,
           "system_post": system_state(), "disk0_before": disk_before, "disk0_after": drive_totals()}
    rows.append(row)
    (out / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(tag, row["generated_tokens"], f"{elapsed:.2f}s", flush=True)
    if result.returncode != 0 or row["generated_tokens"] != tokens:
        raise RuntimeError(f"Incomplete measured generation for {tag}")


for length, count in [(128, 2), (512, 1)]:
    for pair in range(count):
        order = ["router-order", "grouped"] if pair % 2 == 0 else ["grouped", "router-order"]
        for mode in order:
            execute(length, pair + 1, mode)
