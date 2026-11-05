#!/usr/bin/env python3
"""Exercise the real-model teacher-reference CLI, including failure cases."""

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--bench", default=".build/release/QwenFieldfareBench")
    args = parser.parse_args()
    original = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    cases = []
    cases.append(("valid", original, True, "Teacher-forced parity:"))
    wrong_token = copy.deepcopy(original)
    wrong_token["reference_tokens"][0] = (wrong_token["reference_tokens"][0] + 1) % 248320
    cases.append(("wrong token", wrong_token, False, "Teacher-forced parity failed"))
    wrong_prompt = copy.deepcopy(original)
    wrong_prompt["prompt_tokens"][0] = (wrong_prompt["prompt_tokens"][0] + 1) % 248320
    cases.append(("wrong prompt", wrong_prompt, False, "prompt-token mismatch"))
    wrong_hash = copy.deepcopy(original)
    wrong_hash["checkpoint_sha256"]["config.json"] = "0" * 64
    cases.append(("wrong hash", wrong_hash, False, "does not match"))
    invalid_token = copy.deepcopy(original)
    invalid_token["reference_tokens"][0] = -1
    cases.append(("invalid token", invalid_token, False, "Invalid teacher reference"))
    missing_provenance = copy.deepcopy(original)
    missing_provenance["reference_versions"].pop("mlx")
    cases.append(("missing provenance", missing_provenance, False, "Invalid reference provenance"))

    with tempfile.TemporaryDirectory(prefix="qwen-parity-test-") as directory:
        for index, (name, fixture, success, message) in enumerate(cases):
            path = Path(directory) / f"{index}.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            command = [args.bench, "--model", args.model, "--context", "8192",
                       "--cache-slots", "2183", "--teacher-reference", str(path)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            output = result.stdout + result.stderr
            if (result.returncode == 0) != success or message not in output:
                raise RuntimeError(f"{name}: unexpected exit {result.returncode}\n{output}")
            print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()
