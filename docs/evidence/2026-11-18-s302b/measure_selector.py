#!/usr/bin/env python3
"""Measure thread CPU using the actual release selector sources on identical logits."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

HARNESS = r"""
let logits = (0..<248_320).map { index in
    Float((UInt32(index) &* 2_654_435_761) % 65_521) / 8_192
}
let prompt = (0..<300).map { Int32(37 + $0 * 811) }
let configuration = QwenGenerationConfiguration(maxNewTokens: 512, seed: 42)
#if BITMAP
let history = QwenSeenTokenBitmap(tokenCount: logits.count, promptTokens: prompt)
#else
let history = Set(prompt)
#endif
var measured: [UInt64] = []
var raw: [Int32] = []
raw.reserveCapacity(512)
logits.withUnsafeBufferPointer { buffer in
    for round in 0..<5 {
        let start = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)
        for position in 0..<512 {
            #if BITMAP
            let token = QwenLogitSelector.select(logits: buffer.baseAddress!, seenBitmap: history,
                configuration: configuration, samplePosition: position)
            #else
            let token = QwenLogitSelector.select(logits: buffer.baseAddress!, count: buffer.count,
                seenTokenIDs: history, configuration: configuration, samplePosition: position)
            #endif
            if round == 0 { raw.append(token) }
            else { precondition(raw[position] == token) }
        }
        measured.append(clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) - start)
    }
}
let json = try JSONSerialization.data(withJSONObject: ["thread_cpu_ns": measured, "raw": raw])
print(String(decoding: json, as: UTF8.self))
"""


def build(checkout, directory, mode):
    path = checkout / "Sources/QwenFieldfareCore/Runtime/Generation/QwenGenerationSupport.swift"
    source = path.read_text()
    configuration = source[source.index("public enum QwenGenerationConfigurationError"):
                           source.index("public enum QwenSpecialTokens")]
    bitmap = source[source.index("/// Owned by one active-streaming generation."):
                    source.index("/// Shared generation state machine.")] if mode == "bitmap" else ""
    selector = source[source.index("enum QwenLogitSelector"):]
    swift = directory / f"{mode}.swift"
    swift.write_text("\n".join(["import Accelerate\nimport Darwin\nimport Foundation",
                                configuration, bitmap, selector, HARNESS]))
    executable = directory / mode
    flags = ["-D", "BITMAP"] if mode == "bitmap" else []
    subprocess.run(["swiftc", "-O", "-whole-module-optimization", *flags, str(swift),
                    "-o", str(executable)], check=True)
    return executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--bitmap", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="qwen-bitmap-thread-cpu-") as tmp:
        directory = Path(tmp)
        outputs = {
            mode: json.loads(subprocess.check_output([str(build(getattr(args, mode), directory, mode))]))
            for mode in ("set", "bitmap")
        }
        if outputs["set"].pop("raw") != outputs["bitmap"].pop("raw"):
            raise RuntimeError("Synthetic outputs differ")
    print(json.dumps({"choices_per_round": 512, "rounds": outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
