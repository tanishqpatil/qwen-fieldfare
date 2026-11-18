#!/usr/bin/env python3
"""Validate paired selector captures and compute latency and attribution statistics."""

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
from statistics import fmean, median

NANOSECOND_FIELDS = {
    "selector_wall": "selectorWallNs", "selector_thread_cpu": "selectorThreadCPUNs",
    "logit_pointer": "pointerNs", "scan": "scanNs", "sort": "sortNs",
    "weights": "weightsNs", "top_p": "topPNs", "draw": "drawNs",
    "seen_insert": "seenInsertNs", "callback": "previousCallbackNs",
    "final_gpu_wait": "finalWaitNs",
}
MILLISECOND_FIELDS = {
    "cache_lookup": "cacheLookupMs", "miss_io": "missIOms",
    "gpu_wait_all": "allGPUWaitMs", "forward": "forwardWallMs",
}
IDENTICAL_FIELDS = (
    "seed", "promptTokens", "emittedTokens", "finish", "outputSHA256",
    "decodeHits", "decodeMisses", "decodeFetchedBytes",
    "cacheHits", "cacheMisses", "cacheFetchedBytes", "cacheOccupancy",
    "cacheCapacity", "cacheEvictions", "metalBytes",
)


def dist(values):
    values = sorted(values)
    count = len(values)
    return {
        "n": count, "mean": fmean(values), "p50": median(values),
        "p95": values[math.ceil(count * .95) - 1], "min": values[0], "max": values[-1],
    }


def check_pair(control, probe):
    for field in IDENTICAL_FIELDS:
        if control[field] != probe[field]:
            raise ValueError(f"Opt-in changed paired {field}")
    for record in (control, probe):
        target = record["targetTokens"]
        if record["emittedTokens"] != target or len(record["transitionsNs"]) != target - 1:
            raise ValueError("Invalid output or inter-token sample count")
        if record["finish"] != "Output limit" or record["cacheCapacity"] != 2183:
            raise ValueError("Unexpected finish or memory plan")
        if any(record[edge]["pressureLevel"] != 1 for edge in ("memoryBefore", "memoryAfter")):
            raise ValueError("Elevated memory pressure")
        if len(record["phases"]) != (target - 1 if record is probe else 0):
            raise ValueError("Invalid profiling activation or sample count")
    for row in probe["phases"]:
        phase = row["phases"]
        subphases = sum(phase[name] for name in ("scanNs", "sortNs", "weightsNs", "topPNs", "drawNs"))
        if phase["selectorWallNs"] < subphases or phase["selectorThreadCPUNs"] == 0:
            raise ValueError("Inconsistent selector phase timing")


def summarize_group(pairs):
    probe_values = [probe for _, probe in pairs]
    phase_rows = [row for probe in probe_values for row in probe["phases"]]
    phase_values = [row["phases"] for row in phase_rows]
    phase_stats = {
        metric: dist([phase[field] / 1e6 for phase in phase_values])
        for metric, field in NANOSECOND_FIELDS.items()
    }
    phase_stats.update({
        metric: dist([row[field] for row in phase_rows])
        for metric, field in MILLISECOND_FIELDS.items()
    })
    phase_stats["final_gpu_command"] = dist([phase["finalCommandGPUMs"] for phase in phase_values])
    transitions = dist([t / 1e6 for probe in probe_values for t in probe["transitionsNs"]])
    selector_mean = phase_stats["selector_wall"]["mean"]
    transition_mean = transitions["mean"]
    selector_fraction = selector_mean / transition_mean
    return {
        "transitions_ms": transitions, "phases_ms": phase_stats,
        "minimum_searches": dist([phase["minimumSearches"] for phase in phase_values]),
        "zero_selector_latency_reduction_percent": 100 * selector_fraction,
        "zero_selector_throughput_increase_percent": 100 * selector_fraction / (1 - selector_fraction),
        "selector_fraction_needed_for_5_percent_latency_reduction": 0.05 / selector_fraction,
        "prefill_seen_set_ms": dist([probe["prefillSeenNs"] / 1e6 for probe in probe_values]),
        "decode_initial_seen_set_ms": dist([probe["decodeInitialSeenNs"] / 1e6 for probe in probe_values]),
        "pairs": [
            {
                "repeat": probe["repeat"],
                "control_decode_ms": sum(control["transitionsNs"]) / 1e6,
                "profile_decode_ms": sum(probe["transitionsNs"]) / 1e6,
                "profile_overhead_percent": 100 * (
                    sum(probe["transitionsNs"]) / sum(control["transitionsNs"]) - 1
                ),
                "control_request_ms": control["requestNs"] / 1e6,
                "profile_request_ms": probe["requestNs"] / 1e6,
                "profile_selector_wall_ms": fmean(row["phases"]["selectorWallNs"] for row in probe["phases"]) / 1e6,
                "profile_selector_wall_share_percent": 100 * (
                    sum(row["phases"]["selectorWallNs"] for row in probe["phases"])
                    / sum(probe["transitionsNs"])
                ),
            }
            for control, probe in pairs
        ],
    }


def analyze(records):
    workloads = defaultdict(dict)
    for record in records:
        key = record["sampler"], record["targetTokens"]
        capture = record["repeat"], record["mode"]
        if capture in workloads[key]:
            raise ValueError("Duplicate record")
        workloads[key][capture] = record
    if workloads.keys() != {(mode, count) for mode in ("greedy", "app-seeded") for count in (128, 512)}:
        raise ValueError("Expected all four workload configurations")
    results = {}
    for (mode, target), trials in sorted(workloads.items()):
        repeats = range(1, 5 if target == 512 else 4)
        if trials.keys() != {(i, option) for i in repeats for option in ("on", "off")}:
            raise ValueError("Missing an opt-in/control pair")
        pairs = [(trials[i, "off"], trials[i, "on"]) for i in repeats]
        for control, probe in pairs:
            check_pair(control, probe)
        for field in IDENTICAL_FIELDS:
            if len({record[field] for record in trials.values()}) != 1:
                raise ValueError(f"Unstable same-workload {field}")
        results[f"{mode}-{target}"] = summarize_group(pairs)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.records.suffix == ".gz" else open
    with opener(args.records, "rt") as stream:
        result = analyze([json.loads(line) for line in stream])
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
