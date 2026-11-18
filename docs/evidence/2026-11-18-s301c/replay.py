"""Replay the supported partitioned SLRU path and bound adjacent-row read overlap."""
import argparse
from collections import Counter
import gzip
from itertools import accumulate
import json
from pathlib import Path


def require(test, message):
    if not test:
        raise ValueError(message)


class Partition:
    def __init__(self, start, size, protected_limit):
        self.start = start
        self.owner = [-1] * size
        self.age = [0] * size
        self.protected = set()
        self.index = {}
        self.protected_limit = protected_limit

    def oldest(self, choices):
        return min(choices, key=lambda slot: (self.age[slot], slot), default=None)

    def acquire(self, experts, tick):
        locked = set()
        decisions = []
        for expert in experts:
            slot = self.index.get(expert)
            if slot is not None:
                locked.add(slot)
                self.age[slot] = tick
                demoted = None
                if slot not in self.protected:
                    self.protected.add(slot)
                    if len(self.protected) > self.protected_limit:
                        demoted = self.oldest(self.protected - locked)
                        if demoted is not None:
                            self.protected.remove(demoted)
                decisions.append([expert, self.start + slot, 1, -1,
                                  -1 if demoted is None else self.start + demoted])
                continue

            unlocked = [s for s in range(len(self.owner)) if s not in locked]
            require(unlocked, "No available slot in the supported partition")
            slot = next((s for s in unlocked if self.owner[s] < 0), None)
            if slot is None:
                slot = self.oldest(s for s in unlocked if s not in self.protected)
            if slot is None:
                slot = self.oldest(s for s in unlocked if s in self.protected)
            victim = self.owner[slot]
            if victim >= 0:
                del self.index[victim]
            self.owner[slot] = expert
            self.index[expert] = slot
            self.protected.discard(slot)
            self.age[slot] = tick
            locked.add(slot)
            decisions.append([expert, self.start + slot, 0, victim, -1])
        return decisions

    def state(self, layer):
        return [[self.start + slot, layer, expert, self.age[slot], int(slot in self.protected)]
                for slot, expert in enumerate(self.owner) if expert >= 0]


def misses(row):
    return [decision for decision in row["decisions"] if not decision[2]]


def reader(row):
    # Shared work does not consume a cache expert. Only selected hit and miss
    # projections read these slots, and the two GPU commands retire in order.
    return {slot: ("hit" if hit else "tail") for _, slot, hit, _, _ in row["decisions"]}


def row_cost(row):
    router_start, router_end, _, hit_start, hit_end, read_start, read_end, tail_start, tail_end = row["time"]
    service = read_end - read_start if row["reads"] else 0
    return (router_end - router_start, hit_end - hit_start, tail_end - tail_start, service)


def pair_savings(left, right, release, stage_blocked=False):
    """Assume both real router packets are ready, zero host overhead and no resource contention.

    Start left's H and miss service together. GPU tails stay ordered. A right
    read retains its observed completion offset and starts no earlier than
    release and the retirement of the destination's left GPU reader.
    """
    _, hit_left, tail_left, read_left = row_cost(left)
    _, hit_right, _, read_right = row_cost(right)
    left_end = max(hit_left, read_left) + tail_left
    delay = read_left if release == "serial" else 0
    hazard = reader(left)
    retire = {"hit": hit_left, "tail": left_end}
    observed_ends = {expert: end for expert, _, end in right["reads"]}
    done = []
    for expert, slot, _, _, _ in misses(right):
        eligible = 0 if stage_blocked else retire.get(hazard.get(slot), 0)
        done.append(max(delay, eligible) + observed_ends[expert])
    join = read_right - max(observed_ends.values(), default=0)
    effective = max(0, max(done, default=0) + join - left_end)
    baseline_residual = max(0, read_right - hit_right)
    residual = max(0, effective - hit_right)
    return baseline_residual, residual


def verify(capture):
    slots = capture["slots"]
    capacity = sum(slots)
    starts = list(accumulate(slots, initial=0))
    require(len(slots) == 40 and min(slots) >= 8, "Expected forty nonoverlapping partitions")
    partitions = [
        Partition(start, size, max(8, int(float(size) * capture["protectedCapacity"] / capacity)))
        for start, size in zip(starts, slots)
    ]
    rows = capture["rows"]
    require(len(rows) == (capture["reused"] + capture["fresh"]) * len(slots), "Missing rows")
    all_counts = Counter()
    fresh_counts = Counter()
    previous_end = 0
    for tick, row in enumerate(rows, 1):
        position, layer = divmod(tick - 1, len(slots))
        require((row["position"], row["layer"]) == (position, layer), "Changed row/layer order")
        keys = [d[0] for d in row["decisions"]]
        require(len(keys) == len(set(keys)) == 8 and all(0 <= x < 256 for x in keys), "Invalid routed top-eight")
        computed = partitions[layer].acquire(keys, tick)
        location = f"row {position}, layer {layer}"
        require(computed == row["decisions"], f"Policy decision differed at {location}")
        miss_indices = [j for j, d in enumerate(computed) if not d[2]]
        require(miss_indices == row["missPositions"], f"GPU miss list differed at {location}")
        require(sorted(d[0] for d in misses(row)) == [f[0] for f in row["reads"]], f"Read keys differed at {location}")
        a, b, wake, h, hit_end, read_begin, read_end, m, end = row["time"]
        require(previous_end <= a <= b <= wake <= read_begin <= read_end <= m <= end
                and b <= h <= hit_end <= m, f"Clock order differed at {location}")
        require(all(0 <= start <= stop <= read_end - read_begin + 1 for _, start, stop in row["reads"]),
                f"Invalid individual read times at {location}")
        previous_end = end
        updates = Counter(hits=8 - len(miss_indices), misses=len(miss_indices),
                          evictions=sum(d[3] >= 0 for d in computed))
        all_counts.update(updates)
        if position >= capture["reused"]:
            fresh_counts.update(updates)
    require(all_counts == Counter({k: capture[k] for k in ("hits", "misses", "evictions")}),
            "Aggregate counts changed")
    require(all_counts["misses"] * capture["blockBytes"] == capture["logicalBytes"], "Logical bytes changed")
    state = [slot for layer, partition in enumerate(partitions) for slot in partition.state(layer)]
    require(state == capture["finalSlots"], "Final slot mapping, recency or segments changed")
    return dict(fresh_counts)


def analyze_pairs(pairs):
    counts = Counter()
    time = Counter()
    for left, right in pairs:
        counts["pairs"] += 1
        previous = reader(left)
        prior_experts = {d[0] for d in left["decisions"]}
        shared = sum(d[0] in prior_experts for d in right["decisions"])
        counts["shared_experts"] += shared
        counts["pairs_sharing_experts"] += shared > 0
        next_reads = misses(right)
        blocked = 0
        for _, slot, _, victim, _ in next_reads:
            if slot not in previous:
                counts["safe_direct"] += 1
                counts["safe_empty"] += victim == -1
            else:
                counts[f"blocked_until_{previous[slot]}"] += 1
                blocked += 1
        counts["pairs_with_misses"] += bool(next_reads)
        counts["pairs_with_blocked"] += blocked > 0
        if len(next_reads) > blocked:
            _, hit, tail, service = row_cost(left)
            time["serial_gpu_window_us"] += max(0, hit - service) + tail
        counts["peak_blocked"] = max(counts["peak_blocked"], blocked)
        counts["peak_all_staged"] = max(counts["peak_all_staged"], len(next_reads))
        for key, release, staging in (("serial", "serial", False), ("dual", "dual", False),
                                      ("staged_dual", "dual", True)):
            baseline, residual = pair_savings(left, right, release, staging)
            time[f"{key}_saved_us"] += baseline - residual
            time[f"{key}_residual_us"] += residual
            if key == "serial":
                time["baseline_right_residual_us"] += baseline
        time["second_row_router_wake_us"] += right["time"][2] - right["time"][1]
    return dict(counts), dict(time)


def examine(capture, previous):
    fresh = verify(capture)
    require((fresh["hits"], fresh["misses"], fresh["misses"] * capture["blockBytes"])
            == (previous["expertHits"], previous["expertMisses"], previous["fetchedBytes"]),
            "S3-01b fresh request demand changed")
    rows = capture["rows"][40 * capture["reused"]:]
    per_layer = [rows[layer::40] for layer in range(40)]
    sliding, _ = analyze_pairs((left, right) for layer in per_layer for left, right in zip(layer, layer[1:]))
    paired, gains = analyze_pairs((left, right) for layer in per_layer for left, right in zip(layer[::2], layer[1::2]))
    stats = previous["timing"]
    recorded_floor_ms = sum(stats[k] for k in ("routerGPUMs", "hitGPUMs", "tailGPUMs", "rowLocalReadFloorMs"))
    historical = previous["prefillMs"]
    costs = list(map(row_cost, rows))
    local_io_floor = sum(max(0, service - hit) for _, hit, _, service in costs)
    local_gpu = sum(router + hit + tail for router, hit, tail, _ in costs)
    raw_charge = capture["blockBytes"]
    staged_reads = paired.get("blocked_until_hit", 0) + paired.get("blocked_until_tail", 0)
    return {
        "workload": capture["name"], "fresh": capture["fresh"], "seed_rows": capture["reused"],
        "verified_decisions": len(capture["rows"]) * 8, "verified_final_slots": len(capture["finalSlots"]),
        "fresh_cache": {**fresh, "logical_bytes": fresh["misses"] * raw_charge},
        "sliding": sliding, "paired": paired, "timing": gains,
        "recorded_prefill_ms": capture["prefillMs"], "historical_prefill_ms": historical,
        "historical_optimistic_percent": 100 * (1 - recorded_floor_ms / historical),
        "optimistic_with_serial_overlap_percent": 100 * (1 - (recorded_floor_ms - gains["serial_saved_us"] / 1000) / historical),
        "optimistic_with_dual_overlap_percent": 100 * (1 - (recorded_floor_ms - gains["dual_saved_us"] / 1000) / historical),
        "optimistic_with_early_staging_percent": 100 * (1 - (recorded_floor_ms - gains["staged_dual_saved_us"] / 1000) / historical),
        "credited_wake_plus_serial_overlap_percent": 100 * (
            gains["second_row_router_wake_us"] + gains["serial_saved_us"]) / (historical * 1000),
        "same_capture_optimistic_with_serial_percent": 100 * (
            1 - (local_gpu + local_io_floor - gains["serial_saved_us"]) / (capture["prefillMs"] * 1000)),
        "same_capture_optimistic_with_dual_percent": 100 * (
            1 - (local_gpu + local_io_floor - gains["dual_saved_us"]) / (capture["prefillMs"] * 1000)),
        "local_gpu_us": local_gpu,
        "local_residual_before_us": local_io_floor,
        "local_residual_after_serial_us": local_io_floor - gains["serial_saved_us"],
        "optional_staging_peak_bytes": paired.get("peak_blocked", 0) * raw_charge,
        "optional_staging_payload_copy_bytes": staged_reads * raw_charge,
        "metal_bytes": capture["metalBytes"],
    }


def load_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    historical = {x["workload"]: x for x in load_json(args.previous) if x["traced"]}
    results = []
    for label in ("short", "1k", "suffix"):
        path = args.trace_dir / f"{label}.json.gz"
        if not path.is_file():
            path = path.with_suffix("")
        capture = load_json(path)
        require(label == capture["name"], f"Wrong capture {path.name}")
        results.append(examine(capture, historical[label]))
    rendered = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
