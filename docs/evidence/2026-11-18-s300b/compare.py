"""Report raw-logit, pre-norm hidden, ranking and route comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("working", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
modes = ["legacy-cold", "legacy-warm", "router-cold", "router-warm"]
refs = ["mlx-native", "mlx-sequential"]
observed_pairs = {"strict": (80, 14988, 10000), "sustained": (894, 1031, 285)}
quantiles = [0.01, 0.1, 0.5, 0.9, 0.99]


def read_array(directory, position, name, size):
    path = directory / f"{position}-{name}.f32"
    raw = path.read_bytes()
    value = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name} for {directory.name} at {position}")
    return value, hashlib.sha256(raw).hexdigest()


def metrics(original, candidate):
    err = candidate - original
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs": float(np.max(np.abs(err))),
        "relative_l2": float(np.linalg.norm(err) / np.linalg.norm(original)),
    }


def distribution(logits):
    return {"mean": float(np.mean(logits)), "std": float(np.std(logits)),
            "min": float(min(logits)), "max": float(max(logits)),
            "quantiles": dict(zip(map(str, quantiles), np.quantile(logits, quantiles).tolist()))}


def top_indices(logits):
    # Include all ties at the threshold, then use ascending token ID for ties.
    threshold = np.partition(logits, -5)[-5]
    candidates = np.flatnonzero(logits >= threshold)
    return candidates[np.lexsort((candidates, -logits[candidates]))][:5].tolist()


def route_comparison(ref, candidate):
    a = dict(zip(ref["ids"], ref["scores"]))
    b = dict(zip(candidate["ids"], candidate["scores"]))
    common = sorted(a.keys() & b.keys())
    return {"overlap": len(common), "max_shared_weight_abs_error": max((abs(a[i]-b[i]) for i in common), default=None)}


selection = json.loads((args.working / "selection.json").read_text())
histories = json.loads((args.working / "histories.json").read_text())
results = {}
for name, settings in selection.items():
    history = histories[name]
    raw = {}
    records = {}
    for mode in refs + modes:
        directory = args.working / name / mode
        record = json.loads((directory / ("mlx.json" if mode in refs else "swift.json")).read_text())
        teacher = history["teacher"]
        predictions = record.pop("predictions")
        assert len(teacher) == len(predictions)
        record["teacher_prediction_differences"] = [
            [i, a, b] for i, (a,b) in enumerate(zip(teacher, predictions)) if a != b
        ]
        records[mode] = record
        for p in settings["positions"]:
            logits, lhash = read_array(directory, p, "logits", 248320)
            hidden, hhash = read_array(directory, p, "hidden", 2048)
            raw.setdefault(p, {})[mode] = (logits, hidden, lhash, hhash)
    samples = {}
    for p, values in raw.items():
        ranking = {mode: top_indices(arrays[0]) for mode, arrays in values.items()}
        native_top = ranking["mlx-native"]
        best, second = native_top[:2]
        target_position, first_id, second_id = observed_pairs[name]
        sample = {"input_id": (history["prompt"] + history["teacher"][:-1])[p], "modes": {}, "comparisons": {}}
        for mode, (logits, hidden, lhash, hhash) in values.items():
            top = ranking[mode]
            row = {"top5": [[i, logits[i]] for i in top], "own_top2_margin": logits[top[0]] - logits[top[1]],
                   "signed_mlx_native_pair_margin": logits[best] - logits[second],
                   "logit_distribution": distribution(logits), "logits_sha256": lhash, "hidden_sha256": hhash}
            if p == target_position:
                row["historical_pair"] = {"ids": [first_id, second_id], "logits": [logits[first_id], logits[second_id]],
                                          "signed_margin": logits[first_id] - logits[second_id]}
            sample["modes"][mode] = row
        for ref in refs:
            ref_logits, ref_hidden, *_ = values[ref]
            top_ref = ranking[ref]
            sample["comparisons"][ref] = {
                mode: {"logits": metrics(ref_logits, values[mode][0]),
                       "hidden": metrics(ref_hidden, values[mode][1]),
                       "top1_agrees": top_ref[0] == ranking[mode][0],
                       "top5_overlap": len(set(top_ref) & set(ranking[mode])),
                       "ordered_top5_agrees": top_ref == ranking[mode],
                       "signed_reference_pair_margin": float(values[mode][0][top_ref[0]] - values[mode][0][top_ref[1]]),
                       "routing_layer0": route_comparison(records[ref]["layerZero"][str(p)], records[mode]["layerZero"][str(p)])}
                for mode in list(modes) + (["mlx-sequential"] if ref == "mlx-native" else [])
            }
        samples[p] = sample
    aggregate = {}
    for ref in refs:
        aggregate[ref] = {}
        for mode in modes:
            observations = [s["comparisons"][ref][mode] for s in samples.values()]
            aggregate[ref][mode] = {
                "mean_logit_rmse": float(np.mean([r["logits"]["rmse"] for r in observations])),
                "mean_logit_relative_l2": float(np.mean([r["logits"]["relative_l2"] for r in observations])),
                "maximum_logit_abs_error": max(r["logits"]["max_abs"] for r in observations),
                "mean_hidden_relative_l2": float(np.mean([r["hidden"]["relative_l2"] for r in observations])),
                "top1_agreement_count": sum(r["top1_agrees"] for r in observations),
                "ordered_top5_agreement_count": sum(r["ordered_top5_agrees"] for r in observations),
                "top5_mean_overlap": float(np.mean([r["top5_overlap"] for r in observations])),
            }
    results[name] = {"selection": settings, "records": records, "samples": samples, "aggregate": aggregate,
                     "residency": json.loads((args.working / name / "residency.json").read_text())}
args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
