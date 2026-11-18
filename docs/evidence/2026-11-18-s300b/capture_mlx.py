"""Matched native Qwen forwards for two immutable teacher histories."""
import argparse
import copy
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from transformers import AutoTokenizer

register = AutoTokenizer.register
AutoTokenizer.register = lambda config, *a, **k: None if isinstance(config, str) else register(config, *a, **k)
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import wired_limit
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("fixture", type=Path)
parser.add_argument("working", type=Path)
args = parser.parse_args()
fixture = json.loads(args.fixture.read_text())
versions = {name: version(name) for name in ("mlx", "mlx-lm", "transformers", "numpy")}
assert all(versions[k] == v for k, v in fixture["reference_versions"].items())
index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
required = set(index["weight_map"].values()) | {"config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"}
assert set(fixture["checkpoint_sha256"]) == required
for name, expected in fixture["checkpoint_sha256"].items():
    assert Path(name).name == name
    with (args.checkpoint / name).open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == expected, name
print("Checkpoint files verified", len(required), versions, flush=True)

histories = json.loads((args.working / "histories.json").read_text())
assert histories["strict"] == {"prompt": fixture["prompt_tokens"], "teacher": fixture["reference_tokens"]}
assert {k: (len(v["prompt"]), len(v["teacher"])) for k,v in histories.items()} == {
    "strict": (21, 132), "sustained": (430, 512)
}
assert histories["sustained"]["prompt"][:21] != histories["strict"]["prompt"]

model, _ = load(str(args.checkpoint))
norm_inputs = {}
route_logits = {}
text = model.language_model
first = text.model.layers[0].mlp
assert first.top_k == 8 and first.norm_topk_prob
owner_paths = {module.__name__: Path(inspect.getfile(module)) for module in (
    type(model), type(text.model), type(first)
)}
provenance = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in owner_paths.items()}


class Tap(nn.Module):
    def __init__(self, original, capture):
        super().__init__()
        self.original = original
        self.capture = capture

    def __call__(self, x):
        y = self.original(x)
        if self.capture == "hidden":
            norm_inputs["latest"] = x
        else:
            route_logits["latest"] = y
        return y


def capture(logits, row):
    hidden = norm_inputs["latest"]
    gate = route_logits["latest"]
    gates = mx.softmax(gate[:, row:row+1], axis=-1, precise=True)
    ids = mx.argpartition(gates, kth=-8, axis=-1)[..., -8:]
    scores = mx.take_along_axis(gates, ids, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    projected = logits[0, row].astype(mx.float32)
    last = hidden[0, row].astype(mx.float32)
    mx.eval(projected, last, ids, scores)
    return (np.array(projected, dtype="<f4"), np.array(last, dtype="<f4"),
            {"ids": ids.reshape(-1).tolist(), "scores": scores.reshape(-1).tolist()})


def save_sample(folder, position, sample):
    logits, hidden, route = sample
    assert logits.shape == (248320,) and hidden.shape == (2048,)
    logits.tofile(folder / f"{position}-logits.f32")
    hidden.tofile(folder / f"{position}-hidden.f32")
    return route


fixed_positions = {"strict": [0, 1, 20, 80, 151], "sustained": [0, 1, 429, 894, 940]}


def replay(name, history, mode, selection=None):
    prompt, teacher = history["prompt"], history["teacher"]
    sequence = prompt + teacher[:-1]
    start_decode = len(prompt) - 1
    batch = start_decode if mode == "mlx-native" else 0
    folder = args.working / name / mode
    folder.mkdir(parents=True, exist_ok=True)
    fixed = set(fixed_positions[name] if selection is None else selection)
    cache = make_prompt_cache(model)
    dtype = None
    routes = {}
    near = {}
    predictions = []
    started = time.perf_counter()

    if batch:
        logits = model(mx.array([prompt[:batch]]), cache=cache)
        mx.eval(logits, [c.state for c in cache])
        dtype = [str(logits.dtype), str(norm_inputs["latest"].dtype)]
        for p in sorted(fixed & set(range(batch))):
            routes[p] = save_sample(folder, p, capture(logits, p))
        del logits
        norm_inputs.clear()
        route_logits.clear()
        mx.clear_cache()
        print(name, mode, "native prefill", batch, "peak", mx.get_peak_memory(), flush=True)

    for p in range(batch, len(sequence)):
        if p % 64 == 0:
            pressure = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip()
            if pressure != "1":
                raise RuntimeError(f"memory pressure {pressure} at {name}/{mode}/{p}")
        logits = model(mx.array([[sequence[p]]]), cache=cache)
        dtype = [str(logits.dtype), str(norm_inputs["latest"].dtype)]
        best = mx.argmax(logits[0, 0])
        top2 = mx.topk(logits[0, 0], 2).astype(mx.float32)
        mx.eval(logits, [c.state for c in cache], best, top2)
        if p in fixed:
            routes[p] = save_sample(folder, p, capture(logits, 0))
        if p >= start_decode:
            predictions.append(int(best.item()))
        if selection is None and p >= start_decode and p not in fixed:
            half = int(p >= start_decode + len(teacher) // 2)
            margin = abs(float(top2[0].item()) - float(top2[1].item()))
            if half not in near or margin < near[half][0]:
                near[half] = (margin, p, capture(logits, 0))
        if p % 256 == 0:
            print(name, mode, p, "peak", mx.get_peak_memory(), flush=True)
    for _, position, sample in near.values():
        routes[position] = save_sample(folder, position, sample)
    record = {"predictions": predictions, "layerZero": routes, "dtypes": dtype,
              "seconds": time.perf_counter() - started, "peak_bytes": mx.get_peak_memory()}
    (folder / "mlx.json").write_text(json.dumps(record, indent=2) + "\n")
    norm_inputs.clear()
    route_logits.clear()
    return {"positions": sorted(routes), "fixed_positions": sorted(fixed),
            "near_tie_positions": sorted(r[1] for r in near.values())}


selection = {}
with wired_limit(model):
    text.model.norm = Tap(text.model.norm, "hidden")
    first.gate = Tap(first.gate, "route")
    for name, history in sorted(histories.items()):
        mx.reset_peak_memory()
        selection[name] = replay(name, history, "mlx-native")
        mx.clear_cache()
        replay(name, history, "mlx-sequential", selection[name]["positions"])
        mx.clear_cache()

(args.working / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
(args.working / "mlx-device.json").write_text(json.dumps({
    "versions": versions, "source_hashes": provenance, "checkpoint_files": len(required),
    "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256((args.working / "histories.json").read_bytes()).hexdigest(),
    "pressure_check": "level 1 before start and every 64 token positions",
}, indent=2) + "\n")
