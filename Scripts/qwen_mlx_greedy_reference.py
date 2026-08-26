#!/usr/bin/env python3
"""Run the local MLX Qwen checkpoint despite the Transformers 5 register API change."""

import json
import heapq
import argparse
import os
import time

from transformers import AutoTokenizer


_original_register = AutoTokenizer.register


def _compatible_register(config_class, *args, **kwargs):
    if isinstance(config_class, str):
        return None
    return _original_register(config_class, *args, **kwargs)


AutoTokenizer.register = _compatible_register

import mlx.core as mx
from mlx_lm.generate import generate_step, wired_limit
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load


parser = argparse.ArgumentParser(description="Generate a greedy MLX reference for QwenFieldfare parity checks.")
parser.add_argument(
    "--model",
    default=os.environ.get("QWENFIELDFARE_MODEL", os.path.expanduser("~/models/Qwen3.6-35B-A3B-4bit")),
)
parser.add_argument("--prompt", default="Explain HashMap in Java with an example.")
parser.add_argument("--max-tokens", type=int, default=65)
args = parser.parse_args()


model, tokenizer = load(args.model)
prompt_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": args.prompt}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
language_model = model.language_model
first_layer = language_model.model.layers[0]
embedding = language_model.model.embed_tokens(mx.array([[prompt_tokens[0]]]))
normalized = first_layer.input_layernorm(embedding)
qkv_projection = first_layer.linear_attn.in_proj_qkv(normalized)
z_projection = first_layer.linear_attn.in_proj_z(normalized)
mx.eval(embedding, normalized, qkv_projection, z_projection)
print(
    "MLX_DTYPES="
    + json.dumps(
        {
            "embedding": str(embedding.dtype),
            "input_rmsnorm": str(normalized.dtype),
            "qkv_projection": str(qkv_projection.dtype),
            "z_projection": str(z_projection.dtype),
        }
    )
)
sampler = make_sampler(
    0.0,
    1.0,
    0.0,
    1,
    top_k=1,
)

generated = []
finish_reason = "empty"
started = time.perf_counter()
with wired_limit(model):
    for token, logprobs in generate_step(
        mx.array(prompt_tokens),
        model,
        max_tokens=args.max_tokens,
        sampler=sampler,
    ):
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        if len(generated) == 64:
            scores = logprobs.tolist()
            if scores and isinstance(scores[0], list):
                scores = scores[0]
            top = heapq.nlargest(5, enumerate(scores), key=lambda item: item[1])
            print("TOKEN_65_TOP_LOGPROBS=" + json.dumps(top))
            print(
                "TOKEN_65_CANDIDATES="
                + json.dumps({"12": scores[12], "16": scores[16]})
            )
        if token_id in tokenizer.eos_token_ids:
            finish_reason = f"eos:{token_id}"
            break
        generated.append(token_id)
        if len(generated) >= args.max_tokens:
            finish_reason = "length"
            break
elapsed = time.perf_counter() - started

print("=== MLX GREEDY REFERENCE ===")
print(f"Prompt tokens: {len(prompt_tokens)}")
print(f"Generated tokens: {len(generated)}")
print(f"Finish reason: {finish_reason}")
print(f"Generation: {elapsed * 1000:.2f} ms ({len(generated) / elapsed:.2f} tok/s)")
print(f"Peak memory: {mx.get_peak_memory() / 1e9:.3f} GB")
print("TOKEN_IDS_JSON=" + json.dumps(generated))
print("--- OUTPUT ---")
print(tokenizer.decode(generated))
print("--- END OUTPUT ---")
