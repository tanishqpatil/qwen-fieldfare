# Model Provenance — Recursive Flint 40B-A2.5B

## Model Identity

- **Name**: `flint-mixed-3-4bit` (derived from `RecursiveSI/flint-int4` and `RecursiveSI/flint`)
- **Repository Source**: `mlx-community/flint-mixed-3-4bit`
- **Base Checkpoint**: `RecursiveSI/flint-int4`
- **Original Model**: `RecursiveSI/flint`
- **License**: Apache 2.0 (matches original Recursive license)
- **Local Location**: `/Users/tanishq/models/flint-mixed-3-4bit`
- **Local Hugging Face Cache**: `/Users/tanishq/.cache/huggingface/hub/models--mlx-community--flint-mixed-3-4bit`

## Checkpoint Files Inventory

| File | Size (Bytes) | Role / Content |
|---|---|---|
| `config.json` | 30,533 | Model hyperparameter configuration, layer types, quantization dictionary |
| `model.safetensors.index.json` | 159,409 | Shard index mapping 1,751 tensors across 4 shards |
| `model-00001-of-00004.safetensors` | 5,297,553,268 | Embeddings, Layers 0–14 |
| `model-00002-of-00004.safetensors` | 5,363,650,068 | Layers 15–28 |
| `model-00003-of-00004.safetensors` | 5,294,980,456 | Layers 29–42 |
| `model-00004-of-00004.safetensors` | 1,881,556,490 | Layers 43–48, final norm, lm_head |
| `tokenizer.json` | 10,879,128 | HuggingFace fast tokenizer definition (150,144 tokens) |
| `tokenizer_config.json` | 30,439 | Tokenizer special token mappings and parameters |
| `special_tokens_map.json` | 20,192 | Explicit special tokens mapping |
| `chat_template.jinja` | 2,328 | Jinja2 prompt template with native tool calling and thinking tags |
| `generation_config.json` | 202 | Default generation parameters |
| `eval_kl.py` | 3,664 | Evaluation script for KL divergence, perplexity, and uint8 packing patch |
| `experts3_from_int4.py` | 2,760 | Script used to quantize routed experts from 4-bit to 3-bit MSE |

## Total Checkpoint Footprint

- **Disk Usage**: 17,837,740,282 bytes (~17.84 GB / ~16.61 GiB)
- **Total Tensors**: 1,751
  - 1,265 `bfloat16` tensors (norms, router gates, scales, biases, sink offsets)
  - 486 `uint32` packed tensors (embeddings, projections, expert weights, lm_head)
- **Total Model Parameters**: 39,785,285,888

## Checkpoint Integrity & Safety

- Local model files are treated as **strictly immutable** by FlintFieldfare.
- Reads use bounded file descriptor offsets (`preadExact` in `FlintSafetensorsFile`).
- Descriptors are verified with file size, inode, and modification times via `stat`/`fstat` (`verifyUnchanged()`).
- No model weights are staged, committed, or pushed to any Git remote.
- See `docs/flint/PHYSICAL_TENSOR_INVENTORY.md` for complete per-shard and per-expert byte inventories.
