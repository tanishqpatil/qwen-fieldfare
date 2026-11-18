# Recursive Flint 40B-A2.5B Model Specification

## Architectural Overview

- **Model Family**: `flint`
- **Model Class**: `FlintForCausalLM`
- **Total Parameters**: ~39.8B (39,785,285,888)
- **Active Parameters / Token**: ~2.5B (including output projection)
- **Vocabulary Size**: 150,144
- **Hidden Size**: 2,048
- **Total Transformer Layers**: 49 (indices 0..48)
- **Dense MLP Layers**: 2 (Layers 0 and 1; intermediate size 6,144)
- **Sparse MoE Layers**: 47 (Layers 2 through 48)
- **Routed Experts per MoE Layer**: 256
- **Routed Expert Intermediate Size**: 512
- **Active Routed Experts per Token ($K$)**: 6
- **Shared Experts per MoE Layer**: 1 (always active, intermediate size 1,024, ungated)
- **Activation Function**: SiLU (SwiGLU across dense, routed, and shared MLPs)
- **RMSNorm Epsilon**: 1e-6
- **Tie Word Embeddings**: `false`

## Attention Specification

- **Total Attention Layers**: 49 (All layers 0 through 48 feature self-attention)
- **Query Heads**: 32
- **Key/Value Heads**: 4 (Grouped-Query Attention, 8 query heads per KV head)
- **Head Dimension**: 128 (Total Query dim: 4,096; Total KV dim: 512)
- **Attention Scale**: $1 / \sqrt{128} \approx 0.0883883476$
- **Per-Head Q Normalization**: `q_norm` (RMSNorm on head dim 128, eps=1e-6)
- **Per-Head K Normalization**: `k_norm` (RMSNorm on head dim 128, eps=1e-6)
- **V Normalization**: None
- **Learned Attention Sinks**: `softmax_offset` shape `[32]` (BF16, one per query head). Concatenated to query-key logits as extra logit with no corresponding value.
- **Full-Attention Layers (14 layers)**:
  - Indices: `0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48`
  - Positional Embedding: **NONE** (`no_rope_layer_indices`)
  - Window Size: Infinite (causal over full sequence)
- **Sliding-Window Attention Layers (35 layers)**:
  - Indices: All 35 attention layers other than the 14 full-attention layers
  - Positional Embedding: RoPE (`rope_theta = 10000.0`, `traditional = false`, split-half rotation)
  - Window Size: 513 tokens (current token plus 512 preceding tokens)

## Routing Specification

- **Router**: `model.layers.<idx>.mlp.gate.weight` shape `[256, 2048]`, BF16
- **Router Logit Arithmetic**: FP32 precision
- **Activation / Scoring Function**: Sigmoid: $\text{score}_i = \sigma(W_{gate} \cdot x)$
- **Expert Selection**: Top-6 largest scores
- **Score Normalization**: Selected scores normalized to sum to 1 in FP32:
  $$w_i = \frac{\text{score}_i}{\sum_{j \in \text{Top6}} \text{score}_j + 10^{-20}}$$
- **Expert Execution**:
  $$\text{expert}_i(x) = W_{down}^{(i)} \cdot \left( \text{silu}(W_{gate}^{(i)} \cdot x) \odot (W_{up}^{(i)} \cdot x) \right)$$
- **Routed Output Accumulation**:
  $$\text{routed\_out} = \sum_{k=0}^{5} w_{\pi(k)} \cdot \text{expert}_{\pi(k)}(x)$$
  where $\pi(k)$ denotes the $k$-th chosen expert in strictly fixed router-selected order.
- **Shared Expert Execution**:
  $$\text{shared\_out} = W_{down}^{(s)} \cdot \left( \text{silu}(W_{gate}^{(s)} \cdot x) \odot (W_{up}^{(s)} \cdot x) \right)$$
  (ungated, directly added).
- **MoE Block Output**:
  $$\text{moe\_out} = \text{routed\_out} + \text{shared\_out}$$

## Special Tokens & Chat Template

- **BOS Token**: `<|begin_of_text|>` (ID: 150000)
- **EOS Tokens**:
  - `<|end_of_text|>` (ID: 150001)
  - `<|user|>` (ID: 150003)
  - `<|observation|>` (ID: 150005)
  - `<|developer|>` (ID: 150018)
- **Role Markers**:
  - System: `<|system|>` (ID: 150002)
  - Developer: `<|developer|>` (ID: 150018)
  - User: `<|user|>` (ID: 150003)
  - Assistant: `<|assistant|>` (ID: 150004)
  - Tool observation: `<|observation|>` (ID: 150005)
- **Thinking / Reasoning**:
  - Begin: `<think>` (ID: 150006)
  - End: `</think>` (ID: 150007)
- **Native Tool Calling**:
  - Preamble declares available functions inside `<tools>...</tools>`
  - Model outputs function calls as:
    `<tool_call>{name}<arg_key>{k1}</arg_key><arg_value>{v1}</arg_value>...</tool_call>`
  - Client responds with:
    `<|observation|><tool_response>{result}</tool_response>`

## Physical Checkpoint Layout & Per-Expert Record Specification

Refer to `docs/flint/PHYSICAL_TENSOR_INVENTORY.md` for complete physical details.

- **Checkpoints Supported**:
  - Case A: Official INT4 (`bits = 4`, `group_size = 64` uniform across all projections)
  - Case B: Mixed 3/4-bit (`flint-mixed-3-4bit`: 4-bit attention, INT8 embeddings, 3-bit MoE experts with `group_size = 64`)
- **Per-Expert Payload (3-bit MoE)**:
  - `gate_proj.weight`: U32 `[512, 192]` = 393,216 bytes (scales: 32,768 B, biases: 32,768 B)
  - `up_proj.weight`: U32 `[512, 192]` = 393,216 bytes (scales: 32,768 B, biases: 32,768 B)
  - `down_proj.weight`: U32 `[2048, 48]` = 393,216 bytes (scales: 32,768 B, biases: 32,768 B)
  - **Total Per-Expert Record Size**: $1,376,256\text{ bytes} = 1.3125\text{ MiB}$ exactly.
- **Closed-Form Expert Offset for Expert $e \in [0, 255]$**:
  - $\text{offset}_{\text{expert\_weights}} = \text{base\_offset} + e \times 393,216$
  - $\text{offset}_{\text{expert\_scales}} = \text{base\_offset} + e \times 32,768$
  - $\text{offset}_{\text{expert\_biases}} = \text{base\_offset} + e \times 32,768$

