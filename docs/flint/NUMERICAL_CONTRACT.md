# FlintFieldfare Numerical Contract & Verification Protocol

## 1. Mathematical Standards & Precision Conventions

Flint inference requires strict precision control across the entire compute graph:

1. **Dequantization**:
   - Affine transformation: $\text{weight} = \text{scale} \times \text{int\_val} + \text{bias}$.
   - Dequantization scale and bias are stored as `bfloat16`.
   - Arithmetic during dequantization/projection evaluates in FP32 or native FP16/BF16 without clamping artifacts.
2. **Layer Normalization**:
   - RMSNorm evaluated in FP32: $\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}$, with $\epsilon = 10^{-6}$.
   - Normalization applies to hidden dimension 2,048 for input and post-attention layernorms.
   - Normalization applies to head dimension 128 for per-head `q_norm` and `k_norm`.
3. **Rotary Position Embedding (RoPE)**:
   - Evaluated on sliding-window attention layers only.
   - Base $\theta = 10000.0$.
   - Split-half permutation (`traditional=False`): $\text{rotate\_half}(x) = [-x_{d/2..d}, x_{0..d/2}]$.
   - Full-attention layers bypass RoPE completely.
4. **Attention Sinks & Softmax**:
   - Each query head $h \in [0, 31]$ has a learned scalar sink $s_h$ (`softmax_offset[h]`).
   - Sinks participate in softmax denominator:
     $$\text{Denom}_i = \sum_{j=0}^{T-1} \exp(S_{i, j}) + \exp(s_h)$$
   - Softmax weights multiply only value vectors $V_{0..T-1}$; the sink probability mass is discarded.
   - Softmax logits evaluated in FP32 for numerical stability.
5. **Router Scoring & Selection**:
   - Router projection $W_{gate} \cdot x$ evaluated in FP32.
   - Elementwise sigmoid: $\text{score}_i = \frac{1}{1 + \exp(-z_i)}$.
   - Top-6 indices selected by largest scores.
   - Normalization: $\tilde{w}_i = \frac{\text{score}_i}{\sum_{j \in \text{Top6}} \text{score}_j + 10^{-20}}$ in FP32.
6. **Fixed Router-Order Expert Reduction**:
   - Selected experts write their weighted down-projection contributions into `contributions[position * 2048 + row]` where `position \in [0, 5]`.
   - The final reduction sums router positions strictly in order $0 \to 1 \to 2 \to 3 \to 4 \to 5$.
   - Cache hit/miss residency **never** alters the reduction order or grouping.
7. **Shared Expert**:
   - Evaluated independently and added directly to the MoE output: $\text{moe\_out} = \text{routed\_out} + \text{shared\_out}$.
8. **Final Projection & LM Head**:
   - Hidden state normalized by final RMSNorm.
   - Converted to FP32 prior to `lm_head` projection matching the reference serving standard.

## 2. Numerical Tolerances & Release Gates

| Operation Level | Target Metric | Acceptable Tolerance | Gate Type |
|---|---|---|---|
| **Level 1: Weight Dequant** | Max absolute difference vs MLX | $\le 10^{-4}$ | Unit Test |
| **Level 2: Norms & RoPE** | Max absolute difference vs MLX | $\le 10^{-5}$ | Unit Test |
| **Level 3: Router Top-6** | Identical expert index selection | 100% exact match | Kernel Test |
| **Level 3: Router Weights** | Max absolute difference vs reference | $\le 10^{-5}$ | Kernel Test |
| **Level 4: One Expert Forward**| Relative L2 error | $\le 10^{-3}$ | Component Test |
| **Level 5: Attention Layer** | Relative L2 error | $\le 10^{-3}$ | Component Test |
| **Level 6: Full Decoder Layer**| Relative L2 error | $\le 2 \times 10^{-3}$ | Integration Test |
| **Level 7: Model Forward** | Top-1 token agreement | $\ge 99.0\%$ on matched prompt | Teacher Forcing |
| **Level 8: Generation** | Deterministic greedy text output | Match reference golden string | System Gate |

## 3. Matched-Reference Strategy

- Ground truth is generated using the local environment's Python MLX model (`mlx_vlm.models.flint`).
- Teacher-forcing scripts capture intermediate tensors at each layer:
  - Input embeddings
  - Router logits, top-6 indices, and normalized weights
  - Attention output and residual
  - MoE output and residual
  - Final layer norm and logits
- Tests compare Swift/Metal outputs directly against these captured goldens.

## 4. Physical Tensor Byte Checksum Ground Truth (F1-03)

Before numerical arithmetic is evaluated, the raw byte storage is verified against independent Python / `mlx.core.load` SHA-256 reference digests.

| Tensor Identifier | Slice Range | Target SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| `model.layers.2.self_attn.softmax_offset` | `0..<64` | `e55d79b0bd25bc2d585bc4bf6706ae3d6d62e2561c9b0ec8f9918041f6404404` | MATCH |
| `model.layers.0.input_layernorm.weight` (Prefix) | `0..<128` | `f33f5b184b24b4e62c37c757ee2d90396173e9ef2ef28af130b266ca021cc4a5` | MATCH |
| `model.layers.0.input_layernorm.weight` (Full Tensor) | `0..<4096` | `a47a5d39863526d5a24983779dda01852c4ea1f6d6989b695da57f18aa7b24a0` | MATCH |
| `model.embed_tokens.weight` | `0..<256` | `c5b8b890109befaf2efd23d462ece554855d2c73d1c14301a8c62afff75bed71` | MATCH |
| `model.layers.2.mlp.switch_mlp.gate_proj.weight` (Expert 0) | `0..<393216` | `a4fa16c82f9b2f8094462209ea1ed8def8d50730030347d96d582d9ce14f7820` | MATCH |
| `model.layers.20.mlp.switch_mlp.gate_proj.weight` (Expert 128) | `50331648..<50724864` | `a3fcc07e846e6c78e5a57c7e89870e97bbb56afe2a0ddb7979043c127acbd678` | MATCH |
| `model.layers.48.mlp.switch_mlp.down_proj.weight` (Expert 255) | `100270080..<100663296` | `2e823253c72b37404f447e5656a08995746460f39b417474eeed19078d9fd15b` | MATCH |
| `model.layers.2.mlp.switch_mlp.gate_proj.scales` (Prefix) | `0..<64` | `7cc18d29413197ee959987c119bbb6e0ed834af8af8107cb3e9e937d9ee267d7` | MATCH |
| `model.layers.2.mlp.switch_mlp.gate_proj.scales` (Expert 0 Full) | `0..<32768` | `91243a6f1cd6e9489624c85e9780297562c8f94ea9aec390d02eae69b774cb40` | MATCH |
| `model.layers.2.mlp.switch_mlp.gate_proj.biases` (Prefix) | `0..<64` | `4629cee3a9d13041dafcf227f7ac56e5c2388a2619ed197595e176a1fd1bab2d` | MATCH |
| `model.layers.2.mlp.switch_mlp.gate_proj.biases` (Expert 0 Full) | `0..<32768` | `86e7916072ec4e247330d049e2712cce082873727610bfaa9007f74256f4373e` | MATCH |

---

## 5. Permanently Resident Weight Ground Truth (F1-04)

Permanently resident weights loaded into Metal shared memory (`MTLBuffer` with `.storageModeShared`) are verified against independent Python/Safetensors SHA-256 digests across early, middle, and final layers:

| Layer Level | Tensor Identifier | Byte Extent | Target SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Global** | `model.embed_tokens.weight` | `0..<128` | `ef0768498be4264c846d1a62c667875d4b5ed9711bf7fa72fa704f04d391d20f` | MATCH |
| **Global** | `lm_head.weight` | `0..<128` | `201e7d45820db8a313367c551ef61352a7394f5efba3560e10a5466f244f932a` | MATCH |
| **Global** | `model.norm.weight` (Full Tensor) | `0..<4096` | `b254961d001bbee273daa00efeef34e4351b783074fcce83b01146d7a5d90f76` | MATCH |
| **Early (L0)** | `model.layers.0.input_layernorm.weight` (Full Tensor) | `0..<4096` | `a47a5d39863526d5a24983779dda01852c4ea1f6d6989b695da57f18aa7b24a0` | MATCH |
| **Early (L0)** | `model.layers.0.self_attn.softmax_offset` (Full Tensor) | `0..<64` | `47fac1a49ccf7fe080e87522368e90e9246c6d8b5d4d9a84a202858d4160188a` | MATCH |
| **Early (L0)** | `model.layers.0.mlp.gate_proj.weight` | `0..<128` | `f483f5e873a8bef9c125f27fd10224b187474a33096a889621a859f2ee2cb0c3` | MATCH |
| **Early (L1)** | `model.layers.1.mlp.down_proj.scales` | `0..<64` | `55661f4a958e53a0230ad7a65299cd13643c54b5a114ad64d450881c18a0c87b` | MATCH |
| **Middle (L20)** | `model.layers.20.input_layernorm.weight` (Full Tensor) | `0..<4096` | `728d0d938132958627ac3a399711084deac01d73fd6cfd7912b10f054136367d` | MATCH |
| **Middle (L20)** | `model.layers.20.self_attn.q_proj.weight` | `0..<128` | `b72b37ce3d6e60dfad8ad92462d9a26dd54c12a1dbace3e6b226acc329a76d0a` | MATCH |
| **Middle (L20)** | `model.layers.20.mlp.gate.weight` (Prefix) | `0..<128` | `a5daf73c87a5381296e830e461d5b2aa29506b54f8a7a3ba1d3fe11b93073fca` | MATCH |
| **Middle (L20)** | `model.layers.20.mlp.gate.weight` (Full 1MB Router) | `0..<1048576` | `edbf9c3b24e0ca035058b9eceba36ee56f634a82325a0608e2018316f0c2a03b` | MATCH |
| **Middle (L24)** | `model.layers.24.mlp.shared_expert.gate_proj.weight` | `0..<128` | `e265ca6a81ea69368b00517677e35eb361f3c02dd3801ca8c457d8be80672f4a` | MATCH |
| **Final (L48)** | `model.layers.48.post_attention_layernorm.weight` (Full Tensor) | `0..<4096` | `640bc5225fae549b69b5a62c25551a1837e0afab300972c89951f2c2c8c01de4` | MATCH |
| **Final (L48)** | `model.layers.48.self_attn.o_proj.scales` | `0..<64` | `6c876aa5c7744296d504573762df1b8398804fc650d76854576e67017c534fa3` | MATCH |
| **Final (L48)** | `model.layers.48.mlp.shared_expert.down_proj.weight` | `0..<128` | `281d309d2d9a3efa0e0aa7a97f12d9f7d6504f76937086a93e28cf0c9cdf0f51` | MATCH |


