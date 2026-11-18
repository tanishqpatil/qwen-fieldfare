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
