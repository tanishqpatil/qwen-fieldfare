# S3-00 consumed-state contract — rejected candidate

Production Sources/Package/pins have been restored to83d28e2 after the candidate
failed sustained same-Swift equivalence. No cache mode or controls ship. This
document preserves inspected boundary semantics and the proposed candidate
contract,not a validated implementation. See
[failed qualification](evidence/2026-11-18-s300/README.md). Standalone and app-owned
servers still share CoreBackend/one serialized active Q4 runner;all requests reset
and replay. Native chat remains separate/exclusive.

For prompt P of p tokens, forwardToken(P[i],i) updates every layer's state for
that token at absolute position i. Each full-attention layer writes token-major
FP16 K/V only at that position; attention reads positions0...i. Each linear layer
mutates its FP32 Conv history (8192*4*4 bytes) and FP32 DeltaNet state
(32*128*128*4 bytes). These recurrent states cannot rewind by changing a counter.
The expert-weight SLRU cache is a separate resource,not conversation state.

The last prompt forward also predicts the first completion token with the current
request's sampling settings,seen set of **all** prompt tokens and samplePosition0.
GenerationLoop checks cancellation and EOS before publishing/consuming a token.
It publishes a non-EOS token and only feeds it back if another prediction is
needed. Therefore:

- EOS return: represented sequence is P+all returned tokens;EOS itself was never
  consumed or returned. Even an immediate EOS leaves the full prompt consumed.
- Output/context limit: represented sequence is P+returnedTokens.dropLast();the
  last returned token is pending/unconsumed. An empty result/exception must not
  be guessed into a valid boundary.
- During an error,forward may have modified only part of the model. Invalidate
  metadata and require a full reset before another request;never publish partial
  state as a reusable prefix. Cancel/disconnect/shutdown follow the same rule.

A safe in-place hit requires the complete represented token sequence to be an
**exact strict prefix** of the new template-tokenized prompt. Compare IDs,not
message text or visible completion strings. The remaining suffix must include
at least one token:the existing buffers do not preserve reusable logits for a
fully consumed prompt under arbitrary new sampling settings. Otherwise reset.
An LCP shorter than represented state never authorizes recurrent rollback.
These are necessary conditions,not sufficient numerical equivalence guarantees:
the failed sustained case demonstrates expert-history sensitivity even when
the complete consumed-token boundary matches. A future implementation needs a
separately validated residency-independent execution contract as well.

Identity is scoped to the same retained runner/checkpoint/tokenizer/template,
context/memory plan and serialization/thinking policy. Changed schema/policy
must invalidate. Requested sampling affects head selection,not already consumed
hidden state;the final fresh prompt forward must use the new complete prompt's
seen-token set and reset samplePosition0 exactly as replay does. No old pending
prediction may be substituted for a fresh prediction without a separate proof.

Normal no-tool history omits the empty think prefix from earlier assistant turns,
so its next prompt usually diverges before the prior prompt ends. For multiple
tool steps after the same last user,the checkpoint Jinja preserves that thinking
prefix;canonical generated XML may share the prefix,or whitespace/argument order/
BPE seams may still differ. Every real request must be checked;never alter the
chat template or expected tokens to manufacture a hit.

Commit requires successful generation,normal EOS,decoder and complete native-tool
validation,then successful response flush without prior cancellation. Flush is
not remote receipt. A disconnect before that boundary aborts;after a successful
complete flush the expected connection close must not retroactively erase a
committed state. Pending state must not be exposed as valid while streaming.
Clear-cache must invalidate metadata safely without mutating buffers during
current GPU work and must prevent that in-flight transaction from committing.

Rejected candidate:one context-bounded token ledger per runner,existing buffers
in place,no KV/Conv/DeltaNet duplication,no disk snapshots,no multi-session map.
If no natural tool continuation preserves a complete prefix,report the observed
miss boundary and stop for a narrowly justified design rather than guessing a
rewind. A possible future recurrent-only snapshot would be exactly66846720 bytes
(63.75MiB) for30 linear layers;KV rewind would separately require proof that older
positions remain untouched and masking/overwrites are safe. This is not an
implemented snapshot or an approved use of unreserved planner headroom.

The archived `QwenPrefixState` prototype uses locked metadata; GPU work remains serialized
by the existing admission owner. Begin makes metadata uncommitted, successful
forwards append consumed IDs at checked positions, finish verifies the observed
EOS/limit boundary, and commit publishes only the current epoch. Clear/mode
changes advance the epoch without touching in-flight GPU buffers. Errors abort;
the next request resets every state buffer before full replay. Model reload,
context or memory changes construct a new runner and random session identity.
Changed schema/formatter/thinking policy changes the backend's compatibility
digest; changed message tokens force replay even if a shorter LCP exists.

Only token element capacity is additional variable storage; diagnostics report
actual Array capacity times four, excluding its fixed header/lock. No model,
KV, Conv or DeltaNet copy is introduced. Expert-weight caching and its policy
remain independent and unchanged. Diagnostic logical reads/fetched bytes are
not physical SSD traffic. No prompts, IDs or tool arguments enter runtime health
records; synthetic benchmark fixtures are separately reviewed public evidence.

Matching IDs proves the represented sequence, not bitwise hidden-state equality:
expert hit/miss grouping can change floating-point reduction order even in the
existing runtime. Qualification compares raw generated IDs and finish reasons
against full replay in the same Swift runtime. It does not establish universal
numerical equivalence or resolve the independent MLX131/132 blocker. No cache
mode remains in production. Five short tool-cycle pairs matched,but
the first sustained512 output diverged. Full replay itself reproduced expert-
history sensitivity. No promotion or acceptance relaxation follows.
