# S3-00d exact consumed-token state contract

S3-00d restores one opt-in ledger over the fixed router-order production
runner. Both server entry points use the same `CoreBackend` and serialize
all GPU work through one admission owner. Native chat remains a separate
exclusive mode and always replays. See the
[historical rejection](evidence/2026-11-18-s300/README.md) and the
[accepted full-replay baseline](evidence/2026-11-18-s300c/README.md).

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
Test token, full-logit and hidden-state-equivalence separately from this
boundary contract. The previous implementation's residency-dependent sum
could diverge even with a correct token boundary.

Each ledger is scoped to one immutable retained runner and tokenizer,
checkpoint template, absolute context and generation policy. The backend
hash covers the formatter branch, tool-call eligibility and ordered tool
definitions. The ledger also checks all sampling settings and the seed.
An output limit may vary because it does not affect consumed forwards.
The first prediction always runs on the final fresh prompt token with all
prompt IDs in the seen set and sample position zero.

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

The bounded ledger keeps consumed IDs and absolute positions. Existing
attention KV, Conv and DeltaNet buffers preserve their state in place.
Begin removes published validity. Checked forwards append consumed IDs.
Finish verifies the observed EOS or limit boundary. Publication requires
the current request epoch even if a clear arrives while formatting or
tokenizing. Clear and mode changes only invalidate metadata concurrently.
Errors abort, and the next miss resets every sequence-state buffer.
Model replacement, context changes and restart construct a new runner
and session identity.

Only token element capacity is additional variable storage; diagnostics report
actual Array capacity times four, excluding its fixed header/lock. No model,
KV, Conv or DeltaNet copy is introduced. Expert-weight caching and its policy
remain independent and unchanged. Diagnostic logical reads/fetched bytes are
not physical SSD traffic. No prompts, IDs or tool arguments enter runtime health
records; synthetic benchmark fixtures are separately reviewed public evidence.

An output or context limit records the pending boundary and discards reuse
eligibility conservatively. Health exposes the current readiness state,
last invalidation, full-replay and hit/miss counters, reused and fresh
token counts, context capacity, and allocated token-array element bytes.
The context is the maximum represented length. A strict hit always needs
room for a fresh suffix. Existing memory-plan and output limits apply.

Same-runtime qualification uses the original archived short tool cycle and
the 512-output continuation over fixed router-order reduction. Agreement
under these workloads leaves the independent MLX 131/132 release blocker
in place. See the [numerical policy](NUMERICAL_NON_REGRESSION.md).

## Sampling membership lifetime

The active runner builds one current-request bitmap from every prompt ID
before predicting at sample position zero. GPU prefix reuse can skip
forwards. Membership still includes the full new prompt. During generation,
the bitmap follows the same publish-then-mark order as the Set oracle. The
terminal pending GPU token rule and cancellation/clear/replay rules stay
independent of this request-local CPU state. Only consumed GPU state and
its checked ledger can be reused across requests.
