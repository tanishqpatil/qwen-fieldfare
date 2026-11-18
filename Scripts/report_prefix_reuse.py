#!/usr/bin/env python3
"""Summarize captured S3-00 evidence; never runs inference or certifies MLX parity."""
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/2026-11-18-s300'
source = json.loads((EVIDENCE / 'raw-equivalence.json').read_text())
rows = source['records']
cycles = [r for r in rows if r['workload'] == 'tool-cycle']
pairs = []
for index in range(source['rounds']):
    modes = {mode: sorted((r for r in cycles if r['round'] == index and r['mode'] == mode),
                          key=lambda r: r['step']) for mode in ['off', 'exact-prefix']}
    if any(len(value) != 4 for value in modes.values()):
        raise SystemExit(f'Incomplete pair {index}; no completed report written')
    control, candidate = modes['off'], modes['exact-prefix']
    for a, b in zip(control, candidate):
        if (a['prompt_tokens'], a['output_sha256'], a['finish']) != (b['prompt_tokens'], b['output_sha256'], b['finish']):
            raise SystemExit(f'Divergence in pair {index}, step {a["step"]}')
    values = {}
    for mode, records in modes.items():
        values[mode] = {
            'prefill_seconds': sum(r['timing']['prefillMs'] for r in records) / 1000,
            'generation_validation_seconds': sum(r['timing']['overallMs'] for r in records) / 1000,
            'first_generated_token_seconds': sum(r['timing']['firstGeneratedTokenMs'] for r in records) / 1000,
            'fresh_prompt_tokens': sum(r['cache']['processedPromptTokens'] for r in records),
            'reused_tokens': sum(r['cache']['reusedPrefixTokens'] for r in records),
            'expert_misses': sum(r['expert_misses'] for r in records),
            'logical_fetched_bytes': sum(r['expert_fetched_bytes'] for r in records),
        }
    prefill = 100 * (1 - values['exact-prefix']['prefill_seconds'] / values['off']['prefill_seconds'])
    total = 100 * (1 - values['exact-prefix']['generation_validation_seconds'] / values['off']['generation_validation_seconds'])
    pairs.append({'round': index, 'modes': values, 'prefill_reduction_percent': prefill,
                  'generation_validation_reduction_percent': total})

def summary(key):
    values = [pair[key] for pair in pairs]
    mean = statistics.mean(values)
    # Predeclared five pairs, Student-t df=4; do not use this interval for
    # incomplete experiments or imply an external-client performance bound.
    half = 2.776445105 * statistics.stdev(values) / math.sqrt(5) if len(values) == 5 else None
    return {'mean_percent': mean, 'paired_t_95_percent': [mean-half, mean+half] if half is not None else None}

sustained = [r for r in rows if r['workload'] == 'sustained-512']
failures = []
for index in range(2):
    modes = {mode: sorted((r for r in sustained if r['round'] == index and r['mode'] == mode),
                          key=lambda r: r['step']) for mode in ['off', 'exact-prefix']}
    if any(len(records) != 2 for records in modes.values()):
        failures.append(f'Sustained pair {index} not completed; experiment stopped at first discrepancy')
        continue
    for a, b in zip(modes['off'], modes['exact-prefix']):
        if (a['prompt_tokens'], a['output_sha256'], a['finish']) != (b['prompt_tokens'], b['output_sha256'], b['finish']):
            failures.append(f'Sustained raw-token divergence in pair {index}, step {a["step"]}')

result = {'qualification_passed': not failures, 'failures': failures,
          'pairs': pairs, 'paired_prefill_reduction': summary('prefill_reduction_percent'),
          'paired_generation_validation_reduction': summary('generation_validation_reduction_percent'),
          'observed_metal_bytes': sorted(set(r['metal_bytes'] for r in rows)),
          'maximum_observed_footprint_bytes': max(r['footprint_bytes'] for r in rows),
          'maximum_token_element_capacity_bytes': max(r['cache']['tokenStorageBytes'] for r in rows),
          'sustained_continuation_counts': [r['output_tokens'] for r in sustained if r['step'] == 3],
          'limits': ['No hidden-state bitwise assertion', 'No physical SSD measurement',
                     'No exact process-memory peak', 'Same-runner generation+validation timing excludes HTTP/client tools',
                     'Five-pair interval describes the short fixture/host only; failed sustained gate blocks promotion',
                     'Candidate reverted; no cache configuration ships',
                     'Strict MLX 131/132 remains unresolved']}
(EVIDENCE / 'paired-report.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if failures:
    raise SystemExit(1)
