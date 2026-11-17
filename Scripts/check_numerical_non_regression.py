#!/usr/bin/env python3
"""Check known-issue stability; never substitutes for strict MLX parity."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'docs/numerical_non_regression_baseline.json'
EXCLUDED_VALIDATOR = 'Sources/QwenFieldfareCore/Infrastructure/ModelIO/QwenModelDirectory.swift'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_sources(root):
    paths = set()
    for directory in ['Sources/QwenFieldfareCore', 'Sources/QwenFieldfareBench']:
        paths.update(str(p.relative_to(root)) for p in (root / directory).rglob('*')
                     if p.is_file() and p.suffix in {'.swift', '.metal'})
    paths.discard(EXCLUDED_VALIDATOR)
    return paths | {'Package.swift', 'Package.resolved'}


def check_sources(root, baseline):
    expected = baseline['numerical_sources_sha256']
    if numerical_sources(root) != set(expected):
        raise ValueError('Numerical source inventory changed; run an appropriate model-backed gate')
    changed = [name for name, value in expected.items() if digest(root / name) != value]
    if changed:
        raise ValueError('Numerical sources changed; model-backed validation required: ' + ', '.join(changed))


def check_log(output, baseline):
    if 'Reference checkpoint SHA-256 verified;' not in output:
        raise ValueError('Missing checkpoint verification record')
    summaries = re.findall(r'^Teacher-forced parity: (\d+)/(\d+) matched$', output, re.M)
    if len(summaries) != 1:
        raise ValueError('Require exactly one complete, uninstrumented parity result')
    matched, total = map(int, summaries[0])
    mismatches = [tuple(map(int, x)) for x in re.findall(
        r'^Mismatch at reference token (\d+): expected (\d+), got (\d+)$', output, re.M)]
    # Reject malformed mismatch records instead of silently ignoring them.
    if len(mismatches) != sum(line.startswith('Mismatch at reference token ') for line in output.splitlines()):
        raise ValueError('Malformed mismatch record')
    if total != baseline['reference_count'] or matched != total - len(mismatches):
        raise ValueError('Incomplete or inconsistent parity result')
    if len(set(x[0] for x in mismatches)) != len(mismatches):
        raise ValueError('Duplicate mismatch positions')
    known = {tuple(x) for x in baseline['known_mismatches']}
    if not set(mismatches).issubset(known):
        raise ValueError('New or changed numerical mismatches')
    return matched, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--unchanged-inference', action='store_true',
                        help='Also require unchanged numerical sources/dependency pins; selection metadata validator excluded')
    args = parser.parse_args()
    try:
        baseline = json.loads(BASELINE.read_text())
        if digest(ROOT / baseline['fixture']) != baseline['fixture_sha256']:
            raise ValueError('Original fixture identity changed')
        matched, total = check_log(args.log.read_text(), baseline)
        if args.unchanged_inference:
            check_sources(ROOT, baseline)
        print(f'Known-issue stability: {matched}/{total}; no additional mismatch records.')
        print('Strict parity remains a separate gate; this check does not certify numerical accuracy.')
        if args.unchanged_inference:
            print('Numerical sources/dependency pins unchanged; metadata selection validator excluded.')
        print('Log evidence only: no model run or checkpoint hashing performed by this checker.')
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f'Non-regression check failed: {error}\n')


if __name__ == '__main__':
    main()
