#!/usr/bin/env python3
"""Model-free tests for the separate known-issue checker."""
import unittest
import hashlib
from pathlib import Path
import tempfile
from check_numerical_non_regression import check_log, check_sources, numerical_sources

BASELINE = {'reference_count': 132, 'known_mismatches': [[61, 14988, 10000]]}
HEADER = 'Reference checkpoint SHA-256 verified; versions: {}\n'
KNOWN = 'Mismatch at reference token 61: expected 14988, got 10000\n'
SUMMARY = 'Teacher-forced parity: 131/132 matched\n'


class KnownIssueChecks(unittest.TestCase):
    def test_source_guard_rejects_numerical_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['Package.swift', 'Package.resolved', 'Sources/QwenFieldfareCore/Kernels/example.metal']:
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('original')
            baseline = {'numerical_sources_sha256': {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in numerical_sources(root)}}
            check_sources(root, baseline)
            (root / 'Package.resolved').write_text('changed dependency')
            with self.assertRaises(ValueError):
                check_sources(root, baseline)
            (root / 'Package.resolved').write_text('original')
            (root / 'Sources/QwenFieldfareCore/Kernels/new.metal').write_text('new shader')
            with self.assertRaises(ValueError):
                check_sources(root, baseline)

    def test_accepts_exact_known_failure_without_claiming_parity(self):
        self.assertEqual(check_log(HEADER + KNOWN + SUMMARY, BASELINE), (131, 132))

    def test_accepts_no_mismatches_as_non_regression(self):
        self.assertEqual(check_log(HEADER + 'Teacher-forced parity: 132/132 matched\n', BASELINE), (132, 132))

    def test_rejects_an_additional_mismatch(self):
        with self.assertRaises(ValueError):
            check_log(HEADER + KNOWN + 'Mismatch at reference token 62: expected 1, got 2\n'
                      + 'Teacher-forced parity: 130/132 matched\n', BASELINE)

    def test_rejects_changed_actual_at_known_position(self):
        with self.assertRaises(ValueError):
            check_log(HEADER + KNOWN.replace('10000', '10001') + SUMMARY, BASELINE)

    def test_rejects_incomplete_or_inconsistent_evidence(self):
        cases = [KNOWN + SUMMARY, HEADER + KNOWN, HEADER + SUMMARY,
                 HEADER + KNOWN + SUMMARY + SUMMARY,
                 HEADER + KNOWN + KNOWN + 'Teacher-forced parity: 130/132 matched\n',
                 HEADER + KNOWN + 'Teacher-forced parity: 130/131 matched\n',
                 HEADER + KNOWN + 'Mismatch at reference token invalid\n' + SUMMARY]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                check_log(text, BASELINE)


if __name__ == '__main__':
    unittest.main()
