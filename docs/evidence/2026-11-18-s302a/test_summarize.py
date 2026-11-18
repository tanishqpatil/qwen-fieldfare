"""Model-free checks for the paired measurement gate."""

import copy
import unittest

from summarize import IDENTICAL_FIELDS, NANOSECOND_FIELDS, analyze, check_pair, dist


def pair(mode="app-seeded", target=128, repeat=1):
    first = {field: 0 for field in IDENTICAL_FIELDS}
    first.update(
        sampler=mode, targetTokens=target, repeat=repeat, mode="off",
        seed=42, promptTokens=69, emittedTokens=target, finish="Output limit",
        outputSHA256="a" * 64, cacheCapacity=2183,
        memoryBefore={"pressureLevel": 1}, memoryAfter={"pressureLevel": 1},
        phases=[], transitionsNs=[5_000_000] * (target - 1), requestNs=10**9,
    )
    second = copy.deepcopy(first)
    phases = {field: 0 for field in NANOSECOND_FIELDS.values()}
    phases.update(selectorWallNs=500_000, selectorThreadCPUNs=450_000, finalCommandGPUMs=0.5,
                  scanNs=400_000, minimumSearches=100)
    second.update(
        mode="on", prefillSeenNs=300, decodeInitialSeenNs=700,
        phases=[dict(phases=phases, cacheLookupMs=.02, missIOms=.5, allGPUWaitMs=1.0,
                     forwardWallMs=4.9)] * (target - 1),
    )
    return first, second


class SummaryTests(unittest.TestCase):
    def test_percentile_definition(self):
        self.assertEqual(dist(list(range(1, 21))),
                         dict(n=20, mean=10.5, p50=10.5, p95=19, min=1, max=20))

    def test_uses_only_inter_token_steps_for_attribution(self):
        records = [
            record for mode in ("greedy", "app-seeded") for count in (128, 512)
            for repeat in range(1, 5 if count == 512 else 4)
            for record in pair(mode, count, repeat)
        ]
        result = analyze(records)["app-seeded-512"]
        self.assertEqual(result["phases_ms"]["selector_wall"]["n"], 2044)
        self.assertEqual(result["zero_selector_latency_reduction_percent"], 10.0)
        self.assertEqual(result["selector_fraction_needed_for_5_percent_latency_reduction"], .5)

    def test_pair_requires_matching_outputs_and_demand(self):
        for field in ("outputSHA256", "decodeMisses"):
            off, on = pair()
            on[field] = "different"
            with self.assertRaisesRegex(ValueError, field):
                check_pair(off, on)

    def test_pair_rejects_bad_pressure_and_missing_transition(self):
        off, on = pair()
        on["memoryAfter"]["pressureLevel"] = 2
        with self.assertRaisesRegex(ValueError, "Elevated"):
            check_pair(off, on)
        on["memoryAfter"]["pressureLevel"] = 1
        on["phases"].pop()
        with self.assertRaisesRegex(ValueError, "sample count"):
            check_pair(off, on)

    def test_duplicate_trials_cannot_displace_a_pair(self):
        off, on = pair()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze([off, on, on])


if __name__ == "__main__":
    unittest.main()
