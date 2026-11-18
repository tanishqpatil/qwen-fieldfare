import unittest

from replay import Partition, analyze_pairs, pair_savings


def timed_row(decisions, timings, reads):
    return {"decisions": decisions, "time": timings, "reads": reads}


def decision(expert, slot, hit, victim=-1):
    return [expert, slot, hit, victim, -1]


class ReplayTests(unittest.TestCase):
    def test_empty_then_probationary_in_slot_order(self):
        cache = Partition(start=10, size=8, protected_limit=8)
        self.assertEqual(cache.acquire(range(8), 1), [decision(i, 10 + i, 0) for i in range(8)])
        self.assertEqual(cache.acquire([8, 0], 2), [decision(8, 10, 0, 0), decision(0, 11, 0, 1)])
        self.assertEqual(cache.state(2)[:2], [[10, 2, 8, 2, 0], [11, 2, 0, 2, 0]])

    def test_promote_demote_and_preserve_active_slot(self):
        cache = Partition(start=0, size=10, protected_limit=8)
        cache.acquire(range(8), 1)
        cache.acquire([8, 9], 2)
        cache.acquire(range(8), 3)
        self.assertEqual(cache.acquire([0, 9], 4), [decision(0, 0, 1), [9, 9, 1, -1, 1]])
        self.assertEqual(cache.acquire([10], 5), [decision(10, 8, 0, 8)])
        self.assertEqual(cache.acquire([11], 6), [decision(11, 1, 0, 1)])

    def test_protected_victim_when_probationary_is_empty(self):
        cache = Partition(start=0, size=8, protected_limit=8)
        cache.acquire(range(8), 1)
        cache.acquire(range(8), 2)
        self.assertEqual(cache.acquire([0, 9], 3), [decision(0, 0, 1), decision(9, 1, 0, 1)])

    def test_split_reader_retirement_and_staging(self):
        left = timed_row([decision(0, 0, 1), decision(1, 1, 0)], [0, 20, 21, 22, 62, 23, 83, 84, 124], [[1, 0, 60]])
        right = timed_row([decision(4, 0, 0, 0), decision(5, 1, 0, 1), decision(6, 2, 0)],
                          [124, 144, 145, 146, 166, 147, 237, 239, 269],
                          [[4, 0, 70], [5, 0, 30], [6, 0, 90]])
        counts, _ = analyze_pairs([(left, right)])
        self.assertEqual((counts["safe_direct"], counts["safe_empty"], counts["blocked_until_hit"],
                          counts["blocked_until_tail"], counts["peak_blocked"]), (1, 1, 1, 1, 2))
        self.assertEqual(pair_savings(left, right, "serial"), (70, 30))
        self.assertEqual(pair_savings(left, right, "dual"), (70, 10))
        self.assertEqual(pair_savings(left, right, "dual", stage_blocked=True), (70, 0))

    def test_all_hit_right_has_no_prefetch_benefit(self):
        left = timed_row([decision(0, 0, 1)], [0, 20, 21, 22, 62, 23, 24, 63, 103], [])
        right = timed_row([decision(0, 0, 1)], [103, 123, 124, 125, 146, 126, 127, 147, 177], [])
        counts, times = analyze_pairs([(left, right)])
        self.assertEqual(counts["pairs_sharing_experts"], 1)
        self.assertEqual(counts["pairs_with_misses"], 0)
        self.assertEqual(times["serial_saved_us"], 0)


if __name__ == "__main__":
    unittest.main()
