"""Numerical safeguards for measured network-expert performance projections."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network_experts import report as R


class LinkModel(unittest.TestCase):
    def test_serialization_accounts_for_both_directions(self):
        # 125000 bytes take 1ms at 1Gbit/s before base application latency.
        self.assertAlmostEqual(R.ethernet_delay_ns(25000, 100000, 1, .3, 1), 1_300_000)
        self.assertAlmostEqual(R.ethernet_delay_ns(25000, 100000, 1, .3, .5), 2_300_000)

    def test_invalid_link_assumptions_are_not_numbers(self):
        for rate, rtt, efficiency in ((0, .2, .9), (float('nan'), .2, .9), (1, -1, .9),
                                      (1, .2, 0), (1, .2, 1.1)):
            with self.subTest(rate=rate, rtt=rtt, efficiency=efficiency), self.assertRaises(ValueError):
                R.ethernet_delay_ns(1, 1, rate, rtt, efficiency)

    def test_probe_fit_uses_measured_empty_round_trip(self):
        samples = [dict(request_bytes=32, response_bytes=32, roundtrip_ns=2_000_000),
                   dict(request_bytes=5032, response_bytes=5032, roundtrip_ns=2_200_000),
                   dict(request_bytes=10032, response_bytes=10032, roundtrip_ns=2_400_000)]
        model = R.fit_probes(samples)
        self.assertEqual(model.base_ns, 2_000_000)
        self.assertAlmostEqual(model.per_byte_ns, 20)
        self.assertAlmostEqual(model.predict(15032, 15032), 2_600_000)

    def test_failed_or_inadequate_probes_cannot_generate_estimates(self):
        with self.assertRaises(ValueError):
            R.probe_samples({'status': 'failed', 'samples': []})
        with self.assertRaises(ValueError):
            R.fit_probes([dict(request_bytes=32, response_bytes=32, roundtrip_ns=1)])


class CriticalPath(unittest.TestCase):
    @staticmethod
    def request(worker, overlap, wait):
        return dict(send_ns=1000, overlap_ns=overlap, wait_ns=wait,
                    elapsed_ns=1000 + overlap + wait, worker_execute_ns=worker,
                    request_bytes=0, response_bytes=0)

    def test_hidden_network_time_is_not_subtracted(self):
        result = R.request_savings(self.request(1_000_000, 20_000_000, 0),
                                   R.ProbeModel(5_000_000, 0), 1, .2, .9)
        self.assertEqual(result['calibrated_saved_ns'], 0)
        self.assertEqual(result['optimistic_saved_ns'], 0)

    def test_new_link_cannot_remove_cuda_critical_path(self):
        result = R.request_savings(self.request(20_000_000, 2_000_000, 25_000_000),
                                   R.ProbeModel(8_000_000, 0), 1, 1, .9)
        # Even a fast link still leaves 20ms CUDA - 2ms overlap + 1ms link waiting.
        self.assertEqual(result['replacement_wait_ns'], 19_000_000)
        self.assertEqual(result['optimistic_saved_ns'], 6_000_000)
        self.assertEqual(result['calibrated_saved_ns'], 6_000_000)

    def test_unexplained_wait_remains_in_calibrated_estimate(self):
        result = R.request_savings(self.request(1_000_000, 1_000_000, 20_000_000),
                                   R.ProbeModel(2_000_000, 0), 1, .5, .9)
        self.assertEqual(result['calibrated_saved_ns'], 1_500_000)
        self.assertEqual(result['optimistic_saved_ns'], 19_500_000)

    def test_inconsistent_request_clocks_are_rejected(self):
        request = self.request(1, 2, 3)
        request['elapsed_ns'] += 1_000_000
        with self.assertRaises(ValueError):
            R.request_savings(request, R.ProbeModel(1, 0), 1, .2, .9)


class Attribution(unittest.TestCase):
    def test_exclusive_start_inclusive_end_isolates_repetitions(self):
        rows = [dict(type='window', window_id=i, wall_ns=100, success=True) for i in range(1, 5)]
        rows += [dict(type='request', window_id=i, elapsed_ns=80) for i in range(1, 5)]
        requests, windows = R.select_trace(rows, 2, 3)
        self.assertEqual(set(windows), {3})
        self.assertEqual([r['window_id'] for r in requests], [3])

    def test_duplicate_failed_and_orphan_windows_are_rejected(self):
        good = dict(type='window', window_id=1, wall_ns=100, success=True)
        cases = [[good, good], [dict(good, success=False)],
                 [good, dict(type='request', window_id=2, elapsed_ns=50)]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                R.select_trace(rows, 0, 2)

    def test_overlapping_request_lifetimes_are_rejected(self):
        rows = [dict(type='window', window_id=1, wall_ns=10000, success=True),
                dict(type='request', window_id=1, elapsed_ns=7000),
                dict(type='request', window_id=1, elapsed_ns=7000)]
        with self.assertRaises(ValueError):
            R.select_trace(rows, 0, 1)


class Aggregation(unittest.TestCase):
    def test_rates_use_observed_intervals_over_total_time(self):
        rows = [{"variant": "same", "generated_tokens": 2, "decode_ns": 1_000_000_000, "decode_tps": 1},
                {"variant": "same", "generated_tokens": 10, "decode_ns": 3_000_000_000, "decode_tps": 3}]
        result = R.aggregate_runs(rows)[0]
        self.assertEqual(result["token_intervals"], 10)
        self.assertEqual(result["decode_tps"], 2.5)  # Arithmetic mean of the two rates would be 2, not 2.5.

    def test_a_single_token_has_no_invented_aggregate_rate(self):
        rows = [{"variant": "one", "generated_tokens": 1, "decode_ns": 0, "decode_tps": None}]
        result = R.aggregate_runs(rows)[0]
        self.assertIsNone(result["decode_tps"])
        self.assertEqual(result["timed_runs"], 0)


class Completion(unittest.TestCase):
    def test_nonfinal_benchmark_cannot_be_reported_as_success(self):
        for status in ("running", "failed", "unknown", None):
            with self.subTest(status=status), self.assertRaises(ValueError):
                R.make_report({"status": status, "runs": [{"status": "ok", "remote": False}]}, Path("."))

    def test_pending_generation_cannot_be_reported_as_success(self):
        with self.assertRaises(ValueError):
            R.analyze_run({"status": "running"}, Path("."), [1], [.2], .9)


if __name__ == '__main__':
    unittest.main()
