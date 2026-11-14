"""Deterministic arithmetic, safety, framing-error and lock-cleanup tests (no model)."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import struct
import tempfile
import time
import unittest

from tools.network_experts.cluster import (RunError, TraceCursor, benchmark_args, require_idle,
                                          runner_lock, save_json, timing_fields, validate_inference)
from tools.network_experts.probe import (HEADER, HELLO, MAGIC, REPLY, ProbeSession, ProtocolError,
                                        endpoint_address, validate_manifest)


class TimingTests(unittest.TestCase):
    def test_decode_uses_observed_intervals_not_requested_limit(self):
        result = timing_fields(100, 1000, [200, 300, 500])
        self.assertEqual(result["ttft_ns"], 100)
        self.assertEqual(result["request_wall_ns"], 900)
        self.assertEqual(result["decode_ns"], 300)
        self.assertEqual(result["inter_token_ns"], [100, 200])
        self.assertAlmostEqual(result["decode_tps"], 2e9 / 300)
        self.assertEqual(result["inter_token_p50_ns"], 100)
        self.assertEqual(result["inter_token_p95_ns"], 200)
        validate_inference(3, 96, {"generated": 3, "finish": "stop", "reused": 0})

    def test_eos_observed_count_is_separate_from_native_stop_drain(self):
        validate_inference(3, 96, {"generated": 4, "finish": "cancel"}, ended_at_eos=True)
        with self.assertRaises(RunError):
            validate_inference(3, 96, {"generated": 4, "finish": "cancel"})

    def test_zero_or_one_token_has_no_fabricated_decode_rate(self):
        for arrivals in ([], [200]):
            result = timing_fields(100, 300, arrivals)
            self.assertIsNone(result["decode_tps"])
            self.assertEqual(result["inter_token_ns"], [])
            self.assertEqual(result["decode_ns"], 0)
        self.assertIsNone(timing_fields(100, 300, [])["ttft_ns"])

    def test_non_monotonic_arrivals_are_rejected(self):
        for arrivals in ([200, 190], [99], [301]):
            with self.assertRaises(ValueError):
                timing_fields(100, 300, arrivals)

    def test_equal_arrivals_preserve_batched_native_delivery(self):
        result = timing_fields(100, 500, [200, 200, 400])
        self.assertEqual(result["inter_token_ns"], [0, 200])
        self.assertEqual(result["decode_tps"], 1e7)


    def test_failed_done_and_inconsistent_counts_are_not_success(self):
        cases = ((0, {"generated": 0, "finish": "stop"}),
                 (3, {"generated": 2, "finish": "stop"}),
                 (3, {"generated": 3, "finish": "nonfinite"}),
                 (3, {"generated": 3, "finish": "length", "reused": 1}),
                 (3, {"generated": 3, "finish": "stop", "decode_ms": float("nan")}))
        for count, stats in cases:
            with self.subTest(stats=stats), self.assertRaises(RunError):
                validate_inference(count, 96, stats)
        with self.assertRaises(RunError):
            validate_inference(97, 96, {"generated": 97, "finish": "length"})


class SafetyTests(unittest.TestCase):
    def test_busy_unknown_and_queued_services_are_rejected(self):
        for service in ({"up": True, "busy": True}, {"up": True, "busy": None},
                        {"up": True, "busy": False, "queued": 1},
                        {"up": True, "busy": False, "waiting": True}, {}):
            with self.subTest(service=service), self.assertRaises(RunError):
                require_idle({"services": [service]})
        with self.assertRaises(RunError):
            require_idle({"services": [{"up": True, "busy": False}]}, [{"state": "queued"}])
        require_idle({"services": [{"up": True, "busy": False}, {"up": False, "busy": None}]}, [])

    def test_credentials_and_incomplete_override_fail_before_launch(self):
        for args in (["--api-key", "private"], ["--max-context"], ["--prefill", "--spec", "2"]):
            with self.assertRaises(ValueError):
                benchmark_args(args, 4096)

    def test_memory_lock_releases_after_failure(self):
        class Aborted(Exception):
            pass
        with self.assertRaises(Aborted):
            with runner_lock():
                with self.assertRaises(RunError):
                    with runner_lock():
                        self.fail("second runner acquired the memory lock")
                raise Aborted()
        with runner_lock():
            pass

    def test_failed_json_replacement_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            save_json(path, {"status": "failed", "runs": []})
            with self.assertRaises(ValueError):
                save_json(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"status": "failed", "runs": []})


class TraceTests(unittest.TestCase):
    def test_cursor_reads_appends_and_ignores_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            cursor = TraceCursor(path)
            try:
                self.assertEqual(cursor.poll(), 0)
                path.write_text('{"type":"window","window_id":1,"success":true}\n'
                                '{"type":"request","window_id":2}\n'
                                '{"type":"window","window_id":2')
                self.assertEqual(cursor.poll(), 1)
                self.assertEqual(cursor.poll(), 1)
                with path.open("a") as stream:
                    stream.write(',"success":true}\n')
                self.assertEqual(cursor.poll(), 2)
                self.assertEqual(cursor.poll(), 2)
            finally:
                cursor.close()

    def test_cursor_rejects_repeated_completed_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"type":"window","window_id":1}\n{"type":"window","window_id":1}\n')
            cursor = TraceCursor(path)
            try:
                with self.assertRaises(RunError):
                    cursor.poll()
            finally:
                cursor.close()


class ProbeErrorTests(unittest.TestCase):
    def exchange_reply(self, reply: bytes):
        client, peer = socket.socketpair()
        try:
            peer.sendall(reply)
            peer.shutdown(socket.SHUT_WR)
            return ProbeSession(client, timeout=1).exchange(HELLO, b"request")
        finally:
            client.close()
            peer.close()

    def test_wrong_magic_sequence_and_operation_fail(self):
        for magic, op, sequence in ((0, HELLO | REPLY, 1), (MAGIC, HELLO | REPLY, 2), (MAGIC, REPLY | 4, 1)):
            with self.subTest(magic=magic, op=op, sequence=sequence), self.assertRaises(ProtocolError):
                self.exchange_reply(HEADER.pack(magic, op, sequence, 0))

    def test_truncated_frame_does_not_count_as_a_complete_rtt(self):
        with self.assertRaises(ProtocolError):
            self.exchange_reply(HEADER.pack(MAGIC, HELLO | REPLY, 1, 8) + b"short")

    def test_model_identity_and_weight_geometry_are_checked(self):
        identity = "a" * 64
        gu_row, down_row, owned = 832, 640, 128
        blob = 2 * gu_row * 640 + down_row * 2560
        body = identity.encode() + struct.pack("<IIIIIQ", 48, 512, owned, 2560, 640, blob * owned * 48)
        body += struct.pack("<IIQQQ", 21, 20, gu_row, down_row, blob) * 48
        self.assertEqual(validate_manifest(body, identity)["owned"], 128)
        for broken, expected_id in ((body, "b" * 64), (body[:-1], identity),
                                    (body[:84] + struct.pack("<Q", 1) + body[92:], identity)):
            with self.assertRaises(ProtocolError):
                validate_manifest(broken, expected_id)

    def test_probe_cannot_connect_outside_loopback(self):
        for endpoint in ("example.com:50054", "127.0.0.1:0", "127.0.0.1:65536", "bad"):
            with self.assertRaises(ValueError):
                endpoint_address(endpoint)
        self.assertEqual(endpoint_address("127.0.0.1:50554"), ("127.0.0.1", 50554))


if __name__ == "__main__":
    unittest.main()
