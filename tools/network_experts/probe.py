#!/usr/bin/env python3
"""Bounded application RTTs over a model-identified private SSH/TCP expert session.

These timings include TCP/SSH and worker framing; they are not ICMP/network-only
latencies. No inference is performed and the worker cache is not modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import socket
import struct
import sys
import time

MAGIC = 0x3158454E
VERSION = 2
REPLY = 0x80000000
ERROR = 0xFFFFFFFF
HELLO, PROBE = 1, 4
HEADER = struct.Struct("<IIQQ")
MAX_FRAME = 4 << 20
MAX_PROBE = 1 << 20
PAYLOADS = ((0, 0), (10240, 20480), (20480, 51200),
            (81920, 204800), (MAX_PROBE, MAX_PROBE))


class ProtocolError(ValueError):
    pass


def endpoint_address(endpoint: str) -> tuple[str, int]:
    host, colon, port = endpoint.rpartition(":")
    if not colon or host not in ("127.0.0.1", "localhost") or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("endpoint must be loopback HOST:PORT")
    return host, int(port)


def read_exact(sock: socket.socket, size: int, deadline: float) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("expert response deadline expired")
        sock.settimeout(left)
        count = sock.recv_into(view[offset:])
        if count == 0:
            raise ProtocolError("expert worker disconnected before the full response")
        offset += count
    return bytes(data)


def validate_manifest(body: bytes, model_id: str) -> dict:
    fixed = 64 + struct.calcsize("<IIIIIQ")
    if len(body) != fixed + 48 * 32 or body[:64] != model_id.encode("ascii"):
        raise ProtocolError("HELLO manifest identity or length mismatch")
    layers, experts, owned, embedding, ff, weights = struct.unpack_from("<IIIIIQ", body, 64)
    if (layers, experts, embedding, ff) != (48, 512, 2560, 640) or not 0 < owned < experts:
        raise ProtocolError("unsupported HELLO manifest geometry")
    total = 0
    for offset in range(fixed, len(body), 32):
        gu_type, down_type, gu_row, down_row, blob = struct.unpack_from("<IIQQQ", body, offset)
        if not 0 < gu_row <= 65536 or not 0 < down_row <= 65536 or \
                blob != 2 * gu_row * ff + down_row * embedding or blob > 16 << 20:
            raise ProtocolError("invalid HELLO expert layout")
        total += blob * owned
    if total != weights:
        raise ProtocolError("HELLO weight byte count mismatch")
    return {"model_id": model_id, "layers": layers, "experts": experts,
            "owned": owned, "embedding": embedding, "weight_bytes": weights}


class ProbeSession:
    """Sequential v2 frames, with exactly one HELLO on this same TCP session."""
    def __init__(self, sock: socket.socket, timeout: float):
        self.sock, self.timeout, self.sequence = sock, timeout, 0

    def exchange(self, op: int, body: bytes) -> tuple[bytes, int]:
        self.sequence += 1
        sequence = self.sequence
        frame = HEADER.pack(MAGIC, op, sequence, len(body)) + body
        started = time.monotonic_ns()
        deadline = time.monotonic() + self.timeout
        self.sock.settimeout(self.timeout)
        self.sock.sendall(frame)
        magic, reply_op, reply_sequence, size = HEADER.unpack(read_exact(self.sock, HEADER.size, deadline))
        if magic != MAGIC or reply_sequence != sequence or size > MAX_FRAME:
            raise ProtocolError("invalid expert response magic, sequence or size")
        if reply_op == ERROR:
            if size > 4096:
                raise ProtocolError("oversized expert error response")
            read_exact(self.sock, size, deadline)
            # Worker text is untrusted; never echo arbitrary remote data into logs.
            raise ProtocolError("worker rejected the request (check model identity and protocol v2)")
        if reply_op != op | REPLY:
            raise ProtocolError("unexpected expert response operation")
        response = read_exact(self.sock, size, deadline)
        return response, time.monotonic_ns() - started


def probe(endpoint: str, model_id: str, samples: int = 5, timeout: float = 15.0,
          result: dict | None = None) -> dict:
    """Populate result incrementally, preserving all observed RTTs on failure.

    Caller owns serialization, including the partial result when this raises.
    Empty-payload samples are the small raw TCP/SSH-path application RTT.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", model_id):
        raise ValueError("model-id must be a lowercase SHA256")
    if not 1 <= samples <= 100 or not 0 < timeout <= 300:
        raise ValueError("samples must be 1..100 and timeout must be in (0, 300]")
    address = endpoint_address(endpoint)
    result = result if result is not None else {}
    result.update(schema_version=1, protocol_version=VERSION, status="running", endpoint=endpoint,
                  model_id=model_id, timing_kind="application_tcp_ssh_roundtrip", samples=[])
    try:
        connected = time.monotonic_ns()
        with socket.create_connection(address, timeout=timeout) as sock:
            result["connect_ns"] = time.monotonic_ns() - connected
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            session = ProbeSession(sock, timeout)
            manifest, elapsed = session.exchange(HELLO, struct.pack("<I", VERSION) + model_id.encode("ascii"))
            result["hello"] = {"roundtrip_ns": elapsed, "request_bytes": HEADER.size + 68,
                               "response_bytes": HEADER.size + len(manifest)}
            result["manifest"] = validate_manifest(manifest, model_id)
            for request_padding, response_padding in PAYLOADS:
                payload = struct.pack("<II", response_padding, 0) + bytes(request_padding)
                for sample in range(samples):
                    row = {"sample": sample, "request_padding_bytes": request_padding,
                           "response_padding_bytes": response_padding,
                           "request_bytes": HEADER.size + len(payload),
                           "response_bytes": HEADER.size + 8 + response_padding,
                           "status": "running"}
                    result["samples"].append(row)
                    response, elapsed = session.exchange(PROBE, payload)
                    row["roundtrip_ns"] = elapsed
                    row["observed_response_bytes"] = HEADER.size + len(response)
                    if len(response) != 8 + response_padding:
                        raise ProtocolError("PROBE response length mismatch")
                    row["worker_receive_ns"] = struct.unpack_from("<Q", response)[0]
                    if any(response[8:]):
                        raise ProtocolError("PROBE response padding must be zero")
                    row["status"] = "ok"
        result["status"] = "ok"
        return result
    except BaseException as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        if result["samples"] and result["samples"][-1]["status"] == "running":
            result["samples"][-1]["status"] = "failed"
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema_version": 1, "status": "failed"}
    code = 0
    try:
        probe(args.endpoint, args.model_id, args.samples, args.timeout, result)
    except (OSError, ValueError) as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        print(f"expert probe failed: {type(exc).__name__}", file=sys.stderr)
        code = 1
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
