#!/usr/bin/env python3
"""Opt-in, owned-process local/distributed Strata runs from a saved JSON profile."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
try:
    from .probe import probe
except ImportError:
    from probe import probe


class RunError(RuntimeError):
    pass


class Interrupted(RunError):
    pass


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("variant/prompt names must be safe filenames (1..80 characters)")
    return value


def finite_seconds(profile: dict, key: str, default: float) -> float:
    value = float(profile.get(key, default))
    if not math.isfinite(value) or not 0 < value <= 7200:
        raise ValueError(f"{key} must be finite seconds in (0, 7200]")
    return value


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise ValueError("profile schema_version must be 1")
    required = ("base_config", "engine", "python", "ai_switch", "ssh_host", "worker", "weights",
                "worker_port", "tunnel_port", "ram_mib", "vram_mib", "mac_arena_gb", "context", "variants", "prompts")
    if any(key not in profile for key in required):
        raise ValueError("profile is missing required fields")
    # Profiles and artifacts deliberately contain no authentication credentials.
    if any(re.search(r"api.?key|password|secret|authorization", key, re.I) for key in profile):
        raise ValueError("profile must not contain credentials")
    for key in ("base_config", "engine", "python", "ai_switch"):
        expanded = Path(profile[key]).expanduser()
        # Preserve a venv interpreter symlink: resolving it changes sys.prefix.
        profile[key] = os.path.abspath(expanded) if key == "python" else str(expanded.resolve())
        if not Path(profile[key]).is_file():
            raise ValueError(f"profile {key} file does not exist")
    for key in ("worker_port", "tunnel_port"):
        if type(profile[key]) is not int or not 1 <= profile[key] <= 65535:
            raise ValueError(f"{key} must be a port in 1..65535")
    for key in ("context", "ram_mib", "vram_mib"):
        if type(profile[key]) is not int or profile[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(profile["ssh_host"], str) or not re.fullmatch(r"[A-Za-z0-9_@.:-]+", profile["ssh_host"]) or profile["ssh_host"].startswith("-"):
        raise ValueError("ssh_host must be an SSH alias or host")
    seen = set()
    if not isinstance(profile["variants"], list) or not profile["variants"]:
        raise ValueError("profile must name at least one variant")
    for variant in profile["variants"]:
        name = safe_name(variant["name"])
        if name in seen or type(variant["remote"]) is not bool:
            raise ValueError("variant names must be unique and remote must be boolean")
        seen.add(name)
        arena = float(variant.get("mac_arena_gb", profile["mac_arena_gb"]))
        if not math.isfinite(arena) or arena <= 0:
            raise ValueError("mac_arena_gb must be positive and finite")
        for key in ("ram_mib", "vram_mib"):
            value = variant.get(key, profile[key])
            if type(value) is not int or value <= 0:
                raise ValueError(f"variant {key} must be a positive integer")
        target = variant.get("ssh_hostname", profile.get("ssh_hostname"))
        if target is not None and (not isinstance(target, str) or
                                   not re.fullmatch(r"[A-Za-z0-9_.:-]+", target) or target.startswith("-")):
            raise ValueError("ssh_hostname must be a hostname or IP address, not SSH options")
    seen.clear()
    for prompt in profile["prompts"]:
        name = safe_name(prompt["name"])
        if name in seen or not isinstance(prompt["text"], str) or not prompt["text"].strip():
            raise ValueError("prompts must have unique names and nonempty text")
        seen.add(name)
    return profile


def benchmark_args(args: list[str], context: int) -> list[str]:
    """Keep model/projection settings; replace only text benchmark/cache knobs."""
    replacements = {"--max-context": str(context), "--prefill": "64", "--spec": "2",
                    "--suffix-draft": "0", "--prompt-cache": "0", "--prompt-cache-every": "0",
                    "--conversation-cache-mib": "0", "--conversation-cache-slots": "0"}
    remove_flags = {"--vision", "--prompt-cache-tail", "--prompt-cache-spread"}
    remove_values = {"--mmproj", "--vision-model", "--vision-exe", "--vision-max-tokens"}
    result = []
    index = 0
    while index < len(args):
        arg = args[index]
        if not isinstance(arg, str) or re.search(r"api.?key|authorization|password|secret", arg, re.I):
            raise ValueError("engine arguments must not contain credentials")
        flag = arg.split("=", 1)[0]
        if flag in replacements or flag in remove_values:
            if "=" not in arg:
                if index + 1 >= len(args) or args[index + 1].startswith("--"):
                    raise ValueError(f"missing value for {flag}")
                index += 1
        elif flag not in remove_flags:
            result.append(arg)
        index += 1
    for flag, value in replacements.items():
        result.extend((flag, value))
    # The runner is deliberately serial. Parallel IPC has different cache/timing semantics.
    if any(x.split("=", 1)[0] in ("--batch", "--slots", "--layer-split") for x in result):
        raise ValueError("benchmark requires a single serial Metal engine")
    return result


@contextmanager
def runner_lock():
    # One memory lock across all profiles, not just a profile-specific port lock.
    path = Path(tempfile.gettempdir()) / f"strata-network-runner-{os.getuid()}.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunError("another cluster runner owns the machine memory/ports") from None
        yield
    finally:
        os.close(descriptor)


def ai_request(message: dict) -> dict:
    """Talk to the existing daemon only; unlike the ai CLI, never auto-start it."""
    path = Path.home() / "Library/Application Support/ai-switch/ai.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(30)
        connection.connect(str(path))
        connection.sendall(json.dumps(message).encode() + b"\n")
        data = bytearray()
        while not data.endswith(b"\n"):
            chunk = connection.recv(65536)
            if not chunk or len(data) + len(chunk) > 4 << 20:
                raise RunError("invalid ai-switch status response")
            data.extend(chunk)
    reply = json.loads(data)
    if not reply.get("ok", True):
        raise RunError("ai-switch rejected the memory/status request")
    return reply


def ai_status() -> dict:
    reply = ai_request({"cmd": "status"})
    if not isinstance(reply.get("status"), dict):
        raise RunError("ai-switch did not return status")
    return reply["status"]


def require_idle(status: dict, live: list | None = None) -> None:
    services = status.get("services")
    if not isinstance(services, list) or not services:
        raise RunError("ai-switch has no readable service status")
    if live:
        raise RunError("ai-switch has active or queued router work; not acquiring memory")
    for service in services:
        if not isinstance(service, dict) or "up" not in service or \
                (service["up"] and service.get("busy") is not False) or service.get("queued") or service.get("waiting"):
            raise RunError("ai-switch service busy, queued or unknown; not acquiring memory")


def checked_status(profile: dict) -> dict:
    status = ai_status()  # a status request refreshes the daemon's probes
    config = json.loads((Path(profile["ai_switch"]).resolve().parent / "services.json").read_text())
    port = (config.get("monitor") or {}).get("port")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise RunError("ai-switch monitor required to inspect queued router work")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stats?since=0", timeout=15) as response:
        monitor = json.load(response)
    if not isinstance(monitor.get("live"), list):
        raise RunError("ai-switch monitor did not report live router work")
    require_idle(status, monitor["live"])
    return status


def production_request(base: dict, route: str, method: str = "GET") -> dict:
    port = base.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise RunError("base production config needs a valid loopback port")
    headers = {}
    key = base.get("api_key")
    if key and str(key).lower() != "none":
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}",
                                     data=b"{}" if method == "POST" else None,
                                     headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RunError("invalid production service response")
    return result


def unload_idle_production(profile: dict, base: dict) -> None:
    checked_status(profile)
    health = production_request(base, "/health")
    if health.get("service") != "strata" or type(health.get("loaded")) is not bool:
        raise RunError("base production health is not readable Strata state")
    status = production_request(base, "/status")
    if status.get("busy") is not False or status.get("queued") != 0:
        raise RunError("production Strata is busy or queued; not unloading")
    if health["loaded"]:
        # /unload itself acquires the service FIFO without waiting/forcing. A
        # request racing our idle check returns HTTP 409 instead of being stopped.
        result = production_request(base, "/unload", "POST")
        if result.get("status") not in ("unloaded", "not loaded"):
            raise RunError("idle production model did not unload")
        if production_request(base, "/health").get("loaded") is not False:
            raise RunError("production model still holds memory")


def acquire_memory(profile: dict, base: dict) -> None:
    checked_status(profile)
    before = base.get("before_load") or [profile["ai_switch"], "use", "strata"]
    # This runner owns no daemon and must never invoke arbitrary lifecycle/restart hooks.
    if not isinstance(before, list) or len(before) != 3 or \
            Path(before[0]).expanduser().resolve() != Path(profile["ai_switch"]) or before[1:] != ["use", "strata"]:
        raise RunError("before_load must be the configured ai_switch use strata command")
    # Same daemon operation as the verified before_load command, without the CLI's
    # daemon_ask fallback, which would start a missing daemon after a race.
    ai_request({"cmd": "use", "name": "strata"})
    deadline = time.monotonic() + finite_seconds(profile, "memory_timeout_s", 120)
    while True:
        status = checked_status(profile)
        if all(not service.get("holds") for service in status["services"] if service["name"] != "strata"):
            unload_idle_production(profile, base)
            return
        if time.monotonic() >= deadline:
            raise RunError("idle services did not release memory; daemon left unchanged")
        time.sleep(0.5)


class SSHWorker:
    def __init__(self, profile: dict, variant: dict, directory: Path):
        self.profile, self.variant, self.directory = profile, variant, directory
        self.proc = None
        self.log = None
        self.thread = None
        self.ready = threading.Event()
        self.ready_line = None
        self.reader_failed = False

    def start(self) -> None:
        p, v = self.profile, self.variant
        with socket.socket() as check:
            check.bind(("127.0.0.1", p["tunnel_port"]))
        worker = [p["worker"], "--weights", v.get("weights", p["weights"]),
                  "--port", str(p["worker_port"]), "--ram-mib", str(v.get("ram_mib", p["ram_mib"])),
                  "--vram-mib", str(v.get("vram_mib", p["vram_mib"]))]
        if any(not isinstance(arg, str) or "\n" in arg or "\x00" in arg for arg in worker):
            raise ValueError("invalid remote worker argument")
        target = v.get("ssh_hostname", p.get("ssh_hostname"))
        command = ["ssh", "-tt", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                   "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
                   *(["-o", f"Hostname={target}"] if target is not None else []),
                   "-L", f"127.0.0.1:{p['tunnel_port']}:127.0.0.1:{p['worker_port']}",
                   p["ssh_host"], "exec flock -n -F /tmp/strata-network-expert-memory.lock " + shlex.join(worker)]
        self.log = (self.directory / "worker.log").open("w", encoding="utf-8")
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + finite_seconds(p, "worker_ready_timeout_s",
                                                   finite_seconds(p, "startup_timeout_s", 180))
        while not self.ready.wait(0.1):
            if self.proc.poll() is not None or self.reader_failed:
                raise RunError("owned SSH worker exited before READY; see worker.log")
            if time.monotonic() >= deadline:
                raise RunError("owned SSH worker READY deadline expired; see worker.log")
        if self.proc.poll() is not None:
            raise RunError("owned SSH worker exited immediately after READY")

    def _read(self) -> None:
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                line = raw.decode("utf-8", "replace")
                self.log.write(line)
                self.log.flush()
                if re.match(rf"^READY port={self.profile['worker_port']} bind=127\.0\.0\.1\b", line.strip()):
                    self.ready_line = line.strip()
                    self.ready.set()
        except (OSError, ValueError):
            self.reader_failed = True

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                # -tt gives the remote worker an owned controlling tty: ^C is SIGINT
                # to that foreground process, not a broad remote process-name kill.
                try:
                    self.proc.stdin.write(b"\x03")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    self.proc.terminate()  # closing the SSH tty also hangs up its worker
                    try:
                        self.proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=5)
            if self.thread is not None:
                self.thread.join(timeout=5)
            self.proc.stdin.close()
            self.proc.stdout.close()
            self.proc = None
        if self.log is not None:
            self.log.close()
            self.log = None


class TimedInput:
    """Delegate the existing IPC unchanged; timestamp the real GEN write boundary."""
    def __init__(self, stream):
        self.stream = stream
        self.gen_started_ns = None

    def write(self, text):
        if text.startswith("GEN "):
            self.gen_started_ns = time.monotonic_ns()
        return self.stream.write(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def timing_fields(started_ns: int, ended_ns: int, arrivals: list[int]) -> dict:
    if ended_ns < started_ns or any(t < started_ns or t > ended_ns for t in arrivals) or \
            any(b < a for a, b in zip(arrivals, arrivals[1:])):
        raise ValueError("non-monotonic token timings")
    intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
    decode_ns = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0
    ordered = sorted(intervals)
    def percentile(fraction):
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else None
    return {"request_wall_ns": ended_ns - started_ns,
            "ttft_ns": arrivals[0] - started_ns if arrivals else None,
            "decode_ns": decode_ns,
            "decode_tps": (len(arrivals) - 1) * 1e9 / decode_ns if decode_ns > 0 else None,
            "inter_token_ns": intervals, "inter_token_p50_ns": percentile(0.50),
            "inter_token_p95_ns": percentile(0.95), "token_arrival_ns": arrivals,
            "token_arrival_offset_ns": [t - started_ns for t in arrivals]}


def load_tokenizer(base: dict):
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    path = Path(base["tokenizer"])
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for text, index in vocab.items():
        tokens[index] = text
    cfg = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer(tokens, (path / "merges.txt").read_text(encoding="utf-8").split("\n"),
                          json.loads((path / "token_type.json").read_text()),
                          cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)
    template = path / "chat_template.jinja"
    return tokenizer, ChatTemplate(template if template.exists() else ROOT / "serve/chat_template.jinja")


def model_identity(base: dict, profile: dict) -> str:
    args = base["args"]
    native = Path(args[args.index("--native") + 1])
    if not native.is_absolute():
        native = Path(base.get("cwd") or ".") / native
    marker = Path(str(native) + ".done").read_text(encoding="ascii").strip()
    match = re.fullmatch(r"sha256 ([0-9a-f]{64})", marker)
    if not match or profile.get("model_id", match.group(1)) != match.group(1):
        raise RunError("native model verified identity missing or disagrees with profile")
    return match.group(1)


class TraceCursor:
    """Read only newly flushed rows; latest is the last COMPLETED window."""
    def __init__(self, path: Path | None):
        self.path = path
        self.stream = None
        self.latest = 0

    def poll(self) -> int:
        if self.path is None:
            return self.latest
        if self.stream is None:
            if not self.path.exists():
                return self.latest  # a local-only engine has no NetworkExperts trace
            self.stream = self.path.open("rb")
        while True:
            offset = self.stream.tell()
            line = self.stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                self.stream.seek(offset)  # an unflushed/partial row is not completed
                break
            row = json.loads(line)
            if row.get("type") == "window":
                window_id = row.get("window_id")
                if type(window_id) is not int or window_id <= self.latest:
                    raise RunError("trace completed window IDs are not increasing")
                self.latest = window_id
        return self.latest

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def validate_inference(count: int, maximum: int, stats: dict, ended_at_eos: bool = False) -> None:
    native_count = stats.get("generated")
    finishes = ("stop", "length", "eos", "cancel") if ended_at_eos else ("stop", "length", "eos")
    if not 0 < count <= maximum or type(native_count) is not int or \
            (native_count < count if ended_at_eos else native_count != count) or stats.get("finish") not in finishes:
        raise RunError("inference returned no tokens, invalid DONE or inconsistent token count")
    if any(isinstance(value, float) and not math.isfinite(value) for value in stats.values()):
        raise RunError("inference returned nonfinite statistics")
    if stats.get("reused", 0):
        raise RunError("prompt reuse occurred despite benchmark cache disable")


def generate_run(engine, tokenizer, ids: list[int], max_tokens: int, timeout: float,
                 sampling: dict, row: dict, directory: Path, trace: TraceCursor, stop_ids: set[int]) -> None:
    arrivals, token_ids, token_windows = [], [], []
    row.update(status="running", token_ids=token_ids, prompt_tokens=len(ids), generated_tokens=0,
               text="", engine_stats={}, trace_file=str(trace.path) if trace.path else None,
               token_window_ids=token_windows)
    first_window = trace.latest
    ended_at_eos = False
    writer = engine.proc.stdin
    writer.gen_started_ns = None
    cancel, expired = threading.Event(), threading.Event()
    def stop_timeout():
        expired.set()
        cancel.set()
        process = engine.proc
        if process is not None and process.poll() is None:
            process.terminate()  # only this request's owned engine, no hidden retry
    timer = threading.Timer(timeout, stop_timeout)
    timer.daemon = True
    timer.start()
    engine.last = {}
    request_start = time.monotonic_ns()
    try:
        first_window = trace.poll()
        with closing(engine.generate(ids, max_tokens, sampling, cancel)) as native:
            for token in native:
                if token is None:
                    continue
                arrival = time.monotonic_ns()
                if type(token) is not int or not 0 <= token < len(tokenizer.tokens):
                    raise RunError("engine returned a token outside the vocabulary")
                token_ids.append(token)
                arrivals.append(arrival)
                completed_window = trace.poll()
                token_windows.append(completed_window or None)
                if token in stop_ids:
                    ended_at_eos = True
                    break  # closing() lets StrataEngine STOP/drain its own native IPC
        if expired.is_set():
            raise RunError("inference request deadline expired")
        stats = dict(engine.last)
        validate_inference(len(token_ids), max_tokens, stats, ended_at_eos)
        row["status"] = "ok"
    except BaseException as exc:
        row.update(status="failed", error_type=type(exc).__name__)
        if isinstance(exc, RunError):
            row["error"] = str(exc)
        raise
    finally:
        ended = time.monotonic_ns()
        timer.cancel()
        timer.join()
        started = writer.gen_started_ns or request_start
        row.update(timing_fields(started, ended, arrivals))
        row.update(generated_tokens=len(token_ids), text=tokenizer.decode(token_ids),
                   engine_stats={key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
                                 for key, value in engine.last.items()}, gen_started_ns=writer.gen_started_ns,
                   request_ended_ns=ended, observed_finish="eos" if ended_at_eos else engine.last.get("finish"))
        trace_failure = None
        try:
            last_window = trace.poll()
        except Exception as exc:
            trace_failure = exc
            last_window = trace.latest
            row.update(status="failed", trace_error_type=type(exc).__name__)
        row["trace_window_start"] = first_window if trace.path else None
        row["trace_window_end"] = last_window if trace.path else None
        save_json(directory / "run.json", row)
        (directory / "output.txt").write_text(row["text"], encoding="utf-8")
        save_json(directory / "tokens.json", {"token_ids": token_ids, "token_arrival_ns": arrivals,
                                            "token_window_ids": token_windows,
                                            "gen_started_ns": writer.gen_started_ns})
        if trace_failure is not None and sys.exc_info()[0] is None:
            raise RunError("requested trace could not be read") from trace_failure


def environment(base: dict, profile: dict, variant: dict, trace: Path | None) -> dict:
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (base.get("env") or {}).items()})
    # The native child does not need the HTTP server's credentials.
    for key in list(env):
        if re.search(r"api.?key|authorization|password|secret|token$", key, re.I):
            del env[key]
    env.pop("STRATA_NETWORK_EXPERTS", None)
    env.pop("STRATA_NETWORK_TRACE", None)
    env["STRATA_METAL_ARENA_GB"] = str(variant.get("mac_arena_gb", profile["mac_arena_gb"]))
    if variant["remote"]:
        env["STRATA_NETWORK_EXPERTS"] = f"127.0.0.1:{profile['tunnel_port']}"
    if trace is not None:
        env["STRATA_NETWORK_TRACE"] = str(trace)
    return env


def run_variant(profile: dict, base: dict, variant: dict, prompts: list[dict], args,
                summary: dict, summary_path: Path, tokenizer, template) -> None:
    from serve import server
    # Keep existing PID/start-time/starter guards and lifeline process ownership.
    # A benchmark failure must not be hidden by the server's transient-start retry.
    server.START_RETRIES = 0
    server.ENGINE_READY_S = finite_seconds(profile, "engine_ready_timeout_s",
                                           finite_seconds(profile, "startup_timeout_s", 1200))
    directory = summary_path.parent / variant["name"]
    directory.mkdir()
    trace = directory / "trace.jsonl" if not args.no_trace else None
    trace_cursor = TraceCursor(trace)
    engine, worker = None, None
    record = {"name": variant["name"], "remote": variant["remote"], "status": "running",
              "trace_file": str(trace) if trace else None,
              "mac_arena_gb": variant.get("mac_arena_gb", profile["mac_arena_gb"])}
    summary["variants"].append(record)
    probes = {"schema_version": 1, "status": "not_applicable" if not variant["remote"] else "running"}
    try:
        acquire_memory(profile, base)
        if variant["remote"]:
            worker = SSHWorker(profile, variant, directory)
            started = time.monotonic_ns()
            worker.start()
            record.update(worker_startup_ns=time.monotonic_ns() - started, worker_ready=worker.ready_line,
                          ssh_alias=profile["ssh_host"],
                          ssh_hostname_override=variant.get("ssh_hostname", profile.get("ssh_hostname")))
            record["probes_file"] = str(directory / "probes.json")
            probes["before"] = {}
            try:
                probe(f"127.0.0.1:{profile['tunnel_port']}", model_identity(base, profile),
                      samples=profile.get("probe_samples", 5), result=probes["before"])
            finally:
                save_json(directory / "probes.json", probes)
        effective_args = benchmark_args(base["args"], profile["context"])
        record["engine_args"] = effective_args  # no base config, env or API keys copied
        started = time.monotonic_ns()
        # Assign before __init__: SIGINT/failed READY still leaves owned-process cleanup possible.
        engine = server.StrataEngine.__new__(server.StrataEngine)
        server.StrataEngine.__init__(engine, profile["engine"], effective_args, cwd=base.get("cwd"),
                                    log=str(directory / "engine.log"), env=environment(base, profile, variant, trace))
        record["engine_startup_ns"] = time.monotonic_ns() - started
        record["engine_info"] = dict(engine.info)
        engine.silence_s = min(60, finite_seconds(profile, "request_timeout_s", 1200))
        engine.proc.stdin = TimedInput(engine.proc.stdin)
        sampling = {"temperature": 0, "seed": 1, "strata_checkpoint": False}
        stop_ids = server.end_token_ids(tokenizer)
        for repetition in range(args.repetitions):
            for prompt in prompts:
                rendered = template.render([{"role": "user", "content": prompt["text"]}], enable_thinking=False)
                ids = tokenizer.encode(rendered, parse_special=True)
                if not ids or len(ids) + args.max_tokens > engine.max_context:
                    raise RunError("prompt plus bounded generation exceeds benchmark context")
                run_dir = directory / f"{prompt['name']}-{repetition + 1:02d}"
                run_dir.mkdir()
                row = {"variant": variant["name"], "prompt": prompt["name"], "prompt_text": prompt["text"],
                       "repetition": repetition + 1, "remote": variant["remote"],
                       "max_tokens": args.max_tokens, "sampling": sampling, "artifact_dir": str(run_dir),
                       "probe_file": str(directory / "probes.json") if variant["remote"] else None}
                summary["runs"].append(row)
                try:
                    generate_run(engine, tokenizer, ids, args.max_tokens,
                                 finite_seconds(profile, "request_timeout_s", 1200), sampling, row, run_dir,
                                 trace_cursor, stop_ids)
                    if args.command == "run":
                        visible = row["token_ids"]
                        if visible and visible[-1] in stop_ids:
                            visible = visible[:-1]
                        print(tokenizer.decode(visible), flush=True)
                    speed = row["decode_tps"]
                    speed_text = f"{speed:.2f}" if speed is not None else "unavailable"
                    print(f"cluster {variant['name']} / {prompt['name']} / {repetition + 1}: "
                          f"{row['generated_tokens']} tokens, decode {speed_text} tok/s, "
                          f"TTFT {row['ttft_ns'] / 1e6:.1f} ms", flush=True)
                finally:
                    save_json(summary_path, summary)
        engine.close()
        engine = None
        if worker is not None:
            probes["after"] = {}
            try:
                probe(f"127.0.0.1:{profile['tunnel_port']}", model_identity(base, profile),
                      samples=profile.get("probe_samples", 5), result=probes["after"])
            finally:
                save_json(directory / "probes.json", probes)
            probes["status"] = "ok"
        record["status"] = "ok"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        if variant["remote"]:
            probes["status"] = "failed"
        raise
    finally:
        cleanup_errors = []
        trace_cursor.close()
        if engine is not None and getattr(engine, "proc", None) is not None:
            try:
                engine.close()
            except Exception as exc:
                cleanup_errors.append(type(exc).__name__)
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                cleanup_errors.append(type(exc).__name__)
        save_json(directory / "probes.json", probes)
        if cleanup_errors:
            record.update(status="failed", cleanup_errors=cleanup_errors)
        save_json(summary_path, summary)
        if cleanup_errors and sys.exc_info()[0] is None:
            raise RunError("owned process cleanup failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "bench"):
        command = commands.add_parser(name)
        command.add_argument("--variant", action="append", help="select saved variant (repeatable)")
        command.add_argument("--max-tokens", type=positive, default=96)
        command.add_argument("--output", type=Path, required=True, help="new artifact directory")
        command.add_argument("--no-trace", action="store_true", help="disable request/window/GPU trace instrumentation")
        if name == "run":
            command.add_argument("--prompt", required=True)
            command.set_defaults(repetitions=1)
        else:
            command.add_argument("--repetitions", type=positive, default=2)
    args = parser.parse_args()
    summary = {"schema_version": 1, "status": "running", "runs": [], "variants": []}
    summary_path = None
    previous_signals = {}
    try:
        profile = load_profile(args.profile)
        python = Path(profile["python"])
        if str(python) != os.path.abspath(sys.executable):
            os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
        selected = set(args.variant or [v["name"] for v in profile["variants"]])
        variants = [v for v in profile["variants"] if v["name"] in selected]
        if len(variants) != len(selected):
            raise ValueError("selected variant not found in profile")
        if args.max_tokens > 4096 or args.repetitions > 20:
            raise ValueError("bounded runner permits at most 4096 output tokens and 20 repetitions")
        prompts = [{"name": "custom", "text": args.prompt}] if args.command == "run" else profile["prompts"]
        if not prompts or len(prompts) > 32 or any(not p["text"].strip() for p in prompts):
            raise ValueError("provide 1..32 nonempty benchmark prompts")
        args.output = args.output.expanduser().resolve()
        args.output.mkdir(parents=True, exist_ok=False)
        summary_path = args.output / "summary.json"
        summary.update(command=args.command, trace_enabled=not args.no_trace, started_unix_ns=time.time_ns(),
                       profile_path=str(args.profile.expanduser().resolve()),
                       timing_definition="TTFT: GEN write to first yielded token; decode: first to last yielded token, N-1 intervals; startup excluded")
        save_json(summary_path, summary)
        def interrupt(signum, _frame):
            # A second signal must not interrupt bounded owned-process teardown.
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            raise Interrupted(f"received signal {signum}")
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_signals[sig] = signal.signal(sig, interrupt)
        base = json.loads(Path(profile["base_config"]).read_text(encoding="utf-8"))
        tokenizer, template = load_tokenizer(base)
        with runner_lock():
            for variant in variants:
                run_variant(profile, base, variant, prompts, args, summary, summary_path, tokenizer, template)
        summary["status"] = "ok"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="failed", error_type=type(exc).__name__)
        if isinstance(exc, RunError):
            summary["error"] = str(exc)
        # Other exception text may contain captured process/config text; never echo it.
        print(f"cluster failed: {type(exc).__name__}; inspect saved artifacts", file=sys.stderr)
        return 1
    finally:
        summary["ended_unix_ns"] = time.time_ns()
        if summary_path is not None:
            save_json(summary_path, summary)
        for sig, previous in previous_signals.items():
            signal.signal(sig, previous)


if __name__ == "__main__":
    raise SystemExit(main())
