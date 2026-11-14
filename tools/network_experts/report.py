#!/usr/bin/env python3
"""Report actual cluster measurements and explicitly modeled Ethernet scenarios."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics

NS = 1_000_000_000


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("percentile needs observations and a quantile in [0,1]")
    ordered = sorted(values)
    p = (len(ordered) - 1) * q
    lo = int(p)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (p - lo)


def distribution(values):
    if not values:
        return None
    return {"count": len(values), "min": min(values), "mean": statistics.fmean(values),
            "p50": percentile(values, .5), "p95": percentile(values, .95), "max": max(values)}


def nonnegative(record, key):
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid nonnegative measurement {key}: {value!r}")
    return value


@dataclass(frozen=True)
class ProbeModel:
    base_ns: float
    per_byte_ns: float
    residual_p95_ns: float = 0

    def predict(self, request_bytes, response_bytes):
        # Empty PROBE already carries 64 bytes of application framing.
        return self.base_ns + self.per_byte_ns * max(0, request_bytes + response_bytes - 64)


def probe_samples(document):
    sections = [document[k] for k in ("before", "after") if isinstance(document.get(k), dict)]
    if not sections:
        sections = [document]
    samples = []
    for section in sections:
        if section.get("status") in ("failed", "error"):
            raise ValueError("cannot estimate from a failed probe")
        for sample in section.get("samples", []):
            if sample.get("status") in ("failed", "error"):
                raise ValueError("probe contains a failed sample")
            for key in ("request_bytes", "response_bytes", "roundtrip_ns"):
                nonnegative(sample, key)
            if sample["roundtrip_ns"] == 0:
                raise ValueError("zero probe RTT is not a measurement")
            samples.append(sample)
    if not samples:
        raise ValueError("no actual probe samples")
    return samples


def fit_probes(samples):
    groups = defaultdict(list)
    for s in samples:
        groups[s["request_bytes"] + s["response_bytes"]].append(s["roundtrip_ns"])
    if 64 not in groups or len(groups) < 2:
        raise ValueError("need empty and nonempty size-matched probes")
    base = percentile(groups[64], .5)
    points = [(size - 64, percentile(times, .5)) for size, times in groups.items() if size > 64]
    # Fixed measured empty RTT; nonnegative least-squares serialization slope.
    slope = max(0., sum(x * (y - base) for x, y in points) / sum(x * x for x, _ in points))
    residuals = [abs(y - base - slope * x) for x, y in points]
    return ProbeModel(base, slope, percentile(residuals, .95))


def ethernet_delay_ns(request_bytes, response_bytes, gbps, rtt_ms, efficiency):
    if not math.isfinite(gbps) or gbps <= 0 or not math.isfinite(rtt_ms) or rtt_ms < 0 or not 0 < efficiency <= 1:
        raise ValueError("Ethernet rate must be positive, RTT nonnegative, efficiency in (0,1]")
    # gbps*1e9 bits/s -> nanoseconds; full request and response serialized sequentially.
    return rtt_ms * 1_000_000 + 8 * (request_bytes + response_bytes) / (gbps * efficiency)


def request_savings(request, model, gbps, rtt_ms, efficiency):
    """Keep all measured local/worker work; replace only exposed network waiting.

    'calibrated' removes no more than the measured probe model predicts. The
    optimistic bound attributes all wait beyond replacement RPC service to the
    old link. Neither discards CUDA time or subtracts overlapping GPU stages.
    """
    for key in ("send_ns", "overlap_ns", "wait_ns", "elapsed_ns", "worker_execute_ns",
                "request_bytes", "response_bytes"):
        nonnegative(request, key)
    elapsed = request["send_ns"] + request["overlap_ns"] + request["wait_ns"]
    if abs(elapsed - request["elapsed_ns"]) > max(1000, elapsed * .001):
        raise ValueError("request elapsed is not send + overlap + wait")
    delay = ethernet_delay_ns(request["request_bytes"], request["response_bytes"], gbps, rtt_ms, efficiency)
    worker = request["worker_execute_ns"]
    overlap, wait = request["overlap_ns"], request["wait_ns"]
    new_wait = max(0., worker + delay - overlap)
    old_modeled_wait = max(0., worker + model.predict(request["request_bytes"], request["response_bytes"]) - overlap)
    optimistic = max(0., wait - new_wait)
    calibrated = min(optimistic, max(0., old_modeled_wait - new_wait))
    return {"calibrated_saved_ns": calibrated, "optimistic_saved_ns": optimistic,
            "replacement_link_ns": delay, "replacement_wait_ns": new_wait}


def select_trace(records, start, end):
    requests, windows = [], {}
    if start > end:
        raise ValueError("reversed trace window range")
    for row in records:
        if row.get("type") not in ("request", "window"):
            continue
        window = row.get("window_id")
        if not isinstance(window, int) or not start < window <= end:
            continue
        if row["type"] == "window":
            if window in windows:
                raise ValueError("duplicate completed window ID")
            if row.get("success") is not True:
                raise ValueError("inference trace contains a failed window")
            nonnegative(row, "wall_ns")
            windows[window] = row
        else:
            requests.append(row)
    if not windows:
        raise ValueError("no completed windows in the recorded run range")
    by_window = defaultdict(list)
    for row in requests:
        if row["window_id"] not in windows:
            raise ValueError("request has no completed parent window")
        by_window[row["window_id"]].append(row)
    for window, rows in by_window.items():
        # One in-flight request; these lifetimes cannot overlap one another.
        if sum(nonnegative(r, "elapsed_ns") for r in rows) > windows[window]["wall_ns"] + 1000:
            raise ValueError("request lifetimes exceed their parent window")
    return requests, windows


def resolve_path(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def read_trace(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def phase_metrics(requests):
    keys = ("elapsed_ns", "send_ns", "overlap_ns", "wait_ns", "worker_receive_ns",
            "worker_prepare_ns", "worker_execute_ns", "gpu_h2d_stage_ns", "gpu_kernel_ns", "gpu_d2h_ns")
    out = {key: distribution([nonnegative(r, key) for r in requests]) for key in keys}
    hits = sum(nonnegative(r, "cache_hits") for r in requests)
    misses = sum(nonnegative(r, "cache_misses") for r in requests)
    out.update(requests=len(requests), cache_hits=hits, cache_misses=misses,
               cache_hit_fraction=hits / (hits + misses) if hits + misses else None,
               upload_bytes=sum(nonnegative(r, "upload_bytes") for r in requests),
               request_bytes=sum(nonnegative(r, "request_bytes") for r in requests),
               response_bytes=sum(nonnegative(r, "response_bytes") for r in requests))
    return out


def analyze_run(run, base, rates, rtts, efficiency):
    if run.get("status") != "ok":
        raise ValueError("generation must have completed successfully before reporting performance")
    out = {key: run.get(key) for key in ("variant", "prompt", "repetition", "remote", "generated_tokens",
                                       "prompt_tokens", "request_wall_ns", "ttft_ns", "decode_ns", "decode_tps")}
    out["inter_token_ns"] = distribution(run.get("inter_token_ns", []))
    trace_file = run.get("trace_file")
    if not run.get("remote") or not trace_file:
        out["trace_available"] = False
        return out
    records = read_trace(resolve_path(trace_file, base))
    requests, windows = select_trace(records, run["trace_window_start"], run["trace_window_end"])
    out["trace_available"] = True
    out["all_phases"] = phase_metrics(requests)
    samples = probe_samples(json.loads(resolve_path(run["probe_file"], base).read_text()))
    model = fit_probes(samples)
    empty = [s["roundtrip_ns"] for s in samples if s["request_bytes"] + s["response_bytes"] == 64]
    out["probe"] = {"empty_roundtrip_ns": distribution(empty), "fitted_base_ns": model.base_ns,
                    "fitted_ns_per_byte": model.per_byte_ns, "median_fit_residual_p95_ns": model.residual_p95_ns,
                    "serialization_bytes_per_s": NS / model.per_byte_ns if model.per_byte_ns else None}
    # Native token arrival markers exclude first-token work and STOP/drain tail.
    markers = run.get("token_window_ids", [])
    steady_ids = set()
    if markers:
        if len(markers) != run["generated_tokens"] or any(a > b for a, b in zip(markers, markers[1:])):
            raise ValueError("token/window attribution is incomplete or nonmonotonic")
        steady_ids = {wid for wid, w in windows.items() if markers[0] < wid <= markers[-1] and w["phase"] == "decode"}
    steady = [r for r in requests if r["window_id"] in steady_ids]
    out["steady_decode"] = phase_metrics(steady)
    intervals = max(0, run["generated_tokens"] - 1)
    out["steady_decode"]["requests_per_token_interval"] = len(steady) / intervals if intervals else None
    out["steady_decode"]["exposed_wait_ns_per_token_interval"] = sum(r["wait_ns"] for r in steady) / intervals if intervals else None
    trace_steady_ns = sum(windows[wid]["wall_ns"] for wid in steady_ids)
    decode_ns = run.get("decode_ns") or 0
    # Refuse to project a span that the token/trace markers cannot reconcile.
    projection_ok = bool(steady_ids and intervals and decode_ns > 0 and trace_steady_ns <= decode_ns + 1_000_000)
    out["decode_projection_available"] = projection_ok
    out["trace_steady_wall_ns"] = trace_steady_ns
    out["ethernet_scenarios"] = []
    for rate in rates:
        for rtt in rtts:
            savings = [request_savings(r, model, rate, rtt, efficiency) for r in requests]
            steady_savings = [s for r, s in zip(requests, savings) if r["window_id"] in steady_ids]
            total_cal = sum(s["calibrated_saved_ns"] for s in savings)
            total_opt = sum(s["optimistic_saved_ns"] for s in savings)
            request_ns = nonnegative(run, "request_wall_ns")
            if total_opt >= request_ns:
                raise ValueError("modeled savings exceed observed request wall time")
            scenario = {"gbps": rate, "assumed_application_rtt_ms": rtt, "payload_efficiency": efficiency,
                        "calibrated_request_ns": request_ns - total_cal,
                        "optimistic_request_ns": request_ns - total_opt,
                        "calibrated_decode_tps": None, "optimistic_decode_tps": None}
            if projection_ok:
                saved_cal = sum(s["calibrated_saved_ns"] for s in steady_savings)
                saved_opt = sum(s["optimistic_saved_ns"] for s in steady_savings)
                if saved_opt >= decode_ns:
                    raise ValueError("modeled savings exceed observed token interval time")
                scenario["calibrated_decode_tps"] = intervals * NS / (decode_ns - saved_cal)
                scenario["optimistic_decode_tps"] = intervals * NS / (decode_ns - saved_opt)
            out["ethernet_scenarios"].append(scenario)
    return out


def aggregate_runs(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[run["variant"]].append(run)
    result = []
    for variant, rows in groups.items():
        timed = [r for r in rows if r["generated_tokens"] > 1 and r["decode_ns"] > 0]
        intervals = sum(r["generated_tokens"] - 1 for r in timed)
        elapsed = sum(r["decode_ns"] for r in timed)
        rates = [r["decode_tps"] for r in timed]
        combined = {"variant": variant, "runs": len(rows), "timed_runs": len(timed),
                    "generated_tokens": sum(r["generated_tokens"] for r in rows),
                    "token_intervals": intervals, "decode_ns": elapsed,
                    "decode_tps": intervals * NS / elapsed if elapsed else None,
                    "per_run_decode_tps_min": min(rates) if rates else None,
                    "per_run_decode_tps_max": max(rates) if rates else None,
                    "ethernet_scenarios": []}
        scenario_groups = defaultdict(list)
        for row in rows:
            for scenario in row.get("ethernet_scenarios", []):
                key = (scenario["gbps"], scenario["assumed_application_rtt_ms"], scenario["payload_efficiency"])
                scenario_groups[key].append((row, scenario))
        for (rate, rtt, efficiency), pairs in scenario_groups.items():
            if len(pairs) != len(rows):
                raise ValueError("cannot aggregate a partially available Ethernet scenario")
            scenario = {"gbps": rate, "assumed_application_rtt_ms": rtt, "payload_efficiency": efficiency,
                        "calibrated_decode_tps": None, "optimistic_decode_tps": None}
            for kind in ("calibrated", "optimistic"):
                speed_key = kind + "_decode_tps"
                if len(timed) == len(rows) and all(s[speed_key] is not None for _, s in pairs):
                    projected = sum((r["generated_tokens"] - 1) * NS / s[speed_key] for r, s in pairs)
                    scenario[speed_key] = intervals * NS / projected
                scenario[kind + "_request_mean_ns"] = statistics.fmean(s[kind + "_request_ns"] for _, s in pairs)
            combined["ethernet_scenarios"].append(scenario)
        result.append(combined)
    return result


def make_report(summary, base, rates=(1., 2.5, 10.), rtts=(.2, .8), efficiency=.9):
    if summary.get("status") != "ok" or not summary.get("runs"):
        raise ValueError("benchmark must be complete and contain successful real runs")
    runs = [analyze_run(run, base, rates, rtts, efficiency) for run in summary["runs"]]
    return {"schema_version": 1, "status": "ok", "runs": runs, "aggregates": aggregate_runs(runs),
            "assumptions": {
                "measured": "native token arrivals, client request/window durations, worker-local durations and matched SSH/TCP probes",
                "aggregation": "sum actual N-1 token intervals divided by summed first-to-last durations, not the arithmetic mean of rates; raw per-run measurements remain in JSON",
                "future_link": "assumed application-level base RTT plus sequential serialization at the stated payload efficiency",
                "calibrated": "remove only observed exposed wait also explained by the size-calibrated current-link probe model",
                "optimistic": "attribute all observed wait beyond replacement worker/link service to the current link",
                "held_fixed": "model, routing, cache hit/miss trace, CPU/CUDA work, local overlap, host send cost and untraced work",
                "not_additive": "worker prepare overlaps GPU stages; elapsed already includes overlap and wait",
                "limits": "not a wired measurement, confidence interval or guarantee; Wi-Fi loss/jitter and future routing/cache changes are not simulated; faster links require compatible NICs at both ends",
                "token_span": "native streamed first-to-last token interval; exclude TTFT and STOP tail via observed completed-window markers"}}


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(report):
    lines = ["# Distributed expert performance", "", "Observed rates come from real native token arrivals. Aggregate rates divide total observed token intervals by total decode time. Ethernet figures are modeled, not measured.", "",
             "| Variant | Runs | Generated tokens | Aggregate decode tok/s | Per-run decode range tok/s |",
             "|---|---:|---:|---:|---:|"]
    for r in report["aggregates"]:
        lines.append(f"| {r['variant']} | {r['runs']} | {r['generated_tokens']} | {fmt(r['decode_tps'])} | {fmt(r['per_run_decode_tps_min'])}–{fmt(r['per_run_decode_tps_max'])} |")
    lines += ["", "## Handoff decomposition (steady decode)", "", "Request lifetime is not bare network RTT. CPU prepare and GPU transfer stages overlap; do not add columns.", "",
              "| Variant / prompt / rep | Calls/token interval | Empty RPC p50/p95 ms | Exposed wait ms/token | CUDA service p50 ms | H2D stage p50 ms | Kernel p50 ms | Cache hits |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["runs"]:
        if not r.get("trace_available"):
            continue
        s = r["steady_decode"]
        p = r["probe"]["empty_roundtrip_ns"]
        def ms(key):
            return s[key]["p50"] / 1e6 if s[key] else None
        lines.append(f"| {r['variant']} / {r['prompt']} / {r['repetition']} | {fmt(s['requests_per_token_interval'])} | {fmt(p['p50']/1e6)} / {fmt(p['p95']/1e6)} | {fmt(s['exposed_wait_ns_per_token_interval']/1e6 if s['exposed_wait_ns_per_token_interval'] is not None else None)} | {fmt(ms('worker_execute_ns'))} | {fmt(ms('gpu_h2d_stage_ns'))} | {fmt(ms('gpu_kernel_ns'))} | {fmt(s['cache_hit_fraction'] * 100 if s['cache_hit_fraction'] is not None else None)}% |")
    lines += ["", "## Modeled Ethernet", "", "Each range is probe-calibrated to optimistic, not a statistical confidence interval. RTT includes application/SSH overhead. Efficiency defaults to 90%.", "",
              "| Variant | Link Gbit/s | Assumed base RTT ms | Estimated decode tok/s | Mean whole request s (calibrated / optimistic) |",
              "|---|---:|---:|---:|---:|"]
    for r in report["aggregates"]:
        for s in r.get("ethernet_scenarios", []):
            speed = f"{fmt(s['calibrated_decode_tps'])}–{fmt(s['optimistic_decode_tps'])}" if s["calibrated_decode_tps"] is not None else "unavailable: token/window span not reconciled"
            lines.append(f"| {r['variant']} | {s['gbps']:g} | {s['assumed_application_rtt_ms']:g} | {speed} | {fmt(s['calibrated_request_mean_ns']/NS)} / {fmt(s['optimistic_request_mean_ns']/NS)} |")
    lines += ["", "## Interpretation", ""] + [f"- **{key.replace('_', ' ')}:** {value}." for key, value in report["assumptions"].items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="summary.json or benchmark directory")
    parser.add_argument("--gbps", type=float, action="append", help="scenario rate; repeatable (default 1,2.5,10)")
    parser.add_argument("--rtt-ms", type=float, action="append", help="assumed application RTT; repeatable (default .2,.8)")
    parser.add_argument("--efficiency", type=float, default=.9)
    parser.add_argument("--out", type=Path, help="write complete JSON analysis")
    parser.add_argument("--markdown", type=Path, help="write rendered Markdown report")
    args = parser.parse_args()
    source = args.summary / "summary.json" if args.summary.is_dir() else args.summary
    try:
        rates, rtts = args.gbps or [1., 2.5, 10.], args.rtt_ms or [.2, .8]
        for rate in rates:
            for rtt in rtts:
                ethernet_delay_ns(0, 0, rate, rtt, args.efficiency)
        report = make_report(json.loads(source.read_text()), source.resolve().parent, rates, rtts, args.efficiency)
        text = markdown(report)
        if args.out:
            args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.markdown:
            args.markdown.write_text(text)
        print(text, end="")
    except (OSError, KeyError, ValueError, TypeError) as error:
        parser.exit(1, f"network performance report: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
