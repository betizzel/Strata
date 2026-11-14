#!/usr/bin/env python3
"""Export a Qwen Flash-Next expert prefix without dequantizing or changing weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402

MAGIC = b"STRNEXP1"
LAYERS, EXPERTS, EMBEDDING, FEED_FORWARD = 48, 512, 2560, 640


def export(gguf: Path, output: Path, owned: int) -> dict:
    if not 0 < owned < EXPERTS:
        raise ValueError("--experts must be between 1 and 511")
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise ValueError("output or its unfinished .tmp already exists; choose a new output")
    source = GGUFFile(gguf)
    if source.metadata.get("general.architecture") != "qwen4exp":
        raise ValueError("this exporter requires Qwen Flash-Next (qwen4exp)")
    for key, expected in (("block_count", LAYERS), ("embedding_length", EMBEDDING),
                          ("expert_count", EXPERTS)):
        if source.metadata.get("qwen4exp." + key) != expected:
            raise ValueError("unsupported Qwen geometry: " + key)
    if any(key.startswith("prism.hadamard.") for key in source.metadata):
        raise ValueError("Hadamard-folded experts are not supported by this experiment")
    marker = Path(str(gguf) + ".done").read_text(encoding="ascii").strip()
    match = re.fullmatch(r"sha256 ([0-9a-f]{64})", marker)
    if not match:
        raise ValueError("the source needs its verified sha256 .done marker")
    model_id = match.group(1)
    tensors = {t.name: t for t in source.tensors}
    table, roles, total = [], [], 0
    for layer in range(LAYERS):
        selected = [tensors[f"blk.{layer}.ffn_{name}_exps.weight"] for name in ("gate", "up", "down")]
        gate, up, down = selected
        if gate.shape != [EMBEDDING, FEED_FORWARD, EXPERTS] or up.shape != gate.shape or \
                down.shape != [FEED_FORWARD, EMBEDDING, EXPERTS] or up.type_id != gate.type_id:
            raise ValueError(f"unexpected expert role geometry at layer {layer}")
        if gate.type_id not in (18, 21, 22, 23) or down.type_id not in (20, 42):
            raise ValueError(f"unsupported IQ3_S quant pair at layer {layer}")
        gu_block, gu_bytes = BLOCK_GEOMETRY[gate.type_name]
        d_block, d_bytes = BLOCK_GEOMETRY[down.type_name]
        gu_row = EMBEDDING // gu_block * gu_bytes
        d_row = FEED_FORWARD // d_block * d_bytes
        sizes = (gu_row * FEED_FORWARD, gu_row * FEED_FORWARD, d_row * EMBEDDING)
        for t, size in zip(selected, sizes):
            if t.expected_bytes() != size * EXPERTS or source.data_start + t.offset + size * EXPERTS > gguf.stat().st_size:
                raise ValueError(f"tensor payload does not match layout: {t.name}")
        blob_bytes = sum(sizes)
        table.append(struct.pack("<IIQQQ", gate.type_id, down.type_id, gu_row, d_row, blob_bytes))
        roles.append([(source.data_start + t.offset, size) for t, size in zip(selected, sizes)])
        total += owned * blob_bytes
    manifest = model_id.encode("ascii") + struct.pack("<IIIIIQ", LAYERS, EXPERTS, owned,
                                                      EMBEDDING, FEED_FORWARD, total) + b"".join(table)
    header = MAGIC + struct.pack("<I", len(manifest)) + manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    digest = hashlib.sha256()
    digest.update(header)
    written = 0
    with gguf.open("rb", buffering=0) as inp, temporary.open("xb") as out:
        out.write(header)
        for layer, slices in enumerate(roles):
            for expert in range(owned):
                for offset, size in slices:
                    inp.seek(offset + expert * size)
                    data = inp.read(size)
                    if len(data) != size:
                        raise ValueError(f"source ended early at layer {layer}, expert {expert}")
                    out.write(data)
                    digest.update(data)
                    written += size
            print(f"layer {layer + 1}/{LAYERS}: {written / 2**30:.3f} GiB exported", flush=True)
        out.flush()
        os.fsync(out.fileno())
    if written != total:
        raise ValueError("exported expert byte count differs from manifest")
    temporary.replace(output)
    sha256 = digest.hexdigest()
    Path(str(output) + ".sha256").write_text(f"{sha256}  {output.name}\n", encoding="ascii")
    return {"path": str(output), "source_sha256": model_id, "sha256": sha256,
            "owned_per_layer": owned, "experts": owned * LAYERS, "weight_bytes": total,
            "file_bytes": output.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=96, help="owned prefix per layer")
    args = parser.parse_args()
    try:
        report = export(args.gguf, args.out, args.experts)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"expert export: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
