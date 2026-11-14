# Strata on a Mac (Apple Silicon, Metal)

This page is for people with an Apple Silicon Mac (M1 or newer). Strata runs there on its own Metal engine: the NVIDIA kernels
are replaced by Metal ones (`metal/`), and the host code and the server are the PC's. Intel Macs cannot run it.

DeepSeek V4.1-Flash runs on a Mac with 64 GB of memory or more, as text only: [DEEPSEEK_MAC.md](DEEPSEEK_MAC.md). It has its
own engine, separate from the Qwen one described on this page.

> **On this page:** [What you need](#what-you-need) · [Install](#install) · [Defaults](#defaults-and-options) ·
> [Gyro](#gyro-s-and-gyro-m-agentionais-trellis-experts) · [Measured](#measured-m5-max-128-gb) · [Limits](#limits-this-version) ·
> [Switches](#switches) · [Network experts](#experimental-network-expert-worker-local-source-build) ·
> [How it works](#how-it-works) · [History](#history--older-measurements)

## Quick start

There are two ways in:

- **The Strata app (`.dmg`) and the Terminal (`strata launch`).** The Metal engine comes ready-made: the `.dmg` has it inside and
  the Terminal fetches it as an engine component. Nothing is built on your Mac and you need no Command Line Tools. If macOS says
  the app is "damaged", see [Troubleshooting](TROUBLESHOOTING.md#the-desktop-app-and-the-terminal). How the builds are signed:
  [packaging/mac/README.md](../packaging/mac/README.md).
- **From source: `./setup.sh`.** It sets things up the way `START-HERE.bat` does on Windows and builds the engine on your Mac
  (about 1 to 15 minutes).

Pictures are on by default in the app, the Terminal and the source build ([Limits](#limits-this-version)). The NVIDIA, AMD and
Intel engines are unchanged.

**Status:** new. It was measured on three Macs: an M5 Max with 128 GB, where the model stays in memory, and an M4 Mac mini with
32 GB and an M1 Max with 34 GB, where the experts stream from the SSD. Other chips are untested, and reports help
([COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)).

## What you need

- **An Apple Silicon Mac (M1 or newer).**
- **macOS 15 or newer.** The engine keeps its buffers resident through a residency set, which macOS 15 introduced.
- **For a source build only:** Command Line Tools (`xcode-select --install`).
- **Enough unified memory for the pack.** The GPU maps the model's main file and keeps it in memory while the model runs. Metal
  lets one program keep about 3/4 of the memory resident (107.5 GB of 128 GB on the M5 Max).

| Pack | The file the GPU maps | Needs about (with prompt buffers, KV, draft layer) | Recommended from |
| --- | --- | --- | --- |
| Q2_0 | 37.6 GB | 46 GB | 64 GB |
| IQ3_XXS | 47.0 GB | 55 GB | 74 GB (where IQ3_S does not fit) |
| IQ3_S (recommended; the default from 96 GB) | 54.8 GB | 63 GB | 96 GB |
| UD-Q4_K_XL (Unsloth; higher quality) | 82.6 GB | 91 GB | 128 GB |
| UD-Q5_K_XL, UD-Q6_K_XL (Unsloth) | 103.9 / 114.8 GB | streams from the SSD on 128 GB | 128 GB, streamed |
| Gyro-S (agentionai, one 58.5 GB file) | 29.7 GB (the other 28.8 GB is the n-gram table) | 38 GB | 48 GB; streams below that |
| Gyro-M (agentionai, one 92.0 GB file) | 63.2 GB (+ the 28.8 GB table) | 71 GB | 96 GB; streams below that |

- **The n-gram table.** Each pack also has a 28.8 GB n-gram table (the model's second file). It is read from the SSD a few rows at
  a time and does not count against the memory above.
- **Disk.** The original Q2_0 and IQ3 packs download roughly 66 to 84 GB, plus preparation and draft data. Unsloth packs are much
  larger and Gyro uses different files. A streamed model can also need a separate experts file (34 to 40 GB for the examples
  below). Check the selected pack rather than treating these sizes as a universal limit.
- **Setup recommends and never forces.** It recommends the pack by memory. A bigger pack chosen with `--model` is kept, with a
  warning about what it risks.

### A Mac whose memory cannot hold the pack (32 to 48 GB)

The engine streams the experts from the SSD. It decides this itself when the experts are more than 60% of Metal's share.

- It keeps the most-used experts in an arena and reads the rest as each token needs them, with read-ahead for the next layer's
  likely experts. The arena is Metal's share, or the memory macOS reports as available less what it keeps for macOS (7 GB, or 15%
  of the RAM when that is more), whichever is smaller. It counts as free only what macOS can give without swapping, fills the
  arena in 1.5 GB steps and stops early rather than push macOS below its reserve.
- On a Mac with 64 GB or more, the arena, the Metal buffers and the host memory together stay under 60% of the RAM, and the
  arena is wired (`mlock`) step by step as it is filled, so macOS does not compress it during the fill or while the GPU is idle.
  `STRATA_METAL_WIRE_ARENA=0` turns the wiring off; `=1` turns it on for a smaller Mac (the 48 GB M5 Pro serves with it).
  `STRATA_METAL_ARENA_GB` sets the arena yourself, with a warning when it is over the cap. The engine prints the
  planned footprint (`planned resident footprint ... cap ...`) and the footprint after the fill.
- Setup writes a second copy of the experts (`<model>.experts`, 40 GB for IQ3_XXS) in which each expert's three matrices sit
  together, so the SSD reads an expert with one request. On the 32 GB M4 that is about 17% faster decoding and 30% faster prompts
  with the same tokens. Setup writes it only when the disk has room for it plus 10 GB, otherwise it warns, and it does not write
  it on a Mac with 96 GB or more.
- Setup configures such a Mac without the draft layer (one token per step is faster there). A streaming Mac with 96 GB or more
  keeps the draft layer and writes `--spec 3`.
- Prompts stream in chunks of 4096 on a streaming Mac with 96 GB or more, and 1024 on a smaller one. `STRATA_METAL_PROMPT_CHUNK=N`
  sets it.
- The model's file must be on the internal SSD. An external disk is untested and likely much slower.

### Local IQ3_S cache trial (M5 Pro, 48 GiB, 2026-11-12)

The isolated `build-metal-cache/strata` build adds `STRATA_METAL_ARENA_MAX_GB`: a ceiling on automatic sizing,
not a forced allocation. It keeps the shrink and stepped-fill checks; the engine rejects a minimum slot layout
that exceeds the ceiling. Do not combine it with `STRATA_METAL_ARENA_GB`, which bypasses those checks.
The Metal server now prints its expert-stream counters on orderly shutdown, before `_Exit` skips its exit handlers.

Five sequential arms completed 20 requests with 4K context, greedy sampling, 1024-token prompt chunks,
MTP/suffix drafting and prompt reuse off, all routed experts and the existing refusal projection unchanged.
Requested ceilings were 8, 12 and 16 decimal GB; actual arenas ranged from 8.00 to 14.33 GB as available memory changed.
The GPU ceiling was 43008 MiB, with 10 decimal GB kept for macOS. All sampled pressure was normal;
swap increased 2.25 MiB and reported free memory stayed at or above 35%.

At nearly equal actual arenas, the original layout (9.50 GB) read the 3041-token prompt in a mean 11.98 s;
the byte-identical contiguous sidecar (9.58 GB) took 7.22 s. Code decode medians were 12.87 and 13.02 tokens/s.
This is about 40% less prompt time, not a comparable decode improvement or a long-context guarantee.
The sidecar's 8.00 and 14.33 GB arms reported cache hit fractions of 70.24% and 82.39%, but the short,
sequential trials do not establish an optimal cache size. Outputs matched across corresponding requests;
the generated prime functions passed 13 cases each and all retrieval answers were correct.
Disk caches were not purged, and the first request includes deferred engine initialization.

Raw results, the bounded runner, output checks and build/test logs:
`~/Developer/ai-switch/results-expert-cache-20261112-114154/`. The sidecar remains isolated in
`~/Developer/ai-switch/results-flash-iq3s-resume-20261112-003906/`; the production binary and profile were not replaced.

### Follow-on server tuning (same Mac, 2026-11-12)

The next isolated experiment completed 102 requests with the same IQ3_S target, refusal projection and routed experts.
All completed requests with identical inputs returned identical tokens across modes. Prime functions passed 13 cases each;
retrieval and copied JSON matched their expected answers. The prose pair was reviewed and matched token for token.
These are small greedy text fixtures, not a broad quality benchmark, vision test or server soak.

| Comparison | Control | Candidate |
| --- | --- | --- |
| 12,041-token fresh prompt, Metal chunks 1024 versus 4096 | 27.50–28.83 s | 18.29 s |
| Short Python function, no drafter versus native MTP `--spec 4` | 14.28–14.60 tok/s | 27.65 tok/s |
| JSON copying, no drafter versus native MTP `--spec 4` | 17.06–17.34 tok/s | 33.45 tok/s |
| Short prose, matched 64K-configured engines, MTP off versus on | 14.72 tok/s | 17.94 tok/s |

The short decode sweep used 4K context and two measured repetitions. The prose pair used one request per arm.
Automatic arena sizes varied: these are sequential end-to-end measurements, not a fixed-residency kernel comparison.
Suffix lookup alone helped copying (24.74 tok/s) but not the prime function; adding it to MTP did not show a clear gain.

The combined test kept 4096-token chunks, a 12 decimal GB automatic arena ceiling, a 10 GB reserve and 65,536 context.
With MTP off/on, the fresh 65,441-token prompt took 103.06/115.59 s: MTP sped up longer output, but cost cold prompt time.
Both then answered the 65,475-token follow-up correctly, with first-token latency 1.27/1.35 s.
The plain/MTP paths reused 65,448/65,449 tokens, reading only 27/26 new tokens. This is existing conversation caching,
not a newly enabled production feature. Disk caches were not purged. The code and prose tests used fresh short inputs,
not long generations at the end of the 65K prompt.

Higher arena ceilings were not a free win. Two 20 GB-ceiling fills hit macOS warning pressure and were stopped without
additional swap growth; higher arms were skipped. One had allocated 15.21 GB, the other stopped during a 12.7 GB plan.
This is not a hard 16 GB hardware threshold. Wiring a smaller arena succeeded, but its modest speed difference was
confounded by a larger actual cache; wiring after the fill did not remove fill-time pressure risk. It is not recommended
as a new default from this evidence. Across the monitored model trials, swap never exceeded the starting 4358.75 MiB.

Native MTP needs Strata's tensor layout, not a llama.cpp `blk.48.*` draft GGUF. The existing fetch/pack tools prepared
the default Q2_0 **draft-only** experts from pinned source `de4b8e4d43b917e7706784d8bb445c9af86a3540` (about 0.89 GB packed).
The main IQ3_S weights were not requantized. Use `STRATA_METAL_MTP_GGUF` for an explicit draft filename and enable
`mtp` in the server config as well as `--mtp`; an explicit `mtp: false` strips the argument. A native Q4_0 draft attempt
found missing Metal expert kernel entrypoints and was not used for performance claims.

Results, failed trials, raw outputs, source/asset provenance, bounded runner and two tested profile choices:
`~/Developer/ai-switch/results-efficiency-20261112-1157/`. `candidate.json` distinguishes output-heavy MTP chat from
MTP-off long-input/short-output requests. The benchmark kept one conversation checkpoint and disabled parked conversations;
it did not validate production's multi-conversation cache policy, stochastic sampling, thinking, tools or vision.
No production binary/profile was replaced. The 43008 MiB GPU limit is unchanged; all managed models were left unloaded.

### Streaming speed work (same Mac, 2026-11-13)

The same IQ3_S target, refusal projection, all ten routed experts, native MTP `--spec 4` with the Q2_0 draft, and 65,536 context.
Each arm ran a fresh engine on one guarded suite: code, prose, an algorithm, an explanation, a four-turn agent conversation
(a 13K-token first prompt, then short tool-result turns) and one request with the production sampling (seeded).
The expert file was evicted from the file cache before every arm. Greedy and seeded-sampled outputs were identical across
all compared arms (`tools/parity.py`; kernel-profiling arms excluded). These are sequential end-to-end measurements on a
desktop in use, not a long-context or vision benchmark.

Where a 4-token decode window went at the start (6 GB arena, 140 ms): about 51 ms waiting for the SSD (near its ~12-14 GB/s for
2 MB random reads), and GPU segments running slower than their 43 ms of pure compute because the GPU clocks down while it waits.

| Change | Effect (decode tok/s; prompt s for the whole suite) |
| --- | --- |
| Starting point: production settings with 2048-token chunks, 6 GB arena | 20.05; 55.5 s |
| `STRATA_METAL_KEEP_WARM=1`: tiny GPU ticks on a second queue while a decode window waits | GPU segment time 73 → 62 ms a window; 21.2 → 23.6 in an A/B |
| `STRATA_METAL_REGION_FIT=1` + `STRATA_METAL_PROMPT_SMALL=256`: prompt buffers inside the arena; short tool-result prompts leave the arena alone | 23.88 at 8.4 GB (the arena no longer gives up 2.4 GB of separate prompt buffers) |
| `STRATA_METAL_DIRECT=1`: page-aligned bounce reads | Unaligned `F_NOCACHE` reads left about 35% of the bytes in the file cache; the cache then grew to ~18 GB and macOS compressed the arena itself. With direct reads the arena stays resident and can grow |
| Purge fix (code): the host code's 6.2 GB of CUDA-shaped weight copies are really released | Footprint before the arena 13.3 → 7.1 GB, resident 11.1 → 8.1 GB. `setPurgeableState(Empty)` had freed nothing: a buffer that has been in the residency set keeps its pages until `release()` |
| Arena size with the above (decode tok/s / cache hits) | 8.4 GB 24.1 / 0.67; 10.5 GB 26.3 / 0.73; 12.5 GB 28.3 / 0.77; 14 GB 29.3-29.5 / 0.81; 15 GB 30.3-30.6 / 0.82 |
| 4096-token prompt chunks instead of 2048 (14 GB arena, both runs) | prompts 51.1 → 44.4 s, the 13K-token agent prompt 24.8 → 21.5 s, decode unchanged |
| `STRATA_METAL_WIRE_ARENA=1`, now wired step by step during the fill (code) | No compression of the fresh arena: first request 3.2-3.5 s instead of up to 14.3 s; 17.5 GB arenas at normal pressure (32.8 tok/s, hits 0.86) |

Unwired, the fresh arena was compressible anonymous memory: in one 15 GB start with 19 GB free, macOS compressed 13.5 GB of
the just-filled (incompressible) experts within 3 s, and the first request waited 10 s while the GPU wired them back; a 17.5 GB
fill reached warning pressure. Wiring each fill step as it lands (`mlock`, only filled slots) moves that demand to the file
cache and to other programs' compressible pages. Wired, a 17.5 GB arena stayed at normal pressure with no swap growth; 19 GB
(5.5 GB left of 24.5 GB available) reached warning pressure and was stopped. Not adopted: a file-cache tier for misses (+10%
decode, prompts 10-20% slower, compressed other programs), queuing read-ahead before misses, reserving I/O threads, 8 I/O
threads, read-ahead top 8 (within noise), `STRATA_SPEC_COST=1` (it chose 1-token windows: 20 tok/s instead of 28), `--spec 3`
(same as 4) and `--spec 5` (27.7), `--suffix-draft 3` with MTP (28.4 vs 29.5, slower agent turns), cache replacement policies
other than LRU (simulated on routing traces: capacity matters, policy does not), and kernel work (the 4-token window's 1,230
dispatches are a flat profile: the largest class is 7.6%). Changing the verify window (spec 3/5, the cost policy, lookups)
also changes the batched verification's arithmetic: their greedy text differed on the long agent prompt; every arm with the
`--spec 4` window matched the baseline token for token.

The served setup since 2026-11-13: `STRATA_METAL_DIRECT=1 STRATA_METAL_KEEP_WARM=1 STRATA_METAL_REGION_FIT=1
STRATA_METAL_PROMPT_SMALL=256 STRATA_METAL_PROMPT_CHUNK=4096 STRATA_METAL_ARENA_MAX_GB=17.5 STRATA_METAL_KEEP_FREE_GB=8
STRATA_METAL_WIRE_ARENA=1 STRATA_METAL_ARENA_CHUNK_GB=14`, from its own source snapshot (`~/Developer/Strata-iq3s-prod-src`,
commit `9bfd2aac`). The arena is sized at each load from the memory free then (16.1 GB in the final check, 14.5 GB in the live
service with the image encoder beside it), so a busier Mac gets a smaller one. With that build and settings the suite decoded
31.4 tok/s (from 20.05) with 44.5 s of prompts (from 55.5 s), identical tokens. Runner, plans, raw results and tools:
`~/Developer/ai-switch/results-iq3s-speed-20261113/` (`RESULTS.md`).


## Experimental network expert worker (local source build)

This checkout has an opt-in Qwen Flash-Next expert worker. It is not part of the released Strata package and is not
llama.cpp's layer RPC. The Mac keeps routing, attention, KV state, shared experts and its local routed experts on Metal.
A separate CUDA process computes a fixed prefix of routed expert IDs in every layer. Those weights are held in the PC's
anonymous system RAM; a bounded VRAM cache stages requested experts from that RAM. During inference the network carries
normalized activations and unweighted expert results, not weight blobs. The Mac applies each router weight once.

The implemented geometry is the plain-basis, 48-layer Qwen Flash-Next native IQ3_S pack: embedding 2560, expert width 640,
512 experts per layer, ten routed experts per token. It does not support DeepSeek, folded Hadamard experts, tensor
parallelism or remote attention state. Prompts deliberately use the existing windows of at most eight tokens rather than
the local prompt-GEMM path. Speculative verification uses the same windows. The normal path is unchanged when
`STRATA_NETWORK_EXPERTS` is unset.

### Build and prepare the PC-owned weights

Run from this source checkout, after its normal Metal dependencies have been prepared:

```sh
cmake -S metal -B build-metal-network -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMETAL_CPP_DIR="$PWD/third_party/metal-cpp" \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama-pinned"
cmake --build build-metal-network --target strata expert_parity -j 4

MODEL="$HOME/Developer/Strata-new-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
EXPERTS="$HOME/Developer/Strata-network-data/iq3_s-prefix128.strnexp"
python3 tools/network_experts/export.py --gguf "$MODEL" --out "$EXPERTS" --experts 128
```

The exporter reads the existing GGUF header and raw gate/up/down slices. It does not requantize or alter the source files.
The source must have its verified `sha256 ...` `.done` marker. It writes a separate file and checksum, and refuses to
overwrite an existing output. With 128 owned experts per layer the payload is 12,573,081,600 bytes (11.710 GiB),
representing 6,144 experts. Keep this derived file on the PC; remove the temporary Mac copy after verifying its transfer.

For the tested CUDA PC (an SSH alias `pc` here), prepare isolated directories and copy only the worker's source closure:

```sh
ssh pc 'mkdir -p ~/Strata-network-src ~/Strata-network-data'
rsync -rlpRc include/ src/kernels/cuda/ third_party/ggml/ third_party/ggml-tq/ \
  tools/network_experts/ LICENSE pc:Strata-network-src/
rsync -a "$EXPERTS" "$EXPERTS.sha256" pc:Strata-network-data/
ssh pc 'env -C ~/Strata-network-data sha256sum -c iq3_s-prefix128.strnexp.sha256'
ssh pc 'cmake -S ~/Strata-network-src/tools/network_experts \
  -B ~/Strata-network-src/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc && \
  cmake --build ~/Strata-network-src/build --target strata-network-experts -j 2'
```

Source synchronization intentionally does not preserve timestamps: future-dated source files on a PC with a slower clock
cause Ninja to regenerate continuously. The tested PC used CUDA 13.4, GCC 16.2 and an RTX 2070 Super (sm_75).

### Run and verify

Start the private worker through an SSH tunnel:

```sh
ssh -tt -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50554:127.0.0.1:50054 pc \
  'exec flock -n -F /tmp/strata-network-expert-memory.lock \
   ~/Strata-network-src/build/strata-network-experts \
   --weights ~/Strata-network-data/iq3_s-prefix128.strnexp \
   --port 50054 --ram-mib 13312 --vram-mib 5120'
```

The worker binds loopback only. Do not expose this protocol as a public service. Its RAM cap covers the actual expert
allocation, not CUDA context or staging overhead. Startup also requires 8 GiB of available host-RAM headroom and checks
that reserve while loading. The weights are pageable anonymous memory, not locked memory: external memory pressure can
still make Linux swap them. The VRAM budget covers cache slots; fixed buffers and the CUDA context are additional.

In another terminal, compare real CUDA outputs with Metal, including the local ownership-mask skip and weighted merge:

```sh
MODEL="$HOME/Developer/Strata-new-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
build-metal-network/expert_parity --gguf "$MODEL" --layer 17 --ntok 8 \
  --network-endpoint 127.0.0.1:50554
```

The verified matrix uses layers `0,1,2,12,17,21,47`, each at one and eight tokens, covering every actual IQ3_S quant pair.
The existing Metal-versus-double checks remain active. CUDA uses the native Q8 activation arithmetic, so CUDA/Metal
results are not bit-identical; the network check bounds weighted-row and mixed-merge error at relative L2 0.03 and
maximum error relative to peak reference 0.06. It also verifies that remotely assigned rows are not computed locally.

For the Mac engine, set `STRATA_NETWORK_EXPERTS=127.0.0.1:50554` and `STRATA_METAL_STREAM=1`, using the separate
`build-metal-network/strata` binary. Keep the regular `--native`, `--ple-gguf`, `--pack`, `--mmap-experts`,
`--expert-profile`, `--expert-cache`, `--prefill` and `--spec` arguments; a native CLI run requires `--spec` of at least two.
`STRATA_ROUTE_TAIL_SKIP=0` preserves all ten routed experts. `STRATA_METAL_ARENA_GB=24` was used for the bounded trial;
it is not a new production default. The engine's report gives completed remote requests, expert rows and network bytes.
The worker reports its real RAM allocation/RSS, VRAM cache, hits, misses and PC-RAM-to-GPU upload bytes.

In that same terminal, this complete short-generation command uses the tested settings. The supplied prompt asks for
37 + 59; the observed output IDs `24 21 248046` decode to `96` followed by the end-of-turn marker.

```sh
PACK="$HOME/Developer/Strata-new-data/packs/iq3_s"
PLE="$HOME/Developer/Strata-new-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
env STRATA_NETWORK_EXPERTS=127.0.0.1:50554 STRATA_METAL_STREAM=1 \
  STRATA_ROUTE_TAIL_SKIP=0 STRATA_METAL_ARENA_GB=24 \
  build-metal-network/strata --pack "$PACK" --native "$MODEL" --ple-gguf "$PLE" \
  --expert-profile data/expert-profile.bin --expert-cache 64 --mmap-experts \
  --prefill 64 --spec 2 --suffix-draft 0 --max-context 1024 --max-new 8 --stop-eos \
  --tokens 248045,846,198,44712,220,18,22,478,220,20,24,13,17308,440,1132,279,7308,13,248046,198,248045,74455,198,248068,271,248069,271 \
  --stats
```

A failed remote contribution cannot become a successful partial answer. The submitted Metal graph is drained safely and
the engine rejects the failed session; reload the engine before using it again. The worker accepts a fresh connection.
No production alias, server port or launch agent is changed by this experiment. Short Wi-Fi runs establish operation,
not Ethernet performance, long-context quality, vision or projection/abliteration parity.

The initial 96-expert, 24 GB-arena Wi-Fi trial used a 48 GB M5 Pro and the RTX 2070 Super PC. A 67-token coding prompt
generated an 89-token `is_prime` function at 1.82 engine-reported tokens/s; executing it passed 12 negative, boundary, prime,
composite and square cases. That request completed 4,646 remote calls covering 22,047 expert rows. The arithmetic smoke
matched the local-only output IDs. Killing the worker during a real prompt caused a clean nonzero exit with an explicit
disconnect error; restarting the worker and reloading the engine restored the arithmetic result.

### Saved performance runner

The repeatable setup uses `tools/network_experts/cluster.py` and the machine profile
`~/Developer/Strata-network-data/iq3s-mac-pc.json`. It is an opt-in command, not an always-on production service.
It owns its native engine and SSH worker, checks that ai-switch is idle, and releases those processes on completion,
failure or interruption. A local memory lock and PC worker lock prevent overlapping instances. Run it while both devices
are otherwise idle; it does not register a new ai-switch service or prevent somebody from starting another AI app.
No production alias, key, run configuration, daemon or launch agent is replaced.

The saved cases use the same experimental binary, 4K text context, greedy sampling, a spec limit of two and 32 GB
requested Mac expert arena. MTP is off, matching the production run config; native `DONE` recorded no draft tokens.
The allocator can reduce the arena to preserve its memory reserve. Prompt and conversation caching are disabled.
The production config supplies the model and per-layer refusal-direction projection arguments, but no API key is
copied into artifacts. Cases cover Mac-only, 256 MiB and 5 GiB CUDA caches through the SSH alias, and a separate 5 GiB
direct-LAN case. All distributed cases assign expert IDs 0–127 in every layer to PC RAM.

The `pc` alias resolved to the PC's Tailscale address, whose observed direct underlay was the PC's LAN address rather
than a relay. The `remote128-lan-cache5120` case sets `ssh_hostname` to that LAN address.
That optional profile/variant field overrides only OpenSSH's target address, preserving the alias's user, port and
host-key checks. No SSH or Tailscale configuration is changed. For a future cable test, set it to the PC's wired IP and
verify the actual SSH route and negotiated link speed before comparing results.

This arena is close to the Mac's GPU working-set cap; a local run reported under 512 MiB free VRAM. The PC also had
nearly full system swap after the comparisons. Workers were released, not made boot services, and no swap reset or
forced eviction of other programs was attempted. If memory pressure appears, lower the selected variant's
`mac_arena_gb` or PC ownership/cache budget and rerun the comparison; do not infer extra memory from the Mac's GPU limit.

```sh
SRC="$HOME/Developer/Strata-ablation-src"
PY="$HOME/.local/share/strata/terminal/0.1.54/python/bin/python3"
PROFILE="$HOME/Developer/Strata-network-data/iq3s-mac-pc.json"
OUT="$HOME/Developer/Strata-network-data/perf-$(date +%Y%m%d-%H%M%S)"
"$PY" "$SRC/tools/network_experts/cluster.py" --profile "$PROFILE" bench \
  --repetitions 2 --max-tokens 96 --output "$OUT"
"$PY" "$SRC/tools/network_experts/report.py" "$OUT" --gbps 1 --gbps 2.5 \
  --rtt-ms 0.2 --rtt-ms 0.8 --out "$OUT/analysis.json" --markdown "$OUT/report.md"
```

For a custom prompt, use `run --variant remote128-lan-cache5120 --prompt 'your prompt' --max-tokens 96 --output NEW_DIR`.
For an instrumentation-overhead comparison, run the same selected benchmark case with `--no-trace` and a new output
directory. The profile's Python path is used automatically, preserving its installed tokenizer/template dependencies.
The output directory must be new: existing results are never overwritten.

Each case keeps its engine resident across repetitions. Results include exact returned token IDs/text, individual
token-arrival timestamps, native `DONE` statistics, engine/worker logs, before/after application probes and buffered JSONL
request/window traces. Window markers associate observed tokens with completed GPU windows. A failed request is recorded
as failed, returns nonzero and is not silently retried. The reporter refuses unfinished or failed measurements.

Timing definitions:

- Engine startup ends at native `READY`. The first request may still fill/compile the Metal window and arena; that
  deferred work is part of its cold time to first token (TTFT), not steady decode.
- TTFT runs from the actual native `GEN` write to the first yielded token. Decode rate uses the first-to-last arrival
  interval and `N-1` observed token intervals, not the requested output limit. Terminal tokens count; heartbeats do not.
  Speculative tokens can arrive in bursts. These figures are not directly comparable to the initial CLI's aggregate rate.
- A request normally hands off one layer's expert activations, not an entire generated token. Several dozen such calls
  can lie on a token's path. Request lifetime is send time + local-work overlap + exposed result wait, not bare network RTT.
- CUDA event intervals report the H2D stage, quantization/full expert computation and D2H. H2D includes pageable-upload
  enqueue gaps. Host preparation overlaps that stage; do not add it to GPU durations or the inclusive worker service time.
  All durations use local monotonic clocks; no comparison of unsynchronized Mac and PC wall-clock timestamps is needed.
- Empty and size-matched no-compute probes use the same HELLO-gated SSH/TCP connection as inference, before and after the
  engine runs. They do not compute experts or alter its cache. The worker permits one client at a time.
- Ethernet scenarios hold measured routing, cache behavior, CPU/CUDA work, local overlap and untraced work fixed. Only
  exposed wait is reduced. The probe-calibrated estimate preserves unexplained waits; the optimistic estimate attributes
  removable waits to the current link. Assumed application RTT includes SSH/dispatch overhead, with 90% payload efficiency.
  Neither is a wired measurement or a statistical confidence interval.

The physical route is Mac Wi-Fi (`en0`) to the PC's `enp6s0`, currently negotiated at 1 Gbit/s. The PC's Intel I225-V
controller supports up to 2.5 GbE ([Intel specifications](https://www.intel.com/content/www/us/en/products/details/ethernet/gigabit-controllers/i225-controllers/products.html)).
A direct 2.5 GbE scenario requires a compatible Mac adapter/cable and successful link negotiation; faster scenarios
require suitable interfaces at both ends. No measured wired speed or long-context/vision quality is implied.


### Measured performance of the saved setup

The local comparison used the M5 Pro, 48 GB unified memory, RTX 2070 Super 8 GiB and PC system RAM described above.
Every case ran the two saved prompts twice, with a 96-token output limit, 4K context and runtime projection enabled.
Each case returned 370 tokens. These were sequential snapshots, not a randomized test with Wi-Fi conditions held fixed.
Aggregate rates use the total observed token intervals divided by their total duration.

| Case | Aggregate decode tokens/s |
|---|---:|
| Mac only, 32 GB expert arena | 19.19 |
| Wi-Fi + Tailscale SSH, 256 MiB CUDA cache | 2.24 |
| Wi-Fi + Tailscale SSH, 5 GiB CUDA cache | 2.21 |
| Wi-Fi + explicit LAN SSH, 5 GiB CUDA cache | 2.48 |
| Wi-Fi + Tailscale SSH, 5 GiB cache, profiling off | 2.26 |

The four profiling-off runs returned exactly the same token IDs as the matching profiled runs. Their rate difference
was small relative to the observed Wi-Fi run-to-run variation; it is not a controlled estimate of profiler overhead.
The expanded projected model's returned `is_prime` function also passed 12 executed arithmetic/boundary cases.

The 5 GiB cache achieved 88.5–94.4% hits across prompts, versus 0–3.9% with the small cache. On the explicit LAN-over-Wi-Fi
route, each generated-token interval involved about 45.7 layer calls. Median empty application RTT was 3.99 ms, while
actual expert-request median lifetimes were 6.38–6.48 ms. Median CUDA-worker service was 0.17–0.18 ms; the Mac still spent
about 265–276 ms per token interval waiting for remote results. The cache improved worker service time, not overall
Wi-Fi throughput clearly. A generated token is not a single network exchange.

Using that explicit-LAN trace, 90% payload efficiency and assumed application base RTTs of 0.2–0.8 ms:

| Future link | Probe-calibrated decode tokens/s | Optimistic decode tokens/s |
|---|---:|---:|
| 1 GbE | 4.26–4.70 | 6.14–7.11 |
| 2.5 GbE | 4.41–4.81 | 6.47–7.36 |

These ranges vary the assumed RTT; they are not confidence intervals or measured wired speeds. The optimistic column
removes all wait attributed to the old link beyond replacement service. The calibrated column preserves unexplained
waits. Both retain observed GPU work, local work and host overhead. This design extends usable memory; these results
do not establish a speed advantage over the Mac alone. Reducing per-layer handoffs matters more than merely enlarging
the CUDA cache.

The detailed comparison is saved at
`~/Developer/Strata-network-data/comparison-20261030-231810/report.md`, with complete JSON analysis, raw token timestamps,
source-result paths and individual request traces. Original invocations retain their exact profile snapshots; the
profiling-off control is at `~/Developer/Strata-network-data/perf-notrace-20261030-231810/`.
The normal `flash-iq3s-64k-strata` router alias answered `96` to the arithmetic smoke without any worker running, then
returned to its original unloaded state. The temporary Mac export and superseded PC 96-expert file were removed;
the verified PC 128-expert file remains for repeat runs.


## Install

For everyday use follow [the app or Terminal installation](INSTALL.md). The commands below are the source installation, which
builds the Metal engine locally.

```sh
git clone https://github.com/Niko1221/Strata && cd Strata
./setup.sh                 # asks the size; --yes takes the recommended answers
```

What the first run does (each step is skipped when it is already done):

1. **Python.** It needs Python 3.10 or newer, but macOS ships 3.9. If that is all your Mac has, `setup.sh` downloads a standalone
   Python 3.12 (25 MB, pinned by SHA-256) into `.python`. No sudo, no Homebrew.
2. **Checks your Mac:** the chip, the memory, the macOS version, the Command Line Tools and the free disk.
3. **Python packages** into `.venv`. CMake and Ninja come from pip too, installed just before the build.
4. **Builds the engine** (`build-metal/strata`) with Apple's metal-cpp headers (pinned by SHA-256) and llama.cpp's ggml at the
   commit the PC build pins.
5. **Downloads the model.** The download resumes if interrupted. `--gguf-dir` uses files you already have.
6. **Prepares the model:** the pack (seconds) and the MTP draft layer (about 5 GB of the original checkpoint). On a Mac that
   streams, the experts' copy takes the draft layer's place (about 3 minutes on the M4).
7. **Writes `strata-<model>.json` and `run-<model>.sh`, then starts the server.**

After that, `./setup.sh` or `./run-q2_0.sh` starts the model directly. The server is the same as on a PC: the OpenAI-compatible
API on `http://127.0.0.1:8080/v1`, the Anthropic API on `/v1/messages`, and the chat page on `http://127.0.0.1:8080/`.

| Option | Meaning |
| --- | --- |
| `--model Q2_0\|IQ3_XXS\|IQ3_S\|UD-Q4_K_XL\|UD-Q5_K_XL\|UD-Q6_K_XL\|Gyro-S\|Gyro-M` | The pack. The `UD-` files are Unsloth's. |
| `--family qwen\|unsloth\|gyro` | The model family. |
| `--vision yes\|no\|gpu\|cpu` | Picture reader. |
| `--context N`, `--port`, `--host`, `--api-key` | Server settings. |
| `--draft-vocab`, `--no-browser` | Draft vocabulary; do not open the browser. |
| `--data-dir`, `--models-dir`, `--gguf-dir` | Where files go, or where yours are. |
| `--setup`, `--no-start`, `--check`, `--build` | Choose another model or settings; install only; only check the Mac; build the engine again. |

The Qwen Metal engine also accepts the engine's opt-in control-vector flags
`--control-vector-scaled FILE:SCALE`, `--control-vector-layer-range A B`, `--cvec-mode add|project` and
`--cvec-dir per-layer|single:L`. No vector is loaded by default. Projection applies after each selected layer's
feed-forward write, for every residual stream during prompt processing and decoding. A loaded vector can be switched
per request with the engine's `cvec=0|1` setting. These flags do not change the model files. This support is for the
Qwen engine, not DeepSeek.

### The desktop app and `strata launch` on a Mac

The app and `strata launch` build nothing. The Metal engine comes ready-made as an engine component, from the update feed or
inside the app's installer. The model install is `setup.py` in its Control mode (`STRATA_CONTROL_INSTALL=1`). It downloads, or
uses with `--gguf-dir`, and prepares the model files, and writes a run config whose engine is `component:strata-engine`. It does
not run `metal/setup_mac.py`, needs no Command Line Tools, and writes the same Metal flags.

The model list and the recommendation judge a Mac by its unified memory with the rules of `strata_control/mac_fit.py`, the module
`setup_mac.py` uses too. Q2_0, IQ3_XXS, IQ3_S and the Gyro packs are offered. Unsloth's many-shard files need the source
installer's repack for now. A pack that does not stay in memory is listed as "read from the SSD", and a Mac under 16 GB of unified
memory is told the model does not fit.

The ready-made Mac engine is in the app and in the Terminal's engine component, not in the release zips. The zips (for
`START-HERE.bat` and `setup.sh`) have no Mac engine: `./setup.sh` builds it.

## Defaults and options

The config uses the Mac defaults: `--spec 2` with MTP (best on mixed text; an explicit `--spec N` in the config's `args` is kept)
and the Q8_0 hyper-connection read. `STRATA_HC_Q8=0` in the config's `env` gives the exact BF16 read. The Mac engine is
deterministic: two runs give identical numbers.

### Gyro-S and Gyro-M (agentionai's trellis experts)

The Mac engine runs [Gyro](GYRO.md). The routed experts are trellis-coded blocks (ggml types TQ2_T, TQK6, TQK7) in a
Hadamard-rotated basis, read from the published file unchanged. The kernels are in `metal/kernels/quant.metal`, `moe.metal` and
`moe_mm.metal`. Setup offers Gyro-S and Gyro-M next to the other sizes (`./setup.sh --family gyro --model Gyro-S`), never as the
default, and says what the choice risks on your Mac's memory. The 28.8 GB n-gram table is inside the file and is read from the SSD
like every other pack's, so Gyro-S needs 58.5 GB of disk and about 38 GB of memory to stay resident.

Gyro-M runs on the same kernels (checked tensor by tensor) but no Mac tested here could hold it. `metal/tests/gyro_parity.sh`
checks the decoder against the format definition and the expert kernels against a double-precision reference.
`STRATA_METAL_GYRO_TILES=0` turns off the dense-weight tiles.

## Measured (M5 Max, 128 GB)

**Measured on:** Apple M5 Max, 128 GB, macOS 27.2, Qwen3.8-Flash-Next packs, greedy, 200 generated tokens, Strata engine version
not recorded. Decode tok/s for a short question and a story ("prompt1 / story").

| | Q2_0 | IQ3_XXS | IQ3_S |
| --- | --- | --- | --- |
| Setup's defaults (`--spec 2` with MTP, suffix drafter on) | **119.5 / 108.2** | **108.9 / 97.0** | - |
| `--spec 2` with MTP, suffix drafter off | 114.0 / 107.9 | 103.5 / 96.9 | 98.2 / 94.1 |
| No MTP (one token per step) | 87.4 / 87.3 | 78.7 / 78.6 | 74.5 / 74.3 |

The output with MTP is the same as without it, token for token. MTP only proposes tokens, and the full model checks each one.

Prompt reading (time to first token), on the needle-test prompts:

| | 4K tokens | 32K tokens |
| --- | --- | --- |
| Q2_0 | about 1,430 tok/s | about 1,460 tok/s |
| IQ3_XXS | about 1,260 tok/s | about 1,240 tok/s |
| IQ3_S | about 1,320 tok/s | about 1,140-1,190 tok/s |

The first prompt after the Mac starts is slower while macOS reads the model file (IQ3_S: 413 tok/s on the first 4K prompt).
llama.cpp on the same Mac with Q2_0: 56.6 tok/s decode on the story, 887 / 809 tok/s on 4K / 32K prompts.

**Unsloth packs on the same Mac.** IQ3_S and UD-Q4_K_XL are the recommended sizes. Decode is tok/s over 200 tokens (greedy, MTP);
prompts are the warm rate from the second request on.

| | IQ3_S | UD-Q4_K_XL | UD-Q5_K_XL (streamed) | UD-Q6_K_XL (streamed) |
| --- | --- | --- | --- | --- |
| Quality against Q8_0 (top-1 / KL) | 87.9% / 0.142 | 92.8% / 0.055 | not measured | not measured |
| Decode, short question, no MTP / MTP | 75.4 / 101.9 | 60.7 / 94.0 | - | - |
| Decode, story, no MTP / MTP | 74.8 / 94.7 | 59.7 / 73.7 | 28 / 46-51 (`--spec 3`) | not measured / 47-51 (`--spec 3`) |
| Prompt 4K / 32K | about 1,180 / 1,290 | about 1,150 / 1,290 | about 1,070 / 1,110 | about 1,020 / 1,020 |
| llama.cpp Metal, decode (story / code) | 51.1 / 44.1 | 48.4 / 47.3 | 45.1 / 43.2 | does not run (Insufficient Memory) |

UD-Q5 and UD-Q6 cannot stay resident: their repacked file (103.9 and 114.8 GB) is bigger than the largest buffer Metal gives
(86.6 GB). The experts stream from the SSD into an arena of about 65 to 67 GB. The cost is a fixed hand-shake per layer, not the
SSD, so a longer window pays (Q5 story: 28 tok/s without MTP, 41.8 with `--spec 2`, 50.6 with `--spec 3`).

**Measured on:** Apple M4 Mac mini, 32 GB, experts streamed from the SSD (arena 11.5 to 14.7 GB), one token per step (no MTP),
medians of 5 interleaved runs, tok/s.

| | Q2_0 | IQ3_XXS | llama.cpp, CPU only (Q2_0 / IQ3_XXS) |
| --- | --- | --- | --- |
| Decode, short question (200 tokens) | **14.1** | **10.0** | 6.3 / 4.1 |
| Decode, story (200 tokens) | **13.5** | **9.2** | 6.3 / 4.1 |
| Decode after a 6K-token document | **13.1** | **7.7** | 5.1 / 2.8 |
| Prompt, 4K tokens | **93.6** | **65.8** | 21.2 / 8.7 |

The greedy answers equal the M5 Max's (model in memory) on 10 of 10 test prompts. MTP was slower here, because the draft layer's
1.2 GB do more as expert cache. The arena's size matters most: with 5.7 to 9.3 GB (a Mac with less memory free) the same M4 read
about 8.3 tok/s on a story. On an M1 Max with 34 GB (Q2_0, expert cache at 64 slots, kernel fix #2348) decode was 7.9 tok/s and a 9,940-token prompt read at 186 tok/s.

**Measured on:** Apple M4 Mac mini, 32 GB, Gyro-S, experts streamed (arena 11 to 12 GB of the 25.8 GB of experts), no experts' copy,
one token per step, no draft layer. Decode story, 120 tokens, 5 interleaved pairs, medians: **9.14 tok/s** (8.70 with the
dense-weight tiles off). Prompt of 4,065 tokens: **109 tok/s**. The engine held 18.6 GB of the 32 GB. Against the CUDA engine on
the same Gyro-S file (5 text segments, 4,995 positions, RTX 3060) the Mac is as far from either CUDA run as the two CUDA runs are
from each other: top-1 agreement 95.4% against 95.8%, mean KL 0.0207 against 0.0186.

## Limits (this version)

- **Images (vision):** on by default. The app and the Terminal use the `strata-vision` that sits beside `strata` in the engine
  package; `./setup.sh` builds it and asks "Should the model also read images?" (default yes). With `--gpu` it reads on the Mac's
  GPU (ggml's Metal backend), without it on the CPU. A Mac always gets the GPU reader, whatever its memory. The footprint is about
  1.18 GB on the GPU and 1.05 GB on the CPU. `--vision cpu` is still allowed and prints a note. The CPU reader is slow because
  ggml has no vector kernel for BF16 on ARM. If the GPU reader cannot start, the server prints one line and starts it on the CPU.
  Checked on one Mac mini M4 only; the M5 Max is untested for pictures.

  **Measured on:** Mac mini M4, 32 GB, Qwen3.8-Flash-Next BF16 projector.

  | Picture | GPU reader | CPU reader |
  | --- | --- | --- |
  | 260 tokens (640x400) | 0.45 s | 50.7 s |
  | 972 tokens (1600x1200) | 2.4 s | 194 s |

- **Context:** measured up to 32K (needles, KL). Setup recommends 32K below 96 GB and 64K from 96 GB. Up to the model's 262K can
  be set, but it is not measured.
- **One request at a time.** `--parallel` (batch slots) is not ported. Several clients queue.
- **Not ported:** the PC tuning (`--calibrate`).
- **Streaming** is measured at 32 GB only.
- **Families:** the original Qwen3.8-Flash-Next, Unsloth's UD-Q4/Q5/Q6_K_XL and agentionai's Gyro-S are measured. Setup repacks
  Unsloth's shards into one file for the GPU; a Gyro file is used as it is. The Swift and Coder releases are not set up on Mac
  yet, and Qwen3.6 does not run on the Mac engine.

### Quality

- **The Q8_0 hyper-connection read** is the default. Teacher-forced over 28 prompts (code, reasoning, chat, six languages,
  documents), about 6,000 positions per set, BF16 reference: mean KL 0.0007 and top-1 agreement 98.9% on the model's own text,
  99.1% on another model's text, perplexity unchanged. For scale, IQ3_XXS against Q2_0 is a KL of 0.094.
- **The fast prompt path** against reading the prompt through decode windows: mean KL 0.0004, top-1 99.4 to 99.6%.
- **Needles:** a code word hidden in 8K, 16K and 32K tokens at 10/50/90% depth was found 9 of 9 with Q2_0 and with IQ3_XXS.
- **Same answers as on a PC:** against the CUDA engine on a Tesla P100 (IQ3_XXS and IQ3_S) the first 200 tokens of a reply on the
  short question are identical; against the HIP engine on a Radeon AI PRO R9700 (Q2_0), 36 of 39 first tokens are identical. The
  differences are near-ties between the two best tokens, from a different floating-point order.

## Switches

For testing. All are off unless set; the defaults are described above.

| Variable | Effect |
| --- | --- |
| `STRATA_HC_Q8=0` | The BF16 hyper-connection read (exact) instead of Q8_0 |
| `STRATA_METAL_PROMPT=window` | Read the prompt through decode windows (slow, exact fallback; about 190 tok/s) |
| `STRATA_METAL_PROMPT_CHUNK=N` | Prompt chunk size (default 2048 resident; see streaming above) |
| `STRATA_METAL_MTP_Q8=0` | The draft layer's matrices in BF16 instead of Q8_0 |
| `STRATA_METAL_MTP_CHAIN=0` | One GPU submission per draft step instead of one per round |
| `STRATA_METAL_FUSE=0`, `STRATA_METAL_LAZY_FOLD=0` | Turn off two decode fusions (same tokens, slower) |
| `STRATA_METAL_PARTIAL=1`, `STRATA_METAL_ROUTE=1` | Two decode variants that measured no faster. `ROUTE` is also wrong on an M1 Max (stale routing logits) |
| `STRATA_METAL_PROMPT_EXPERTS=v3` | The M5 layout for the prompt's expert GEMMs. Fails on an M1 Max, so it is off there (default v2) |
| `STRATA_METAL_STREAM=1` / `=0` | Stream the experts from the SSD even when the pack fits / never (the default decides by size) |
| `STRATA_METAL_CONTIG=0` | Streaming: read the model file even when the experts' copy is there |
| `STRATA_METAL_ARENA_GB=N`, `STRATA_METAL_KEEP_FREE_GB=N` | Streaming: force the arena's size (bypasses shrink/growth safeguards), or set the memory left to macOS |
| `STRATA_METAL_ARENA_MAX_GB=N` | Streaming: cap automatic sizing in decimal GB while retaining memory safeguards; minimum 1 GB, cannot combine with `STRATA_METAL_ARENA_GB`; an oversized minimum slot layout fails before allocation |
| `STRATA_METAL_LEAVE_SHARE=0.4` | Streaming, 64 GB or more: the share of the RAM left to macOS when the footprint cap is set |
| `STRATA_METAL_WIRE_ARENA=0` / `=1` | Streaming: leave the arena pageable / wire it as it is filled (default: wired on 64 GB or more) |
| `STRATA_METAL_IGNORE_REPACK=1` | Start although a model repack or setup's preparation is running (the engine refuses by default) |
| `STRATA_METAL_KEEP_HOST_WEIGHTS=1` | Streaming: keep the host code's own device copies of the weights (6.2 GB for IQ3_S: the pack pool, the native head and projections); by default they are released before the arena is sized |
| `STRATA_METAL_DIRECT=1` | Streaming from the experts' copy: read each expert as whole pages into a per-thread buffer, so nothing lands in macOS's file cache (unaligned `F_NOCACHE` reads kept about 35% of the bytes cached, which pushed the arena itself into the compressor) |
| `STRATA_METAL_KEEP_WARM=1` | Streaming: while a decode window waits for the SSD, a second queue runs tiny GPU ticks so the GPU stays clocked up between segments (`STRATA_METAL_WARM_TG`, `STRATA_METAL_WARM_ITERS` size a tick) |
| `STRATA_METAL_REGION_FIT=1` | Streaming: lay the biggest slot class out last, so a prompt's buffers fit in a region of the arena instead of their own allocation |
| `STRATA_METAL_PROMPT_SMALL=N` | Streaming: prompts of at most N tokens get a small buffer set of their own and leave the arena's region (and its experts) alone |
| `STRATA_METAL_IO_THREADS=N`, `STRATA_METAL_READ_SPLIT=N`, `STRATA_METAL_PREFETCH=K`, `STRATA_METAL_PREFETCH2=K` | Streaming: I/O threads (default 6), requests per expert (1), read-ahead top K of the next layer (6 for a big model such as Flash-Next, else 10) and of the one after (0) |
| `STRATA_METAL_ROUTE_TRACE=path`, `STRATA_METAL_SEG_TIMING=1`, `STRATA_METAL_KPROF_STREAM=1` | Streaming diagnostics: binary routing trace for cache simulations, per-segment GPU/host/wait times, per-kernel times of a segmented window |
| `STRATA_METAL_TRACE=1`, `STRATA_METAL_PROFILE=1`, `STRATA_METAL_KPROF=T` | Per-window traces, a time breakdown, per-kernel GPU times |

## How it works

The code is in `metal/` (the engine's side in `metal/src/engine`, the kernels in `metal/kernels`).

- **Host code:** the PC host code compiles unchanged for macOS. `metal/include/metal_compat` maps the CUDA runtime calls it makes
  onto a small Metal runtime (`metal/src/gpu`).
- **Decode and prompt work** run in Metal kernels: one command buffer per verify window, experts read in place from the mapped
  model file, hyper-connections, GDN state and QSA attention on the GPU, and a whole MTP draft round in one command buffer.
- **The PC builds are byte-identical** with this code present. Every shared file's change is behind `STRATA_GPU_METAL` or `__APPLE__`.

## History / older measurements

**Why `--spec 2`.** Measured on: Apple M5 Max, 128 GB, decode tok/s with MTP on the two prompts (prompt1 / story).

| Window | Q2_0 | IQ3_XXS |
| --- | --- | --- |
| `--spec 2` | 111 / 106 | 102 / 95 |
| `--spec 3` | 116 / 92 | 108 / 95 |
| `--spec 4` | 111 / 81 | 108 / 83 |
| `--spec 5-6` | slower | slower |

**The experts' copy.** Measured on: Apple M4 Mac mini, 32 GB, IQ3_XXS after a 6K-token document, arena held at 12 GB, medians of 5
interleaved runs, identical tokens in all 10 runs.

| | Decode | The 6K prompt | SSD wait per token |
| --- | --- | --- | --- |
| The model file (three reads per expert) | 7.37 tok/s | 64.7 tok/s | 46.6 ms |
| The experts' copy (one read per expert) | **8.62 tok/s** | **84.1 tok/s** | 31.5 ms |

The 4K prompt read 72.9 tok/s with the copy against 62.5 without. A 34-minute server run with the copy (47 requests, context
34,816) got every answer right and did not slow down. With the model file, an arena of 7.5 GB gave 5.5 tok/s, 10.5 GB gave 6.7 and
12 GB gave 7.5. On the 128 GB M5 Max the copy for Q5 did not pay (decode 50.3 against 50.2 tok/s). The SSD alone measures 2.97 GB/s
for one 1.75 MB read per expert against 2.30 GB/s for three 0.58 MB reads. The Q2_0 copy was not measured.

**Prompt chunk on a streaming Mac.** Measured on: M5 Max, UD-Q5, warm, 2 interleaved rounds. Chunks of 1024, 2048 and 4096 gave 700,
920 and 1,100 tok/s on the 4K prompt (32K: 710, 970, 1,100). Decode did not change. On a resident model 4096 instead of 2048 gained
only 2 to 4% (Q2_0), so that default stays.

**Memory footprint cap.** Measured on: M5 Max, 128 GB. Before the cap, a streaming arena of 100 GB compressed 19 GB of itself in 7
seconds, and the kernel watchdog panicked three times. With 40% of the RAM left to macOS and the arena wired, a 40 GB file write
beside a 16 GB program caused no swap-out. A resident pack (UD-Q4_K_XL, wired 91.7 GB) took the same load with nothing compressed,
so resident packs are limited to 70% of the RAM.

**Gyro-S kernel profile.** Measured on: M4 Mac mini, 32 GB. A decode window of two tokens takes 111 ms: the routed experts about
40 ms, the dense weights (Q5_K and Q6_K, 3.6 GB read a token) about 66 ms before the tile change. The kernel for one expert layer
(ten experts, one token) takes 278 us at 37.5 GB/s; IQ3_XXS takes 333 us and Q2_0 166 us. The trellis decode is limited by latency
and ALU, not by memory. The prompt kernels run 68 ms a layer against 62 ms for IQ3_XXS at 1,024 tokens.

**Quality against the CUDA engine, Gyro-S file.** `tools/quant_quality.py`, teacher-forced over 4,995 positions, CUDA = RTX 3060.

| Reference -> run | top-1 agreement | mean KL | KL p99 | perplexity |
| --- | --- | --- | --- | --- |
| CUDA GPU experts -> CUDA CPU experts (noise floor) | 95.80% | 0.0186 | 0.195 | 4.462 / 4.477 |
| CUDA GPU experts -> Mac | 95.46% | 0.0207 | 0.241 | 4.486 / 4.477 |
| CUDA CPU experts -> Mac | 95.38% | 0.0187 | 0.272 | 4.486 / 4.462 |
