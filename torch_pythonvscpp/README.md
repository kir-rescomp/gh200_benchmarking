# PyTorch (Python) vs LibTorch (C++) benchmark — parity gate & timing harness

A rigorous comparison of PyTorch Python and LibTorch C++ on ResNet/CIFAR-10,
built for Isambard-AI GH200 nodes (4 × Grace-Hopper superchip per node, GPUs
coupled by NVLink). This repository contains the two foundational components:
the **parity gate** (proves both implementations do identical work) and the
**timing harness** (measures them identically). Nothing else is valid without
these two.

## Why this exists

A widely-forked reference benchmark reported a "3–4× C++ speedup" that was an
artefact: its C++ pipeline omitted data augmentation and the LR scheduler, so
it trained faster by doing less work, and its test accuracy diverged (~71% C++
vs ~88% Python) — proof the two pipelines were not equivalent. It also used
wall-clock timing around asynchronous CUDA calls, a single run, single-image
latency, and no warmup discipline. Every one of those is a correctness defect,
not a tuning detail.

This design fixes all of them. The central principle: **the GPU work must be
provably identical; only the host-side driver (Python interpreter vs compiled
C++) varies.** The experiment then measures where host-side overhead is, and
is not, hidden behind kernel time.

## The parity gate (hard precondition)

Both languages consume the *same bytes*: one exported set of initial weights,
one fixed input batch, and one reference forward/backward result. The gate
runs one forward + backward from the shared weights on the fixed batch and
asserts that loss, logits, and every gradient match to tolerance.

Exchange format is a TorchScript bundle (`torch.jit.save` of a module whose
buffers are the tensors), which `torch::jit::load` reads cleanly on the C++
side — robust across LibTorch versions, unlike a pickled `state_dict`.

- `config/experiment.json` — single source of truth for both languages.
- `python/model.py`, `cpp/include/resnet.h` — the model, defined once per
  language and required to mirror each other exactly (including submodule
  registration names, so `state_dict` keys align).
- `parity/export_parity_artifacts.py` — run once; writes the shared bundles.
- `parity/parity_check_python.py` — Python self-check (anchors determinism).
- `cpp/src/parity_check.cpp` — C++ check against the Python reference.

If the gate fails on a name/shape mismatch, `model.py` and `resnet.h` have
diverged. If it fails on values, a numerical path differs. **No timing may be
reported until the gate passes.** `scripts/run_gated_benchmark.sh` enforces
this ordering.

The parity gate runs in FP32 with deterministic cuDNN so the comparison is
bitwise-close. The timing arms run in the precision under test (FP32 / TF32 /
BF16-AMP).

## The timing harness

Structurally identical on both sides (`python/harness.py`,
`cpp/src/harness.cpp`), sharing these invariants:

- **CUDA events** for all timing, never wall-clock around async calls.
- **Warmup excluded**: first epoch and first N steps dropped; cuDNN autotune,
  allocator warmup, and clock ramp never enter a statistic.
- **`cudnn.benchmark = true`** on both sides so re-autotuning is not measured.
- **Repeats + statistics**: ≥5 seeded repeats; median + IQR for throughput
  (never the mean — throughput is right-skewed), full p50/p90/p95/p99/p99.9
  for latency (the tail is the entire low-latency argument).

Single-GPU arms provided here:

- `train_throughput` — steady-state images/sec and per-step time.
- `inference_latency` — batch-1 latency distribution.
- `inference_throughput` — large-batch images/sec.

Distributed training (DDP vs C10D), the precision sweep beyond the config
default, the input-pipeline stress arm, and the real-dataset accuracy-parity
run are deliberately separate so these single-GPU numbers stay uncontaminated.

## Build & run

Python side needs the target torch build active. C++ side:

```bash
cd cpp
cmake -B build \
  -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Critical:** build against the *same* libtorch that ships with the Python
wheel you benchmark, inside the same container/module environment that
provides the `sm_90` (Hopper) build on Isambard-AI. Linking a different
libtorch means comparing builds, not languages.

Then, from the repo root:

```bash
scripts/run_gated_benchmark.sh results 5
```

This exports the parity artefacts, runs the gate (aborting if it fails), then
runs 5 repeats of each arm for both languages and aggregates to
`results/summary.json`.

## Reading the results

`scripts/aggregate.py` prints Python vs C++ side by side with a signed
`cpp_vs_python%` (positive = C++ better, sign-aware for latency). A difference
is only meaningful if it exceeds the per-run spread shown for each language —
if the spread overlaps the difference, the two are indistinguishable at this
sample size.

## Expected findings (so results don't surprise you)

Training throughput near-parity on GH200 (single-digit-% C++ edge at most,
shrinking with batch and model size); measurable C++ advantage in batch-1
tail latency — likely erased by CUDA graphs, which is itself a finding;
multi-GPU often favouring Python DDP on developer-achievable performance. The
useful conclusion for facility users is generally "language barely affects
throughput; it affects tail latency and developer effort," not a headline
speedup ratio.

## Validated so far

The Python model, exporter, self-parity check, all three harness arms, the
config reader used by the C++ side, and the aggregator have been run and pass.
The C++ parity and harness sources are written to mirror the Python logic and
compile against libtorch; they must be built and the gate run on a
GH200 node before any timing is trusted.

## Next components (not in this deliverable)

- Real-dataset accuracy-parity run (the second half of the gate: both reach
  statistically indistinguishable CIFAR-10 test accuracy across seeds).
- Slurm submission template for the GH200 partition (exclusive allocation,
  NUMA/affinity pinning to the GPU-local Grace, clock/power recording).
- Distributed arm (DDP vs C10D) and the NVLink-then-Slingshot scaling study.
- CUDA-graph inference variant.
