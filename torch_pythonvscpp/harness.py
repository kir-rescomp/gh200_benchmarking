"""Timing harness, Python side.

Design invariants (mirrored exactly in cpp/src/harness.cpp):
  * Timing uses CUDA events, never wall-clock around async CUDA calls.
  * Warmup epochs and warmup steps are excluded from every statistic.
  * Every measured quantity is repeated and reported as median + IQR, never
    a single number and never the mean (throughput is right-skewed).
  * Inference latency is reported as a full percentile distribution
    (p50/p90/p95/p99/p99.9), because tail latency is the whole point of the
    low-latency argument.
  * The GPU is synchronised only at explicit measurement boundaries.

This file provides the measurement primitives and a CLI that runs the three
single-GPU arms: train throughput, inference latency, inference throughput.
It deliberately does NOT implement distributed training -- that is arm D and
lives separately so the single-GPU numbers stay uncontaminated.
"""
import argparse
import json
import statistics
import sys
import time

import torch

from model_path_shim_local import load_model_builder

build_model = load_model_builder()


# --------------------------------------------------------------------------
# measurement primitives
# --------------------------------------------------------------------------
class CudaTimer:
    """Elapsed GPU time in milliseconds between start() and stop(), using
    CUDA events. On CPU falls back to perf_counter (for local logic tests)."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        if self.cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.cuda:
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.ms = self._start.elapsed_time(self._end)
        else:
            self.ms = (time.perf_counter() - self._t0) * 1e3


def percentiles(values, ps):
    s = sorted(values)
    out = {}
    n = len(s)
    for p in ps:
        if n == 1:
            out[p] = s[0]
            continue
        rank = p / 100.0 * (n - 1)
        lo = int(rank)
        frac = rank - lo
        hi = min(lo + 1, n - 1)
        out[p] = s[lo] * (1 - frac) + s[hi] * frac
    return out


def summarize(values):
    return {
        "n": len(values),
        "median": statistics.median(values),
        "iqr": (percentiles(values, [75])[75] - percentiles(values, [25])[25]),
        "min": min(values),
        "max": max(values),
    }


# --------------------------------------------------------------------------
# precision / backend setup
# --------------------------------------------------------------------------
def apply_precision(cfg):
    prec = cfg["precision"]
    torch.backends.cudnn.benchmark = cfg["cudnn_benchmark"]
    if prec == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return None
    if prec == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return None
    if prec == "bf16_amp":
        return torch.bfloat16
    raise ValueError(f"unknown precision {prec}")


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def synthetic_loader(cfg, device, n_batches):
    """Synthetic data so the harness measures compute, not disk. The real
    dataset run uses the same loop with a torchvision loader; the synthetic
    path isolates the language overhead from the input pipeline (that is a
    separate arm, arm C)."""
    bs = cfg["train"]["batch_size"]
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    batches = []
    for _ in range(n_batches):
        x = torch.randn(bs, 3, 32, 32, generator=g)
        y = torch.randint(0, cfg["num_classes"], (bs,), generator=g)
        batches.append((x.to(device), y.to(device)))
    return batches


def arm_train_throughput(cfg, device):
    amp_dtype = apply_precision(cfg)
    h = cfg["harness"]
    model = build_model(cfg["model"], cfg["num_classes"],
                        cfg["cifar_stem"]).to(device).train()
    opt = torch.optim.SGD(model.parameters(),
                          lr=cfg["train"]["lr"],
                          momentum=cfg["train"]["momentum"],
                          weight_decay=cfg["train"]["weight_decay"])

    measured = h.get("measured_steps", 200)
    steps = h["warmup_steps_per_epoch"] + measured
    data = synthetic_loader(cfg, device, steps)
    bs = cfg["train"]["batch_size"]

    step_ms = []
    for i, (x, y) in enumerate(data):
        with CudaTimer(device) as t:
            opt.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = torch.nn.functional.cross_entropy(model(x), y)
            else:
                loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        if i >= h["warmup_steps_per_epoch"]:
            step_ms.append(t.ms)

    s = summarize(step_ms)
    s["images_per_sec_median"] = bs / (s["median"] / 1e3)
    return {"arm": "train_throughput", "batch_size": bs,
            "precision": cfg["precision"], "step_ms": s}


def arm_inference_latency(cfg, device):
    amp_dtype = apply_precision(cfg)
    h = cfg["harness"]
    model = build_model(cfg["model"], cfg["num_classes"],
                        cfg["cifar_stem"]).to(device).eval()
    x = torch.randn(h["latency_batch_size"], 3, 32, 32, device=device)

    lat_ms = []
    with torch.no_grad():
        for i in range(h["inference_warmup_iters"] + h["inference_latency_iters"]):
            with CudaTimer(device) as t:
                if amp_dtype is not None:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        _ = model(x)
                else:
                    _ = model(x)
            if i >= h["inference_warmup_iters"]:
                lat_ms.append(t.ms)

    pct = percentiles(lat_ms, [50, 90, 95, 99, 99.9])
    return {"arm": "inference_latency", "batch_size": h["latency_batch_size"],
            "precision": cfg["precision"], "iters": len(lat_ms),
            "latency_ms_percentiles": pct,
            "latency_ms_median": statistics.median(lat_ms)}


def arm_inference_throughput(cfg, device):
    amp_dtype = apply_precision(cfg)
    h = cfg["harness"]
    bs = h["throughput_batch_size"]
    model = build_model(cfg["model"], cfg["num_classes"],
                        cfg["cifar_stem"]).to(device).eval()
    x = torch.randn(bs, 3, 32, 32, device=device)

    measured = h.get("measured_steps", 200)
    warm = 50
    batch_ms = []
    with torch.no_grad():
        for i in range(warm + measured):
            with CudaTimer(device) as t:
                if amp_dtype is not None:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        _ = model(x)
                else:
                    _ = model(x)
            if i >= warm:
                batch_ms.append(t.ms)
    s = summarize(batch_ms)
    s["images_per_sec_median"] = bs / (s["median"] / 1e3)
    return {"arm": "inference_throughput", "batch_size": bs,
            "precision": cfg["precision"], "batch_ms": s}


ARMS = {
    "train_throughput": arm_train_throughput,
    "inference_latency": arm_inference_latency,
    "inference_throughput": arm_inference_throughput,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../config/experiment.json")
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeat-index", type=int, default=0,
                    help="which repeat this is; recorded in output for the "
                         "aggregator to compute cross-run medians")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    torch.manual_seed(cfg["seed"] + args.repeat_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = ARMS[args.arm](cfg, device)
    result["language"] = "python"
    result["torch_version"] = torch.__version__
    result["device"] = "cuda" if device.type == "cuda" else "cpu"
    result["repeat_index"] = args.repeat_index
    result["model"] = cfg["model"]

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
