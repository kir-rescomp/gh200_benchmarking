"""Parity gate, Python side.

Two roles:
  1. Self-check: reload the exported bundles, re-run one fwd/bwd from the
     shared init weights on the fixed batch, and confirm Python reproduces
     its own reference. This anchors determinism and validates the compare
     logic that the C++ side mirrors.
  2. Reference loader: exposes load_bundle() used by other tooling.

Exit non-zero on any mismatch so it can gate a Slurm dependency chain.
"""
import argparse
import json
import sys

import torch

from model_path_shim import load_model_builder

build_model = load_model_builder()


def load_bundle(path):
    """Load a TorchScript tensor bundle back into a name->tensor dict,
    reversing the '__' -> '.' name mangling."""
    m = torch.jit.load(path, map_location="cpu")
    out = {}
    for name, buf in m.named_buffers():
        out[name.replace("__", ".")] = buf
    return out


def compare(a, b, atol, rtol, label, failures):
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    diff = (a - b).abs()
    max_abs = diff.max().item()
    max_rel = (diff / b.abs().clamp_min(1e-12)).max().item()
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    print(f"  {label:28s} max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../config/experiment.json")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p = cfg["parity"]
    atol_f, rtol_f = p["forward_atol"], p["forward_rtol"]
    atol_g, rtol_g = p["grad_atol"], p["grad_rtol"]

    init = load_bundle("../parity/init_bundle.pt")
    batch = load_bundle("../parity/fixed_batch_bundle.pt")
    ref = load_bundle("../parity/reference_bundle.pt")

    model = build_model(cfg["model"], cfg["num_classes"],
                        cfg["cifar_stem"]).to(device).float()
    # Load shared init weights by exact state_dict name.
    sd = model.state_dict()
    missing = [k for k in sd if k not in init]
    if missing:
        print(f"[parity-py] MISSING init weights: {missing[:5]} ...")
        sys.exit(2)
    model.load_state_dict({k: init[k].to(device) for k in sd}, strict=True)

    x = batch["input"].to(device)
    y = batch["label"].to(device).long()

    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    print(f"[parity-py] device={device}")
    print(f"[parity-py] recomputed loss = {loss.item():.8f}")
    print(f"[parity-py] reference  loss = {ref['loss'].item():.8f}")

    failures = []
    compare(loss.detach().cpu().reshape(1), ref["loss"], atol_f, rtol_f,
            "loss", failures)
    compare(logits.detach().cpu(), ref["logits"], atol_f, rtol_f,
            "logits", failures)
    for name, pr in model.named_parameters():
        if pr.grad is None:
            continue
        gk = f"grad.{name}"
        if gk in ref:
            compare(pr.grad.detach().cpu(), ref[gk], atol_g, rtol_g,
                    gk, failures)

    if failures:
        print(f"[parity-py] {len(failures)} FAILED: {failures[:5]}")
        sys.exit(1)
    print("[parity-py] ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
