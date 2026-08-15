"""Export shared parity artefacts as TorchScript bundles.

Why TorchScript bundles rather than torch.save pickles: a Python-pickled
state_dict is not reliably loadable from C++ across LibTorch versions. Wrapping
tensors as buffers on a scripted module and using torch.jit.save produces an
archive that torch::jit::load reads cleanly on the C++ side, keyed by the same
names. This is the robust cross-language exchange format.

Produces, in ../parity/:
  init_bundle.pt            weights keyed by state_dict name
  fixed_batch_bundle.pt     {"input", "label"}
  reference_bundle.pt       {"loss", "logits", "grad.<param name>", ...}
  weight_manifest.json      names + shapes, for human/debug inspection

Run once, commit the .pt bundles. Both languages consume identical bytes.
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from model_path_shim import load_model_builder

build_model = load_model_builder()


class TensorBundle(nn.Module):
    """A scripted container whose buffers are the tensors to exchange.

    Buffer names cannot contain '.', so we substitute '.' -> '__' on the way
    out and the C++ side reverses it. named_buffers() then yields the exact
    names both sides agree on.
    """

    def __init__(self, tensors: dict):
        super().__init__()
        self._keymap = {}
        for k, v in tensors.items():
            safe = k.replace(".", "__")
            self._keymap[safe] = k
            self.register_buffer(safe, v.detach().clone())

    def forward(self):  # required for scripting; unused
        return torch.tensor(0)


def save_bundle(tensors: dict, path: str):
    bundle = TensorBundle(tensors)
    scripted = torch.jit.script(bundle)
    torch.jit.save(scripted, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../config/experiment.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    seed = cfg["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.deterministic = cfg["cudnn_deterministic_for_parity"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = build_model(cfg["model"], cfg["num_classes"],
                        cfg["cifar_stem"]).to(device).float()

    os.makedirs("../parity", exist_ok=True)

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_bundle(sd, "../parity/init_bundle.pt")

    with open("../parity/weight_manifest.json", "w") as f:
        json.dump([{"name": k, "shape": list(v.shape)}
                   for k, v in sd.items()], f, indent=2)

    bs = cfg["train"]["batch_size"]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(bs, 3, 32, 32, generator=g)
    y = torch.randint(0, cfg["num_classes"], (bs,), generator=g)
    save_bundle({"input": x, "label": y.to(torch.int64)},
                "../parity/fixed_batch_bundle.pt")

    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(x.to(device))
    loss = torch.nn.functional.cross_entropy(logits, y.to(device))
    loss.backward()

    ref = {"loss": loss.detach().cpu().reshape(1),
           "logits": logits.detach().cpu()}
    for name, p in model.named_parameters():
        if p.grad is not None:
            ref[f"grad.{name}"] = p.grad.detach().cpu()
    save_bundle(ref, "../parity/reference_bundle.pt")

    print(f"[export] device={device} seed={seed} model={cfg['model']}")
    print(f"[export] reference loss = {loss.item():.8f}")
    print(f"[export] params={sum(1 for _ in model.parameters())} "
          f"grad tensors={sum(1 for _,p in model.named_parameters() if p.grad is not None)}")
    print("[export] wrote init_bundle / fixed_batch_bundle / reference_bundle")


if __name__ == "__main__":
    main()
