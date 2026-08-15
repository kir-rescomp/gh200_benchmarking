"""Tiny shim so parity scripts can import build_model from ../python/model.py
without packaging fuss, whatever the working directory is."""
import importlib.util
import os


def load_model_builder():
    here = os.path.dirname(os.path.abspath(__file__))
    model_py = os.path.join(here, "..", "python", "model.py")
    spec = importlib.util.spec_from_file_location("bench_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model
