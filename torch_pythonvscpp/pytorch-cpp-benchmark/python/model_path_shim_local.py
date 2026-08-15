"""Local shim so harness.py can import build_model from model.py in the same dir."""
import importlib.util
import os


def load_model_builder():
    here = os.path.dirname(os.path.abspath(__file__))
    model_py = os.path.join(here, "model.py")
    spec = importlib.util.spec_from_file_location("bench_model_local", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model
