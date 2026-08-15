#!/usr/bin/env bash
# Gate-enforced benchmark runner.
#
# The parity gate is a HARD precondition: no timing arm runs until both the
# Python self-check and the C++ check pass against the SAME exported
# artefacts. This makes it structurally impossible to publish a speedup from
# non-equivalent pipelines -- the failure mode that invalidated the reference
# repository.
#
# Usage:
#   scripts/run_gated_benchmark.sh <results_dir> <repeats>
#
# Assumes:
#   - Python env with the target torch build active
#   - C++ binaries built at cpp/build/parity_check and cpp/build/harness
#   - run from the repo root
set -euo pipefail

RESULTS_DIR="${1:-results}"
REPEATS="${2:-5}"
CONFIG="config/experiment.json"

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo " STEP 1: export shared parity artefacts"
echo "=============================================="
( cd parity && python3 export_parity_artifacts.py --config "../$CONFIG" )

echo "=============================================="
echo " STEP 2: PARITY GATE (must pass to proceed)"
echo "=============================================="
echo "--- Python self-parity ---"
( cd parity && python3 parity_check_python.py --config "../$CONFIG" )

echo "--- C++ parity vs Python reference ---"
if [[ ! -x cpp/build/parity_check ]]; then
  echo "ERROR: cpp/build/parity_check not found. Build the C++ side first:"
  echo "  cd cpp && cmake -B build -DCMAKE_PREFIX_PATH=\$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
  exit 3
fi
# parity binary expects the bundles at parity/ relative to CWD
( cd parity && ../cpp/build/parity_check )

echo "=============================================="
echo " PARITY GATE PASSED -- timing is now valid"
echo "=============================================="

ARMS=(train_throughput inference_latency inference_throughput)

echo "=============================================="
echo " STEP 3: timing arms, $REPEATS repeats each"
echo "=============================================="
for r in $(seq 0 $((REPEATS-1))); do
  for arm in "${ARMS[@]}"; do
    echo "--- python $arm repeat $r ---"
    ( cd python && python3 harness.py --config "../$CONFIG" --arm "$arm" \
        --repeat-index "$r" --out "../$RESULTS_DIR/py_${arm}_r${r}.json" ) >/dev/null

    echo "--- cpp $arm repeat $r ---"
    ( ./cpp/build/harness --config "$CONFIG" --arm "$arm" \
        --repeat-index "$r" --out "$RESULTS_DIR/cpp_${arm}_r${r}.json" ) >/dev/null
  done
done

echo "=============================================="
echo " STEP 4: aggregate"
echo "=============================================="
python3 scripts/aggregate.py --results-dir "$RESULTS_DIR" \
    --out "$RESULTS_DIR/summary.json"
