#!/bin/bash -e

module purge

export PATH="/apps/kir/eb/hpc-utils/aarch64:$PATH"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONNOUSERSITE=1
module use /apps/eb/el9/2025a/aarch64/modules/all/

# Set these env variables to a shared path
# Otherwise, uv will cache python binary to home directory .i.e. can not share from home
export UV_PYTHON_INSTALL_DIR=""
export UV_CACHE_DIR=""

uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu126
uv pip install cmake
uv pip install torch_scatter torch_sparse \
    --no-build-isolation \
    -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

# pyg-lib for aarch64 has to be compiled from source as there
# aren't any pre-compiled binaries
cd pyg-lib
git clone https://github.com/pyg-team/pyg-lib.git
git submodule update --init --recursive

TORCH_CUDA_ARCH_LIST="9.0" \
    MAX_JOBS=4 \
    uv pip install . --no-build-isolation -v

# few dependencies require fortran compiler. Provide it via GCC
module load GCC/14.2.0
uv pip install jax[cuda12] nichecompass[all]
uv pip install squidpy gdown
