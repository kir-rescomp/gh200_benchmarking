"""
GH200 Test 1: FSDP forward+backward across 2x GH200 144GB GPUs.
Model  : Qwen2.5-72B
Run    : torchrun --nproc_per_node=2 fsdp_llm_train.py
Requires: torch>=2.4, transformers, flash-attn
"""

import os
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "/well/kir/mirror/LLM/huggingface/Qwen-Qwen2.5-72B"
SEQ_LEN      = 8192   # long context — exercises HBM3e bandwidth
BATCH_SIZE   = 8      # per GPU
N_STEPS      = 100
WARMUP_STEPS = 10
# ─────────────────────────────────────────────────────────────────────────────


def rank_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)


def mem_str(device):
    alloc = torch.cuda.memory_allocated(device) / 1e9
    peak  = torch.cuda.max_memory_allocated(device) / 1e9
    return f"alloc={alloc:.1f}GB | peak={peak:.1f}GB"


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    rank_print(rank, f"\n{'='*64}")
    rank_print(rank, f"GH200 Test 1 — FSDP Forward Pass (inference mode)")
    rank_print(rank, f"Model      : {MODEL_NAME}")
    rank_print(rank, f"GPUs       : {world_size}")
    rank_print(rank, f"Seq len    : {SEQ_LEN} | Batch/GPU: {BATCH_SIZE}")
    rank_print(rank, f"Steps      : {N_STEPS} ({WARMUP_STEPS} warmup discarded)")
    rank_print(rank, f"{'='*64}\n")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.use_cache = False

    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    rank_print(rank, "Instantiating model on meta device ...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    rank_print(rank, "Wrapping with FSDP ...")
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=rank,
        param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
        limit_all_gathers=True,
        use_orig_params=True,
    )

    # ── Gradient checkpointing ───────────────────────────────────────────────
    # Recomputes MLP/attention intermediates during backward instead of storing
    # them. Drops activation memory from ~160GB to ~11GB across 80 layers.
    #rank_print(rank, "Applying activation checkpointing ...")
    #apply_activation_checkpointing(
    #    model,
    #    checkpoint_wrapper_fn=partial(
    #        checkpoint_wrapper,
    #        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    #    ),
    #    check_fn=lambda m: isinstance(m, Qwen2DecoderLayer),
    #)
    # ─────────────────────────────────────────────────────────────────────────

    rank_print(rank, f"Model ready | {mem_str(device)}\n")

    step_times, throughputs = [], []

    for step in range(N_STEPS):
        input_ids = torch.randint(
            0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device
        )

        t0 = time.perf_counter()

        with torch.no_grad():
            out  = model(input_ids=input_ids, use_cache=False)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        tokens  = BATCH_SIZE * SEQ_LEN * world_size
        tps     = tokens / elapsed

        if step >= WARMUP_STEPS:
            step_times.append(elapsed)
            throughputs.append(tps)

        rank_print(rank,
            f"step {step+1:>3}/{N_STEPS} | "
            f"time={elapsed:.2f}s | tokens/s={tps:,.0f} | {mem_str(device)}"
        )

    if throughputs:
        mean_time = sum(step_times) / len(step_times)
        mean_tps  = sum(throughputs) / len(throughputs)
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e9

        rank_print(rank, f"\n{'─'*64}")
        rank_print(rank, f"Results (post-warmup, steps {WARMUP_STEPS+1}–{N_STEPS}):")
        rank_print(rank, f"  Mean step time : {mean_time:.2f}s")
        rank_print(rank, f"  Mean tokens/s  : {mean_tps:,.0f}")
        rank_print(rank, f"  Peak VRAM      : {peak_vram:.1f} GB / 144 GB per GPU")
        rank_print(rank, f"{'─'*64}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
