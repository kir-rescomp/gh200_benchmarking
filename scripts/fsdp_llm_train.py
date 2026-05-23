"""
GH200 Test 1: FSDP LLM fine-tuning across 2× GH200 GPUs.
Run with: torchrun --nproc_per_node=2 fsdp_llm_train.py
Requires: torch>=2.3, transformers, flash-attn (optional but recommended)
"""

import os, time, math
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy # FSDP Utility
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from functools import partial

# ── Config ─────────────────────────────────────────────────────────────────
# Qwen-Qwen2.5-72B was downloaded from huggingface
# increase batch / seq_len to fill VRAM.
MODEL_NAME   = "/gpfs3/well/kir/projects/mirror/LLM/huggingface/Qwen-Qwen2.5-72B"
SEQ_LEN      = 8192    # long context — exercises memory bandwidth
BATCH_SIZE   = 1       # per GPU; 2×8192×2 = 32k tokens per step
N_STEPS      = 30
WARMUP_STEPS = 5
# ───────────────────────────────────────────────────────────────────────────


def rank_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)


def mem_str(device):
    a = torch.cuda.memory_allocated(device) / 1e9
    r = torch.cuda.memory_reserved(device) / 1e9
    m = torch.cuda.max_memory_allocated(device) / 1e9
    return f"alloc={a:.1f}GB | reserved={r:.1f}GB | peak={m:.1f}GB"


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    rank_print(rank, f"\n{'='*64}")
    rank_print(rank, f"GH200 Test 1 — FSDP LLM Training")
    rank_print(rank, f"World size: {world_size} GPUs | Model: {MODEL_NAME}")
    rank_print(rank, f"Seq len: {SEQ_LEN} | Batch/GPU: {BATCH_SIZE}")
    rank_print(rank, f"{'='*64}\n")

    # Load config to derive wrap policy
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # Determine the decoder-layer class for FSDP wrapping
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
            config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )

    rank_print(rank, "Wrapping with FSDP ...")
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=rank,
        # Materialise parameters on GPU from meta device
        param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
        limit_all_gathers=True,
        use_orig_params=True,
    )

    rank_print(rank, f"Model loaded | {mem_str(device)}")


    step_times, throughputs = [], []
    for step in range(N_STEPS):
        input_ids = torch.randint(
            0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device
        )

        t0 = time.perf_counter()
        out  = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        tokens  = BATCH_SIZE * SEQ_LEN * world_size
        tps     = tokens / elapsed

        if step >= WARMUP_STEPS:
            step_times.append(elapsed)
            throughputs.append(tps)

        rank_print(rank,
            f"step {step+1:>3}/{N_STEPS} | loss={loss.item():.4f} | "
            f"time={elapsed:.2f}s | tokens/s={tps:,.0f} | {mem_str(device)}"
        )

    if throughputs:
        rank_print(rank, f"\n--- Results (post-warmup) ---")
        rank_print(rank, f"Mean step time : {sum(step_times)/len(step_times):.2f}s")
        rank_print(rank, f"Mean tokens/s  : {sum(throughputs)/len(throughputs):,.0f}")
        rank_print(rank, f"Peak GPU VRAM  : {torch.cuda.max_memory_allocated(device)/1e9:.1f} GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
