"""
GH200 Test 3: HBM3e bandwidth and TFLOPS sweep via cuBLAS GEMM.

Measures effective memory bandwidth (GB/s) and compute throughput (TFLOPS)
across a range of matrix sizes in float16, bfloat16, and float32.
Compares to GH200 theoretical peaks.

Requires: torch>=2.2, numpy
"""

import torch
import time
import math

# GH200 144GB theoretical peaks
THEORETICAL_HBM_BW_TBS  = 4.0     # TB/s HBM3e bandwidth
THEORETICAL_BF16_TFLOPS  = 1979.0  # TFLOPS (tensor core BF16)
THEORETICAL_FP32_TFLOPS  = 66.9    # TFLOPS (FP32 non-TC)

DEVICE = torch.device("cuda:0")
WARMUP_ITERS = 5
BENCH_ITERS  = 20


def bench_gemm(M: int, N: int, K: int, dtype: torch.dtype) -> dict:
    """Single GEMM benchmark: C = A @ B."""
    A = torch.randn(M, K, dtype=dtype, device=DEVICE)
    B = torch.randn(K, N, dtype=dtype, device=DEVICE)

    # Warmup
    for _ in range(WARMUP_ITERS):
        C = torch.mm(A, B)
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH_ITERS):
        C = torch.mm(A, B)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms  = start.elapsed_time(end) / BENCH_ITERS
    elapsed_s   = elapsed_ms / 1000

    flops       = 2 * M * N * K
    tflops      = flops / elapsed_s / 1e12

    bytes_per_el = torch.finfo(dtype).bits // 8
    bytes_read   = (M * K + K * N) * bytes_per_el
    bytes_write  = M * N * bytes_per_el
    bw_gbs       = (bytes_read + bytes_write) / elapsed_s / 1e9

    del A, B, C
    torch.cuda.empty_cache()

    return {
        "M": M, "N": N, "K": K,
        "dtype": str(dtype).split(".")[-1],
        "elapsed_ms": elapsed_ms,
        "tflops": tflops,
        "bw_gbs": bw_gbs,
        "mem_gb": (M * K + K * N + M * N) * bytes_per_el / 1e9,
    }


def print_header():
    print(f"\n{'Shape (M=N=K)':>16} | {'dtype':>7} | {'Time(ms)':>9} | "
          f"{'TFLOPS':>8} | {'% Peak':>7} | {'BW(GB/s)':>9} | {'% BW Peak':>10} | {'Mem(GB)':>8}")
    print("-" * 95)


def print_row(r: dict):
    shape_str = f"{r['M']}³"
    peak_tflops = THEORETICAL_FP32_TFLOPS if r["dtype"] == "float32" else THEORETICAL_BF16_TFLOPS
    pct_tflops  = r["tflops"]  / peak_tflops * 100
    pct_bw      = r["bw_gbs"] / (THEORETICAL_HBM_BW_TBS * 1000) * 100
    print(f"{shape_str:>16} | {r['dtype']:>7} | {r['elapsed_ms']:>9.2f} | "
          f"{r['tflops']:>8.1f} | {pct_tflops:>6.1f}% | {r['bw_gbs']:>9.0f} | "
          f"{pct_bw:>9.1f}% | {r['mem_gb']:>8.2f}")


if __name__ == "__main__":
    print("GH200 Test 3 — HBM3e Bandwidth & TFLOPS Sweep")
    dev_props = torch.cuda.get_device_properties(DEVICE)
    print(f"Device: {dev_props.name}")
    print(f"VRAM:   {dev_props.total_memory / 1e9:.1f} GB")
    print(f"SM count: {dev_props.multi_processor_count}")
    print(f"\nTheoretical BF16: {THEORETICAL_BF16_TFLOPS} TFLOPS | "
          f"HBM3e BW: {THEORETICAL_HBM_BW_TBS} TB/s\n")

    # Square GEMMs: small (cache-friendly) → large (memory-bound → compute-bound)
    sizes = [512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 32768]

    print_header()
    for n in sizes:
        mem_gb = 3 * n * n * 2 / 1e9  # 3 matrices in float16
        if mem_gb > 130:
            break  # stay within single-GPU VRAM for this test
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            mem_for_dtype = 3 * n * n * torch.finfo(dtype).bits // 8 / 1e9
            if mem_for_dtype > 130:
                continue
            try:
                r = bench_gemm(n, n, n, dtype)
                print_row(r)
            except torch.cuda.OutOfMemoryError:
                print(f"{n}³ | {dtype} | OOM")
                break

    # Non-square: tall-skinny (attention-like) and wide-flat (projection)
    print(f"\n--- Attention-shaped (m=batch×seq, k=d_model, n=d_model) ---")
    print_header()
    attention_shapes = [
        # (M,    N,    K)    # description
        (131072, 4096, 4096),   # 32k seq × 4096 d_model
        (65536,  8192, 8192),   # 16k seq × 8192 (Llama-3-70B)
        (32768, 16384, 16384),  # 8k  seq × 16384
    ]
    for M, N, K in attention_shapes:
        for dtype in [torch.bfloat16]:
            mem_gb = (M * K + K * N + M * N) * 2 / 1e9
            if mem_gb > 130:
                print(f"  ({M},{N},{K}) skipped — {mem_gb:.0f} GB")
                continue
            try:
                r = bench_gemm(M, N, K, dtype)
                print_row(r)
            except torch.cuda.OutOfMemoryError:
                print(f"  ({M},{N},{K}) OOM")
    print(f"Expected GH200 peaks: BF16 ~60–80% of ~989 TFLOPS dense theoretical")
    print(f"  (1979 TFLOPS is sparse TC peak; practical dense GEMM ceiling ~989 TFLOPS)")
