- `size (B)` — the total message size in bytes being reduced across all GPUs. 
  This is what each rank contributes; the actual data moving over the network is a function of this.

- `count (elements)` — the number of individual elements in the buffer. Since we're using float (4 bytes each), `count = size / 4`.
- `type` — the datatype (float in your case). NCCL supports float, half, int, etc.
- `redop` — the reduction operation. sum means each GPU's buffer is summed element-wise across all ranks. Other options are min, max, prod.
- `root` — only relevant for rooted collectives like broadcast or reduce. -1 here means not applicable for allreduce.
- `time (us)` — the average time in microseconds to complete one allreduce operation across all 8 GPUs, measured from the calling rank's perspective.
- `algbw (GB/s)` — algorithmic bandwidth: size / time. This is the application-level throughput — how fast data was processed from the perspective of a single rank. It doesn't account for the actual traffic on the wire.
  
- `busbw (GB/s)` — bus bandwidth: algbw × 2(n-1)/n where n is the number of ranks. This scales algbw by the fraction of data that actually travels between GPUs in a ring allreduce, making it comparable to the hardware's rated link speed. This is the number to compare against your 200 Gb/s IB links. Your ~40 GB/s was the meaningful result.
- `#wrong` — number of elements that failed the correctness check (comparison against expected values). Zero is what you want. Any non-zero value means something is broken in the communication.
- `out-of-place vs in-place` — whether the result is written to a separate output buffer (out-of-place) or back into the same input buffer (in-place). Both are tested because some hardware/software paths handle them differently. In practice the numbers should be similar.


# NCCL `all_reduce_perf` benchmark — GH200 NVL2 (4 nodes, 8 GPUs)

## Hardware

| Component | Detail |
|---|---|
| Nodes | `compgh000`–`compgh003` (4 nodes) |
| GPU | NVIDIA GH200 144G HBM3e × 2 per node (NVL2 dual-superchip) |
| Intra-node link | NVLink-C2C (Grace CPU ↔ Hopper GPU per superchip; NV18 between superchips) |
| IB fabric | 2× ConnectX-7 HDR200 (200 Gb/s) per node — `mlx5_0`, `mlx5_1` |
| CUDA | 12.8 (system: `/usr/local/cuda`) |
| NCCL | 2.27.7 |
| MPI | OpenMPI 5.0.7-GCC-14.2.0 |

## Job configuration

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2   # one rank per GPU
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
```

One MPI rank per GPU, with `--gpu-bind=closest` so each rank is bound to its
NUMA-local Hopper GPU. With 2 ranks per node × 4 nodes = 8 ranks total.

## NCCL environment variables

Two variables were tested to see whether they improved performance over NCCL's
defaults:

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_NET_GDR_LEVEL=5
```

**`NCCL_IB_HCA`** explicitly tells NCCL which InfiniBand HCAs to use. Without
it, NCCL probes all available network devices. On these nodes `mlx5_2` and
`mlx5_3` are 2.5 Gb/s Ethernet adapters (not IB), so setting this variable
prevents NCCL from wasting time probing them.

**`NCCL_NET_GDR_LEVEL`** controls the threshold for enabling GPU Direct RDMA
(GDR) — the ability for the IB HCA to read/write GPU HBM directly without
staging data through host CPU memory. A value of `5` corresponds to the `SYS`
level, meaning GDR is enabled regardless of the PCIe/NVLink topology distance
between the GPU and the HCA.

**In practice, neither variable changed the result.** NCCL's debug output
(`NCCL_DEBUG=INFO`) confirmed that on these nodes NCCL auto-selected both
`mlx5_0` and `mlx5_1`, created a bonded virtual device at 400 Gb/s, and
enabled GPUDirect RDMA via the NVLink-C2C path — all without any explicit
hints. The achieved bus bandwidth was identical in both runs (~40.49 GB/s).

These variables can be safely omitted on the GH200 nodes. They are documented
here because they are commonly recommended for multi-NIC IB setups and were
explicitly verified to have no effect in this environment.

## Transport stack confirmed by NCCL debug output

- Both IB HCAs detected and bonded: `mlx5_0+mlx5_1 speed=400000`
- GPUDirect RDMA enabled on all 8 GPUs via NVLink-C2C link:
  `GPU Direct RDMA Enabled for GPU X / HCA 2 (distance 3 <= 9)`
- 4 RDMA channels per rank pair, using the combined HCA device
- Out-of-band bootstrap via `ib-bond0`

## Results

Test command:

```bash
srun --gpu-bind=closest ./nccl-tests/build/all_reduce_perf \
  -b 1G -e 128G -f 2 \
  -n 1000 -w 10 \
  -d float -o sum
```

> Note: NCCL reduced `maxBytes` from 128 GB to ~46 GB due to HBM memory
> already in use by the driver and other processes.

| Message size | Time (ms) | algbw (GB/s) | busbw (GB/s) | Errors |
|---:|---:|---:|---:|---:|
| 1 GB | 46.5 | 23.10 | 40.43 | 0 |
| 2 GB | 92.8 | 23.15 | 40.52 | 0 |
| 4 GB | 185.4 | 23.17 | 40.55 | 0 |
| 8 GB | 371.0 | 23.16 | 40.52 | 0 |
| 16 GB | 742.7 | 23.13 | 40.48 | 0 |
| ~34 GB | 1486.3 | 23.12 | 40.46 | 0 |
| **Average** | | **23.14** | **40.49** | **0** |

**algbw** (algorithmic bandwidth) is `message_size / time` — the
application-level throughput from a single rank's perspective.

**busbw** (bus bandwidth) scales algbw by the ring allreduce communication
factor `2(n−1)/n`, making it directly comparable to the hardware's rated link
speed. This is the meaningful number for fabric characterisation.

## Interpretation

The theoretical unidirectional IB peak per node is 2 × 200 Gb/s = 400 Gb/s =
**50 GB/s**. The achieved busbw of **40.49 GB/s represents ~81% efficiency**,
which is strong for an 8-rank ring allreduce — the ring algorithm pipelines
traffic across all nodes rather than fully saturating both NICs on any one node
simultaneously. The result is flat across all message sizes from 1 GB to 34 GB,
indicating no congestion or thermal throttling during the test.
