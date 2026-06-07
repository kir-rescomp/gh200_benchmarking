- `size (B)` — the total message size in bytes being reduced across all GPUs. 
  This is what each rank contributes; the actual data moving over the network is a function of this.

- `count (elements)` — the number of individual elements in the buffer. Since we're using float (4 bytes each), `count = size / 4`.
type — the datatype (float in your case). NCCL supports float, half, int, etc.
redop — the reduction operation. sum means each GPU's buffer is summed element-wise across all ranks. Other options are min, max, prod.
root — only relevant for rooted collectives like broadcast or reduce. -1 here means not applicable for allreduce.
time (us) — the average time in microseconds to complete one allreduce operation across all 8 GPUs, measured from the calling rank's perspective.
algbw (GB/s) — algorithmic bandwidth: size / time. This is the application-level throughput — how fast data was processed from the perspective of a single rank. It doesn't account for the actual traffic on the wire.
busbw (GB/s) — bus bandwidth: algbw × 2(n-1)/n where n is the number of ranks. This scales algbw by the fraction of data that actually travels between GPUs in a ring allreduce, making it comparable to the hardware's rated link speed. This is the number to compare against your 200 Gb/s IB links. Your ~40 GB/s was the meaningful result.
#wrong — number of elements that failed the correctness check (comparison against expected values). Zero is what you want. Any non-zero value means something is broken in the communication.
out-of-place vs in-place — whether the result is written to a separate output buffer (out-of-place) or back into the same input buffer (in-place). Both are tested because some hardware/software paths handle them differently. In practice the numbers should be similar.
