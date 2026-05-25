Peak HBM3e memory bandwidth (~4 TB/s theoretical) and
tensor core TFLOPS (BF16 ~2000 TFLOPS theoretical). A GEMM sweep across
matrix sizes approaching 144 GB reveals the compute-bound vs memory-bound
crossover, confirming HBM3e is functioning correctly.

Nearly all HPC workloads (MD force calculations,
neural network layers, dense linear algebra, FFTs) are ultimately bottlenecked
by memory bandwidth. This confirms the HBM3e is operating at spec.
