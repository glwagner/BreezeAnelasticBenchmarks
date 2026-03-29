# Weak Scaling Results

## Hardware

- 4x NVIDIA A100-SXM4-80GB
- NV12 NVLink between all GPU pairs
- Julia 1.12.5, NCCL 2.28.3
- NCCLDistributed architecture (all comm via NCCL, no GPU-aware MPI needed)

## Full Results (all optimizations: comm_stream + async overlap + pipelined RK3)

### WENO5 + BuoyancyTracer, 1024×1024×128 per GPU (large grid)

| GPUs | Float32 (ms) | F32 eff | Float64 (ms) | F64 eff |
|------|-------------|---------|-------------|---------|
| 1    | 356.8       | 100%    | 487.7       | 100%    |
| 2    | 419.0       | 85.2%   | 594.0       | 82.1%   |
| 4    | 420.9       | 84.8%   | 606.1       | 80.5%   |

### WENO5 + BuoyancyTracer, 200×200×80 per GPU (small grid)

| GPUs | Float64 (ms) | F64 eff |
|------|-------------|---------|
| 1    | 14.31       | 100%    |
| 2    | 21.95       | 65.2%   |
| 4    | 21.87       | 65.4%   |

### ERF-like: Centered + ScalarDiffusivity(ν=200,κ=200), 50×400×80 per GPU

| GPUs | Float32 (ms) | F32 eff | Float64 (ms) | F64 eff |
|------|-------------|---------|-------------|---------|
| 1    | 3.87        | 100%    | 6.81        | 100%    |
| 2    | 14.93       | 25.9%   | 19.29       | 35.3%   |
| 4    | 18.47       | 20.9%   | 19.84       | 34.3%   |

## Analysis

### Why scaling efficiency depends on grid size

The distributed overhead is ~70 ms of GPU idle time at synchronization points
(cuStreamWaitEvent waits for NCCL to complete). This is approximately constant
regardless of grid size. At large grids, compute dominates:

| Grid/GPU | Compute (ms) | Overhead (ms) | Efficiency |
|----------|-------------|--------------|-----------|
| 1024×1024×128 F32 | 357 | ~63 | 85% |
| 1024×1024×128 F64 | 488 | ~106 | 82% |
| 200×200×80 F64 | 14 | ~8 | 65% |
| 50×400×80 F64 | 7 | ~12 | 35% |

### Where the overhead comes from (nsight profiling)

Per timestep, the GPU is idle at 33 synchronization points totaling ~70 ms:
- `synchronize_communication!`: 3 per step (1 per RK3 substage)
- Pressure solver transposes: 6 per step (2 per substage)
- Velocity/pressure halo fills: ~4 per step

Each wait averages 2-5 ms (NVLink transfer latency for the data volume).

### What's already optimized

1. All NCCL on dedicated comm_stream (overlaps with GPU compute)
2. Async halo fills overlap with interior tendency computation
3. Pipelined RK3: tracer halos start before pressure solve
4. cuMemcpy2D for halo pack (DMA engine, frees compute units)
5. Multi-field batched NCCL groups (reduce kernel launches)
6. sync_device! eliminated (NCCL is stream-native)

### Remaining optimization opportunities

1. **Pipelining across RK3 substages** — start next substage's halo comm while
   current substage's pressure solver runs. Requires double-buffered fields.
2. **Communication-avoiding pressure solver** — multigrid or local iterative
   method that doesn't need global FFT transposes.
3. **Larger grids** — efficiency naturally improves as compute grows with N³
   while comm overhead grows with N².
