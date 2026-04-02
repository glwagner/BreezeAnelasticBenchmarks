# cuFFTMp vs Custom Distributed FFT: Analysis

## Current Distributed FFT Architecture (Oceananigans)

The `DistributedFourierTridiagonalPoissonSolver` uses a "transpose + local FFT" approach:

```
z-local data → pack → Alltoall(z→y) → unpack → local FFT(y) →
               pack → Alltoall(y→x) → unpack → local FFT(x) →
               tridiagonal solve(z) →
               pack → Alltoall(x→y) → unpack → local iFFT(y) →
               pack → Alltoall(y→z) → unpack → local iFFT(x)
```

Each RK3 substep does 1 pressure solve = **4 Alltoall transposes**.
3 substeps per step = **12 Alltoall transposes per time step**.

With NCCL, each Alltoall is implemented as grouped Send/Recv pairs
(N sends + N receives per rank, where N = #ranks in the sub-communicator).

## Profiling Breakdown (NHM 8×1, 1600×1600×100, 3 steps)

| Component | Time | % of GPU |
|-----------|------|----------|
| NCCL Send/Recv (total) | 27,788 ms | 70.5% |
|   └ Transpose Alltoall (>10ms transfers) | 24,499 ms | **62.2%** |
|   └ Halo communication (<10ms transfers) | 3,289 ms | 8.3% |
| permutedims (layout conversion) | 1,123 ms | 2.8% |
| Pack/unpack buffers | 384 ms | 1.0% |
| FFT compute (cuFFT kernels) | 489 ms | 1.2% |
| Tendency kernels | 8,542 ms | 21.7% |
| Other | 581 ms | 1.5% |

**The distributed FFT solver consumes 67% of GPU time** (Alltoall + permutedims + pack/unpack + FFT).
The FFT compute itself is only 1.2% — the bottleneck is entirely data movement.

## What cuFFTMp Would Change

cuFFTMp replaces the entire transpose + local FFT pipeline with a single
optimized multi-GPU FFT call. Key advantages:

1. **Eliminates explicit transposes**: cuFFTMp handles data redistribution
   internally using NCCL or NVSHMEM, fusing communication with FFT compute.

2. **Pipelining**: cuFFTMp can overlap communication of one slab with FFT
   computation of another, hiding latency.

3. **Eliminates pack/unpack + permutedims**: The buffer packing (1.0%) and
   layout transpose (2.8%) kernels are folded into cuFFTMp's internal operations.

4. **Optimized collective patterns**: cuFFTMp uses slab or pencil decomposition
   with communication patterns tuned for the specific GPU topology (NVLink, PCIe).

## Expected Performance

### Optimistic estimate (cuFFTMp achieves full overlap)

If cuFFTMp fully pipelines communication with FFT compute:
- Transpose Alltoall: 24,499 ms → hidden behind FFT compute (effectively 0)
- permutedims: 1,123 ms → 0 (eliminated)
- Pack/unpack: 384 ms → 0 (eliminated)
- FFT compute: 489 ms → ~500 ms (same or slightly larger due to different layout)
- Net savings: ~25,500 ms over 3 steps = **~2,835 ms/step**

NHM step time would drop from 437.7 ms to ~154 ms (**3.5× faster**, 97% efficiency).

### Realistic estimate (cuFFTMp with partial overlap)

cuFFTMp typically achieves 2-5× speedup over manual transpose for multi-GPU FFTs:
- Transpose + FFT: 26,495 ms → ~8,000 ms (3.3× faster)
- Net savings: ~18,500 ms over 3 steps = ~2,056 ms/step

NHM step time: 437.7 - 206 = ~232 ms (**1.9× faster**, 64% efficiency).

### Conservative estimate (cuFFTMp limited by NVLink bandwidth)

The transpose moves ~32M complex floats × 4 transposes × 3 substeps = 384M transfers
at 8 bytes each = 3.1 GB per step. NVLink on A100 provides ~600 GB/s bisection.
Bandwidth limit: 3.1 GB / 600 GB/s ≈ 5 ms.
Current time: ~8,166 ms/step for transposes.

Even at 50% NVLink efficiency: ~10 ms, vs current ~2,720 ms/step.
This suggests **cuFFTMp could be 100-270× faster** for the transpose alone.

## Installation Requirements

cuFFTMp requires:
- NVIDIA HPC SDK, or separate `nvidia-cufftmp` package
- NCCL or NVSHMEM backend
- Not available via pip/conda on this machine
- Available on Perlmutter via `module load cufftmp` or NVIDIA NGC containers

## Recommended Next Steps

1. **Benchmark on Perlmutter** where cuFFTMp is available via HPC SDK
2. **Write Julia wrapper** for `cufftMpMakePlan3d` / `cufftMpExecC2C`
   (small C wrapper + ccall, ~100 lines)
3. **Compare**: Run NHM pressure solve with cuFFTMp vs current approach
4. **Integrate**: If cuFFTMp is significantly faster, add as optional backend
   to `DistributedFourierTridiagonalPoissonSolver`

## Alternative: NCCL-native Alltoall Optimization

Even without cuFFTMp, the current NCCL Alltoall can be improved:
- Current: grouped Send/Recv (N sends + N receives per rank)
- Better: use NCCL's `ncclAlltoAll` (available since NCCL 2.20)
- This avoids the per-pair overhead and uses optimized ring/tree algorithms

The NCCL.jl package may need to be updated to expose `ncclAlltoAll`.
