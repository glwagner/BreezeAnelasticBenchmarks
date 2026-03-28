# Weak Scaling Results

## Hardware

- 4x NVIDIA A100-SXM4-80GB
- NV12 NVLink between all GPU pairs
- Open MPI 4.1.6 (no GPU-aware MPI — NCCL handles all GPU communication)
- Julia 1.12.5

## Configuration

- Grid: 200x200x80 per GPU (weak scaling)
- Topology: Periodic x Periodic x Bounded
- Advection: WENO(order=5)
- Buoyancy: BuoyancyTracer, tracers = :b
- Timestepper: RK3, dt = 0.1 s
- Warmup: 30 steps, Benchmark: 20 steps

## Results (2026-03-28, NCCL extension with multi-field batching)

### NonhydrostaticModel + WENO5 + BuoyancyTracer

| GPUs | ms/step | Scaling efficiency |
|------|---------|-------------------|
| 1    | 13.37   | 100% (baseline)   |
| 2    | 23.57   | 56.7%             |
| 4    | 23.87   | 56.0%             |

### NonhydrostaticModel + Centered + BuoyancyTracer (simpler advection)

| GPUs | ms/step | Scaling efficiency |
|------|---------|-------------------|
| 1    | 8.9     | 100% (baseline)   |
| 2    | 16.0    | 56%               |
| 4    | 16.0    | 56%               |

### Pressure solver only (isolated)

| GPUs | ms/solve |
|------|----------|
| 1    | 0.95     |
| 2    | 2.82     |
| 4    | 2.95     |

## Nsight profile breakdown (4 GPUs, WENO5, 10 timesteps)

| Category | % GPU time | Total (ms) |
|----------|-----------|-----------|
| NCCL communication | 26.9% | 295 |
| Tendencies (WENO5) | 26.1% | 286 |
| FFT (pressure solver) | 13.2% | 145 |
| Broadcast (pack/unpack) | ~5% | ~55 |
| Other (RK3, cache, hydro) | ~29% | ~315 |

NCCL steady-state: 594 calls, 142 ms (0.24 ms/call avg).
6 outlier calls >2 ms total 153 ms (NCCL init/GC).

## Notes

- 2→4 GPU scaling is nearly flat (23.6→23.9 ms), confirming overhead is from distributed path
- ~10 ms distributed overhead = pack/unpack (~3-4 ms) + NCCL comm (~1.4 ms steady) + NCCL outliers (~1.5 ms) + solver/FFT overhead (~2 ms) + extra kernel launches (~2 ms)
- All 3 halo fills per RK3 substep are algorithmically necessary (can't merge)
- Multi-field NCCL batching already applied (all fields in one NCCL group per fill call)
- Remaining optimization: comm/computation overlap (Phase 5)
