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

## Large grid Float32 results (1024×1024×128/GPU)

| GPUs | ms/step | Scaling efficiency |
|------|---------|-------------------|
| 1    | 357.1   | 100% (baseline)   |
| 2    | 435.4   | 82.0%             |
| 4    | 429.4   | 83.2%             |

## Overhead analysis (nsight, 2 GPUs, 1024×1024×128 F32)

| Source | ms/step | Stream | Status |
|--------|---------|--------|--------|
| NCCL comm (async halos) | 30.4 | comm_stream | Overlapping with compute ✓ |
| NCCL comm (sync solver) | 3.4 | comm_stream | Overlapping via events ✓ |
| NCCL comm (residual default) | 20.4 | default | Batched single-field fills |
| Pack/unpack (halo buffers) | 30.0 | default | Fundamental cost |
| Pack/unpack (FFT transpose) | 26.2 | default | Fundamental cost |

Total overhead: ~78 ms = pack/unpack (~56 ms) + residual NCCL + event sync.
Pack/unpack is the dominant remaining cost — requires kernel fusion to eliminate.

## Notes

- 2→4 GPU scaling is nearly flat, confirming overhead is from distributed path not comm volume
- All NCCL operations route through a dedicated comm_stream for maximum overlap
- Async halo fills (from update_state!) overlap with interior tendency computation
- Multi-field NCCL batching applied for synchronous single-field fills
- All 3 halo fills per RK3 substep are algorithmically necessary
- Further improvement requires fusing pack/unpack into compute kernels (deep refactor)
