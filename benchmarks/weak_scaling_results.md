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

## Results (2026-03-28, current NCCL extension)

### NonhydrostaticModel + WENO5 + BuoyancyTracer

| GPUs | ms/step | Scaling efficiency |
|------|---------|-------------------|
| 1    | 13.38   | 100% (baseline)   |
| 2    | 21.20   | 63%               |
| 4    | 23.78   | 56%               |

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

## Notes

- WENO5 has more halo points (3 vs 1 for Centered), increasing communication volume
- The 2→4 GPU scaling is nearly flat (21→24 ms), suggesting communication is the bottleneck
- 1-GPU is non-distributed (different code path, no halo communication)
- Distributed overhead: ~1.6-1.8x for Centered, ~1.6-1.8x for WENO5
