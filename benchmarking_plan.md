# NCCL Distributed Benchmarking Plan

## Goal

Characterize performance of NCCLDistributed vs vanilla MPI Distributed
for full Oceananigans simulations, and identify remaining bottlenecks.

## Benchmark Configuration

- **Grid:** 200×200×80 per GPU (weak scaling), Periodic×Periodic×Bounded
- **Physics:** NonhydrostaticModel, BuoyancyTracer, WENO5 advection
- **Time stepping:** RK3, dt = 1.0 s, 20 warmup + 20 timed steps
- **Hardware:** A100-SXM4-80GB, NV12 NVLink (4 GPUs per node)

## Results (2027-03-27, A100-80GB, NVLink)

| GPUs | NCCLDistributed (ms/step) | Non-distributed (ms/step) | Overhead |
|------|---------------------------|---------------------------|----------|
| 1    | —                         | 8.9                       | baseline |
| 2    | 53.5                      | —                         | 6.0x     |
| 4    | 21.9                      | —                         | 2.5x     |

### Pressure solver only (isolated)

| GPUs | NCCL (ms/solve) |
|------|-----------------|
| 1    | 0.95            |
| 2    | 2.82            |
| 4    | 2.95            |

The pressure solver scales well. The full timestep overhead must come from
halo communication, distributed field operations, or other distributed overhead.

## Next Steps

### 1. Nsight Systems profiling (in progress)

Profile 4-GPU run to identify:
- Time in NCCL Send/Recv (halo + transpose)
- Time in GPU kernels (advection, tendencies, pressure solve)
- Idle gaps between kernels (pipeline stalls)
- Number of `sync_device!` calls that remain (should be zero for NCCL path)
- Any remaining MPI calls that shouldn't be there

### 2. Component breakdown

Time each phase of a time step separately:
- `calculate_tendencies!` (advection + buoyancy + closures)
- `fill_halo_regions!` (all fields)
- `solve!` (pressure solver)
- `update_state!`
- `time_step!` (total)

### 3. Apples-to-apples MPI vs NCCL (Perlmutter)

On Perlmutter (GPU-aware MPI available):
- Same grid, same physics
- Run with `Distributed(GPU())` (MPI path)
- Run with `NCCLDistributed(GPU())` (NCCL path)
- Compare ms/timestep and component breakdown

### 4. Weak scaling curve

| GPUs | Nodes | Grid |
|------|-------|------|
| 1    | 1     | 200×200×80 |
| 2    | 1     | 400×200×80 |
| 4    | 1     | 800×200×80 |
| 8    | 2     | 1600×200×80 |
| 16   | 4     | 3200×200×80 |

### 5. Comparison with ERF (Perlmutter)

ERF uses 50×400×80 per GPU with 8 GPUs on 2 nodes.
Run same configuration with Breeze + NCCLDistributed.

## Analysis Questions

1. Why is 2-GPU (53.5 ms) slower than expected given 4-GPU (21.9 ms)?
   - Possible: NCCL peer initialization overhead amortized over more ranks
   - Possible: 2-GPU has suboptimal NVLink topology mapping
   - Profile will reveal

2. What fraction of the 21.9 ms timestep is communication vs compute?
   - Pressure solver: ~3 ms (from isolated benchmark)
   - Remaining ~19 ms: halos, advection, tendencies

3. Where are the remaining `sync_device!` calls?
   - Our extension sets `sync_device!` to no-op for NCCLDistributed
   - But there may be `CUDA.synchronize()` calls elsewhere in Oceananigans
