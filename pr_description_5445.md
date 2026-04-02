## Summary

Add `OceananigansNCCLExt` that replaces all MPI-based GPU communication with NCCL. NCCL operations are GPU-stream-native, eliminating `sync_device!` pipeline stalls.

## Optimization history

### Phase 1: NCCL pressure solver transposes (`0dcf95496`)
Replace `sync_device! + MPI.Alltoall` with NCCL grouped `Send/Recv` in `NCCLDistributedFFTSolver`.
- Pressure solver: 2.82 ms/solve (2 GPU), 2.95 ms/solve (4 GPU)
- MPI baseline (Derecho): 36.2 ms/solve (2 GPU)

### Phase 2: NCCLDistributed architecture + halo communication (`e7849b683`)
`NCCLDistributed` drop-in replacement for `Distributed`. All halo fills use NCCL.
`NCCLCommunicator` wraps NCCL + MPI comms; MPI forwarding for reductions/init.
- First full distributed simulation runs end-to-end

### Phase 3: Fix transpose dispatch (`10f7ffd5a`)
`twin_grid` creates new architectures without `NCCLCommunicator`. Fixed by checking `yfield` (not `xfield`) for NCCL arch in transpose overrides.
- Full simulation: 76.3 ms/step (2 GPU) — first working result

### Phase 4: Per-field NCCL batching (`eaa633f7f`)
Pack all sides for one field, then one NCCL group, then unpack all sides.
- 2 GPUs: 76.3 → 17.4 ms/step (4.4x improvement)

### Phase 5: Multi-field NCCL batching (`afa36cab0`)
Pack ALL fields, one NCCL group for all fields' Send/Recv, unpack all.
- 2 GPUs: 17.4 → 16.0 ms/step
- 4 GPUs: 19.5 → 16.0 ms/step (near-perfect weak scaling)

### Phase 6: Clean up + proper dispatch (`646087dce`, `0b4cbf809`)
- `NCCLDistributedArch`, `NCCLDistributedGrid`, `NCCLDistributedField` type aliases
- Extend `distributed_fill_halo_event!` on `NCCLDistributedGrid` (not overwrite)
- Remove `__precompile__(false)`
- No type piracy, no underscore-prefixed functions, explicit return statements

## Benchmark results

### WENO5 weak scaling (200×200×80/GPU, A100-SXM4-80GB, NV12 NVLink)

| GPUs | ms/step | Efficiency |
|------|---------|-----------|
| 1 (non-distributed) | 13.37 | 100% |
| 2 | 21.30 | 63% |
| 4 | 24.00 | 56% |

### ERF-like weak scaling (50×400×80/GPU, Centered + diffusion)

| GPUs | ms/step | Efficiency |
|------|---------|-----------|
| 1 (non-distributed) | 6.85 | 100% |
| 2 | 18.22 | 38% |
| 4 | 16.49 | 42% |

### Nsight profile (4 GPUs, WENO5, 10 timesteps)

| Category | % GPU time |
|----------|-----------|
| NCCL communication | 27% |
| Tendencies (WENO5) | 26% |
| FFT (pressure solver) | 13% |
| Pack/Unpack buffers | 5% |

## Dependencies

Requires NCCL.jl with Complex type support: JuliaGPU/NCCL.jl#67

## Usage

```julia
using NCCL
NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
arch = NCCLExt.NCCLDistributed(GPU(); partition = Partition(Ngpus, 1))
grid = RectilinearGrid(arch, ...)
model = NonhydrostaticModel(grid, ...)
# All communication automatically uses NCCL
```
