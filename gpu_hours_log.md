# GPU Hours Log

Account: UMIT0049 (Derecho)

## Completed Jobs

| Job ID | Date | GPUs | Walltime (req) | Walltime (actual) | GPU-hrs | Description |
|--------|------|------|----------------|-------------------|---------|-------------|
| 5608630 | 2026-03-25 | 1 | 1:00:00 | ~0:05 | 0.08 | Distributed 1 GPU (old script) |
| 5608759 | 2026-03-25 | 1 | 1:00:00 | ~0:05 | 0.08 | Distributed 1 GPU (old script) |
| 5608760 | 2026-03-25 | 2 | 1:00:00 | ~0:05 | 0.17 | Distributed 2 GPU |
| 5608761 | 2026-03-25 | 4 | 1:00:00 | ~0:05 | 0.33 | Distributed 4 GPU |
| 5608762 | 2026-03-25 | 8 | 1:00:00 | ~0:05 | 0.67 | Distributed 8 GPU |
| 5615935 | 2026-03-26 | 1 | 0:30:00 | ~0:03 | 0.05 | Diagnostic perf (default pool) |
| 5615936 | 2026-03-26 | 1 | 0:30:00 | ~0:03 | 0.05 | Diagnostic perf (pool=none) |
| 5615937 | 2026-03-26 | 1 | 0:30:00 | ~0:03 | 0.05 | Diagnostic perf (pool=cuda) |
| 5616279 | 2026-03-26 | 1 | 1:00:00 | ~0:05 | 0.08 | Distributed 1 GPU |
| 5616280 | 2026-03-26 | 2 | 1:00:00 | ~0:02 | 0.07 | Distributed 2 GPU (FAILED: cuIpcGetMemHandle) |
| 5616281 | 2026-03-26 | 8 | 1:00:00 | ~0:02 | 0.27 | Distributed 8 GPU (FAILED: cuIpcGetMemHandle) |
| 5616391 | 2026-03-26 | 4 | 1:00:00 | ~0:05 | 0.33 | Distributed 4 GPU (pool=none) |
| 5616392 | 2026-03-26 | 2 | 1:00:00 | ~0:05 | 0.17 | Distributed 2 GPU (pool=none) |
| 5616393 | 2026-03-26 | 4 | 1:00:00 | ~0:05 | 0.33 | Distributed 4 GPU (pool=none) |
| 5616519 | 2026-03-26 | 1 | 0:30:00 | ~0:05 | 0.08 | Diag distributed 1 GPU (default pool) |
| 5616520 | 2026-03-26 | 1 | 0:30:00 | ~0:05 | 0.08 | Diag distributed 1 GPU (pool=none) |

**Total estimated GPU-hours used: ~2.9 hrs**

| 5627352 | 2026-03-26 | 1 | 0:30:00 | ~0:01 | 0.02 | Profile v1 (failed: missing import) |
| 5627368 | 2026-03-26 | 1 | 0:30:00 | ~0:01 | 0.02 | Profile v2 (failed: CPU→GPU broadcast) |
| 5627505 | 2026-03-26 | 1 | 0:30:00 | ~0:05 | 0.08 | Profile v3 (SUCCESS: found 92% unaccounted) |
| 5627521 | 2026-03-26 | 1 | 0:30:00 | ~0:05 | 0.08 | Profile v3b (duplicate, no pool=none) |

| 5627654 | 2026-03-26 | 1 | 0:30:00 | ~0:10 | 0.17 | Profile with fill_corners! fix (SUCCESS) |

| 5628432-34 | 2026-03-26 | 1-4 | 0:30:00 | ~0:05 | 0.25 | Pressure solver baseline (1,2,4 GPU) |
| 5628445 | 2026-03-26 | 2 | 0:30:00 | ~0:05 | 0.17 | Pressure solver component breakdown |
| 5628476-77 | 2026-03-26 | 1-2 | 0:30:00 | ~0:05 | 0.17 | Pressure solver Hyp A test |
| 5628534-36 | 2026-03-26 | 1-4 | 0:30:00 | ~0:05 | 0.25 | Pressure solver Hyp A+B test (2x improvement!) |

**Total estimated GPU-hours used: ~4.2 hrs** (excluding scaling suite jobs)

## Pending Jobs

| Job ID | Date | GPUs | Walltime (req) | GPU-hrs (req) | Description |
|--------|------|------|----------------|---------------|-------------|
| 5627796 | 2026-03-26 | 1 | 1:00:00 | 1.0 | Weak scaling 1 GPU (with fix) |
| 5627797 | 2026-03-26 | 2 | 1:00:00 | 2.0 | Weak scaling 2 GPUs |
| 5627798 | 2026-03-26 | 4 | 1:00:00 | 4.0 | Weak scaling 4 GPUs |
| 5627799 | 2026-03-26 | 8 | 1:00:00 | 8.0 | Weak scaling 8 GPUs (2 nodes) |
| 5627800 | 2026-03-26 | 16 | 1:00:00 | 16.0 | Weak scaling 16 GPUs (4 nodes) |
| 5627801 | 2026-03-26 | 32 | 1:00:00 | 32.0 | Weak scaling 32 GPUs (8 nodes) |
| 5627802 | 2026-03-26 | 64 | 1:00:00 | 64.0 | Weak scaling 64 GPUs (16 nodes) |

**Note:** Max potential cost 127 GPU-hrs if all run to walltime. Actual usage will be ~5-10% of this.
