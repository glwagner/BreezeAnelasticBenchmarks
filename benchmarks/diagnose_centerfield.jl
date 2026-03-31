# Minimal reproducer for CenterField(grid) hang on multi-node.
#
# Usage:
#   srun -n 8 --gpus 8 julia --project=. benchmarks/diagnose_centerfield.jl

using Logging
disable_logging(Logging.Warn)

using MPI
MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)

t0 = time()
ts() = "$(round(time()-t0, digits=1))s"
log(msg) = rank == 0 && (println("$(ts()): $msg"); flush(stdout))

using Oceananigans
using Oceananigans.Grids: topology, halo_size
using Oceananigans.Fields: validate_field_data, validate_indices, validate_boundary_conditions,
                           offset_data, new_data
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, construct_boundary_conditions_kernels
using Oceananigans.DistributedComputations: inject_halo_communication_boundary_conditions,
                                            communication_buffers
using CUDA

# Set Float type BEFORE grid creation
FT = Float32
Oceananigans.defaults.FloatType = FT

log("packages loaded ($nranks ranks)")

arch = Distributed(GPU(); partition = Partition(nranks, 1))
log("Distributed arch created")

grid = RectilinearGrid(arch,
                       size = (50 * nranks, 400, 80),
                       x = (0, 21000.0 * nranks),
                       y = (0, 168000.0),
                       z = (0, 20000.0),
                       halo = (1, 1, 1),
                       topology = (Periodic, Periodic, Bounded))

log("grid created (eltype=$(eltype(grid)))")

MPI.Barrier(MPI.COMM_WORLD)
log("barrier after grid")

loc = (Center(), Center(), Center())
indices = (Colon(), Colon(), Colon())

# Step 1: new_data (GPU allocation)
log("allocating data...")
data = new_data(FT, grid, loc, indices)
log("data allocated")

MPI.Barrier(MPI.COMM_WORLD)
log("barrier after data")

# Step 2: FieldBoundaryConditions with regularization (same path as CenterField)
log("creating regularized BCs...")
bcs = FieldBoundaryConditions(grid, loc, indices)
log("regularized BCs created")

MPI.Barrier(MPI.COMM_WORLD)
log("barrier after BCs")

# Step 3: inject_halo_communication_boundary_conditions
log("injecting halo communication BCs...")
new_bcs = inject_halo_communication_boundary_conditions(bcs, loc, arch.local_rank, arch.connectivity, topology(grid))
log("halo BCs injected")

MPI.Barrier(MPI.COMM_WORLD)
log("barrier after inject")

# Step 4: communication_buffers
log("creating communication buffers...")
offdata = offset_data(parent(data), grid, loc, indices)
buffers = communication_buffers(grid, offdata, new_bcs)
log("buffers created")

MPI.Barrier(MPI.COMM_WORLD)
log("barrier after buffers")

# Step 5: construct_boundary_conditions_kernels
log("constructing BC kernels...")
final_bcs = construct_boundary_conditions_kernels(new_bcs, offdata, grid, loc, indices)
log("BC kernels constructed")

MPI.Barrier(MPI.COMM_WORLD)
log("barrier after kernels")

# Step 6: Full CenterField (should be fast now — all types already compiled)
log("calling CenterField(grid)...")
f = CenterField(grid)
log("CenterField created!")

MPI.Barrier(MPI.COMM_WORLD)
log("ALL DONE!")
