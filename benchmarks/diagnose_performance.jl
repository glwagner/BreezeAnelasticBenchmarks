using CUDA
using Printf

println("=== GPU & CUDA Info ===")
println("CUDA runtime: ", CUDA.runtime_version())
println("CUDA driver:  ", CUDA.driver_version())
println("Device:       ", CUDA.name(CUDA.device()))
println("Capability:   ", CUDA.capability(CUDA.device()))
println("Memory pool:  ", get(ENV, "JULIA_CUDA_MEMORY_POOL", "default (binned)"))
println("Julia version:", VERSION)
println()

# Quick allocation benchmark to test memory pool impact
println("=== Memory allocation benchmark ===")
# Warmup
for _ in 1:10
    x = CUDA.rand(Float32, 100, 100)
    CUDA.unsafe_free!(x)
end
CUDA.synchronize()

# Time many small allocations (sensitive to memory pool)
t_alloc = @elapsed begin
    for _ in 1:1000
        x = CUDA.rand(Float32, 400, 400)
        CUDA.unsafe_free!(x)
    end
    CUDA.synchronize()
end
@printf("1000 small allocations: %.3f s\n", t_alloc)

# Time a few large allocations
t_large = @elapsed begin
    for _ in 1:100
        x = CUDA.rand(Float32, 400, 400, 80)
        CUDA.unsafe_free!(x)
    end
    CUDA.synchronize()
end
@printf("100 large allocations:  %.3f s\n", t_large)
println()

# Now run the actual benchmark
using BreezeAnelasticBenchmarks

println("=== Benchmark ===")
arch = GPU()
model = setup_supercell(arch)

# Warmup
elapsed_warmup = @elapsed run_benchmark!(model)
CUDA.synchronize()
@printf("Warmup:  %.3f s\n", elapsed_warmup)

elapsed1 = @elapsed run_benchmark!(model)
CUDA.synchronize()
@printf("Trial 1: %.3f s\n", elapsed1)

elapsed2 = @elapsed run_benchmark!(model)
CUDA.synchronize()
@printf("Trial 2: %.3f s\n", elapsed2)

elapsed3 = @elapsed run_benchmark!(model)
CUDA.synchronize()
@printf("Trial 3: %.3f s\n", elapsed3)
