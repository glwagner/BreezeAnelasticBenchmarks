module BreezeAnelasticBenchmarks

using PrecompileTools

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula

using Oceananigans
using Oceananigans.Units

using CUDA
using MPI

export setup_supercell, setup_supercell_erf, setup_supercell_compressible, run_benchmark!,
       GPU, CPU, Distributed, Partition


"""
    setup_supercell(arch; kw...)

Build a DCMIP2016 supercell `AtmosphereModel` on architecture `arch`.

Keyword arguments
=================
- `FT = Float32`: floating-point type
- `Nx, Ny, Nz = 400, 400, 80`: grid points
- `Lx, Ly, Lz = 168km, 168km, 20km`: domain extent
"""
function setup_supercell(arch;
                         FT = Float32,
                         Nx = 400, Ny = 400, Nz = 80,
                         Lx = 168kilometers, Ly = 168kilometers, Lz = 20kilometers)

    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(arch,
                           size = (Nx, Ny, Nz),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (0, Lz),
                           halo = (5, 5, 5),
                           topology = (Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

    reference_state = ReferenceState(grid, constants,
                                     surface_pressure = 100000,
                                     potential_temperature = 300)

    dynamics = AnelasticDynamics(reference_state)

    # DCMIP2016 supercell background profiles
    g  = constants.gravitational_acceleration
    cpd = constants.dry_air.heat_capacity

    theta0 = 300       # K - surface potential temperature
    thetap = 343       # K - tropopause potential temperature
    zp     = 12000     # m - tropopause height
    Tp     = 213       # K - tropopause temperature
    zs     = 5kilometers  # shear layer height
    us     = 30        # m/s - maximum shear wind speed
    uc     = 15        # m/s - storm motion (Galilean translation speed)

    function theta_background(z)
        thetat = theta0 + (thetap - theta0) * (z / zp)^(5/4)
        thetas = thetap * exp(g / (cpd * Tp) * (z - zp))
        return (z <= zp) * thetat + (z > zp) * thetas
    end

    H_background(z) = (1 - 3/4 * (z / zp)^(5/4)) * (z <= zp) + 1/4 * (z > zp)

    function u_background(z)
        ul = us * (z / zs) - uc
        ut = (-4/5 + 3 * (z / zs) - 5/4 * (z / zs)^2) * us - uc
        uu = us - uc
        return (z < (zs - 1000)) * ul +
               (abs(z - zs) <= 1000) * ut +
               (z > (zs + 1000)) * uu
    end

    dtheta = 3              # K - perturbation amplitude
    rbh    = 10kilometers   # bubble horizontal radius
    rbv    = 1500           # m - bubble vertical radius
    zb     = 1500           # m - bubble center height
    xb     = Lx / 2
    yb     = Ly / 2

    function theta_init(x, y, z)
        theta_bar = theta_background(z)
        r = sqrt((x - xb)^2 + (y - yb)^2)
        R = sqrt((r / rbh)^2 + ((z - zb) / rbv)^2)
        theta_prime = ifelse(R < 1, dtheta * cos((pi / 2) * R)^2, 0.0)
        return theta_bar + theta_prime
    end

    u_init(x, y, z) = u_background(z)
    H_init(x, y, z) = H_background(z)

    microphysics = DCMIP2016KesslerMicrophysics()
    advection = WENO(order=5)
    model = AtmosphereModel(grid; dynamics, microphysics, advection,
                            thermodynamic_constants = constants)

    set!(model, θ=theta_init, ℋ=H_init, u=u_init)

    return model
end

"""
    setup_supercell_erf(arch; kw...)

ERF-equivalent supercell benchmark: Centered(2) advection, ScalarDiffusivity,
smaller per-GPU grid (200×200×80), halo=(1,1,1).

Designed to match ERF weak scaling test configuration for comparison.
"""
function setup_supercell_erf(arch;
                              FT = Float32,
                              Nx = 200, Ny = 200, Nz = 80,
                              Lx = 84kilometers, Ly = 84kilometers, Lz = 20kilometers)

    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(arch,
                           size = (Nx, Ny, Nz),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (0, Lz),
                           halo = (1, 1, 1),
                           topology = (Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

    reference_state = ReferenceState(grid, constants,
                                     surface_pressure = 100000,
                                     potential_temperature = 300)

    dynamics = AnelasticDynamics(reference_state)

    # Same DCMIP2016 background profiles as setup_supercell
    g  = constants.gravitational_acceleration
    cpd = constants.dry_air.heat_capacity

    theta0 = 300
    thetap = 343
    zp     = 12000
    Tp     = 213
    zs     = 5kilometers
    us     = 30
    uc     = 15

    function theta_background(z)
        thetat = theta0 + (thetap - theta0) * (z / zp)^(5/4)
        thetas = thetap * exp(g / (cpd * Tp) * (z - zp))
        return (z <= zp) * thetat + (z > zp) * thetas
    end

    H_background(z) = (1 - 3/4 * (z / zp)^(5/4)) * (z <= zp) + 1/4 * (z > zp)

    function u_background(z)
        ul = us * (z / zs) - uc
        ut = (-4/5 + 3 * (z / zs) - 5/4 * (z / zs)^2) * us - uc
        uu = us - uc
        return (z < (zs - 1000)) * ul +
               (abs(z - zs) <= 1000) * ut +
               (z > (zs + 1000)) * uu
    end

    dtheta = 3
    rbh    = 10kilometers
    rbv    = 1500
    zb     = 1500
    xb     = Lx / 2
    yb     = Ly / 2

    function theta_init(x, y, z)
        theta_bar = theta_background(z)
        r = sqrt((x - xb)^2 + (y - yb)^2)
        R = sqrt((r / rbh)^2 + ((z - zb) / rbv)^2)
        theta_prime = ifelse(R < 1, dtheta * cos((pi / 2) * R)^2, 0.0)
        return theta_bar + theta_prime
    end

    u_init(x, y, z) = u_background(z)
    H_init(x, y, z) = H_background(z)

    microphysics = DCMIP2016KesslerMicrophysics()
    advection = Centered(order=2)
    closure = ScalarDiffusivity(ν=200, κ=200)
    model = AtmosphereModel(grid; dynamics, microphysics, advection, closure,
                            thermodynamic_constants = constants)

    set!(model, θ=theta_init, ℋ=H_init, u=u_init)

    return model
end

"""
    setup_supercell_compressible(arch; kw...)

Compressible dynamics supercell benchmark: fully explicit time stepping,
Centered(2) advection, ScalarDiffusivity, no Poisson pressure solve.
Designed for ERF weak scaling comparison (no global communication in pressure).
"""
function setup_supercell_compressible(arch;
                                      FT = Float32,
                                      Nx = 200, Ny = 200, Nz = 80,
                                      Lx = 84kilometers, Ly = 84kilometers, Lz = 20kilometers)

    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(arch,
                           size = (Nx, Ny, Nz),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (0, Lz),
                           halo = (1, 1, 1),
                           topology = (Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

    g  = constants.gravitational_acceleration
    cpd = constants.dry_air.heat_capacity

    theta0 = 300
    thetap = 343
    zp     = 12000
    Tp     = 213
    zs     = 5kilometers
    us     = 30
    uc     = 15

    function theta_background(z)
        thetat = theta0 + (thetap - theta0) * (z / zp)^(5/4)
        thetas = thetap * exp(g / (cpd * Tp) * (z - zp))
        return (z <= zp) * thetat + (z > zp) * thetas
    end

    # Fully explicit compressible dynamics (no pressure solve)
    dynamics = CompressibleDynamics(;
        surface_pressure = 100000,
        reference_potential_temperature = theta_background)

    function u_background(z)
        ul = us * (z / zs) - uc
        ut = (-4/5 + 3 * (z / zs) - 5/4 * (z / zs)^2) * us - uc
        uu = us - uc
        return (z < (zs - 1000)) * ul +
               (abs(z - zs) <= 1000) * ut +
               (z > (zs + 1000)) * uu
    end

    dtheta = 3
    rbh    = 10kilometers
    rbv    = 1500
    zb     = 1500
    xb     = Lx / 2
    yb     = Ly / 2

    function theta_init(x, y, z)
        theta_bar = theta_background(z)
        r = sqrt((x - xb)^2 + (y - yb)^2)
        R = sqrt((r / rbh)^2 + ((z - zb) / rbv)^2)
        theta_prime = ifelse(R < 1, dtheta * cos((pi / 2) * R)^2, 0.0)
        return theta_bar + theta_prime
    end

    u_init(x, y, z) = u_background(z)

    advection = Centered(order=2)
    closure = ScalarDiffusivity(ν=200, κ=200)
    model = AtmosphereModel(grid; dynamics, advection, closure,
                            thermodynamic_constants = constants)

    # Initialize density from reference state, set θ and u
    ref = model.dynamics.reference_state
    set!(model, θ=theta_init, u=u_init, ρ=ref.density)

    return model
end

"""
    run_benchmark!(model, Nt=10, Δt=0.1)

Take `Nt` time steps of size `Δt` and return `nothing`.
"""
function run_benchmark!(model, Nt=10, Δt=0.1)
    for n = 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

# ---- Precompilation workloads ------------------------------------------------
#
# All three configs (WENO, ERF, compressible) × all architectures
# (GPU, Distributed, NCCLDistributed) are precompiled here.
#
# Run with 2+ GPUs under MPI to cache everything:
#   srun -n 2 --gpus 2 julia --project -e 'using MPI; MPI.Init(); using Pkg; Pkg.precompile()'

function precompile_all_configs!(arch; Nx, Ny, Nz, Lx, Ly, Lz)
    run_benchmark!(setup_supercell(arch; Nx, Ny, Nz, Lx, Ly, Lz), 1)
    run_benchmark!(setup_supercell_erf(arch; Nx, Ny, Nz, Lx, Ly, Lz), 1)
    run_benchmark!(setup_supercell_compressible(arch; Nx, Ny, Nz, Lx, Ly, Lz), 1)
end

@setup_workload begin
    N = 8
    Lx = 168_000.0
    Ly = 168_000.0
    Lz = 20_000.0

    @compile_workload begin
        is_rank0 = !MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0

        kw = (; Nx=N, Ny=N, Nz=N, Lx, Ly, Lz)

        # Serial GPU workload (rank 0 only)
        if is_rank0 && CUDA.functional()
            precompile_all_configs!(GPU(); kw...)
        elseif is_rank0
            run_benchmark!(setup_supercell(CPU(); kw...), 1)
        end

        # Distributed workloads (all ranks)
        if MPI.Initialized() && CUDA.functional()
            Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
            dist_kw = (; Nx = N * Ngpus, Ny=N, Nz=N, Lx = Lx * Ngpus, Ly, Lz)

            # MPI Distributed
            mpi_arch = Distributed(GPU(); partition = Partition(Ngpus, 1))
            precompile_all_configs!(mpi_arch; dist_kw...)
            MPI.Barrier(MPI.COMM_WORLD)

            # NCCL Distributed
            try
                using NCCL
                ext = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
                if ext !== nothing
                    nccl_arch = ext.NCCLDistributed(GPU(); partition = Partition(Ngpus, 1))
                    precompile_all_configs!(nccl_arch; dist_kw...)
                    MPI.Barrier(MPI.COMM_WORLD)
                end
            catch
            end
        end
    end
end

end # module
