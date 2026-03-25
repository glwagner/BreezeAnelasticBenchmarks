using MPI
MPI.Init()

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znodes

using CUDA
using Printf

FT = if "--float-type" in ARGS
    i = findfirst(==("--float-type"), ARGS)
    Dict("Float32" => Float32, "Float64" => Float64)[ARGS[i+1]]
else
    Float32
end

Oceananigans.defaults.FloatType = FT

## Weak scaling: each GPU gets 400×400×80 grid points.
## X-decomposition only: total Nx = 400 * Ngpus.
Ngpus = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
arch = Distributed(GPU(); partition=Partition(Ngpus, 1))

Nx_per_gpu = 400
Ny, Nz = 400, 80
Lx_per_gpu = 168kilometers
Ly, Lz = 168kilometers, 20kilometers

Nx = Nx_per_gpu * Ngpus
Lx = Lx_per_gpu * Ngpus

if rank == 0
    @info "Weak scaling benchmark" Ngpus FT Nx Ny Nz
end

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

θ₀ = 300       # K - surface potential temperature
θᵖ = 343       # K - tropopause potential temperature
zᵖ = 12000     # m - tropopause height
Tᵖ = 213       # K - tropopause temperature
zˢ = 5kilometers  # m - shear layer height
uˢ = 30           # m/s - maximum shear wind speed
uᶜ = 15           # m/s - storm motion (Galilean translation speed)

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

function θ_background(z)
    θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)
    θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    return (z <= zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

ℋ_background(z) = (1 - 3/4 * (z / zᵖ)^(5/4)) * (z <= zᵖ) + 1/4 * (z > zᵖ)

function u_background(z)
    uˡ = uˢ * (z / zˢ) - uᶜ
    uᵗ = (-4/5 + 3 * (z / zˢ) - 5/4 * (z / zˢ)^2) * uˢ - uᶜ
    uᵘ = uˢ - uᶜ
    return (z < (zˢ - 1000)) * uˡ +
           (abs(z - zˢ) <= 1000) * uᵗ +
           (z > (zˢ + 1000)) * uᵘ
end

Δθ = 3              # K - perturbation amplitude
rᵇʰ = 10kilometers  # m - bubble horizontal radius
rᵇᵛ = 1500          # m - bubble vertical radius
zᵇ = 1500           # m - bubble center height
xᵇ = Lx / 2         # m - bubble center x-coordinate
yᵇ = Ly / 2         # m - bubble center y-coordinate

function θᵢ(x, y, z)
    θ̄ = θ_background(z)
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ′ = ifelse(R < 1, Δθ * cos((π / 2) * R)^2, 0.0)
    return θ̄ + θ′
end

uᵢ(x, y, z) = u_background(z)

microphysics = DCMIP2016KesslerMicrophysics()
advection = WENO(order=5)
model = AtmosphereModel(grid; dynamics, microphysics, advection, thermodynamic_constants=constants)

ℋᵢ(x, y, z) = ℋ_background(z)
set!(model, θ=θᵢ, ℋ=ℋᵢ, u=uᵢ)

function many_time_steps!(model, Nt=10)
    for n = 1:Nt
        time_step!(model, 0.1)
    end
    return nothing
end

MPI.Barrier(MPI.COMM_WORLD)

## Warmup (includes compilation)
elapsed₁ = @elapsed many_time_steps!(model)
MPI.Barrier(MPI.COMM_WORLD)

elapsed₂ = @elapsed many_time_steps!(model)
MPI.Barrier(MPI.COMM_WORLD)

elapsed₃ = @elapsed many_time_steps!(model)
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    @info @sprintf("Warmup:  %.3f seconds", elapsed₁)
    @info @sprintf("Trial 1: %.3f seconds", elapsed₂)
    @info @sprintf("Trial 2: %.3f seconds", elapsed₃)
end
