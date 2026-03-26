using Breeze
using BreezeAnelasticBenchmarks
using Printf

FT = if "--float-type" in ARGS
    i = findfirst(==("--float-type"), ARGS)
    Dict("Float32" => Float32, "Float64" => Float64)[ARGS[i+1]]
else
    Float32
end

@show FT

model = setup_supercell(GPU(); FT,
                         Nx = 400, Ny = 400, Nz = 80)

@time run_benchmark!(model)
@time run_benchmark!(model)
@time run_benchmark!(model)
