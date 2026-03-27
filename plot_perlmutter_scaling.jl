using CairoMakie

# ============================================================
# Derecho data: NT=100 steps, all optimized, 50×400×80/GPU, Ry=1
# ============================================================

# WENO5 (400×400×80/GPU, WENO5+Kessler, halo=5)
derecho_weno_gpus = [1,    2,     4,     8,     16,    20]
derecho_weno_ms   = [95.9, 462.2, 491.6, 666.6, 765.9, 803.3]

# ERF anelastic (50×400×80/GPU, Centered(2)+ScalarDiffusivity, halo=1)
derecho_erf_gpus  = [1,    2,     4,     8,     16,    20,    40]
derecho_erf_ms    = [51.2, 137.0, 140.5, 172.4, 180.6, 204.9, 233.3]

# Compressible (50×400×80/GPU, CompressibleDynamics, no Poisson, halo=1)
derecho_comp_gpus = [1,   2,    4,    8,    16,    20,    40]
derecho_comp_ms   = [4.4, 62.7, 72.0, 95.4, 110.1, 123.7, 137.4]

# ============================================================
# Perlmutter data: NT=10 steps, Trial 2
# ============================================================

# WENO5 (400×400×80/GPU, halo=5, x-only partition)
# Average of two runs (Trial 2 values: 62.8/64.1, 80.4/81.4, 85.2/88.3)
perlmutter_weno_gpus = [1,    2,     4]
perlmutter_weno_ms   = [63.5, 80.9,  86.8]

# ERF anelastic (50×400×80/GPU, Centered(2)+ScalarDiffusivity, halo=1, x-only)
# Average of two runs (Trial 2 values: 13.1/9.4, 19.6/20.0, 21.9/21.8)
perlmutter_erf_gpus  = [1,    2,     4]
perlmutter_erf_ms    = [11.3, 19.8,  21.9]

# TODO: Fill in multi-node results when jobs complete (pending in queue)
# perlmutter_weno_gpus = [1, 2, 4, 8, 16, 32]
# perlmutter_erf_gpus  = [1, 2, 4, 8, 16, 20, 40]

# ============================================================
# Figure: Two columns (Derecho | Perlmutter), two rows (ms/step | efficiency)
# ============================================================

fig = Figure(size = (1200, 900), fontsize = 14)

common_xticks = [1, 2, 4, 8, 16, 32, 64]
common_yticks = [1, 5, 10, 50, 100, 500, 1000]
colors = (weno = :purple, erf = :dodgerblue, comp = :seagreen)
markers = (weno = :diamond, erf = :circle, comp = :utriangle)

function plot_timing!(ax, weno_gpus, weno_ms, erf_gpus, erf_ms;
                      comp_gpus=nothing, comp_ms=nothing)
    scatter!(ax, weno_gpus, weno_ms, color=colors.weno, marker=markers.weno,
             markersize=12, label="WENO5 + Kessler (anelastic)")
    lines!(ax, weno_gpus, weno_ms, color=colors.weno, linewidth=2)

    scatter!(ax, erf_gpus, erf_ms, color=colors.erf, marker=markers.erf,
             markersize=12, label="Centered(2) + diffusion (anelastic)")
    lines!(ax, erf_gpus, erf_ms, color=colors.erf, linewidth=2)

    if !isnothing(comp_gpus)
        scatter!(ax, comp_gpus, comp_ms, color=colors.comp, marker=markers.comp,
                 markersize=12, label="Centered(2) + diffusion (compressible)")
        lines!(ax, comp_gpus, comp_ms, color=colors.comp, linewidth=2)
    end

    vlines!(ax, [4], color=:gray80, linestyle=:dashdot, linewidth=0.8)
    text!(ax, 4.3, minimum(filter(x -> x > 0, vcat(weno_ms, erf_ms))),
          text="1 node", fontsize=10, color=:gray50)
end

function plot_efficiency!(ax, weno_gpus, weno_ms, erf_gpus, erf_ms;
                          comp_gpus=nothing, comp_ms=nothing, baseline_gpus=8)

    hlines!(ax, [100], color=:gray70, linestyle=:dash, linewidth=1, label="Ideal")

    # Use the first GPU count >= baseline_gpus, or the last one if none qualify
    weno_base = findfirst(>=(baseline_gpus), weno_gpus)
    erf_base  = findfirst(>=(baseline_gpus), erf_gpus)
    weno_base = isnothing(weno_base) ? length(weno_gpus) : weno_base
    erf_base  = isnothing(erf_base)  ? length(erf_gpus)  : erf_base

    weno_eff = 100 .* weno_ms[weno_base] ./ weno_ms
    scatter!(ax, weno_gpus, weno_eff, color=colors.weno, marker=markers.weno,
             markersize=12, label="WENO5 anelastic")
    lines!(ax, weno_gpus, weno_eff, color=colors.weno, linewidth=2)

    erf_eff = 100 .* erf_ms[erf_base] ./ erf_ms
    scatter!(ax, erf_gpus, erf_eff, color=colors.erf, marker=markers.erf,
             markersize=12, label="ERF-like anelastic")
    lines!(ax, erf_gpus, erf_eff, color=colors.erf, linewidth=2)

    if !isnothing(comp_gpus)
        comp_base = findfirst(>=(baseline_gpus), comp_gpus)
        comp_base = isnothing(comp_base) ? length(comp_gpus) : comp_base
        comp_eff = 100 .* comp_ms[comp_base] ./ comp_ms
        scatter!(ax, comp_gpus, comp_eff, color=colors.comp, marker=markers.comp,
                 markersize=12, label="Compressible")
        lines!(ax, comp_gpus, comp_eff, color=colors.comp, linewidth=2)
    end

    vlines!(ax, [4], color=:gray80, linestyle=:dashdot, linewidth=0.8)
end

# --- Derecho (left column) ---
ax1 = Axis(fig[1, 1],
           ylabel = "Time per step (ms)",
           title = "Derecho (A100-40GB)",
           xscale = log2, yscale = log10,
           xticks = common_xticks, yticks = common_yticks,
           xminorticksvisible = false, yminorticksvisible = false)

plot_timing!(ax1, derecho_weno_gpus, derecho_weno_ms,
             derecho_erf_gpus, derecho_erf_ms;
             comp_gpus=derecho_comp_gpus, comp_ms=derecho_comp_ms)
axislegend(ax1, position=:lt, framevisible=true, labelsize=11)

ax3 = Axis(fig[2, 1],
           xlabel = "GPUs",
           ylabel = "Weak scaling efficiency (%)",
           title = "Efficiency vs 2 nodes (8 GPUs)",
           xscale = log2,
           xticks = common_xticks,
           xminorticksvisible = false,
           limits = (nothing, (0, 110)))

plot_efficiency!(ax3, derecho_weno_gpus, derecho_weno_ms,
                 derecho_erf_gpus, derecho_erf_ms;
                 comp_gpus=derecho_comp_gpus, comp_ms=derecho_comp_ms)
axislegend(ax3, position=:rt, framevisible=true, labelsize=11)

# --- Perlmutter (right column) ---
ax2 = Axis(fig[1, 2],
           title = "Perlmutter (A100-80GB)",
           xscale = log2, yscale = log10,
           xticks = common_xticks, yticks = common_yticks,
           xminorticksvisible = false, yminorticksvisible = false)

plot_timing!(ax2, perlmutter_weno_gpus, perlmutter_weno_ms,
             perlmutter_erf_gpus, perlmutter_erf_ms)
axislegend(ax2, position=:lt, framevisible=true, labelsize=11)

ax4 = Axis(fig[2, 2],
           xlabel = "GPUs",
           title = "Efficiency vs 1 node (4 GPUs)",
           xscale = log2,
           xticks = common_xticks,
           xminorticksvisible = false,
           limits = (nothing, (0, 110)))

plot_efficiency!(ax4, perlmutter_weno_gpus, perlmutter_weno_ms,
                 perlmutter_erf_gpus, perlmutter_erf_ms;
                 baseline_gpus=4)
axislegend(ax4, position=:rt, framevisible=true, labelsize=11)

save("scaling_comparison.png", fig, px_per_unit = 3)
println("Saved scaling_comparison.png")
