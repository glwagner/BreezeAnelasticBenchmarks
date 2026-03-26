using CairoMakie

# ============================================================
# Data: NT=100 steps, all optimized, 50×400×80/GPU, Ry=1 (x-only)
# ============================================================

# WENO5 (400×400×80/GPU, WENO5+Kessler, halo=5)
weno_gpus  = [1,    2,     4,     8,     16,    20]
weno_ms    = [95.9, 462.2, 491.6, 666.6, 765.9, 803.3]

# ERF anelastic (50×400×80/GPU, Centered(2)+ScalarDiffusivity, halo=1)
erf_gpus   = [1,    2,     4,     8,     16,    20,    40]
erf_ms     = [51.2, 137.0, 140.5, 172.4, 180.6, 204.9, 233.3]

# Compressible (50×400×80/GPU, CompressibleDynamics, no Poisson, halo=1)
comp_gpus  = [1,   2,    4,    8,    16,    20,    40]
comp_ms    = [4.4, 62.7, 72.0, 95.4, 110.1, 123.7, 137.4]

# ============================================================
# Figure
# ============================================================

fig = Figure(size = (1000, 500), fontsize = 14)

# --- Left panel: ms/step vs GPUs ---
ax1 = Axis(fig[1, 1],
           xlabel = "GPUs",
           ylabel = "Time per step (ms)",
           title = "Weak Scaling on Derecho A100s (50×400×80/GPU)",
           xscale = log2,
           yscale = log10,
           xticks = [1, 2, 4, 8, 16, 20, 40],
           yticks = [1, 5, 10, 50, 100, 500, 1000],
           xminorticksvisible = false,
           yminorticksvisible = false)

scatter!(ax1, weno_gpus, weno_ms, color = :purple, marker = :diamond, markersize = 12,
         label = "WENO5 + Kessler (anelastic)")
lines!(ax1, weno_gpus, weno_ms, color = :purple, linewidth = 2)

scatter!(ax1, erf_gpus, erf_ms, color = :dodgerblue, marker = :circle, markersize = 12,
         label = "Centered(2) + diffusion (anelastic)")
lines!(ax1, erf_gpus, erf_ms, color = :dodgerblue, linewidth = 2)

scatter!(ax1, comp_gpus, comp_ms, color = :seagreen, marker = :utriangle, markersize = 12,
         label = "Centered(2) + diffusion (compressible)")
lines!(ax1, comp_gpus, comp_ms, color = :seagreen, linewidth = 2)

vlines!(ax1, [4], color = :gray80, linestyle = :dashdot, linewidth = 0.8)
text!(ax1, 4.3, 2, text = "1 node", fontsize = 10, color = :gray50)
vlines!(ax1, [8], color = :gray80, linestyle = :dashdot, linewidth = 0.8)
text!(ax1, 8.5, 2, text = "2 nodes", fontsize = 10, color = :gray50)

axislegend(ax1, position = :lt, framevisible = true, labelsize = 11)

# --- Right panel: Efficiency ---
ax2 = Axis(fig[1, 2],
           xlabel = "GPUs",
           ylabel = "Weak scaling efficiency (%)",
           title = "Efficiency vs 1 GPU",
           xscale = log2,
           xticks = [1, 2, 4, 8, 16, 20, 40],
           xminorticksvisible = false,
           limits = (nothing, (0, 110)))

hlines!(ax2, [100], color = :gray70, linestyle = :dash, linewidth = 1, label = "Ideal")

weno_eff = 100 .* weno_ms[1] ./ weno_ms
scatter!(ax2, weno_gpus, weno_eff, color = :purple, marker = :diamond, markersize = 12,
         label = "WENO5 anelastic")
lines!(ax2, weno_gpus, weno_eff, color = :purple, linewidth = 2)

erf_eff = 100 .* erf_ms[1] ./ erf_ms
scatter!(ax2, erf_gpus, erf_eff, color = :dodgerblue, marker = :circle, markersize = 12,
         label = "ERF-like anelastic")
lines!(ax2, erf_gpus, erf_eff, color = :dodgerblue, linewidth = 2)

comp_eff = 100 .* comp_ms[1] ./ comp_ms
scatter!(ax2, comp_gpus, comp_eff, color = :seagreen, marker = :utriangle, markersize = 12,
         label = "Compressible")
lines!(ax2, comp_gpus, comp_eff, color = :seagreen, linewidth = 2)

vlines!(ax2, [4], color = :gray80, linestyle = :dashdot, linewidth = 0.8)
vlines!(ax2, [8], color = :gray80, linestyle = :dashdot, linewidth = 0.8)

axislegend(ax2, position = :rt, framevisible = true, labelsize = 11)

save("scaling_results.png", fig, px_per_unit = 3)
println("Saved scaling_results.png")
