using CairoMakie

# ============================================================
# Data
# ============================================================

# 1-GPU reference at 400×400×80 (ERF 2-node equivalent)
t_1gpu_400 = 0.614  # seconds for 10 steps

# Ry=1 (x-only partition, 50×400×80 per GPU)
ry1_gpus   = [1,   2,     4,     8,     16,    20,    40]
ry1_nodes  = [1,   1,     1,     2,     4,     5,     10]
ry1_time   = [0.547, 1.313, 0.965, 1.855, 1.707, 2.197, 2.363]

# Ry=2 (100×200×80 per GPU)
ry2_gpus   = [2,     4,     8,     16,    32]
ry2_nodes  = [1,     1,     2,     4,     8]
ry2_time   = [0.774, 1.375, 1.606, 2.031, 2.918]

# WENO5 (400×400×80 per GPU, x-only, pre-optimization baseline)
weno_gpus  = [1,   2,     4,     8,     16,    20]
weno_time  = [0.940, 7.711, 6.743, 7.714, 10.291, 11.526]

# ============================================================
# Figure
# ============================================================

fig = Figure(size = (1000, 500), fontsize = 14)

# --- Left panel: Time vs GPUs ---
ax1 = Axis(fig[1, 1],
           xlabel = "GPUs",
           ylabel = "Time for 10 steps (s)",
           title = "Weak Scaling: Anelastic Supercell on Derecho A100s",
           xscale = log2,
           yscale = log2,
           xticks = [1, 2, 4, 8, 16, 32, 40],
           yticks = [0.5, 1, 2, 4, 8],
           xminorticksvisible = false,
           yminorticksvisible = false)

# Ideal weak scaling line (flat)
hlines!(ax1, [t_1gpu_400], color = :gray70, linestyle = :dash, linewidth = 1,
        label = "Ideal (1-GPU 400×400×80)")

# WENO5 baseline (pre-optimization)
scatter!(ax1, weno_gpus, weno_time, color = :gray50, marker = :diamond, markersize = 10,
         label = "WENO5 baseline (pre-opt)")
lines!(ax1, weno_gpus, weno_time, color = :gray50, linestyle = :dot, linewidth = 1)

# ERF Ry=1 (optimized)
scatter!(ax1, ry1_gpus, ry1_time, color = :dodgerblue, marker = :circle, markersize = 12,
         label = "ERF Ry=1 (50×400×80/GPU)")
lines!(ax1, ry1_gpus, ry1_time, color = :dodgerblue, linewidth = 2)

# ERF Ry=2 (optimized)
scatter!(ax1, ry2_gpus, ry2_time, color = :orangered, marker = :utriangle, markersize = 12,
         label = "ERF Ry=2 (100×200×80/GPU)")
lines!(ax1, ry2_gpus, ry2_time, color = :orangered, linewidth = 2)

axislegend(ax1, position = :lt, framevisible = true, labelsize = 11)

# Mark node boundaries
vlines!(ax1, [4], color = :gray80, linestyle = :dashdot, linewidth = 0.8)
text!(ax1, 4.3, 0.55, text = "1 node", fontsize = 10, color = :gray50)
vlines!(ax1, [8], color = :gray80, linestyle = :dashdot, linewidth = 0.8)
text!(ax1, 8.5, 0.55, text = "2 nodes", fontsize = 10, color = :gray50)

# --- Right panel: Efficiency ---
ax2 = Axis(fig[1, 2],
           xlabel = "GPUs",
           ylabel = "Weak scaling efficiency (%)",
           title = "Efficiency vs 1 GPU",
           xscale = log2,
           xticks = [1, 2, 4, 8, 16, 32, 40],
           xminorticksvisible = false,
           limits = (nothing, (0, 110)))

# Ideal
hlines!(ax2, [100], color = :gray70, linestyle = :dash, linewidth = 1, label = "Ideal")

# ERF Ry=1
ry1_eff = 100 .* ry1_time[1] ./ ry1_time
scatter!(ax2, ry1_gpus, ry1_eff, color = :dodgerblue, marker = :circle, markersize = 12,
         label = "ERF Ry=1")
lines!(ax2, ry1_gpus, ry1_eff, color = :dodgerblue, linewidth = 2)

# ERF Ry=2 — use 2-GPU Ry=2 as 100% reference
ry2_eff = 100 .* ry2_time[1] ./ ry2_time
scatter!(ax2, ry2_gpus, ry2_eff, color = :orangered, marker = :utriangle, markersize = 12,
         label = "ERF Ry=2")
lines!(ax2, ry2_gpus, ry2_eff, color = :orangered, linewidth = 2)

# WENO5
weno_eff = 100 .* weno_time[1] ./ weno_time
scatter!(ax2, weno_gpus, weno_eff, color = :gray50, marker = :diamond, markersize = 10,
         label = "WENO5 baseline")
lines!(ax2, weno_gpus, weno_eff, color = :gray50, linestyle = :dot, linewidth = 1)

axislegend(ax2, position = :rt, framevisible = true, labelsize = 11)

vlines!(ax2, [4], color = :gray80, linestyle = :dashdot, linewidth = 0.8)
vlines!(ax2, [8], color = :gray80, linestyle = :dashdot, linewidth = 0.8)

save("scaling_results.png", fig, px_per_unit = 3)
println("Saved scaling_results.png")
