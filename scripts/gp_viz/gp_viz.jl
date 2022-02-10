using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using CairoMakie
using AbstractGPs
using AbstractGPsMakie
using Distributions
using ColorSchemes
sb = ColorSchemes.seaborn_colorblind

n_data = 10
x = rand(n_data) * 3 .- 1.5
x_test = -2.5:0.01:2.5
g(x) = sinc(x)
σ = 0.05
y = g.(x) .+ randn(n_data) * σ

fig = Figure(resolution=(900, 300))

k = with_lengthscale(Matern32Kernel(), 1.0)
gp = GP(k)
f = gp(x, σ * 0.1)
f_post = posterior(f, y)

GP_theme = Theme(palette=sb)
fig_path = joinpath(@__DIR__, "..", "..", "thesis", "chapters", "2_background", "figures")

with_theme(GP_theme, linewidth=25.0) do
    g_lw = 1.0
    ax1 = Axis(fig[1, 1], title="GP Prior")
    plot!(ax1, x_test, g, linewidth=g_lw, label="sinc(x)")
    plot!(ax1, x_test, gp, color=(sb[1], 0.3), label="f")

    ax2 = Axis(fig[1, 2], title="GP Posterior")
    plot!(ax2, x_test, g, linewidth=g_lw, label="sinc(x)")
    
    plot!(ax2, x_test, f_post, color=(sb[1], 0.3), linewidth=3.5)
    scatter!(ax2, x, y, color=sb[2], label="observations")
    linkyaxes!(ax1, ax2)
    tightlimits!(ax1)
    tightlimits!(ax2)
    save(joinpath(fig_path, "GP_example.pdf"), fig)
    display(fig)
end
