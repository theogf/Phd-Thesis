using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using CairoMakie
using AbstractGPs
using AbstractGPsMakie
using Distributions
using ColorSchemes
using AugmentedGPLikelihoods
using ApproximateGPs
using LinearAlgebra

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
    # save(joinpath(fig_path, "GP_example.pdf"), fig)
    display(fig)
end

## 

n_data = 100
x = rand(n_data) * 3 .- 1.5
lik = BernoulliLikelihood()
f_latent = LatentGP(gp, lik, 1e-6)
f, y = rand(f_latent(x))

function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(m, S)))
end

function cavi!(fz::AbstractGPs.FiniteGP, x, y, m, S, qΩ; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    for _ in 1:niter
        post_u = u_posterior(fz, m, S)
        post_fs = marginals(post_u(x))
        aux_posterior!(qΩ, lik, y, post_fs)
        S .= inv(Symmetric(inv(K) + Diagonal(only(expected_auglik_precision(lik, qΩ, y)))))
        m .= S * (only(expected_auglik_potential(lik, qΩ, y)) - K \ mean(fz))
    end
    return m, S
end

m = zeros(n_data)
S = Matrix{Float64}(I(n_data))
qΩ = init_aux_posterior(lik, n_data)
fz = gp(x, 1e-8);

cavi!(fz, x, y, m, S, qΩ; niter=4);

q = u_posterior(fz, m, S)
##
fig = Figure(resolution=(900, 300))
with_theme(GP_theme, linewidth=20.0) do 
    g_lw = 1.0
    ax1 = Axis(fig[1, 1], title="Latent GP representation")
    scatter!(ax1, x, y, color=(sb[2], 0.5), label=L"y")
    plot!(ax1, x_test, q(x_test), color=(sb[1], 0.3), linewidth=g_lw, label=L"q(f)")
    plot!(ax1, x, f, color=sb[3], linewidth=g_lw, label=L"f_{\mathrm{true}}")
    axislegend(ax1)
    ax2 = Axis(fig[1, 2], title="p(y|f)")
    scatter!(ax2, x, y, color=(sb[2], 0.5), label=L"y")
    p_m =  mean(lik(mean(q(x_test))))
    lines!(ax2, x_test, p_m, color=sb[1], linewidth=5.0, label=L"E_{q(f)}[p(y|f)]")
    axislegend(ax2)
    save(joinpath(fig_path, "GP_classification_example.pdf"), fig)
    fig
end
