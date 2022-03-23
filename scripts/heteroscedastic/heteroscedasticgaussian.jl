# Heteroscedastic Gaussian Regression

# We load all the necessary packages
using AbstractGPs
using ApproximateGPs
using ArraysOfArrays
using AugmentedGPLikelihoods
using Distributions
using LinearAlgebra
using ColorSchemes
using SplitApplyCombine
using CairoMakie
using LaTeXStrings
using LogExpFunctions
using AbstractGPsMakie
using Random: seed!

# We create some random data (sorted for plotting reasons)
seed!(45)

N = 100
x = collect(range(-10, 10; length=N))
λ = 3.0
lik = HeteroscedasticGaussianLikelihood(InvScaledLogistic(3.0))
SplitApplyCombine.invert(x::ArrayOfSimilarArrays) = nestedview(flatview(x)')
X = MOInput(x, nlatent(lik))
kernel = 5.0 * with_lengthscale(SqExponentialKernel(), 2.0)
gpf = GP(kernel)
μ₀g = -3.0
gpg = GP(μ₀g, kernel)
gps = [gpf, gpg]
gpm = GP(IndependentMOKernel(kernel))
fs = rand(gpm(X, 1e-6))
fs = nestedview(reduce(hcat, Iterators.partition(fs, N)))
fs[2] .+= μ₀g
py = lik(invert(fs))
y = rand(py)

# We write our CAVI algorithmm
function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(copy(m), S)))
end

function opt_lik(lik::HeteroscedasticGaussianLikelihood, (qf, qg)::AbstractVector{<:AbstractVector{<:Normal}}, y::AbstractVector{<:Real})
    ψ = AugmentedGPLikelihoods.second_moment.(qf .- y) / 2
    c = sqrt.(AugmentedGPLikelihoods.second_moment.(qg))
    σ̃g = AugmentedGPLikelihoods.approx_expected_logistic.(-mean.(qg), c)
    λ = length(y) / (dot(ψ, 1 .- σ̃g))
    return HeteroscedasticGaussianLikelihood(InvScaledLogistic(λ))
end

function cavi!(fzs, x, y, ms, Ss, qΩ, lik; niter=10)
    K = ApproximateGPs._chol_cov(fzs[1])
    for _ in 1:niter
        posts_u = u_posterior.(fzs, ms, Ss)
        posts_fs = marginals.([p_u(x) for p_u in posts_u])
        lik = opt_lik(lik, posts_fs, y)
        aux_posterior!(qΩ, lik, y, invert(posts_fs))
        ηs, Λs = expected_auglik_potential_and_precision(lik, qΩ, y, last.(invert(posts_fs)))
        Ss .= inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(Λs)))
        ms .= Ss .* (ηs .+ Ref(K) .\ mean.(fzs))
    end
    return lik
end
# Now we just initialize the variational parameters
ms = nestedview(zeros(N, nlatent(lik)))
Ss = [Matrix{Float64}(I(N)) for _ in 1:nlatent(lik)]
qΩ = init_aux_posterior(lik, N)
fzs = [gpf(x, 1e-8), gpg(x, 1e-8)];
x_te = -10:0.01:10
# We run CAVI for 3-4 iterations
new_lik = cavi!(fzs, x, y, ms, Ss, qΩ, lik; niter=20);
# And visualize the obtained variational posterior
f_te = u_posterior(fzs[1], ms[1], Ss[1])(x_te)
g_te = u_posterior(fzs[2], ms[2], Ss[2])(x_te)

# Gibbs Sampling
const α = 0.3
const β = 0.1
function gibbs_sample(fzs, fs, Ω; nsamples=200)
    K = ApproximateGPs._chol_cov(fzs[1])
    Σ = [zeros(N, N) for _ in 1:nlatent(lik)]
    μ = [zeros(N) for _ in 1:nlatent(lik)]
    return map(1:nsamples) do _
        λ = rand(Gamma(α + length(y) / 2, inv(β + sum(
            logistic.(fs[2]) .* abs2.(fs[1] - y)
            ) / 2)))
        lik = HeteroscedasticGaussianLikelihood(InvScaledLogistic(λ))
        aux_sample!(Ω, lik, y, invert(fs))
        Σ .= inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(auglik_precision(lik, Ω, y, fs[2]))))
        μ .= Σ .* (auglik_potential(lik, Ω, y, fs[2]) .+ Ref(K) .\ mean.(fzs))
        rand!.(MvNormal.(μ, Σ), fs) # this corresponds to f -> g
        return (copy(fs), λ)
    end
end;
# We initialize our random variables
fs_init = nestedview(randn(N, nlatent(lik)))
Ω = init_aux_variables(lik, N);
# Run the sampling for default number of iterations (200)
fs_samples = gibbs_sample(fzs, fs_init, Ω, nsamples = 100);
# And visualize the samples overlapped to the variational posterior
# that we found earlier.



## 
sb = ColorSchemes.seaborn_colorblind
GPtheme = Theme(palette=sb, linewidth=10.0)
fig = Figure(resolution=(900, 600))
lw = 3.0
with_theme(GPtheme, linewidth=3.0) do 
    ax1 = Axis(fig[1, 2], title=L"y | f, g", titlesize=25.0)
    ax2 = Axis(fig[1, 3], title="Latent GPs", titlesize=19.0)
    ax3 = Axis(fig[2, 2], title="")
    ax4 = Axis(fig[2, 3], title="")

    
    y_qm = mean(f_te)
    y_qσ = sqrt.(lik.invlink.(mean(g_te)))
    band!(ax1, x_te, y_qm - y_qσ, y_qm + y_qσ, color=(sb[3], 0.5))
    lines!(ax1, x_te, y_qm, label=L"E_{q(f,g)}[p(y|f,g)]", linewidth=3.0, color=sb[3])
    
    plot!(ax2, f_te, color=(sb[1], 0.3), linestyle=:dash, label=L"q(f)", linewidth=lw)
    plot!(ax2, g_te, color=(sb[2], 0.3), linestyle=:dash, label=L"q(g)", linewidth=lw)
    
    alpha = 1e-2
    for (fs, λ) in fs_samples
        lik = HeteroscedasticGaussianLikelihood(InvScaledLogistic(λ))
        fσ = sqrt.(lik.invlink.(fs[2]))
        band!(ax3, x, fs[1] - fσ, fs[1] + fσ, color=(sb[3], alpha))
        lines!(ax3, x, fs[1], color=(sb[3], alpha * 10))
        for i in 1:nlatent(lik)
            plot!(ax4, x, fs[i]; color=(sb[i], 0.01), lw=1.0)
        end
    end
    
    y_σ = sqrt.(lik.invlink.(fs[2]))
    lines!(ax1, x, fs[1] - y_σ, color=(sb[1], 0.5), linestyle=:dash, linewidth=lw)
    lines!(ax1, x, fs[1] + y_σ, color=(sb[1], 0.5), linestyle=:dash, linewidth=lw)
    # band!(ax3, x, fs[1] - y_σ, fs[1] + y_σ, color=(sb[1], 0.5))
    lines!(ax1, x, fs[1], label=L"p(y|f,g)", linewidth=3.0, color=sb[1])
    lines!(ax3, x, fs[1] - y_σ, color=(sb[1], 0.5), linestyle=:dash, linewidth=lw)
    lines!(ax3, x, fs[1] + y_σ, color=(sb[1], 0.5), linestyle=:dash, linewidth=lw)
    lines!(ax3, x, fs[1], label=L"p(y|f,g)", linewidth=3.0, color=sb[1])
    
    lines!(ax2, x, fs[1], label=L"f", color=sb[1], linewidth=lw)
    lines!(ax4, x, fs[1], label=L"f", color=sb[1], linewidth=lw)
    lines!(ax2, x, fs[2], label=L"g", color=sb[2], linewidth=lw)
    lines!(ax4, x, fs[2], label=L"g", color=sb[2], linewidth=lw)
    
    scatter!(ax1, x, y; label="y", color=sb[2], markersize=6.0)
    scatter!(ax3, x, y; label="y", color=sb[2], markersize=6.0)
    tightlimits!.((ax1, ax2, ax3, ax4))
    linkyaxes!(ax1, ax3)
    linkyaxes!(ax2, ax4)

    elem_1 = [
        LineElement(color = (sb[1], 0.5), linestyle=:dash, points = Point2f[(0, 1), (1,1)]),
        LineElement(color = sb[1], linewidth=lw),
        LineElement(color = (sb[1], 0.5), linestyle=:dash, points = Point2f[(0, 0), (1,0)]),
    ]
    elem_2 = [
        LineElement(color = sb[3], linewidth=lw),
        PolyElement(color = (sb[3], 0.5), strokewidth=0.0)
    ]
    elem_3 = MarkerElement(color=sb[2], marker=:circle)
    elem_4 = [
        LineElement(color = (sb[1], 0.5), points = Point2f[(0, 2//3), (1, 2//3)], linewidth=lw),
        LineElement(color = (sb[2], 0.5), points = Point2f[(0, 1//3), (1, 1//3)], linewidth=lw),
    ]
    Legend(fig[1, 2], [ elem_3, elem_1, elem_2], [L"y", L"p(y|f,g)", L"E_{q(f,g)}[p(y|f,g)]"], tellwidth=false, halign=:center, valign=:top, margin=(10, 10, 10, 10), nbanks=3)
    aem = Char(0x200B)
    Legend(fig[2, 2], [elem_2],  [latexstring("""{\$p(y|f_i,g_i)\$}\$$(aem)_{i=1}^S\$""")], tellwidth=false, halign=:center, valign=:top, margin=(10, 10, 10, 10), framevisible=false)
    Legend(fig[2, 3], [elem_4], [latexstring("""{\$f_i,g_i\$}\$$(aem)_{i=1}^S\\sim p(f,g|y)\$""")],tellwidth=false, halign=:center, valign=:top, margin=(10, 10, 10, 10),framevisible=false)
    # axislegend(ax1)
    axislegend(ax2, nbanks=2)
    Label(fig[1, 1], text="Variational\nInference", textsize=20, rotation = pi/2, tellheight=false)
    Label(fig[2, 1], text="Gibbs\nSampling", textsize=20, rotation = pi/2, tellheight=false)
    fig_path = joinpath(@__DIR__, "..", "..", "thesis", "chapters", "8_discussions", "figures")
    save(joinpath(fig_path, "heteroscedastic.pdf"), fig)
    fig
end