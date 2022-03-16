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
using AbstractGPsMakie
using Random: seed!

# We create some random data (sorted for plotting reasons)
seed!(42)

N = 100
Nclass = 3
x = collect(range(-10, 10; length=N))

# We will present simultaneously the bijective and non-bijective likelihood
liks = [CategoricalLikelihood(BijectiveSimplexLink(LogisticSoftMaxLink(zeros(Nclass)))), CategoricalLikelihood(LogisticSoftMaxLink(zeros(Nclass)))]

# This is small hack until https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl/pull/68 is merged
AugmentedGPLikelihoods.nlatent(::CategoricalLikelihood{<:BijectiveSimplexLink}) = Nclass - 1
AugmentedGPLikelihoods.nlatent(::CategoricalLikelihood{<:LogisticSoftMaxLink}) = Nclass

SplitApplyCombine.invert(x::ArrayOfSimilarArrays) = nestedview(flatview(x)')
# We deifne the models
kernel = 5.0 * with_lengthscale(SqExponentialKernel(), 2.0)
gp = GP(kernel)
fz = gp(x, 1e-8);
# We use a multi-output GP to generate our (N-1) latent GPs
gpm = GP(IndependentMOKernel(kernel))
X = MOInput(x, Nclass-1);

# We sample (N-1) latent GPs to force the setting of the bijective version
# which the non-bijective version should also be able to recover
fs = rand(gpm(X, 1e-6))
fs = nestedview(reduce(hcat, Iterators.partition(fs, N)))
lik_true = liks[1](invert(fs)) # the likelihood
y = rand(lik_true);

# We build the one-hot encoding for each likelihood (different)
Ys = map(liks) do lik
    Y = nestedview(sort(unique(y))[1:nlatent(lik)] .== permutedims(y))
    return Y
end;

# We write our CAVI algorithmm
function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(copy(m), S)))
end


function cavi!(fz::AbstractGPs.FiniteGP, lik, x, Y, ms, Ss, qΩ; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    for _ in 1:niter
        posts_u = u_posterior.(Ref(fz), ms, Ss)
        posts_fs = marginals.([p_u(x) for p_u in posts_u])
        aux_posterior!(qΩ, lik, Y, SplitApplyCombine.invert(posts_fs))
        Ss .= inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(expected_auglik_precision(lik, qΩ, Y))))
        ms .= Ss .* (expected_auglik_potential(lik, qΩ, Y) .- Ref(K \ mean(fz)))
    end
    return ms, Ss
end
# Now we just initialize the variational parameters
ms_Ss = map(liks, Ys) do lik, Y
    m = nestedview(zeros(N, nlatent(lik)))
    S = [Matrix{Float64}(I(N)) for _ in 1:nlatent(lik)]
    qΩ = init_aux_posterior(lik, N)
    fz = gp(x, 1e-8);
    cavi!(fz, lik, x, Y, m, S, qΩ; niter=20);
    return (;m, S)
end

# Gibbs Sampling
function gibbs_sample(fz, lik, Y, fs, Ω; nsamples=200)
    K = ApproximateGPs._chol_cov(fz)
    Σ = [zeros(N, N) for _ in 1:nlatent(lik)]
    μ = [zeros(N) for _ in 1:nlatent(lik)]
    return map(1:nsamples) do _
        aux_sample!(Ω, lik, Y, invert(fs))
        Σ .= inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(auglik_precision(lik, Ω, Y))))
        μ .= Σ .* (auglik_potential(lik, Ω, Y) .- Ref(K \ mean(fz)))
        rand!.(MvNormal.(μ, Σ), fs)
        return copy(fs)
    end
end;
# We initialize our random variables
samples = map(liks, Ys, plts) do lik, Y, plt
    fs_init = nestedview(randn(N, nlatent(lik)))
    Ω = init_aux_variables(lik, N);
    # Run the sampling for default number of iterations (200)
    return gibbs_sample(fz, lik, Y, fs_init, Ω);
    # And visualize the samples overlapped to the variational posterior
    # that we found earlier.
end;


## 
sb = ColorSchemes.seaborn_colorblind
GPtheme = Theme(palette=sb, linewidth=10.0)
fig = Figure(resolution=(900, 600))
lw = 3.0
for i in 1:2
    with_theme(GPtheme, linewidth=3.0) do 
        ax1 = Axis(fig[1, 2], title=L"y | f, g", titlesize=25.0)
        ax2 = Axis(fig[1, 3], title="Latent GPs", titlesize=19.0)
        ax3 = Axis(fig[2, 2], title="")
        ax4 = Axis(fig[2, 3], title="")

        # Plot the truth
        scatter!(ax1, x, y / Nclass; color=sb[y], label=[1 2 3 4], msw=0.0)
        ps_true = getproperty.(lik_true.v, :p)
        for k in 1:Nclass
            lines!(ax1, x, invert(ps_true)[k], color=sb[k], lw =3.0)
            lines!(ax3, x, invert(ps_true)[k], color=sb[k], lw =3.0)
        end

        for f in vcat(fs, [zeros(N)])
            lines!(ax2, x, f, color=sb[k], lw=3.0)
            lines!(ax4, x, f, color=sb[k], lw=3.0)
        end

        # Plot variational and sampling results
        lik_pred = liks[i](invert(mean.([u_post(x) for u_post in u_posterior.(Ref(fz), ms_Ss[i].m, ms_Ss[i].S)])))
        ps_pred = getproperty.(lik_pred.v, :p)
        samp_pred = map(samples[i]) do s
            getproperty(lik(invert(s)).v, :p)
        end
        α = 1e-3
        for k in 1:Nclass
            lines!(ax1, x, invert(ps_true)[k], color=sb[k], lw =3.0)
            for s_p in samp_pred
                lines!(ax3, x, invert(ps_true)[k], color=(sb[k], α), lw =3.0)
            end
        end


        y_qσ = sqrt.(lik.invlink.(mean(g_te)))
        band!(ax1, x_te, y_qm - y_qσ, y_qm + y_qσ, color=(sb[3], 0.5))
        lines!(ax1, x_te, y_qm, label=L"E_{q(f,g)}[p(y|f,g)]", linewidth=3.0, color=sb[3])
        
        plot!(ax2, f_te, color=(sb[1], 0.3), linestyle=:dash, label=L"q(f)", linewidth=lw)
        plot!(ax2, g_te, color=(sb[2], 0.3), linestyle=:dash, label=L"q(g)", linewidth=lw)
        
        alpha = 1e-2
        for fs in fs_samples
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
        Legend(fig[1, 2], [ elem_3, elem_1, elem_2], [L"y", L"p(y|f,g)", L"E_{q(f,g)}[p(y|f,g)]"], tellwidth=false, halign=:right, valign=:top, margin=(10, 10, 10, 10))
        Legend(fig[2, 2], [elem_2],  [L"""\left(p(y|f_i,g_i)\right)_{i=1}^S"""], tellwidth=false, halign=:right, valign=:top, margin=(10, 10, 10, 10))
        Legend(fig[2, 3], [elem_4], [L"""\left(f_i,g_i\right)_{i=1}^S\sim p(f,g|y)"""],tellwidth=false, halign=:right, valign=:top, margin=(10, 10, 10, 10))
        # axislegend(ax1)
        axislegend(ax2, nbanks=2)
        Label(fig[1, 1], text="Variational\nInference", textsize=20, rotation = pi/2, tellheight=false)
        Label(fig[2, 1], text="Gibbs\nSampling", textsize=20, rotation = pi/2, tellheight=false)
        fig_path = joinpath(@__DIR__, "..", "..", "thesis", "chapters", "8_discussions", "figures")
        save(joinpath(fig_path, "categorical.pdf"), fig)
        display(fig)
    end
end