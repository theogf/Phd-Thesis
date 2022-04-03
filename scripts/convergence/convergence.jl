using AugmentedGPLikelihoods
using CairoMakie
using Random
using KernelFunctions
using ApproximateGPs
using LinearAlgebra
using Distributions
using ColorSchemes
using LsqFit
using Formatting
using LaTeXStrings
# We create some random data (sorted for plotting reasons)
N = 100
x = range(-10, 10; length=N)
kernel = with_lengthscale(SqExponentialKernel(), 2.0)
gp = GP(kernel)

# ## CAVI Updates
# We write our CAVI algorithmm
function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(m, S)))
end
function aug_elbo(lik, u_post, x, y)
    qf = marginals(u_post(x))
    qΩ = aux_posterior(lik, y, qf)
    return expected_logtilt(lik, qΩ, y, qf) - aux_kldivergence(lik, qΩ, y) -
           kldivergence(u_post.approx.q, u_post.approx.fz) # approx.fz is the prior and approx.q is the posterior 
end
function cavi!(fz::AbstractGPs.FiniteGP, lik, x, y, m, S, qΩ; niter=10, storage)
    K = ApproximateGPs._chol_cov(fz)
    push!(storage, (;m=deepcopy(m), S=deepcopy(S)))
    for _ in 1:niter
        post_u = u_posterior(fz, m, S)
        post_fs = marginals(post_u(x))
        aux_posterior!(qΩ, lik, y, post_fs)
        S .= inv(Symmetric(inv(K) + Diagonal(only(expected_auglik_precision(lik, qΩ, y)))))
        m .= S * (only(expected_auglik_potential(lik, qΩ, y)) + K \ mean(fz))
        push!(storage, (;m=deepcopy(m), S=deepcopy(S), qΩ=deepcopy(qΩ)))
    end
    return m, S
end;
liks = Dict(
    :bernoulli => BernoulliLikelihood(),
    :studentt => StudentTLikelihood(3.0, 1.0),
    :laplace => LaplaceLikelihood(),
    :poisson => PoissonLikelihood(ScaledLogistic(3.0)) 
)
lik_label = Dict(
    :bernoulli => "Bernoulli",
    :studentt => "Student-T",
    :laplace => "Laplace",
    :poisson => "Poisson",
)
# lik = StudentTLikelihood(4.0, 1.0)



sb = ColorSchemes.seaborn_colorblind
GPtheme = Theme(palette=sb, linewidth=10.0)
fig = Figure(resolution=(1000, 1200))
lnames = [:bernoulli, :studentt, :laplace, :poisson]

for (i, lname) in enumerate(lnames)
    lik = liks[lname]
    lf = LatentGP(gp, lik, 1e-6)
    @info "Working on $lname"
    f, y = rand(lf(x));

    # Now we just initialize the variational parameters
    m = zeros(N)
    S = Matrix{Float64}(I(N))
    qΩ = init_aux_posterior(lik, N)
    fz = gp(x, 1e-8);
    # And visualize the current posterior
    x_te = -10:0.01:10
    # We run CAVI for 3-4 iterations
    storage = []
    cavi!(fz, lik, x, y, m, S, qΩ; niter=20, storage);


    aug_elbo(lik, u_posterior(fz, m, S), x, y)
    ##
    opt = last(storage)
    opt_L = aug_elbo(lik, u_posterior(fz, opt.m, opt.S), x, y)
    trace = storage[1:end-1]
    trace_S = map(trace) do (m, S)
        norm(S - opt.S)
    end
    trace_m = map(trace) do (m, S)
        norm(m - opt.m)
    end
    trace_L = map(trace) do (m, S)
        abs(aug_elbo(lik, u_posterior(fz, m, S), x, y) - opt_L)
    end

    ## Figure

    lw = 5.0
    aff(t, p) = p[1] .+  p[2] * t
    # Treat m
    ax_m = Axis(fig[i, 2], yscale=log10, title=i == 1 ? L"|m^t - m*|" : "", titlesize=30.0, xlabelsize=25.0, xlabel=i==length(lnames) ? "Iteration t" : "")
    t_m = trace_S[1:findlast(!=(0), trace_m)]
    x_m = 0:(length(t_m) - 1)
    lines!(ax_m, x_m, t_m, color=sb[1], linewidth=lw)
    ps = curve_fit(aff, 1:length(t_m), log.(t_m), rand(2)).param
    lines!(ax_m, x_m, x->exp(ps[1]) * exp(ps[2] * x),color=sb[3], linestyle=:dash, linewidth=lw, label=L"C_0 \exp(-c t)")
    mid_point = length(t_m) ÷ 2
    annotations!(ax_m, ["c = $(Formatting.format(ps[2], precision=2))"], [Point2(x_m[mid_point], t_m[2])])
    # axislegend(ax_m, labelsize=27.0)
    # Treat s
    ax_S = Axis(fig[i, 3], yscale=log10, title=i == 1 ? L"|S^t - S*|" : "", titlesize=30.0,  xlabelsize=25.0, xlabel=i==length(lnames) ? "Iteration t" : "")
    t_S = trace_S[1:findlast(!=(0), trace_S)]
    x_S = 0:(length(t_S) - 1)
    lines!(ax_S, x_S, t_S,color=sb[1], linewidth=lw)
    ps = curve_fit(aff, 1:length(t_S), log.(t_S), rand(2)).param
    lines!(ax_S, x_S, x->exp(ps[1]) * exp(ps[2] * x), color=sb[3], linestyle=:dash, linewidth=lw, label=L"C_0 \exp(-ct)")
    mid_point = length(t_S) ÷ 2
    annotations!(ax_S, ["c = $(Formatting.format(ps[2], precision=2))"], [Point2(x_S[mid_point], t_S[2])])

    Label(fig[i, 1], text=lik_label[lname], textsize=30.0, rotation = pi/2, tellheight=false)
    # axislegend(ax_S, labelsize=27.0)
    # Treat ELBO
    ax_L = Axis(fig[i, 4], yscale=log10, title=i == 1 ? L"|F^t - F*|" : "", titlesize=30.0,  xlabelsize=25.0, xlabel=i==length(lnames) ? "Iteration t" : "")
    t_L = trace_L[1:findlast(!=(0), trace_L)]
    x_L = 0:(length(t_L) - 1)
    lines!(ax_L, x_L, t_L, linewidth=lw, color=sb[1])
    ps = curve_fit(aff, 1:length(t_L), log.(t_L), rand(2)).param
    lines!(ax_L, x_L, x->exp(ps[1]) * exp(ps[2] * x), color=sb[3], linestyle=:dash, linewidth=lw)
    mid_point = length(t_L) ÷ 2
    annotations!(ax_L, ["c = $(Formatting.format(ps[2], precision=2))"], [Point2(x_L[mid_point], exp((log(t_L[1]) + log(t_L[2]))/2))])

end

fig
fig_path = joinpath(@__DIR__, "..", "..", "thesis", "chapters", "8_discussions", "figures")
save(joinpath(fig_path, "convergence.pdf"), fig)
fig