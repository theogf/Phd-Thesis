using GLMakie
using CairoMakie
CairoMakie.activate!()
using AugmentedGPLikelihoods
using GPLikelihoods
using AbstractGPs
using AbstractGPsMakie
using ApproximateGPs
using KernelFunctions
using Optim
using LinearAlgebra
using Distributions
using ParameterHandling
using Zygote
using Random: seed!

seed!(42)

N = 200
x = range(-5, 5, length=N)
kernel = with_lengthscale(SqExponentialKernel(), 2.0)
gp = GP(kernel)
lik = BernoulliLikelihood()
lf = LatentGP(gp, lik, 1e-6)
f, y = rand(lf(x))

function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(m, S)))
end

function u_posterior_grad(fz, params)
    return posterior(SparseVariationalApproximation(fz, MvNormal(unflatten(params)...)))
end 

function cavi!(fz::AbstractGPs.FiniteGP, x, y, m, S, qΩ; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    for _ in 1:niter
        post_u = u_posterior(fz, m, S)
        post_fs = marginals(post_u(x))
        aux_posterior!(qΩ, lik, y, post_fs)
        S .= inv(Symmetric(inv(K) + Diagonal(only(expected_auglik_precision(lik, qΩ, y)))))
        m .= S * (only(expected_auglik_potential(lik, qΩ, y)) + K \ mean(fz))
    end
    return m, S
end

m = zeros(N)
S = Matrix{Float64}(I(N))
qΩ = init_aux_posterior(lik, N)
fz = gp(x, 1e-8);
niter = 10

cavi!(fz, x, y, m, S, qΩ; niter);
post = u_posterior(fz, m, S)
x_te = -10:0.01:10

raw_initial_params = (
    m=zeros(N),
    S=positive_definite(Matrix{Float64}(I, N,  N)),
)

params, unflatten = value_flatten(raw_initial_params)

function opt(fz::AbstractGPs.FiniteGP, x, y, params; niter=10)
    function loss(params)
        m, S = unflatten(params)
        svgp = SparseVariationalApproximation(fz, MvNormal(m, S))
        lfz = AbstractGPs.LatentFiniteGP(fz, lik)
        -elbo(svgp, lfz, y)
    end
    opt = optimize(
        loss,
        only ∘ Base.Fix1(Zygote.gradient, loss),
        params,
        LBFGS(;
            alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
            linesearch=Optim.LineSearches.BackTracking(),
        ),
        Optim.Options(; iterations=niter);
        inplace=false,
    )
    return opt.minimizer
end

params = opt(fz, x, y, params; niter)

fig = Figure()

aug_ax = fig[2, 1] = Axis(fig; title="CAVI Updates")
scatter!(aug_ax, x, y)
lines!(aug_ax, x, f)
plot!(aug_ax, x, post, color=(:darkgreen, 0.4))

grad_ax = fig[2, 2] = Axis(fig; title="Gradient updates")
scatter!(grad_ax, x, y)
lines!(grad_ax, x, f)
plot!(grad_ax, x, u_posterior_grad(fz, params), color=(:darkgreen, 0.4))

title = Label(fig[1, :], "What's up")

fig