using GLMakie
using CairoMakie
CairoMakie.activate!()
using AugmentedGPLikelihoods
using AugmentedGPLikelihoods: logistic
using GPLikelihoods
using AbstractGPs
using AbstractGPsMakie
using AdvancedHMC
using ApproximateGPs
using KernelFunctions
using Optim
using LinearAlgebra
using Distributions
using ParameterHandling
using Zygote
using Random: seed!

seed!(42)

include(joinpath(@__DIR__, "..", "attributes", "theme.jl"))
set_theme!(my_theme)
target_path = joinpath(@__DIR__, "fig")
mkpath(target_path)

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

t = Observable(1)
niter = 1

post_cavi = map(t->u_posterior(fz, m, S), t)
post_grad = map(t->u_posterior_grad(fz, params), t)
# params = opt(fz, x, y, params; niter)
fig = Figure()

linewidth = 5.0

aug_ax = fig[2, 1] = Axis(fig; title="CAVI Updates")
scatter!(aug_ax, x, y; color=(sb[1], 0.6))
lines!(aug_ax, x, f; color=sb[1], linewidth)
plot!(aug_ax, x, post_cavi; color=(sb[3], 0.5), linewidth)

grad_ax = fig[2, 2] = Axis(fig; title="Gradient updates")
scatter!(grad_ax, x, y; color=(sb[1], 0.6))
lines!(grad_ax, x, f; color=sb[1], linewidth)
plot!(grad_ax, x, post_grad; color=(sb[3], 0.5), linewidth)
title = Label(fig[1, :], "Binary classification", textsize=30.0)

record(fig, joinpath(target_path, "convergence_vi.mp4"), 1:20, framerate=2) do i
 t[] = i
 if i > 2
    global params = opt(fz, x, y, params; niter)
    cavi!(fz, x, y, m, S, qΩ; niter);
 end
end

fig

## Now time to tackle the sampling

function gibbs_sample(fz, f, Ω; nsamples=200)
    K = ApproximateGPs._chol_cov(fz)
    Σ = zeros(length(f), length(f))
    μ = zeros(length(f))
    return map(1:nsamples) do _
        aux_sample!(Ω, lik, y, f)
        Σ .= inv(Symmetric(inv(K) + Diagonal(only(auglik_precision(lik, Ω, y)))))
        μ .= Σ * (only(auglik_potential(lik, Ω, y)) - K \ mean(fz))
        rand!(MvNormal(μ, Σ), f)
        return copy(f)
    end
end;

function sample_with_nuts(init_θ, logπ, metric; nsamples=200)
    hamiltonian = Hamiltonian(metric, logπ, Zygote)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, init_θ)
    integrator = Leapfrog(initial_ϵ)

    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    samples, _ = sample(hamiltonian, proposal, init_θ, nsamples, adaptor, 100; progress=true)
    return Ref(L) .* getindex.(samples, Ref(1:N))
end

f_init = randn(N)
const L = ApproximateGPs._chol_cov(fz).L
function logπ(θ)
    v = θ[1:N]
    return loglikelihood(Normal(), v) + sum(logpdf.(Bernoulli.(logistic.(L * v)), y))
end
D = N
logπ(f_init)
metric = DiagEuclideanMetric(D)
n_s = 200
t_hmc = @elapsed samples_hmc = sample_with_nuts(copy(f_init), logπ, metric; nsamples=n_s)
Ω = init_aux_variables(lik, N)
t_gibbs = @elapsed samples_gibbs = gibbs_sample(fz, copy(f_init), Ω; nsamples=n_s)

## Put it on a figure

fig = Figure()

linewidth = 5.0

aug_ax = fig[2, 1] = Axis(fig; title="Gibbs Sampling")
scatter!(aug_ax, x, y; color=(sb[1], 0.6))
lines!(aug_ax, x, f; color=sb[1], linewidth)

grad_ax = fig[2, 2] = Axis(fig; title="NUTS Sampling")
scatter!(grad_ax, x, y; color=(sb[1], 0.6))
lines!(grad_ax, x, f; color=sb[1], linewidth)
title = Label(fig[1, :], "Binary classification", textsize=30.0)

t_per_s = (t_hmc, t_gibbs) ./ 200

tot_t = ceil(max(t_hmc, t_gibbs))
fr = 24
tot_i = Int(tot_t * fr)
Δt = tot_t / tot_i
current_t = Observable(1)
alpha = 

alpha_gibbs = map(current_t) do t
    curr_time = t * Δt
    T = min(n_s, ceil(Int, curr_time / t_per_s[2]))
    α = vcat(0.1 * ones(T), zeros(n_s - T))
end;
alpha_hmc = map(current_t) do t
    curr_time = t * Δt
    T = min(n_s, ceil(Int, curr_time / t_per_s[1]))
    α = vcat(0.1 * ones(T), zeros(n_s - T))
end;

for i in 1:n_s
    lines!(aug_ax, x, samples_gibbs[i]; linewidth, color=@lift((sb[3], $(alpha_gibbs)[i])))
    lines!(grad_ax, x, samples_hmc[i]; linewidth, color=@lift((sb[3], $(alpha_hmc)[i])))
end
# series!(aug_ax, samp_gibbs; solid_color=(sb[3], 0.1), linewidth)
# series!(grad_ax, samp_hmc; solid_color=(sb[3], 0.1), linewidth)

record(fig, joinpath(target_path, "sampling_binary.mp4"), 1:tot_i; framerate=fr) do i
    @info i
    current_t[] = i
end

fig