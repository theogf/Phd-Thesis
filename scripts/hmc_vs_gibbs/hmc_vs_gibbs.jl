using Pkg
Pkg.activate(@__DIR__)
using AugmentedGPLikelihoods
using ApproximateGPs
using Distributions
using AbstractGPs
using LinearAlgebra
using MCMCChains
using AdvancedHMC
using Random
using ForwardDiff
using Zygote
using MLDataUtils
using MLDatasets
using CairoMakie

rng = MersenneTwister(42)
ϵ = 1e-5

## Problem setting
use_toy = false

ν = 3.0
lik = StudentTLikelihood(ν, 1.0)
kernel = SqExponentialKernel()
gp = GP(kernel)
if use_toy
    N = 100
    x = range(-10, 10, length=N)
    f = rand(rng, gp(x, ϵ))
    y = rand(rng, lik(f))
else
    x = MLDatasets.BostonHousing.features()
    rescale!(x; obsdim=2)
    x = ColVecs(x)
    y = vec(MLDatasets.BostonHousing.targets())
    rescale!(y)
    N = length(y)
end
fx = gp(x, ϵ)
K = cov(fx)
const L = ApproximateGPs._chol_cov(fx).L

## Gibbs sampling part
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

f_init = randn(rng, N)
Ω = init_aux_variables(lik, N)
t_gibbs = @elapsed fs = gibbs_sample(fx, copy(f_init), Ω; nsamples=2500)
gibbs_chain = Chains(fs[501:end])

## HMC Part

function sample_with_nuts(init_θ, logπ, metric)
    hamiltonian = Hamiltonian(metric, logπ, Zygote)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, init_θ)
    integrator = Leapfrog(initial_ϵ)

    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    t_nuts = @elapsed samples, stats = sample(hamiltonian, proposal, init_θ, 2000, adaptor, 500; progress=true)
    nuts_chain = Chains(Ref(L) .* getindex.(samples, Ref(1:N)))
    return t_nuts, nuts_chain
end

function sample_with_hmc(init_θ, logπ, metric)
    hamiltonian = Hamiltonian(metric, logπ, Zygote)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, init_θ)
    integrator = Leapfrog(initial_ϵ)

    proposal = StaticTrajectory(integrator, 30)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    t_hmc = @elapsed samples, stats = sample(hamiltonian, proposal, init_θ, 2000, adaptor, 500; progress=true)
    hmc_chain = Chains(Ref(L) .* getindex.(samples, Ref(1:N)))
    return t_hmc, hmc_chain
end
## Run the Augmented model

π_0 = aux_prior(lik)
function logπ(θ)
    v = θ[1:N]
    ω = exp.(θ[N+1:end])
    return loglikelihood(Normal(), v) + sum(logtilt.(Ref(lik), ω, y, L * v)) + sum(logpdf.(π_0, ω)) 
end
D = 2N
init_θ = vcat(L \ f_init, log.(rand(rng, N)))
logπ(init_θ)
metric = DiagEuclideanMetric(D)
t_nuts_aug, nuts_aug_chain = sample_with_nuts(init_θ, logπ, metric)
t_hmc_aug, hmc_aug_chain = sample_with_hmc(init_θ, logπ, metric)

## HMC Chain

function logπ(θ)
    v = θ[1:N]
    return loglikelihood(Normal(), v) + sum(logpdf.(LocationScale.(L * v, 1.0, TDist(ν)), y))
end
D = N
init_θ = L \ f_init
logπ(init_θ)
metric = DiagEuclideanMetric(D)
t_nuts, nuts_chain = sample_with_nuts(init_θ, logπ, metric)
t_hmc, hmc_chain = sample_with_hmc(init_θ, logπ, metric)

## Exploration part 
lags = 1:10

autocors = map((gibbs_chain, hmc_aug_chain, hmc_chain, nuts_aug_chain, nuts_chain)) do chain
    autocor_matrix = Array(DataFrame(autocor(chain[1:2:end]; lags)[1])[!, lags .+ 1])
    return mean(eachrow(autocor_matrix))
    # return autocor_matrix[rand(1:N),:]
end
fig = Figure()
ax = Axis(fig[1, 1], title = "Autocorrelation", titlesize=33.0, xlabel="Lag", xlabelsize=30.0)
ps = map(zip(autocors, ["Gibbs Sampling", "HMC (aug. model)", "HMC", "NUTS (aug. model)", "NUTS"])) do (ac, name)
    lines!(ax, lags * 2, ac, label=name, linewidth=5.0)
end
axislegend(ax; labelsize=30.0)
resize_to_layout!(fig)

fig_path = joinpath(@__DIR__, "..", "..", "thesis", "chapters", "8", "figures")
mkpath(fig_path)
save(joinpath(fig_path, "autocorrelation.pdf"), fig)
fig
##
# plot(lags, collect(autocors), lw=3.0, title="Autocorrelation", label=[], xlabel="lag", ylims=(0, 0.4), legendfontsize=13.0)

for chain in (:gibbs, :nuts_aug, :nuts, :hmc_aug)
    ess_val = mean(Matrix(DataFrame(ess_rhat(eval(Symbol(chain, "_chain"))[1:2:end, :, :])))[:, 2]) + mean(Matrix(DataFrame(ess_rhat(eval(Symbol(chain, "_chain"))[2:2:end, :, :])))[:, 2])
    println("$chain\n\tess: $(ess_val)")
    println("\ttime: $(eval(Symbol("t_", chain)))")
    println("\tess/s: $(ess_val / eval(Symbol("t_", chain)))")
end