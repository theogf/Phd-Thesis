pkg_active = @isdefined(pkg_active) ? false : true

using Pkg
if pkg_active
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using CairoMakie
using FileIO
using Makie
using MeasureTheory
using Random
using Colors
using ColorSchemes
sb = ColorSchemes.seaborn_colorblind
cg = ColorScheme(map(x->RGB(x...), include(joinpath(@__DIR__, "..", "colors", "color_flare"))))
mc = :black

const c = 1/20216.335877
logp(x) = - (x[1]^2 * x[2]^2 + x[1]^2 + x[2]^2 - 8x[1] - 8x[2]) / 2
p(x1, x2) = c * exp(- (x1^2 * x2^2 + x1^2 + x2^2 - 8x1 - 8x2) / 2)
cond_p(x) = Normal(4 / (x^2 + 1), 1 / (x^2 + 1))

rng = MersenneTwister(42)

function plot_dist!(fig, f; n=1000, xlabel="x₁", ylabel="x₂")
    xlin = LinRange(-1, 7, n)
    ax = Axis(fig[1,1]; xlabel, ylabel, xlabelsize=30.0, ylabelsize=30.0, yticks=-1:1:7, xticks=-1:1:7)
    contourf!(ax, xlin, xlin, (x,z)->f(x, z), colormap=cg)
    resize_to_layout!(fig)
    fig |> display
    return ax
end

function create_path_gibbs!(ax, n=10)
    x1 = randn(rng)
    x2 = randn(rng)
    xs = []
    push!(xs, [x1, x2])
    for i in 1:n
        x1 = rand(rng, cond_p(x2))
        push!(xs, [x1, x2])
        x2 = rand(rng, cond_p(x1))
        push!(xs, [x1, x2])
    end
    lines!(ax, first.(xs), last.(xs), color=mc)
    scatter!(ax, first.(xs), last.(xs), color=mc)
end

fig_path = joinpath(@__DIR__, "..", "..", "thesis", "chapters", "2_background", "figures")
mkpath(fig_path)

fig = Figure(resolution=(600, 600));
ax = plot_dist!(fig, p)
create_path_gibbs!(ax, 20)
# save(joinpath(fig_path, "gibbs_sampling.pdf"), fig)
fig

## 

using ForwardDiff
using LinearAlgebra
grad(x) = ForwardDiff.gradient(x->-logp(x), x)
function leapfrog(x, p, N, ϵ)
    int_states = zeros(2, N+1)
    for i in 1:N
        p = p - ϵ / 2 * grad(x)
        int_states[:, i] .= x
        x = x + ϵ * p
        p = p - ϵ / 2 * grad(x)
    end
    int_states[:, end] .= x
    return x, int_states
end
function create_path_hmc!(ax, n=10, N=100, ϵ=1e-2)
    x = randn(rng, 2)
    xs = []
    int_states = []
    push!(xs, copy(x))
    for i in 1:n
        p = randn(2)
        x, int_state = leapfrog(x, p, N, ϵ)
        push!(xs, copy(x))
        push!(int_states, int_state)
    end
    int_states = reduce(hcat, int_states)
    lines!(ax, int_states[1, :], int_states[2, :], color=:grey)
    scatter!(ax, first.(xs), last.(xs), color=mc)
end

fig = Figure(resolution=(600, 600));
ax = plot_dist!(fig, p)
create_path_hmc!(ax, 20)
save(joinpath(fig_path, "hmc_sampling.pdf"), fig)
fig
