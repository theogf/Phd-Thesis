using Distributions
using CairoMakie
using LinearAlgebra

include(joinpath(@__DIR__, "..", "attributes", "theme.jl"))
set_theme!(my_theme)
target_path = joinpath(@__DIR__, "fig")
mkpath(target_path)
linewidth=5.0
## Create data

# Make a simple regression (BMI vs size)
N = 20
sizes = filter!(<(2.0), randn(N) .* 0.2 .+ 1.7)
true_w = 50
weight = sizes * 50 .+ randn(length(sizes)) * 5.0

fig = Figure()
ax = Axis(fig[1,1], xlabel="Height [m]", ylabel="Weight [kg]")
scatter!(ax, sizes, weight; color=sb[1])
save(joinpath(target_path, "data.png"), fig)
for i in 1:3
    w = randn() * 5 + true_w
    ablines!(ax, [0.0], [w]; linewidth)
end
save(joinpath(target_path, "possible_weight.png"), fig)

fig
##

sigma_prior = 5
prior_w = Normal(40, sigma_prior)
xs = range(10, 100, length=300)
noise = 10
Λ = dot(sizes, sizes) + inv(sigma_prior^2)
μ = inv(Λ) * (40 / sigma_prior^2 + dot(sizes, weight))
sigma = inv(sum(abs2, sizes) + inv(sigma_prior))
posterior_w = Normal(μ, noise^2 * inv(Λ))
we_prior = Observable(0.0)
we_lik = Observable(0.0)
we_post = Observable(0.0)
alpha_prior = Observable(1.0)
alpha_lik = Observable(0.0)
alpha_post = Observable(0.0)

t_dev = 3
t_wait = 2
framerate = 24

fig = Figure()
ax = Axis(fig[1,1], xlabel=L"w", ylabel=L"p(w)")
plot!(ax, xs, @lift(x->$we_prior * pdf(prior_w, x)); color=@lift((sb[1], $alpha_prior)), linewidth, label=L"prior $p(w)$")
plot!(ax, xs, @lift(x->$we_lik * sum(pdf.(Normal.(x .* sizes, noise), weight))); color=@lift((sb[2], $alpha_lik)), linewidth, label=L"likelihood $p(x|w)$")
plot!(ax, xs, @lift(x->$we_post * pdf(posterior_w, x)); color=@lift((sb[3], $alpha_post)), linewidth, label=L"posterior $p(w|x)$", labelcolor=:blue)
leg= axislegend(ax, labelcolor=(:black, 0.0))
anim = Animation(1, 0.0, sineio(), t_dev * framerate, 1.0)
ylims!(ax, -0.05, 0.8)
leg.entrygroups[][1][2][1].labelcolor[] = (:black, 1.0)


record(fig, joinpath(target_path, "evolution_bayes.mp4"); framerate) do io
    for i in 1:(t_dev * framerate)
        we_prior[] = anim(i)
        recordframe!(io)
    end
    [recordframe!(io) for _ in 1:(t_wait * framerate)]
    alpha_lik[] = 1.0
    leg.entrygroups[][1][2][2].labelcolor[] = (:black, 1.0)

    for i in 1:(t_dev * framerate)
        we_lik[] = anim(i)
        alpha_prior[] = anim(t_dev * framerate - ceil(Int, i / 2))
        recordframe!(io)
    end
    [recordframe!(io) for _ in 1:(t_wait * framerate)]
    alpha_post[] = 1.0
    leg.entrygroups[][1][2][3].labelcolor[] = (:black, 1.0)

    for i in 1:(t_dev * framerate)
        we_post[] = anim(i)
        alpha_lik[] = anim(t_dev * framerate - ceil(Int, i / 2))
        recordframe!(io)
    end
end
fig
##



for i in 1:3
    fig = Figure()
    ax = Axis(fig[1,1], xlabel=L"w", ylabel=L"p(w)")
    if i == 1
        leg = axislegend(ax)
        save(joinpath(target_path, "prior.png"), fig)
        continue
    end
    plot!(ax, xs, x->sum(pdf.(Normal.(x .* sizes, noise), weight)); color=sb[2], linewidth, label=L"likelihood $p(x|w)$")
    if i == 2
        leg = axislegend(ax)
        save(joinpath(target_path, "likelihood.png"), fig)
        continue
    end
    plot!(ax, xs, x->pdf(posterior_w, x); color=sb[3], linewidth, label=L"posterior $p(w|x)$")
    if i == 3
        leg = axislegend(ax)
        save(joinpath(target_path, "posterior.png"), fig)
        continue
    end
end
fig
## Now show the different sampled Options


fig = Figure()
ax = Axis(fig[1,1], xlabel="Height [m]", ylabel="Weight [kg]")
scatter!(ax, sizes, weight; color=sb[1])
save(joinpath(target_path, "data.png"))
n_samples = 50
ablines!(ax, zeros(n_samples), rand(posterior_w, n_samples); color=(sb[2], 0.05), linewidth)
save(joinpath(target_path, "posterior_samples.png"), fig)
fig
