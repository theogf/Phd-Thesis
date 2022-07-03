using Distributions
using CairoMakie

include(joinpath(@__DIR__, "..", "attributes", "theme.jl"))
set_theme!(my_theme)
target_path = joinpath(@__DIR__, "fig")
mkpath(target_path)
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
xs = range(10, 100, length=200)
noise = 10
sigma = inv(sum(abs2, sizes) + inv(sigma_prior))
m = sigma * (inv(sigma_prior) * 40  + dot(sizes, weight))
posterior_w = Normal(m, noise * sigma) 
for i in 1:3
    fig = Figure()
    ax = Axis(fig[1,1], xlabel=L"w", ylabel=L"p(w)")
    plot!(ax, xs, x->pdf(prior_w, x); color=sb[1], linewidth, label=L"prior $p(w)$")
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

##
loglikelihood(x) = sum(logpdf(Normal(x, noise), data))
likelihood(x) = exp(loglikelihood(x))
likelihood(1.5)

joint(x) = likelihood(x) * pdf(prior_height, x)

x = 0:0.01:3

mean(data)
mean(abs2, data)



fig, axis, plt = lines(x, pdf.(prior_height, x))
lines!(axis, x, joint.(x))
fig