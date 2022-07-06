using CairoMakie
CairoMakie.activate!()
using SpecialFunctions
using Distributions

## Basis functions
x = -5:0.01:5

linewidth = 10.0
lw2 = 5.0
target_path = joinpath(@__DIR__, "fig")
mkpath(target_path)
include(joinpath(@__DIR__, "..", "attributes", "theme.jl"))
sb2 = ColorScheme(RGB)
# Workaround for the current issue with GLMakie
alt_lines! = lines!

## First part, a simple mixture of means
fig = Figure(;resolution)
μs = [0.5, 0.8, -1.0, 2.0, 1.2, -2.2]

axnormal = fig[1, 1] = Axis(fig, title=L"\sum_i \pi_i N(x|\mu_i, \sigma)", xlabel=L"x", ylabel=L"p(x)")
axtransform = fig[1, 2] = Axis(fig, title=L"\pi_i (\mu_i)", xlabel=L"\mu", ylabel=L"\pi")

πs = Observable(zeros(length(μs)))
πs[][1] = 1.0
norm_πs = @lift $πs / sum($πs)
μs_points = map(norm_πs) do n_π
    reduce(vcat, [[Point2(μ, 0), Point2(μ, n_π[i])] for (i, μ) in enumerate(μs)])
end
linesegments!(axtransform, μs_points; colormap=sbhide, color=sb[1:length(μs)], linewidth)
fig


xlims!(axnormal, extrema(x)...)
xlims!(axtransform, -3, 3)
ylims!(axtransform, -0.1, 1.1)

σ = 0.5
for (i, μ) in enumerate(μs)
    lines!(axnormal, x, @lift(x->$(norm_πs)[i] * pdf(Normal(μ, σ), x)), color=sb[i])
end
mix = @lift x->sum($(norm_πs)[i] * pdf(Normal(μs[i], σ), x) for i in 1:length(μs))
lines!(axnormal, x, mix; linewidth, color=:black)

ylims!(axnormal, -0.05, 1.0)
t_m = 2
framerate = 24
anim = Animation(1, 0.0, sineio(), t_m * framerate, 1.0)

record(fig, joinpath(target_path, "mixture_means_transform.mp4"); framerate) do io
    for i in 1:length(μs)
        notify(index)
        for j in 1:(t_m * framerate)
            πs[][i] = anim(j)
            notify(πs)
            recordframe!(io)
        end
    end
end

fig

## Now we move to discrete scale
fig = Figure(;resolution)
σs = [1.0, 2.0, 5.0, 0.1, 3.2]

axnormal = fig[1, 1] = Axis(fig, title=L"\sum_i \pi_i N(x|0, \sigma_i)", xlabel=L"x", ylabel=L"p(x)")
axtransform = fig[1, 2] = Axis(fig, title=L"\pi_i (\sigma_i)", xlabel=L"\sigma", ylabel=L"\pi")

πs = Observable(zeros(length(σs)))
πs[][1] = 1.0
norm_πs = @lift $πs / sum($πs)
σs_points = map(norm_πs) do n_π
    reduce(vcat, [[Point2(σ, 0), Point2(σ, n_π[i])] for (i, σ) in enumerate(σs)])
end
linesegments!(axtransform, σs_points; color=sb[1:length(σs)], linewidth)
fig
xlims!(axnormal, extrema(x)...)
xlims!(axtransform, 0, 6.0)
ylims!(axtransform, -0.05, 1.0)
ylims!(axnormal, -0.05, 1.2)

for (i, σ) in enumerate(σs)
    lines!(axnormal, x, @lift(x->$(norm_πs)[i] * pdf(Normal(0, σ), x)), color=sb[i])
end
mix = @lift x->sum($(norm_πs)[i] * pdf(Normal(0, σs[i]), x) for i in 1:length(σs))
lines!(axnormal, x, mix; linewidth, color=:black)


t_m = 2
framerate = 24
anim = Animation(1, 0.0, sineio(), t_m * framerate, 1.0)

record(fig, joinpath(target_path, "mixture_scale_transform.mp4"); framerate) do io
    for i in 1:length(σs)
        for j in 1:(t_m * framerate)
            πs[][i] = anim(j)
            notify(πs)
            recordframe!(io)
        end
    end
end

fig

## Now move to the continuous setting
fig = Figure(;resolution)
ν = 3.0
p_w = InverseGamma(ν / 2, ν / 2)
p_x = TDist(ν)
σs = rand(p_w, 50)

axnormal = fig[1, 1] = Axis(fig, title=L"\int N(x|0, \sigma^2)p(\sigma^2)d\sigma^2", xlabel=L"x", ylabel=L"p(x)")
axtransform = fig[1, 2] = Axis(fig, title=L"p(\sigma^2) = \mathrm{IG}(\sigma^2|\frac{\nu}{2},\frac{\nu}{2})", xlabel=L"\sigma^2", ylabel=L"p(\sigma^2)")

πs = Observable(zeros(length(σs)))
πs[][1] = 1.0
norm_πs = @lift $πs / sum($πs)
σs_points = map(norm_πs) do n_π
    reduce(vcat, [[Point2(σ, 0), Point2(σ, n_π[i])] for (i, σ) in enumerate(σs)])
end


linesegments!(axtransform, σs_points; color=[mod1(i, length(sb)) for i in 1:length(σs)], colormap=sb, linewidth)
fig
lines!(axtransform, 0:0.01:6, x->pdf(p_w, x); color=:black, linewidth)
xlims!(axnormal, extrema(x)...)
xlims!(axtransform, 0, 6.0)
ylims!(axtransform, -0.05, 1.05)
ylims!(axnormal, -0.05, 0.5)

for (i, σ) in enumerate(σs)
    lines!(axnormal, x, @lift(x->$(norm_πs)[i] * pdf(Normal(0, sqrt(σ)), x)), color=sb[mod1(i, length(sb))])
end
mix = @lift x->sum($(norm_πs)[i] * pdf(Normal(0, sqrt(σs[i])), x) for i in 1:length(σs))
lines!(axnormal, x, mix; linewidth, color=:black)
lines!(axnormal, x, x->pdf(p_x, x); linewidth, color=sb[1], linestyle=:dash)
lines!(axtransform, x, zeros(length(x)); linewidth, color=:white)

t_m = 0.5
framerate = 24
per_p = floor(t_m * framerate)
anim = Animation(1, 0.0, sineio(), per_p, 1.0)

record(fig, joinpath(target_path, "mixture_student_t.mp4"); framerate) do io
    for i in 1:length(σs)
        for j in 1:per_p
            πs[][i] = anim(j)
            notify(πs)
            recordframe!(io)
        end
    end
end

fig
