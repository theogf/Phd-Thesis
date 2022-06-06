using Distributions
using GLMakie
using Animations
using Colors
using AdvancedVI
using LinearAlgebra
import ColorSchemes.seaborn_colorblind as sb
a = 0.1
b = 6

logrosen(x,y) = -0.5 * (abs2(x * y) + abs2(x) + abs2(y) - b * (x + y))

c =  1/20216.335877
rosen(x, y) = c * exp(logrosen(x, y))

function sample_rosen(n)
    x = randn(2)
    samples = zeros(2, n)
    samples[:, 1] .= x
    for i in 2:n
        x = samples[:, i-1]
        x1 = rand(Normal(b/2/(x[2]^2 + 1), sqrt(inv(x[2]^2 + 1))))
        x2 = rand(Normal(b/2/(x1^2 + 1), sqrt(inv(x1^2 + 1))))
        samples[:, i] .= [x1, x2]
    end
    return samples
end

X = sample_rosen(1000)
m = vec(mean(X, dims=2))
plot(eachrow(X)...)

p = MvNormal(m, 1.5 * ones(2))



x = -1:0.1:5
fig, ax, plt = contourf(x, x, rosen, levels= 10, linewidth=1.0, colormap=:heat)
contourf!(ax, x, x, (x,y)->pdf(p, [x, y]), colormap=:heat)
## This will create an animation transforming a Gaussian to 
n_frames_prob = 100
a_prob = Animation(1, 0.0, sineio(), n_frames_prob, 1.0)
i_prob = Observable(1)
f = @lift (x,y)->(1-a_prob($i_prob)) * pdf(p, [x, y]) + a_prob($i_prob) * rosen(x, y)
fig, ax, plt = contourf(x, x, f, levels= 10, linewidth=1.0, colormap=:heat)
display(fig)
for i in 1:n_frames_prob
    i_prob[] = i
end

fig

## We now show samples coming from there
fig, ax, plt = contourf(x, x, f, levels= 10, linewidth=1.0, colormap=:heat)
i_prob[] = n_frames_prob
n_s = Observable(0)
n_frames_s = 300
a_s = Animation(0, 0.0, polyin(3), n_frames_s, float(size(X, 2)))
a_alpha_s = Animation(0, 1.0, sineio(), n_frames_s, 0.5)
S = @lift X[:, 1:$n_s]
s_color = @lift RGBA(sb[1], a_alpha_s($n_s))
points = @lift Point2.(eachrow($S)...)
scatter!(ax, points, color=s_color)
xlims!(ax, extrema(x))
ylims!(ax, extrema(x))
display(fig)
for i in 1:n_frames_s
    sleep(1/24)
    @info a_s(i)
    n_s[] = floor(Int, a_s(i))
end
fig
## We show now a fitting of a Gaussian distribution on it
fig, ax, plt = contourf(x, x, f, levels= 10, linewidth=1.0, colormap=:heat)
i_prob[] = n_frames_prob
n_vi = Observable(0)
# m_init = [3, 3, log(0.1), log(0.1)]
m_init = [3.0, 3.0]
logπ(x) = logrosen(x...)
# getq(m) = MvNormal(m[1:2], exp.(m[3:4]))
getq(m) = MvNormal(m[1:2], 0.1 * ones(2))
q = getq(m_init)
nσ = 3
θ = range(0, 2π, length=100)
opt = DecayedADAGrad()
Θ = vcat(cos.(θ)', sin.(θ)')
vals = map(n_vi) do _
    μ = mean(q)
    L = cholesky(cov(q)).L
    mapreduce(vcat, 1:nσ) do iσ
        Point2f.(eachcol(μ .+ iσ * L * Θ))
    end
end
linesegments!(ax, vals, linewidth=2.0, color=sb[1])
xlims!(ax, extrema(x))
ylims!(ax, extrema(x))
display(fig)
for i in 1:100
    sleep(0.01)
    advi = ADVI(20, 1)
    global q = vi(logπ, advi, getq, m_init; optimizer=opt)
    m_init[1:2] .= mean(q)
    # @show m_init[3:4] .= log.(var(q))
    n_vi[] = i
end

fig