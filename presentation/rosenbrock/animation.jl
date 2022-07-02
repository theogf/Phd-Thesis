using Distributions
using CairoMakie
using LinearAlgebra
using ForwardDiff
using Optimisers
a = 0.1
b = 6

include(joinpath(@__DIR__, "..", "attributes", "theme.jl"))
set_theme!(my_theme)

logrosen(x,y) = -0.5 * (abs2(x * y) + abs2(x) + abs2(y) - b * (x + y))

c =  1/20216.335877
rosen(x, y) = c * exp(logrosen(x, y))

target_path = joinpath(@__DIR__, "fig")
mkpath(target_path)

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
record(fig, joinpath(target_path, "from_gaussian_to_rosenbrock.mp4"), 1:n_frames_prob; framerate=15) do i
# for i in 1:n_frames_prob
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
record(fig, joinpath(target_path, "sampling_rosenbrock.mp4"), 1:n_frames_s; framerate=15) do i
# for i in 1:n_frames_s
    # sleep(1/24)
    # @info a_s(i)
    n_s[] = floor(Int, a_s(i))
end
fig
## We show now a fitting of a Gaussian distribution on it
fig, ax, plt = contourf(x, x, f, levels= 10, linewidth=1.0, colormap=:heat)
i_prob[] = n_frames_prob
n_vi = Observable(0)
m = [2.0, 3.0]
Γ = [0.1, 0.1]
q = MvNormal(m, Diagonal(abs2.(Γ)))
# m_init = [3.0, 3.0]
logπ(x) = logrosen(x...)
nσ = 3
θ = range(0, 2π, length=100)
Θ = vcat(cos.(θ)', sin.(θ)')
vals = map(n_vi) do _
    μ = mean(q)
    L = cholesky(cov(q)).L
    mapreduce(vcat, 1:nσ) do iσ
        Point2f.(eachcol(μ .+ iσ * L * Θ))
    end
end
cent = map(n_vi) do _
    [Point2(mean(q)...)]
end
linesegments!(ax, vals, linewidth=3.0, color=sb[1])
scatter!(ax, cent, color=sb[1])
xlims!(ax, extrema(x))
ylims!(ax, extrema(x))
n_rand = 10
opt = ADAGrad(0.25)
state = Optimisers.setup(opt, (m, Γ))
record(fig, joinpath(target_path, "vi_rosenbrock.mp4"), 1:n_frames_s; framerate=20) do i
# for i in 1:100
    global q = MvNormal(m, Diagonal(abs2.(Γ)))
    x_0 = randn(2, n_rand)
    x_p = m .+ Γ .* x_0
    Δ = ForwardDiff.gradient(x->sum(-logπ.(eachcol(x))), x_p)
    global state, (m, Γ) =  Optimisers.update(state, (m, Γ), (vec(mean(Δ, dims = 2)), (vec(mean(Δ .* x_0, dims=2)) - inv.(Γ))))
    n_vi[] = i
end

fig