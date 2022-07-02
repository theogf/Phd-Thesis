using CairoMakie
CairoMakie.activate!()
using SpecialFunctions

include(joinpath(@__DIR__, "..", "attributes", "theme.jl"))

## Basis functions
sqexp(x) = exp(-x^2)
lap(x) = exp(-abs(x))
trans_sqexp(x) = exp(-x^2/4) / (sqrt(2))
trans_lap(x) = sqrt(2/π) / (abs2(x) + 1)
x = -5:0.01:5
y = sqexp.(x)

lw = 5.0
target_path = joinpath(@__DIR__, "fig")
mkpath(target_path)
set_theme!(my_theme)

# Workaround for the current issue with GLMakie
alt_lines! = lines!

## First part, demonstration of the Fourier transform (discrete first)
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title=L"Function $f$")
axtransform = fig[1, 2] = Axis(fig, title=L"Fourier Transform $\hat{f}$")

ns = 1:5
N = Observable(1)
Ns = @lift ns[1:$N]
sumcos = @lift x->sum(cos(n * x) / n for n in $Ns)

alt_lines!(axnormal, x, sumcos, linewidth=lw, color=:black)
c = √(π/2)
xlims!(axnormal, extrema(x))

linkxaxes!(axnormal, axtransform)

record(fig, joinpath(target_path, "basic_fourier_transform.mp4"), ns; framerate=1) do n
# for n in ns
    N[] = n
    alt_lines!(axnormal, x, x->cos(n * x) / n, color=@lift RGBA(sb[n], n <= $N ? 0.6 : 0.0))
    linesegments!(axtransform, vcat(Point2.(-n, [0.0, c / n]), Point2.(n, [0.0, c / n])), color=@lift(RGBA(sb[n], n <= $N)), linewidth=lw)
end


fig


## Fourier transform in continuous setting

### Show sq. exp. to laplace
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title=L"Function $f$")
axtransform = fig[1, 2] = Axis(fig, title=L"Fourier Transform $\hat{f}$")

to_lap = Observable(1)
n_frames_lap = 10
mycmap = range(sb[1], sb[4], length=n_frames_lap)
a_lap = Animation(1, 0.0, sineio(), n_frames_lap, 1.0)


f_transform(to_lap) = x -> (1 - a_lap(to_lap)) * trans_sqexp(x) + a_lap(to_lap) * trans_lap(x)
f_normal(to_lap) = x -> (1 - a_lap(to_lap)) * sqexp(x) + a_lap(to_lap) * lap(x)
# annotations!(axtransform, ["f̂(x) = exp(-x²/2)/√2 -> √(2/π)/(x² + 1)"], [Point2(-4.5, 0.85)])
tightlimits!(axnormal, Left(), Right())
tightlimits!(axtransform, Left(), Right())
alt_lines!(axtransform, x, f_transform(1); color=mycmap[1])
alt_lines!(axnormal, x, f_normal(1); color=mycmap[1])
ylims!(axtransform, -0.05, 1.0)

# Save the fig
record(fig, joinpath(target_path, "fourier_continuous.mp4"), 1:n_frames_lap; framerate=2) do i
    alt_lines!(axnormal, x, f_normal(i); color=mycmap[i])
    alt_lines!(axtransform, x, f_transform(i); color=mycmap[i])
    # autolimits!(axtransform)
end

fig

## Do the same thing with a Laplace transform

# First discrete version
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title=L"Function $f")
axtransform = fig[1, 2] = Axis(fig, title=L"Laplace Transform $\hat{f}$")

step = 0.01
x = step:step:5
ns = 1:5
N = Observable(1)
Ns = @lift ns[1:$N]
sumexp = @lift x->sum(exp(-n * x) / n for n in $Ns)

alt_lines!(axnormal, x, sumexp, linewidth=lw, color=:black)
xlims!(axnormal, extrema(x))
linkxaxes!(axnormal, axtransform)
record(fig, joinpath(target_path, "basic_laplace_transform.mp4"), ns; framerate=1) do n
# for n in ns
    N[] = n
    alt_lines!(axnormal, x, x->exp(-n * x) / n, color=@lift RGBA(sb[n], n <= $N))
    linesegments!(axtransform, Point2.(n, [0.0, 1 / n]), color=@lift(RGBA(sb[n], n <= $N)), linewidth=lw)
end

fig

## Now the discrete version with x^2
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title=L"Function $f(x^2)$")
axtransform = fig[1, 2] = Axis(fig, title=L"Laplace Transform $\hat{f}$")

step = 0.01
x = -5:step:5
ns = 1:5
N = Observable(1)
Ns = @lift ns[1:$N]
sumexp = @lift x->sum(exp(-n * x^2) * exp(-(n-1) / 10) for n in $Ns)
xlims!(axnormal, extrema(x))
c = √(π/2)

alt_lines!(axnormal, x, sumexp, linewidth=lw, color=:black)
record(fig, joinpath(target_path, "basic_laplace_transform_squared.mp4"), ns; framerate=1) do n
# for n in ns
    N[] = n
    alt_lines!(axnormal, x, x->exp(-n * x^2) * exp(-(n-1)/ 10), color=@lift RGBA(sb[n], n <= $N ? 0.6 : 0.0))
    linesegments!(axtransform, Point2.(n, [0.0, exp(- (n-1) / 10)]), color=@lift(RGBA(sb[n], n <= $N)), linewidth=lw)
end

fig

## Now do a continuous version

fig
## Show sq. exp. to laplace
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title=L"Function $f(x)$")
axtransform = fig[1, 2] = Axis(fig, title=L"Laplace transform $\hat{f}$")

to_lap = Observable(1)
n_frames_lap = 10
a_lap = Animation(1, 0.0, sineio(), n_frames_lap, 1.0)
mycmap = range(sb[1], sb[4], length=n_frames_lap)


ν = 3.0
st(x) = (1 + abs2(x)/ν)^(-(ν + 1) / 2)

fl_lap(x) = exp(-1/(4x)) / (2√(π) * x^(3/2))
fl_st(x) = ν * exp(-ν * x) * (ν * x)^((ν + 1) / 2 - 1) / gamma((ν + 1)/ 2 )

f_transform(to_lap) = x -> (1 - a_lap(to_lap)) * fl_lap(x) + a_lap(to_lap) * fl_st(x)
f_normal(to_lap) = x -> (1 - a_lap(to_lap)) * lap(x) + a_lap(to_lap) * st(x)

x_transform = 0:0.01:6
alt_lines!(axtransform, x_transform, f_transform(1); color=mycmap[1])
alt_lines!(axnormal, x, f_normal(1); color=mycmap[1])

# Save the fig
record(fig, joinpath(target_path, "laplace_continuous.mp4"), 1:n_frames_lap; framerate=2) do i
# for i in 2:n_frames_lap
    alt_lines!(axtransform, x_transform, f_transform(i); color=mycmap[i])
    alt_lines!(axnormal, x, f_normal(i); color=mycmap[i])
    to_lap[] = i
end
# annotations!(axnormal, ["f(x) = exp(-√(x²))"], [Point2(-5, 0.6)])
# annotations!(axtransform, ["f(x) = exp(-1 / 4x) / (2√(π)x^3/2"], [Point2(0.6, 0.6)])

fig