using GLMakie
using FFTW
using Animations
import ColorSchemes.seaborn_colorblind as sb
using Colors

## Basis functions
sqexp(x) = exp(-x^2)
lap(x) = exp(-abs(x))
trans_sqexp(x) = exp(-x^2/4) / (sqrt(2))
trans_lap(x) = sqrt(2/π) / (abs2(x) + 1)
x = -5:0.01:5
y = sqexp.(x)

lw = 5.0

# Workaround for the current issue with GLMakie
alt_lines! = lines!

## First part, demonstration of the Fourier transform (discrete first)

fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title="Function")
axtransform = fig[1, 2] = Axis(fig, title="Fourier Transform")

ns = 1:5
N = Observable(5)
Ns = @lift ns[1:$N]
sumcos = @lift x->sum(cos(n * x) / n for n in $Ns)

alt_lines!(axnormal, x, sumcos, linewidth=lw)
c = √(π/2)

for n in ns
    alt_lines!(axnormal, x, x->cos(n * x) / n, color=@lift RGBA(sb[n], n <= $N))
    linesegments!(axtransform, vcat(Point2.(-n, [0.0, c / n]), Point2.(n, [0.0, c / n])), color=@lift(RGBA(sb[n], n <= $N)), linewidth=lw)
end


xlims!(axnormal, extrema(x))

linkaxes!(axnormal, axtransform)
fig

## Fourier transform in continuous setting
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title="Function")
axtransform = fig[1, 2] = Axis(fig, title="Fourier transform")

to_lap = Observable(1)
n_frames_lap = 100
a_lap = Animation(1, 0.0, sineio(), n_frames_lap, 1.0)


f_transform = @lift x -> (1 - a_lap($to_lap)) * trans_sqexp(x) + a_lap($to_lap) * trans_lap(x)
f_normal = @lift x -> (1 - a_lap($to_lap)) * sqexp(x) + a_lap($to_lap) * lap(x)

alt_lines!(axtransform, x, f_transform)
annotations!(axtransform, ["f(x) = exp(-x²/2)"], [Point2(-4, 0.6)])
# Save the fig with this continuous version

alt_lines!(axnormal, x, f_normal)
# Show the result
fig
## Show sq. exp. to laplace
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title="Function")
axtransform = fig[1, 2] = Axis(fig, title="Fourier Transform")

to_lap = Observable(1)
n_frames_lap = 100
a_lap = Animation(1, 0.0, sineio(), n_frames_lap, 1.0)


f_transform = @lift x -> (1 - a_lap($to_lap)) * trans_sqexp(x) + a_lap($to_lap) * trans_lap(x)
f_normal = @lift x -> (1 - a_lap($to_lap)) * sqexp(x) + a_lap($to_lap) * lap(x)

alt_lines!(axtransform, x, f_transform)

# Save the fig
alt_lines!(axnormal, x, f_normal)
for i in 1:n_frames_lap
    to_lap[] = i
end

fig

## Do the same thing with a Laplace transform

# First discrete version
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title="Function")
axtransform = fig[1, 2] = Axis(fig, title="Laplace Transform")

step = 0.01
x = step:step:5
ns = 1:5
N = Observable(5)
Ns = @lift ns[1:$N]
sumcos = @lift x->sum(exp(-n * x) / n for n in $Ns)

alt_lines!(axnormal, x, sumcos, linewidth=lw)

for n in ns
    alt_lines!(axnormal, x, x->exp(-n * x) / n, color=@lift RGBA(sb[n], n <= $N))
    linesegments!(axtransform, Point2.(n, [0.0, 1 / n]), color=@lift(RGBA(sb[n], n <= $N)), linewidth=lw)
end


xlims!(axnormal, extrema(x))

linkxaxes!(axnormal, axtransform)
fig

## Now the discrete version with x^2
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title="Function")
axtransform = fig[1, 2] = Axis(fig, title="Laplace Transform")

step = 0.01
x = -5:step:5
ns = 1:5
N = Observable(5)
Ns = @lift ns[1:$N]
sumcos = @lift x->sum(exp(-n * x^2) * exp(-n / 10) for n in $Ns)

alt_lines!(axnormal, x, sumcos, linewidth=lw)

for n in ns
    alt_lines!(axnormal, x, x->exp(-n * x^2) / n, color=@lift RGBA(sb[n], n <= $N))
    linesegments!(axtransform, Point2.(n, [0.0, exp(- n / 10)]), color=@lift(RGBA(sb[n], n <= $N)), linewidth=lw)
end


xlims!(axnormal, extrema(x))
c = √(π/2)

fig

## Now do a continuous version

fig
## Show sq. exp. to laplace
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title="Function")
axtransform = fig[1, 2] = Axis(fig, title="Laplace transform")

to_lap = Observable(1)
n_frames_lap = 100
a_lap = Animation(1, 0.0, sineio(), n_frames_lap, 1.0)


f_transform = x -> exp(-1/(4x)) / (2√(π) * x^(3/2))
f_normal = x -> exp(-abs(x))

alt_lines!(axtransform, 0:0.01:6, f_transform)

# Save the fig
alt_lines!(axnormal, x, f_normal)
for i in 1:n_frames_lap
    to_lap[] = i
end
annotations!(axnormal, ["f(x) = exp(-√(x²))"], [Point2(-5, 0.6)])
annotations!(axtransform, ["f(x) = exp(-1 / 4x) / (2√(π)x^3/2"], [Point2(0.6, 0.6)])

fig