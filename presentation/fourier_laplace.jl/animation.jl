using GLMakie
using FFTW
using Animations

sqexp(x) = exp(-x^2)
lap(x) = exp(-abs(x))
trans_sqexp(x) = exp(-x^2/4) / (sqrt(2))
trans_lap(x) = sqrt(2/π) / (abs2(x) + 1)
x = -5:0.01:5
y = sqexp.(x)

## 

fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title="f")
axtransform = fig[1, 2] = Axis(fig, title="transform")

ns = 1:5

lines!(axnormal, x, x->sum(cos(n * x) / n for n in ns))
xlims!(axnormal, extrema(x))
c = √(π/2)

points = mapreduce(vcat, ns) do n
    vcat(Point2.(-n, [0.0, c / n]), Point2.(n, [0.0, c / n]))
end
linesegments!(axtransform, points)
linkaxes!(axnormal, axtransform)
fig
## Show sq. exp. to laplace
fig = Figure()
axnormal = fig[1, 1] = Axis(fig, title="f")
axtransform = fig[1, 2] = Axis(fig, title="transform")

to_lap = Observable(1)
n_frames_lap = 100
a_lap = Animation(1, 0.0, sineio(), n_frames_lap, 1.0)


f_transform = @lift x -> (1 - a_lap($to_lap)) * trans_sqexp(x) + a_lap($to_lap) * trans_lap(x)
f_normal = @lift x -> (1 - a_lap($to_lap)) * sqexp(x) + a_lap($to_lap) * lap(x)

lines!(axtransform, x, f_transform)

# Save the fig
lines!(axnormal, x, f_normal)
for i in 1:n_frames_lap
    to_lap[] = i
    display(fig)
end

fig