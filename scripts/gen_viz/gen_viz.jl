using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1])