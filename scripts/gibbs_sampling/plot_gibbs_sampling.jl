using CairoMakie
using Makie
using MeasureTheory
using Soss

function plot_dist!(fig, f; n=1000, xlabel="x", ylabel="z")
    xlin = LinRange(-1, 7, n)
    ax = Axis(fig[1,1]; xlabel, ylabel, xlabelsize=30.0, ylabelsize=30.0, yticks=-1:1:7, xticks=-1:1:7)
    contourf!(ax, xlin, xlin, (x,z)->f(x, z))
    resize_to_layout!(fig)
    fig |> display
    return ax
end

function create_path!(ax, n=10)
    x1 = randn()
    x2 = randn()
    xs = []
    push!(xs, [x1, x2])
    for i in 1:n
        x1 = rand(cond_p(x2))
        push!(xs, [x1, x2])
        x2 = rand(cond_p(x1))
        push!(xs, [x1, x2])
    end
    lines!(ax, first.(xs), last.(xs), color=:black)
    scatter!(ax, first.(xs), last.(xs), color=:black)
end

const c = 1/20216.335877
p(x1, x2) = c * exp(- (x1^2 * x2^2 + x1^2 + x2^2 - 8x1 - 8x2) / 2)
cond_p(x) = Normal(4 / (x^2 + 1), 1 / (x^2 + 1))


fig_path = joinpath(@__DIR__, "..", "..", "thesis", "chapters", "1_introduction", "figures")
mkpath(fig_path)

fig = Figure(resolution=(600, 600));
ax= plot_dist!(fig, p)
create_path!(ax, 4)
fig
save(joinpath(fig_path, "gibbs_sampling.pdf"), f)
