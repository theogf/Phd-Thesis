using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using CairoMakie
using MeasureTheory
using Soss
using ColorSchemes
sb = ColorSchemes.seaborn_colorblind

m1 = @model begin
    z ~ Normal(0, 3)
    x ~ Normal(0, exp(z/2))
end

m2 = @model begin
    z ~ Normal()
    x ~ Normal()
end

logdensity(m1(), (x=1, z=2))
logdensity(m2(), (x=1, z=2))


cg = ColorScheme(map(x->RGB(x...), include(joinpath(@__DIR__, "..", "colors", "color_flare"))))

function plot_dist!(f, model; n=1000, xlabel="x", ylabel="z")
    xlin = LinRange(-5, 5, n)
    ax = Axis(f[1,1]; xlabel, ylabel, xlabelsize=30.0, ylabelsize=30.0, yticks=-5:1:5, xticks=-5:1:5)
    contourf!(ax, xlin, xlin, (x,z)->exp(logdensity(model(), (;x, z))), colormap=cg)
    resize_to_layout!(f)
    f |> display
end

fig_path = joinpath(@__DIR__, "..", "..", "thesis", "chapters", "1_introduction", "figures")
mkpath(fig_path)

f = Figure(resolution=(600, 600));
plot_dist!(f, m1)
save(joinpath(fig_path, "neals_funnel_centered.pdf"), f)
f = Figure(resolution=(600, 600));
plot_dist!(f, m2; xlabel="x̃", ylabel="z̃")
save(joinpath(fig_path, "neals_funnel_non_centered.pdf"), f)
