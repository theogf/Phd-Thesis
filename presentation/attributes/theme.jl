using Colors
import ColorSchemes.seaborn_colorblind as sb
using ColorSchemes
using LaTeXStrings
using Animations

sbhide = ColorScheme(vcat(RGB.(collect(sb)), RGBA(colorant"white", 0.0)))
large_dims = (28, 12)
dpi = 100 / 2.54
resolution = large_dims .* dpi

my_theme = Theme(
    background_color=:white,
    palette = (
        colors = sb,
    ),
    Axis = (
        titlesize=28,
        xlabelsize=26,
        ylabelsize=30,
        xticklabelsize=22,
        yticklabelsize=22,
    ),
    Legend = (
        labelsize=40.0,
        framevisible=false,
    ),
    Figure = (
        resolution = large_dims .* dpi,
    ),
)

set_theme!(my_theme)
