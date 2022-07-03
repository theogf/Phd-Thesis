using Colors
import ColorSchemes.seaborn_colorblind as sb
using LaTeXStrings
using Animations

my_theme = Theme(
    Axis = (
        titlesize=28.0,
        xlabelsize=26.0,
        ylabelsize=30.0,
        xticklabelsize=22.0,
        yticklabelsize=22.0,
    ),
    Legend = (
        labelsize=40.0,
        framevisible=false,
    )
)