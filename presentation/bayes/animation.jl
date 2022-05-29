using Distributions
using ConjugatePriors

prior_height = Normal(1.5, 0.5)
data = [
    1.65,
    1.7,
    1.8,
    1.92,
    1.54,
    1.65,
    1.82,
]

x = 0:0.01:3

plot(x, pdf.(prior_height, x))
