using Distributions
using GLMakie
# Make a simple regression (BMI vs size)


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

noise = 0.5

loglikelihood(x) = sum(logpdf(Normal(x, noise), data))
likelihood(x) = exp(loglikelihood(x))
likelihood(1.5)

joint(x) = likelihood(x) * pdf(prior_height, x)

x = 0:0.01:3

mean(data)
mean(abs2, data)

Normal()  Normal()


fig, axis, plt = lines(x, pdf.(prior_height, x))
lines!(axis, x, joint.(x))
fig