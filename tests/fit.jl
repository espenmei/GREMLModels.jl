using GREMLModels, LinearAlgebra, DataFrames, StatsModels, Test, Random

const rng = MersenneTwister(123)

n, m = 1000, 2000
v = randn(rng, n, m)
R1 = Symmetric(v * v' / m)
R2 = Diagonal(ones(n))
V = 2R1 + 4R2
y = cholesky(V).L * randn(rng, n)
r = [R1, R2]
  
dat = DataFrame(y = y, x = rand(rng, n), z = rand(rng, n))
m = fit(GREMLModel, @formula(y ~ 1 + x + z), dat, r)

@test loglikelihood(m) ≈ -2292.68483 rtol = 1.0e-5