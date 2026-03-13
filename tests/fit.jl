using Revise, GREMLModels, LinearAlgebra, DataFrames, StatsModels, Test, Random, MixedModels
import MixedModels: LinearMixedModel
import GREMLModels: vcov, stderror, objective

const rng = MersenneTwister(123)

n, nsnp = 1000, 2000
v = randn(rng, n, nsnp)
R1 = Symmetric(v * v' / nsnp)
R2 = Diagonal(ones(n))
V = 2R1 + 4R2
y = cholesky(V).L * randn(rng, n)
r = [R1, R2]

dat = DataFrame(y = y, x = rand(rng, n), z = rand(rng, n))

m = fit(GREMLModel, @formula(y ~ 1 + x + z), dat, r, verbose = false)
m2 = fit(GREMLModel, @formula(y ~ 1), dat, r, verbose = false)

# --- coeftable smoke tests ---
@testset "coeftable" begin
    @test_nowarn coeftable(m2)
    @test occursin("(Intercept)", sprint(show, coeftable(m2)))
end

# --- variance component estimates close to true values (δ₁≈2, δ₂≈4) ---
@testset "variance component estimates" begin
    @test isapprox(m.δ[1], 2.0, rtol = 0.25)
    @test isapprox(m.δ[2], 4.0, rtol = 0.25)
end

# --- wrss matches reference formula ---
@testset "wrss correctness" begin
    ϵ = m.data.y - m.μ
    ref = dot(ϵ, m.Λ \ ϵ)
    @test isapprox(GREMLModels.wrss(m), ref, rtol = 1e-10)
end

# --- hessian restores model state ---
@testset "hessian restores θ" begin
    θ_before = copy(m.θ)
    H = hessian(m)
    @test m.θ ≈ θ_before
    @test m.μ ≈ m.data.X * m.β   # internal state consistent
end

# --- NLoptSolver customization ---
@testset "NLoptSolver algorithm" begin
    m_cobyla = fit(GREMLModel, @formula(y ~ 1), dat, r,
                   solver = NLoptSolver(algorithm = :LN_COBYLA), verbose = false)
    @test isapprox(m_cobyla.δ[1], m2.δ[1], rtol = 0.05)
    @test isapprox(m_cobyla.δ[2], m2.δ[2], rtol = 0.05)
end

@testset "NLoptSolver maxfeval" begin
    m_limited = fit(GREMLModel, @formula(y ~ 1), dat, r,
                    solver = NLoptSolver(maxfeval = 5), verbose = false)
    @test m_limited.opt.ret == :MAXEVAL_REACHED
end

# --- refitting guard ---
@testset "refit guard" begin
    @test_throws ArgumentError fit!(m)
end

# --- vcov and stderror sanity ---
@testset "vcov and stderror" begin
    V_fe = vcov(m)
    @test V_fe ≈ V_fe'   # symmetric up to floating point
    @test isposdef(V_fe)
    @test all(stderror(m) .> 0)
end

# --- model with single identity component (degenerate / OLS-like) ---
@testset "single component model" begin
    m_id = fit(GREMLModel, @formula(y ~ 1 + x), dat, [Matrix(R2)], verbose = false)
    @test m_id.opt.ret ∉ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY]
    @test length(coef(m_id)) == 2
end

# --- REML vs ML ---
@testset "REML vs ML" begin
    m_ml   = fit(GREMLModel, @formula(y ~ 1), dat, r, reml = false, verbose = false)
    m_reml = fit(GREMLModel, @formula(y ~ 1), dat, r, reml = true,  verbose = false)
    # Estimates should be close but objectives differ
    @test isapprox(m_ml.δ[1], m_reml.δ[1], rtol = 0.10)
    @test objective(m_ml) != objective(m_reml)
end

# --- MixedModels.jl comparison ---
# Random intercept model: y = Xβ + Zᵍu + ε
#   u  ~ N(0, σ²_g I),  ε ~ N(0, σ²_e I)
#   Cov(y) = σ²_g * ZZ' + σ²_e * I
# GREMLModels with R1 = ZZ', R2 = I must give the same ML log-likelihood.
@testset "MixedModels comparison" begin
    rng_mm = MersenneTwister(99)
    n_g, n_per = 40, 15
    n_mm = n_g * n_per
    group = repeat(1:n_g, inner = n_per)

    # Group indicator matrix Z (n × n_g), so ZZ' is the random-intercept covariance
    Z = zeros(n_mm, n_g)
    for (i, g) in enumerate(group)
        Z[i, g] = 1.0
    end

    x_mm = randn(rng_mm, n_mm)
    V_mm = 1.5 .* (Z * Z') .+ 2.0 .* Matrix(I, n_mm, n_mm)
    y_mm = cholesky(Symmetric(V_mm)).L * randn(rng_mm, n_mm)

    dat_mm = DataFrame(y = y_mm, x = x_mm, g = string.(group))
    R1_mm = Symmetric(Z * Z')
    R2_mm = Matrix{Float64}(I, n_mm, n_mm)

    m_greml = fit(GREMLModel, @formula(y ~ 1 + x), dat_mm, [R1_mm, R2_mm],
                  reml = false, verbose = false)
    m_mm = fit(LinearMixedModel, @formula(y ~ 1 + x + (1|g)), dat_mm, REML = false)

    @test isapprox(loglikelihood(m_greml), loglikelihood(m_mm), rtol = 1e-4)
end
