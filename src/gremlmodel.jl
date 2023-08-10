"""
`GREMLData` holds input data of a variance component model
# Fields
- `y`: 'n' vector of responses
- `X`: 'n × p' matrix of covariates
- `r`: 'q' vector of 'n × n' correlation matrices
- `dims`: tuple of n = dimension of y, p = columns of X, q = dimension of r
"""
struct GREMLData{T<:AbstractFloat} 
    y::Vector{T}
    X::Matrix{T}
    r::Vector{<:AbstractMatrix} # Change to tuple
    dims::NamedTuple{(:n, :p, :q), NTuple{3, Int}}
    nms::Tuple
end

function GREMLData(y::Vector{T}, X::VecOrMat{T}, r::Vector{<:AbstractMatrix}, nms) where T<:AbstractFloat
    X = reshape(X, :, size(X, 2)) # Make sure X is a matrix
    n, p = size(X)
    dims = (n = n, p = p, q = length(r))
    GREMLData(y, X, r, dims, nms)
end

function GREMLData(y::Vector{T}, X::VecOrMat{T}, r::Vector{<:AbstractMatrix}) where T<:AbstractFloat
    nms = ("y", "x" .* string.(1:size(X, 2)))
    GREMLData(y, X, r, nms)
end

"""
`GREMLModel` holds data, parameters and optimization info of a variance component model
# Fields
- `data`: GREMLData
- `θ`: 'q' vector of scalar variance component parameters
- `Λ`: cholesky factorization of the 'n × n' model implied covariance matrix
- `μ`: 'n' vector of model implied means
- `opt`: GREMLOpt with optimization info
"""
struct GREMLModel{T<:AbstractFloat} <:StatsBase.StatisticalModel
    data::GREMLData{T} # Type!?
    θ::Vector{T}
    δ::Vector{T}
    Λ::Cholesky{T, Matrix{T}}
    β::Vector{T}
    μ::Vector{T}
    opt::GREMLOpt{T}
    invVX::Matrix{T}
end

function GREMLModel(d::GREMLData, θ::Vector{T},  θ_lb::Vector{T}, reml::Bool = false) where T<:AbstractFloat
    n, p, _ = d.dims
    m = GREMLModel(
        d,
        copy(θ), # Make a copy to avoid modifying input
        copy(θ),
        Cholesky(zeros(T, n, n), :U, 0),
        zeros(T, p),
        zeros(T, n),
        GREMLOpt(:LN_BOBYQA, copy(θ), θ_lb, reml),
        zeros(T, n, p)
        )
    update!(m, m.θ) # Gjør et update her?
    m.opt.finitial = objective(m)
    copyto!(m.opt.xfinal, m.opt.xinitial)
    m.opt.ffinal = m.opt.finitial
    m
end

function GREMLModel(d::GREMLData, θ_lb::Vector{<:AbstractFloat}, reml::Bool = false)
    GREMLModel(
    d,
    initialvalues(d),
    θ_lb,
    #(θ::Vector{T}) -> θ, # identity
    reml
    )
end

function initialvalues(d::GREMLData)
    n, _, q = d.dims
    X, y = d.X, d.y
    β = X \ y
    msse = sum(abs2, y - X * β) / n
    fill(msse / q, q)
end

isreml(m::GREMLModel) = m.opt.reml
transform(θ::Vector) = transform!(similar(θ), θ)
transform!(δ::Vector, θ::Vector) = copyto!(δ, θ)

function update!(m::GREMLModel, θ::Vector)
    updateμ!(updateΛ!(setθ!(m, θ)))
    m
end

function setθ!(m::GREMLModel, θ::Vector)
    copyto!(m.θ, θ)
    m
end

function updateΛ!(m::GREMLModel)
    δ = m.δ
    transform!(δ, m.θ)
    Λfac = m.Λ.factors 
    fill!(Λfac, zero(eltype(δ)))
    @inbounds for i ∈ 1:m.data.dims.q
        muladduppertri!(Λfac, δ[i], m.data.r[i]) #mul!(m.Λ.factors, δ[i], m.data.r[i], 1, 1) axpy!(δ[i], m.data.r[i], Λfac)
    end
    cholesky!(Symmetric(Λfac, :U), check = false) # Compute the cholesky factorization object (Tar mest tid)
    # Can be harder to debug withut check
    m
end

# GLS for β - Pawitan p. 440 (X'V⁻¹X)β = X'V⁻¹y
function updateμ!(m::GREMLModel)
    y, X = m.data.y, m.data.X
    invVX = m.invVX
    β = m.β
    ldiv!(invVX, m.Λ, X)
    # P/H = X, (X' * invVX) \ (invVX') -> "Hat matrix"
    mul!(β, invVX', y)
    ldiv!(bunchkaufman!(Symmetric(X'invVX)), β)
    #β = Symmetric(X'invVX) \ (invVX'y)
    mul!(m.μ, X, β)
end

function dfresidual(m::GREMLModel)::Int
    n, p, _ = m.data.dims
    isreml(m) ? n - p : n
end

# http://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/10-chol/chol.html#Multivariate-normal-density
# Weighted residual sums of squares
# (y - Xβ)'V^-1(y - Xβ)
# Same as y'Py in Lynch & Walsh
# Same as trace(V^-1(y - Xβ)(y - Xβ)')
function wrss(m::GREMLModel)
    ϵ = m.data.y - m.μ # Allocates
    invVϵ = m.Λ \ ϵ 
    dot(ϵ, invVϵ)
end

# Adjustment for reml likelihood
# Pawitan p. 441
# X' * V^-1 * X - It's computed in μ but cheap 
function rml(m::GREMLModel)
    m.data.X'm.invVX # logdet will give -Inf if not X full rank
end

# -2 × log-likelihood
function objective(m::GREMLModel)
    val = log(2π) * dfresidual(m) + logabsdet(m) + wrss(m)
    isreml(m) ? val + first(logabsdet(rml(m))) : val
end

# covariance of fixed effects
# Same as reml so make one X' * (m.Λ \ X) function 
function vcov(m::GREMLModel)
    X = m.data.X
    inv(X' * (m.Λ \ X))
end

# covariance of variance components
# scale to minimum of -2L
function vcovvc(m::GREMLModel)
    H = m.opt.H
    any(ismissing.(H)) ? H : inv(0.5 * H)
end

function stderror(m::GREMLModel)
    sqrt.(diag(vcov(m)))
end

function stderrorvc(m::GREMLModel)
    sqrt.(diag(vcovvc(m)))
end

function fixef!(v::Vector{T}, m::GREMLModel{T}) where T
    copyto!(v, m.data.X \ m.μ)
    v
end

function fixef(m::GREMLModel{T}) where T
    fixef!(zeros(T, m.data.dims.p), m)
end

function ranef!(W::Matrix, m::GREMLModel)
    δ = transform(m.θ)
    invVϵ = m.Λ \ (m.data.y - m.μ) # V^-1(y - Xβ) 
    for i ∈ 1:m.data.dims.q
        W[:, i] = δ[i] * m.data.r[i] * invVϵ
    end
    W
end

function ranef(m::GREMLModel{T}) where T
    W = zeros(T, m.data.dims.n, m.data.dims.q)
    ranef!(W, m)
end

# Implements
# StatsAPI
function StatsAPI.fit(::Type{GREMLModel}, f::FormulaTerm, df::DataFrame, r::Vector, reml::Bool=false)
    sch = schema(f, df)
    form = apply_schema(f, sch)
    y, X = modelcols(form, df)
    nms = coefnames(form)
    d = GREMLData(y, X, r, nms)
    θ_lb = fill(0.0, length(r))
    m = GREMLModel(d, θ_lb, reml)
    fit!(m)
end

function StatsAPI.fit!(m::GREMLModel)
    if m.opt.feval > 0
        throw(ArgumentError("This model has already been fitted"))
    end
    # Det er jo egentlig gjort ett update når modellen ble laget. Men da må du stole på at modellen ikke har blitt klussa med.
    function obj(θ::Vector, g)
        val = objective(update!(m, θ))
        update!(m.opt, θ, val)
        showiter(m.opt)
        val
    end
    opt = Opt(m.opt)
    min_objective!(opt, obj)
    minf, minx, ret = optimize!(opt, m.θ)
    m.opt.ret = ret
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    m
end


StatsAPI.coef(m::GREMLModel) = fixef(m)

function StatsAPI.coeftable(m::GREMLModel)
    co = fixef(m)
    se = stderror(m)
    z = co ./ se
    pval = ccdf.(Chisq(1), abs2.(z))
    xnms = m.data.nms[2]
    tab = hcat(co, se, z, pval)
    CoefTable(
    tab, # value cols
    ["Coef.", "Std. Error", "z", "Pr(>|z|)"], # Colnames
    xnms, # rownames
    4, # pvalcol
    3, # zcol
    )
end

StatsAPI.deviance(m::GREMLModel) = objective(m)

StatsAPI.dof(m::GREMLModel) = m.data.dims.p + m.data.dims.q

# Error for reml?
StatsAPI.loglikelihood(m::GREMLModel) = -0.5 * objective(m)

# StatsBase
function StatsBase.dof_residual(m::GREMLModel)::Int
    n, p, q = m.data.dims
    n - p - q
end

StatsBase.modelmatrix(m::GREMLModel) = m.data.X

StatsBase.nobs(m::GREMLModel) = m.data.dims.n

StatsBase.response(m::GREMLModel) = m.data.y

# StatsModels
# It is diffiult to check rterms when constraints are imposed by manipulationg any R structure
function StatsModels.isnested(m1::GREMLModel, m2::GREMLModel; atol::Real = 0.0)
    response = m1.data.y == m2.data.y
    criterion = m1.opt.reml == m2.opt.reml
    fterms = issubset(m1.data.X, m2.data.X)
    if m1.opt.reml && m2.opt.reml
        fterms = m1.data.X == m2.data.X
    end
    response && criterion && fterms
end

# Base
function Base.show(io::IO, m::GREMLModel)
    if m.opt.feval <= 0
        @warn("This model has not been fitted.")
        return nothing
    end
    # Fit measures
    oo = objective(m)
    oovals = Ryu.writefixed.([-0.5 * oo, oo, aic(m), aicc(m), bic(m)], 4)
    fieldwd = max(maximum(textwidth.(oovals)) + 1, 11)
    for i ∈ ["logLik", "-2 logLik", "AIC", "AICc", "BIC"]
        print(io, rpad(i, fieldwd))
    end
    println(io)
    for i ∈ oovals
        print(io, rpad(i, fieldwd))
    end
    # Variance components
    println(io, "\n\n Variance component parameters:")
    vcvals = Ryu.writefixed.(m.θ, 4)
    vcse = stderrorvc(m)
    vcsevals = any(ismissing.(vcse)) ? fill('-', length(vcse)) : Ryu.writefixed.(vcse, 4)
    for i ∈ ["Comp.", "Est.", "Std. Error"]
        print(io, i, "\t")
    end
    println(io)
    for i ∈ eachindex(vcvals)
        print(io, "θ" * Char(0x2080 + i), "\t", vcvals[i], "\t", vcsevals[i], "\n")
    end
    # Fixed effects
    println(io, "\n Fixed-effects parameters:")
    show(io, coeftable(m))
end