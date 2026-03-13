
"""Solver configuration for NLopt-based optimization. All fields have sensible defaults."""
struct NLoptSolver
    algorithm::Symbol
    xtol_abs::Float64
    xtol_rel::Float64
    ftol_abs::Float64
    ftol_rel::Float64
    maxfeval::Int
end

NLoptSolver(;
    algorithm::Symbol = :LN_BOBYQA,
    xtol_abs::Real    = 1e-10,
    xtol_rel::Real    = 0.0,
    ftol_abs::Real    = 1e-8,
    ftol_rel::Real    = 1e-12,
    maxfeval::Int     = -1
) = NLoptSolver(algorithm, xtol_abs, xtol_rel, ftol_abs, ftol_rel, maxfeval)

#- `reml`: boolean indicator for reml
#- `H`: matrix with missing or twice inverse covariance matrix of θ
mutable struct GREMLOpt{T<:AbstractFloat}
    xlb::Vector{T}
    feval::Int
    xinitial::Vector{T}
    xfinal::Vector{T}
    finitial::T
    ffinal::T
    ret::Symbol
    reml::Bool
    H::Matrix{Union{Missing, T}}
    ∇::Vector{Union{Missing, T}}
end

function GREMLOpt(xinitial::Vector{T}, xlb::Vector{T}, reml::Bool = false) where {T<:AbstractFloat}
    q = length(xlb)
    GREMLOpt(
        xlb,
        0,
        xinitial,
        copy(xinitial),
        T(0),
        T(0),
        :FAILURE,
        reml,
        Matrix{Union{Missing, T}}(missing, q, q),
        Vector{Union{Missing, T}}(missing, q)
    )
end

function update!(o::GREMLOpt, θ::AbstractVector{T}, val::T) where T<:AbstractFloat
    o.feval += 1
    o.xfinal = θ
    o.ffinal = val
    o
end

function NLopt.Opt(solver::NLoptSolver, opt::GREMLOpt)
    o = NLopt.Opt(solver.algorithm, length(opt.xlb))
    NLopt.lower_bounds!(o, opt.xlb)
    NLopt.xtol_abs!(o, solver.xtol_abs)
    NLopt.xtol_rel!(o, solver.xtol_rel)
    NLopt.ftol_abs!(o, solver.ftol_abs)
    NLopt.ftol_rel!(o, solver.ftol_rel)
    NLopt.maxeval!(o, solver.maxfeval)
    o
end

#function Base.show(io::IO, ::MIME"text/plain", o::VCOpt)
   # for i ∈ 1:fieldcount(VCOpt)
  #      name = fieldname(VCOpt, i)
 #       val = getfield(o, i)
#        println(io, "$name = $val")
#    end
#end
#Base.show(io::IO, o::VCOpt) = Base.show(io, MIME"text/plain"(), o)
function Base.show(io::IO, o::GREMLOpt)
    for i ∈ 1:fieldcount(GREMLOpt)
        name = fieldname(GREMLOpt, i)
        val = getfield(o, i)
        println(io, "$name = $val")
    end
end

function showvector(io, v::AbstractVector)
    print(io, "[")
    for (i, elt) in enumerate(v)
        i > 1 && print(io, ", ")
        print(io, elt)
    end
    print(io, "]")
end

function showiter(io, o::GREMLOpt)
    print(io, "iteration: ",o.feval)
    print(io, ", objective: ", o.ffinal)
    print(io, ", θ: ", o.xfinal)
    if !any(ismissing.(o.∇))
        print(io, ", ∇: ")
        showvector(io, o.∇)
    end
    println(io)
end

showiter(o::GREMLOpt) = showiter(IOContext(stdout, :compact => true), o)