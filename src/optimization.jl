
#- `reml`: boolean indicator for reml
#- `H`: matrix with missing or twice inverse covariance matrix of θ
mutable struct GREMLOpt{T<:AbstractFloat}
    optimizer::Symbol
    xlb::Vector{T}
    xtol_abs::T
    xtol_rel::T
    ftol_abs::T
    ftol_rel::T
    feval::Int
    maxfeval::Int
    xinitial::Vector{T}
    xfinal::Vector{T}
    finitial::T
    ffinal::T
    ret::Symbol
    reml::Bool
    H::Matrix{Union{Missing, T}}
    ∇::Vector{Union{Missing, T}}
end

function GREMLOpt(optimizer::Symbol, xinitial::Vector{T}, xlb::Vector{T}, reml::Bool = false) where {T<:AbstractFloat}
    q = length(xlb)
    GREMLOpt(
        optimizer,
        xlb,
        T(10^-10),
        T(0),
        T(10^-8),
        T(10^-12),
        0,
        -1,
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

function NLopt.Opt(o::GREMLOpt)
    opt = NLopt.Opt(o.optimizer, length(o.xlb))
    NLopt.lower_bounds!(opt, o.xlb)
    NLopt.xtol_abs!(opt, o.xtol_abs)
    NLopt.xtol_rel!(opt, o.xtol_rel)
    NLopt.ftol_abs!(opt, o.ftol_abs)
    NLopt.ftol_rel!(opt, o.ftol_rel)
    NLopt.maxeval!(opt, o.maxfeval)
    opt
end

function converged(o::GREMLOpt)
    conv = true    
    f_abs = abs(o.ffinal - o.finitial)
    x_abs = abs.(o.xfinal - o.xinitial)
    
    if f_abs < o.ftol_abs ||  f_abs < o.ftol_rel * o.ffinal
        o.ret = :FTOL_REACHED
    elseif all(x_abs .< o.xtol_abs) || x_abs < o.xtol_rel * o.xfinal
        o.ret = :XTOL_REACHED
    elseif o.maxfeval >= 0 && o.feval >= o.maxfeval
        o.ret = :MAXEVAL_REACHED
    else
        conv = false
    end
    conv
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