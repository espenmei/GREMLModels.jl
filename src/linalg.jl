
# Symmetric Tag slows this down. Is it the indexing?
# Antakeligvis fordi den lagrer :U. Så iterer heller den.
function muladduppertri!(C::AbstractMatrix{T}, α::Number, B::AbstractMatrix{T}) where {T<:AbstractFloat}
    for i ∈ axes(C, 2)
        for j ∈ 1:i
           C[j,i] += α * B[j,i]
       end
   end
   C
end

function muladduppertri!(C::Matrix{T}, α::Number, B::Diagonal{T, Vector{T}}) where {T<:AbstractFloat}
    for i ∈ axes(C, 2)
        C[i,i] += α * B[i,i]
    end
    C
end

function LinearAlgebra.dot(A::Symmetric{T}, B::Diagonal{T, Vector{T}}) where {T<:AbstractFloat}
    dot(A, Symmetric(B))
end

function LinearAlgebra.dot(A::Diagonal{T, Vector{T}}, B::Symmetric{T}) where {T<:AbstractFloat}
    dot(Symmetric(A), B)
end

# This should not be necessary - The cov matrix should be pd
function LinearAlgebra.logabsdet(m::GREMLModel)
    ld = zero(eltype(m.Λ))
    @inbounds for i ∈ diagind(m.Λ.factors)
        ld += log(abs(m.Λ.factors[i]))
    end
    ld + ld
end