# presteps.jl
# ---------------------------------------------------------------------------
# Preprocessing for GREMLModels.
#
# In practice the genetic relationship matrix (GRM) is already computed and
# stored as its upper triangle in an HDF5 file. This script focuses on the
# steps from there:
#
#   1. Read the GRM from HDF5 into a dense Symmetric matrix (memory-safe:
#      the packed vector is never held alongside the full matrix).
#   2. Select trios, pruning closely related parents.
#   3. Build the trio-specific relationship matrices that feed a GREMLModel.
#
# The GRM computation/storage helpers (grm_blocks, write_grm) are kept only so
# the bundled example can generate an HDF5 file to read back.
#
# This is a standalone, versioned helper script. It is not included by the
# GREMLModels package.
#
# Dependencies: SnpArrays, HDF5, LinearAlgebra.
# (The downstream modelling step additionally uses GREMLModels, DataFrames,
#  StatsModels.)
# ---------------------------------------------------------------------------

using LinearAlgebra
using SnpArrays
using HDF5

# ===========================================================================
# Reading the GRM from HDF5
#   Layout "grm_u": the upper triangle, column-major vectorized.
#   Element (i, j) with i ≤ j sits at linear position div(j*(j-1), 2) + i.
# ===========================================================================

# Read a GRM stored in the "grm_u" layout into a dense Symmetric matrix.
# The matrix is filled one column at a time directly from the dataset, so only
# the result (plus a single-column buffer) is ever in memory — the full packed
# vector is never materialized alongside it.
function read_grm(filename::AbstractString; nme::AbstractString = "grm_u")
    h5open(filename, "r") do file
        d = file[nme]
        p = length(d)
        k = div(isqrt(8p + 1) - 1, 2)
        p == div(k * (k + 1), 2) || throw(ArgumentError("dataset length $p is not a triangular number"))
        G = Matrix{eltype(d)}(undef, k, k)
        base = 0
        @inbounds for j in 1:k                # column j holds upper rows 1:j
            G[1:j, j] = d[base + 1 : base + j]
            base += j
        end
        Symmetric(G, :U)
    end
end

# ===========================================================================
# Trio selection and trio-specific GRMs
# ===========================================================================

# From candidate trios (aligned offspring/mother/father index vectors, one entry
# per trio), keep only trios where BOTH parents survive kinship pruning (removes
# closely related parent pairs). Returns the surviving (offspring, mother, father)
# index vectors.
#
# Pruning only concerns relationships among parents, so we prune on a view of the
# parents×parents block of A (no large copy). `kinship_pruning` comes from
# SnpArrays and returns a BitVector over the rows it is given, here ordered
# [mothers; fathers].
function sel_trio(A, oid, mid, fid; method::Symbol = :bottom_up, cutoff::Real = 0.1)
    nm = length(mid)
    parents = vcat(mid, fid)
    keep = kinship_pruning(view(A, parents, parents); method = method, cutoff = cutoff)
    sel = findall(view(keep, 1:nm) .& view(keep, nm + 1:lastindex(keep)))
    (oid[sel], mid[sel], fid[sel])
end

# Build the seven trio-specific relationship matrices from the GRM A, indexed at
# the (surviving) trio individuals. All blocks are small (trio-sized) dense
# copies; R is the residual identity term.
function computeGRMs(A, oid, mid, pid)
    Amm = A[mid, mid]
    App = A[pid, pid]
    Aoo = A[oid, oid]
    Dpm = A[pid, mid] + A[mid, pid]
    Dom = A[oid, mid] + A[mid, oid]
    Dop = A[oid, pid] + A[pid, oid]
    R = Diagonal(ones(length(oid)))
    [Amm, App, Aoo, Dpm, Dom, Dop, R]
end

# ===========================================================================
# GRM computation and storage (only needed to generate an HDF5 file to read)
# ===========================================================================

# Partition 1:size into `nblocks` contiguous ranges; any remainder goes to the
# first block.
function block_indices(size::Int, nblocks::Int)
    if nblocks > size
        throw(ArgumentError("Number of blocks can not be greater than the number of indices"))
    end
    blocksize = div(size, nblocks)
    rest = mod(size, nblocks)
    indices = Vector{UnitRange{Int}}(undef, nblocks)
    indices[1] = 1:blocksize + rest
    for i ∈ 2:nblocks
        first = last(indices[i - 1]) + 1
        indices[i] = first:first + blocksize - 1
    end
    indices
end

# Compute the GRM 2Φ = G·G' / nsnps by accumulating rank-k updates over blocks
# of SNPs. Only a single npeople×blocksize buffer is held in addition to the
# npeople×npeople result, so peak memory is dominated by the result itself.
function grm_blocks(s::SnpArray, nblocks::Int)
    T = Float64
    npeople, nsnps = size(s)
    indices = block_indices(nsnps, nblocks)
    Φ = zeros(T, npeople, npeople)
    α = inv(nsnps)   # folds the factor 2 in: 2 * (G·G' / 2nsnps) = G·G' / nsnps

    G = zeros(T, npeople, maximum(length, indices))   # one buffer, sized to largest block
    for i ∈ eachindex(indices)
        idx = indices[i]
        Gv = @view G[:, 1:length(idx)]
        @views copyto!(Gv, s[:, idx], model = ADDITIVE_MODEL, impute = true, center = true, scale = true)
        BLAS.syrk!('U', 'N', α, Gv, one(T), Φ)   # Φ ← α·Gv·Gv' + Φ (upper triangle)
        println("Block $i / $nblocks")
    end
    Symmetric(Φ, :U)
end

# Write G to `<filename>.h5` in the "grm_u" layout, streaming a column at a time
# so the full packed vector is never materialized. Stores G's element type as-is
# (pass a Float32 matrix to halve the file).
function write_grm(G::Symmetric{T}, filename::AbstractString;
                   nme::AbstractString = "grm_u") where {T<:AbstractFloat}
    G.uplo == 'U' || throw(ArgumentError("expected upper-stored Symmetric(.., :U)"))
    P = parent(G)
    k = size(P, 1)
    p = div(k * (k + 1), 2)
    h5open(filename * ".h5", "w") do file
        d = create_dataset(file, nme, T, (p,); chunk = (min(p, 1 << 22),))
        base = 0
        @inbounds for j in 1:k                # column j holds upper rows 1:j
            d[base + 1 : base + j] = view(P, 1:j, j)
            base += j
        end
    end
end

# Print the in-memory size of a dense rows×cols Float64 matrix in GiB.
function matsize_gb(rows, cols)
    gb = rows * cols * 8 / (1024^3)
    println("Gb: $gb")
    gb
end

# ===========================================================================
# Example pipeline
# ===========================================================================

# Step 0 (one-time): compute the GRM from PLINK data and store it. In practice
# this already exists, so normally you start at step 1.
trio = SnpData(joinpath(@__DIR__, "..", "data", "trio"))
matsize_gb(trio.people, trio.people)              # size of the dense GRM in memory
A0 = grm_blocks(trio.snparray, 100)
write_grm(A0, "grm4")

# 1. Read the GRM from HDF5 into a dense Symmetric matrix.
A = read_grm("grm4.h5")

# 2. Define candidate trios (this toy dataset is laid out mothers|fathers|offspring)
#    and prune closely related parents.
n = size(A, 1)
k = div(n, 3)
mid = collect(1:k)          # mothers
fid = collect(k+1:2k)       # fathers
oid = collect(2k+1:3k)      # offspring
oid, mid, fid = sel_trio(A, oid, mid, fid)

# 3. Build the trio-specific relationship matrices.
grms = computeGRMs(A, oid, mid, fid)

# `grms` is now the vector of relationship matrices to pass to a GREMLModel.
