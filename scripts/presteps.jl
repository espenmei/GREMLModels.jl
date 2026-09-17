# presteps.jl
# ---------------------------------------------------------------------------
# Preprocessing for GREMLModels.
#
# Pipeline:
#   1. Compute a genetic relationship matrix (GRM, 2Φ) from PLINK SNP data,
#      blocked over SNPs to keep memory bounded.
#   2. Store it compactly in HDF5 (upper triangle only, column-vectorized).
#   3. Read back ONLY the individuals you actually need (e.g. selected trios)
#      without materializing the full matrix, and build the trio-specific
#      relationship matrices that feed into a GREMLModel.
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
# 1. GRM computation from SNP data (blocked over SNPs)
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

# ===========================================================================
# 2. Compact GRM storage in HDF5
#    Layout "grm_u": the upper triangle, column-major vectorized.
#    Element (i, j) with i ≤ j is stored at linear position div(j*(j-1),2) + i.
# ===========================================================================

# Unwrap to the dense array whose upper triangle holds the data, so bulk column
# copies can bypass Symmetric's element-wise getindex.
function _upperparent(M::AbstractMatrix)
    if M isa Symmetric
        M.uplo == 'U' || throw(ArgumentError("expected upper-stored Symmetric(.., :U)"))
        return parent(M)
    end
    M
end

# Vectorize the upper triangle of M (column-major) into a vector.
function ugrm2vec(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    k = size(M, 1)
    p = div(k * (k + 1), 2)
    v = Vector{T}(undef, p)
    P = _upperparent(M)
    stride = size(P, 1)
    off = 0
    @inbounds for i in 1:k                       # column i, upper part = rows 1:i
        copyto!(v, off + 1, P, (i - 1) * stride + 1, i)
        off += i
    end
    v
end

# Rebuild a dense (upper-filled) matrix from an upper-triangle vector.
function vec2ugrm(v::AbstractVector{T}) where {T<:AbstractFloat}
    k = length(v)
    p = div(isqrt(8k + 1) - 1, 2)
    G = zeros(T, p, p)
    off = 0
    @inbounds for i in 1:p                        # column i, rows 1:i
        copyto!(G, (i - 1) * p + 1, v, off + 1, i)
        off += i
    end
    G
end

# Write G to `<filename>.h5`, streaming chunks of each column into a chunked
# dataset. Never materializes the full upper-triangle vector, and chunking makes
# the partial reads in `read_grm_subset` efficient. Stores G's element type
# as-is (pass a Float32 matrix to halve the file).
function write_grm(G::AbstractMatrix{T}, filename::AbstractString;
                   nme::AbstractString = "grm_u") where {T<:AbstractFloat}
    P = _upperparent(G)
    k = size(P, 1)
    p = div(k * (k + 1), 2)
    stride = size(P, 1)
    h5open(filename * ".h5", "w") do file
        chunklen = min(p, 1 << 22)
        d = create_dataset(file, nme, T, (p,); chunk = (chunklen,))
        buf = Vector{T}(undef, chunklen)
        dpos = 0
        @inbounds for i in 1:k                    # column i, upper part = rows 1:i
            srcpos = (i - 1) * stride + 1
            remaining = i
            while remaining > 0
                ncopy = min(remaining, chunklen)
                copyto!(buf, 1, P, srcpos, ncopy)
                d[dpos + 1 : dpos + ncopy] = view(buf, 1:ncopy)
                srcpos += ncopy
                dpos += ncopy
                remaining -= ncopy
            end
        end
    end
end

# Read the full GRM back into a dense Symmetric matrix.
# Warning: materializes the whole matrix — for large GRMs prefer read_grm_subset.
function read_grm(filename::AbstractString; nme::AbstractString = "grm_u")
    g = h5open(filename, "r") do file
        read(file, nme)
    end
    Symmetric(vec2ugrm(g), :U)
end

# Linear position of element (row, col), row ≤ col, in the "grm_u" layout.
@inline _uidx(row::Integer, col::Integer) = div(col * (col - 1), 2) + row

# Read only the submatrix S[keep, keep] from an HDF5 GRM without materializing
# the full matrix. `keep` holds global individual indices (e.g. the union of a
# set of trios). Returns a dense k×k Symmetric matrix, k = length(unique(keep)).
#
# For each needed column j, all rows 1..j are stored contiguously, so we read
# one hyperslab per kept column and scatter the kept rows into place.
function read_grm_subset(filename::AbstractString, keep::AbstractVector{<:Integer};
                         nme::AbstractString = "grm_u")
    kp = sort(unique(keep))
    k = length(kp)
    isempty(kp) && throw(ArgumentError("no individuals selected for the GRM subset"))
    lo = first(kp)
    h5open(filename, "r") do file
        d = file[nme]
        T = eltype(d)
        S = Matrix{T}(undef, k, k)
        @inbounds for lj in 1:k
            j = kp[lj]
            base = div(j * (j - 1), 2)
            col = d[base + lo : base + j]         # column j's upper part (rows lo..j)
            for li in 1:k
                i = kp[li]
                i > j && break                    # kp sorted → remaining rows are > j
                val = col[i - lo + 1]
                S[li, lj] = val
                S[lj, li] = val
            end
        end
        Symmetric(S, :U)
    end
end

# ===========================================================================
# 3. Trio selection and trio-specific GRMs
# ===========================================================================

# From candidate trios (aligned offspring/mother/father index vectors, one entry
# per trio), keep only trios where BOTH parents survive kinship pruning (removes
# closely related parent pairs). Returns the surviving (offspring, mother, father)
# index vectors.
#
# Kinship pruning only needs relationships AMONG the parents, so only the
# parents×parents block is read from the HDF5 GRM — never the full matrix. The
# offspring do not participate in pruning. `kinship_pruning` comes from SnpArrays
# and returns a BitVector over the rows of the block it is given.
function sel_trio(filename::AbstractString, oid, mid, fid;
                  method::Symbol = :bottom_up, cutoff::Real = 0.1,
                  nme::AbstractString = "grm_u")
    parents = sort(unique(vcat(mid, fid)))              # global indices of all parents
    A_par = read_grm_subset(filename, parents; nme = nme)  # parents block only, from disk
    keepmask = kinship_pruning(A_par; method = method, cutoff = cutoff)
    kept = Set(parents[keepmask])                       # surviving parent global indices
    sel = findall(i -> (mid[i] in kept) && (fid[i] in kept), eachindex(oid))
    (oid[sel], mid[sel], fid[sel])
end

# Build the seven trio-specific relationship matrices from the (sub)GRM S.
# The three direct blocks are views into S; the three cross-sums are freshly
# allocated. R is the residual identity term.
# Note: S must outlive the returned views (do not mutate S afterwards).
function computeGRMs(S, oid, mid, pid)
    Amm = @view S[mid, mid]
    App = @view S[pid, pid]
    Aoo = @view S[oid, oid]

    Dpm = Matrix(@view S[pid, mid]); Dpm .+= @view S[mid, pid]
    Dom = Matrix(@view S[oid, mid]); Dom .+= @view S[mid, oid]
    Dop = Matrix(@view S[oid, pid]); Dop .+= @view S[pid, oid]

    R = Diagonal(ones(length(oid)))
    [Amm, App, Aoo, Dpm, Dom, Dop, R]
end

# ===========================================================================
# utilities
# ===========================================================================

# Print the in-memory size of a dense rows×cols Float64 matrix in GiB.
function matsize_gb(rows, cols)
    gb = rows * cols * 8 / (1024^3)
    println("Gb: $gb")
    gb
end

# ===========================================================================
# Example pipeline (runs on the bundled data/trio PLINK files)
# ===========================================================================

trio = SnpData(joinpath(@__DIR__, "..", "data", "trio"))
matsize_gb(trio.people, trio.people)              # size of the dense GRM in memory

# 1. Compute the GRM (2Φ) from SNP data, blocked over SNPs.
@time A = grm_blocks(trio.snparray, 100)

# (optional) validate against SnpArrays' built-in grm — doubles memory/compute,
# only feasible on small data.
@time A_ref = 2 * grm(trio.snparray; method = :GRM, minmaf = 0)
@assert isapprox(A, A_ref)

# 2. Store to HDF5 (upper triangle, column-vectorized, chunked).
#    Tip: write_grm(Symmetric(Float32.(parent(A)), :U), "grm4") to halve the
#    file — read it back and widen to Float64 for the model.
write_grm(A, "grm4")

# (optional, small data only) validate the round trip. This materializes the
# whole matrix, so skip it for large GRMs.
@assert isapprox(read_grm("grm4.h5"), A)

# Define candidate trios (this toy dataset is laid out mothers|fathers|offspring).
n = trio.people
k = div(n, 3)
mid = collect(1:k)          # mothers
fid = collect(k+1:2k)       # fathers
oid = collect(2k+1:3k)      # offspring

# 3. Prune related trios. sel_trio reads ONLY the parents×parents block from
#    disk — the full GRM is never materialized.
oid, mid, fid = sel_trio("grm4.h5", oid, mid, fid)

# 4. Read ONLY the surviving trio individuals from disk, remap to local indices,
#    and build the trio-specific GRMs on the small submatrix.
keep = sort(unique(vcat(oid, mid, fid)))
loc  = Dict(g => i for (i, g) in enumerate(keep))
Asub = read_grm_subset("grm4.h5", keep)

grms = computeGRMs(Asub,
    getindex.(Ref(loc), oid),
    getindex.(Ref(loc), mid),
    getindex.(Ref(loc), fid))

# `grms` is now the vector of relationship matrices to pass to a GREMLModel.
