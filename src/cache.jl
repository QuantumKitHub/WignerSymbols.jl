# Caches for Wigner symbols, indexed by their canonical key (see symmetries.jl).
#
# Exact values `(r, s)`, representing `s * √r`, are stored in an LRU cache. Values that are
# requested as `Float64` or `Float32` are stored as correctly rounded `Float64` values in a
# separate `FloatCache`, such that the exact values do not need to be kept and no
# conversion is required upon retrieval.

"""
    DenseTable{N}

Thread safe table of `Float64` values indexed by non-increasing `NTuple{N,Int}` keys,
using the combinatorial number system. The table is stored contiguously and holds all keys
with leading entry `k₁ ≤ K` for some `K ≤ maxleading`; it is grown upon first access of a
key with larger `k₁`. Entries that have not yet been computed are `NaN`.

The table is grown by copying into a new vector, so that concurrent readers can keep using
the previous one without a lock. Values that are written into the previous vector during
growth are lost and will be recomputed when requested again. Concurrent computation of the
same entry by different threads results in the same value being written more than once.
"""
mutable struct DenseTable{N}
    @atomic data::Vector{Float64}
    @atomic maxleading::Int # keys with k₁ > maxleading are not stored in the table
    const lock::ReentrantLock
end
DenseTable{N}(maxleading::Int) where {N} = DenseTable{N}(Float64[], maxleading, ReentrantLock())

# grow the table such that it holds all keys with leading entry k₁
function _grow!(t::DenseTable{N}, k₁::Int) where {N}
    return lock(t.lock) do
        data = @atomic t.data
        length(data) >= _tablelength(N, k₁) && return data
        # grow at least by a factor 3/2 in length, but not beyond maxleading
        K = k₁
        Kmax = @atomic t.maxleading
        while K < Kmax && 2 * _tablelength(N, K) < 3 * length(data)
            K += 1
        end
        newdata = Vector{Float64}(undef, _tablelength(N, K))
        copyto!(newdata, data)
        fill!(view(newdata, (length(data) + 1):length(newdata)), NaN)
        @atomic t.data = newdata
        return newdata
    end
end

"""
    FloatCache{K,N}

Cache of `Float64` values for keys of type `K`, which are mapped onto non-increasing
`NTuple{N,Int}` via `_combinatorialkey`: a `DenseTable` for keys with small leading entry,
and an `LRU` cache for the remaining keys.
"""
struct FloatCache{K, N}
    table::DenseTable{N}
    lru::LRU{K, Float64}
end
FloatCache{K, N}(maxleading::Int, maxsize::Int) where {K, N} =
    FloatCache{K, N}(DenseTable{N}(maxleading), LRU{K, Float64}(; maxsize = maxsize))

# get the value for `key`, or compute it as `f(key)` and store it
@inline function _get!(f, c::FloatCache{K, N}, key::K) where {K, N}
    ckey = _combinatorialkey(key)::NTuple{N, Int}
    k₁ = ckey[1]
    if k₁ <= @atomic c.table.maxleading
        data = @atomic c.table.data
        i = _tableindex(ckey)
        if i > length(data)
            data = _grow!(c.table, k₁)
        end
        v = @inbounds data[i]
        if isnan(v)
            v = f(key)
            @inbounds data[i] = v
        end
        return v
    else
        v = get(c.lru, key, NaN)
        if isnan(v)
            v = f(key)
            c.lru[key] = v
        end
        return v
    end
end

# `Float64` value of s * √r, correctly rounded up to an intermediate precision of 256 bits
function _float64(r::Rational{BigInt}, s::Rational{BigInt})
    prec = 256
    return Float64(BigFloat(s; precision = prec) * sqrt(BigFloat(r; precision = prec)))
end
