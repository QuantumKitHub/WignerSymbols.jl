module WignerSymbols
export δ, Δ, clebschgordan, wigner3j, wigner6j, racahV, racahW, HalfInteger

using HalfIntegers
using RationalRoots
using LRUCache
const RRBig = RationalRoot{BigInt}
import RationalRoots: _convert

include("growinglist.jl")
include("primefactorization.jl")
convert(BigInt, primefactorial(401)) # trigger compilation and generate some fixed data

include("symmetries.jl")
include("cache.jl")

const Wigner3j = LRU{Key3j,Tuple{Rational{BigInt},Rational{BigInt}}}(; maxsize = 10^6)
const Wigner6j = LRU{Key6j,Tuple{Rational{BigInt},Rational{BigInt}}}(; maxsize = 10^6)
const Wigner3jFloat = FloatCache{Key3j, 5}(40, 10^6) # dense table for all jᵢ ≤ 20
const Wigner6jFloat = FloatCache{Key6j, 6}(30, 10^6) # dense table for all jᵢ ≤ 15

_exactcache(::Key3j) = Wigner3j
_exactcache(::Key6j) = Wigner6j
_floatcache(::Key3j) = Wigner3jFloat
_floatcache(::Key6j) = Wigner6jFloat

# types for which values are retrieved from the `Float64` caches
const CachedFloat = Union{Float32, Float64}

"""
    WignerSymbols.set_buffer3j_size!(; maxsize)

Set the maximal number of 3j symbols that are stored in the LRU caches, both for exact
values and for `Float64` values that are not stored in the dense table.
"""
function set_buffer3j_size!(; maxsize)
    resize!(Wigner3j; maxsize = maxsize)
    return resize!(Wigner3jFloat.lru; maxsize = maxsize)
end
"""
    WignerSymbols.set_buffer6j_size!(; maxsize)

Set the maximal number of 6j symbols that are stored in the LRU caches, both for exact
values and for `Float64` values that are not stored in the dense table.
"""
function set_buffer6j_size!(; maxsize)
    resize!(Wigner6j; maxsize = maxsize)
    return resize!(Wigner6jFloat.lru; maxsize = maxsize)
end

"""
    WignerSymbols.set_table3j_size!(; jmax)

Store all 3j symbols with `jᵢ ≤ jmax` that are requested as floating point numbers in a
dense table, which is allocated incrementally upon first use and requires up to
`8 * binomial(2jmax + 5, 5)` bytes of memory. Other 3j symbols are stored in an LRU cache.
Memory that was already allocated for the table is not freed when decreasing `jmax`.
"""
function set_table3j_size!(; jmax)
    @atomic Wigner3jFloat.table.maxleading = convert(Int, 2 * jmax)
    return jmax
end
"""
    WignerSymbols.set_table6j_size!(; jmax)

Store all 6j symbols with `jᵢ ≤ jmax` that are requested as floating point numbers in a
dense table, which is allocated incrementally upon first use and requires up to
`8 * binomial(2jmax + 6, 6)` bytes of memory. Other 6j symbols are stored in an LRU cache.
Memory that was already allocated for the table is not freed when decreasing `jmax`.
"""
function set_table6j_size!(; jmax)
    @atomic Wigner6jFloat.table.maxleading = convert(Int, 2 * jmax)
    return jmax
end

# check integerness and correctness of (j,m) angular momentum
ϵ(j, m) = (abs(m) <= j && ishalfinteger(j) && isinteger(j-m) && isinteger(j+m))

# check triangle condition
"""
    δ(j₁, j₂, j₃) -> ::Bool

Checks the triangle conditions `j₃ <= j₁ + j₂`, `j₁ <= j₂ + j₃` and `j₂ <= j₃ + j₁`.
"""
δ(j₁, j₂, j₃) = (j₃ <= j₁ + j₂) && (j₁ <= j₂ + j₃) && (j₂ <= j₃ + j₁) && isinteger(j₁+j₂+j₃)

# triangle coefficient
"""
    Δ(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃) -> ::T

Computes the triangle coefficient
`Δ(j₁, j₂, j₃) = √((j₁+j₂-j₃)!*(j₁-j₂+j₃)!*(j₂+j₃-j₁)! / (j₁+j₂+j₃+1)!)`
as a type `T` real number.

Returns `zero(T)` if the triangle condition `δ(j₁, j₂, j₃)` is not satisfied, but
throws a `DomainError` if the `jᵢ`s are not (half)integer
"""
Δ(j₁, j₂, j₃) = Δ(RRBig, j₁, j₂, j₃)
function Δ(T::Type{<:Real}, j₁, j₂, j₃)
    for jᵢ in (j₁, j₂, j₃)
        (ishalfinteger(jᵢ) && jᵢ >= zero(jᵢ)) || throw(DomainError(jᵢ, "invalid jᵢ"))
    end
    if !δ(j₁, j₂, j₃)
        return zero(T)
    end
    n, d = Δ²(j₁, j₂, j₃)
    r = Base.unsafe_rational(n, d)
    return convert(T, signedroot(RationalRoot{BigInt}, r))
end

"""
    wigner3j(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₂-m₁) -> ::T

Compute the value of the Wigner-3j symbol
    ⎛ j₁  j₂  j₃ ⎞
    ⎝ m₁  m₂  m₃ ⎠
as a type `T` real number. By default, `T = RationalRoot{BigInt}` and `m₃ = -m₁-m₂`.

Returns `zero(T)` if the triangle condition `δ(j₁, j₂, j₃)` is not satisfied, but
throws a `DomainError` if the `jᵢ`s and `mᵢ`s are not (half)integer or `abs(mᵢ) > jᵢ`.
"""
wigner3j(j₁, j₂, j₃, m₁, m₂, m₃ = -m₁-m₂) = wigner3j(RRBig, j₁, j₂, j₃, m₁, m₂, m₃)
function wigner3j(T::Type{<:Real}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₁-m₂)
    # check angular momenta
    for (jᵢ,mᵢ) in ((j₁, m₁), (j₂, m₂), (j₃, m₃))
        ϵ(jᵢ, mᵢ) || throw(DomainError((jᵢ, mᵢ), "invalid combination (jᵢ, mᵢ)"))
    end
    return _wigner3j(T, HalfInteger.((j₁, j₂, j₃, m₁, m₂, m₃))...)
end

function _wigner3j(T::Type{<:Real}, j₁::HalfInteger, j₂::HalfInteger, j₃::HalfInteger,
                                    m₁::HalfInteger, m₂::HalfInteger, m₃::HalfInteger)
    # check triangle condition and m₁+m₂+m₃ == 0
    if !δ(j₁, j₂, j₃) || !iszero(m₁+m₂+m₃)
        return zero(T)
    end

    # canonical form under all 72 symmetries, using the Regge square
    R = ReggeSquare(j₁, j₂, j₃, m₁, m₂, m₃)
    key, odd = canonicalize(R)
    # odd permutations of rows or columns of R yield a sign (-1)^J
    return _value(T, key, odd & isodd(sumJ(R)))
end

# value of the Wigner symbol with canonical key `key`, with an additional sign if `flip`
function _value(T::Type{<:CachedFloat}, key, flip::Bool = false)
    v = _get!(_floatcache(key), key) do key
        r, s = get(() -> _exact(key), _exactcache(key), key) # does not store exact value
        return _float64(r, s)
    end
    return convert(T, (flip & !iszero(v)) ? -v : v)
end
function _value(T::Type{<:Real}, key, flip::Bool = false)
    cache = _exactcache(key)
    rs = get(cache, key, nothing)
    if isnothing(rs)
        rs = _exact(key)
        cache[key] = rs
    end
    r, s = rs
    return _convert(T, flip ? -s : s) * convert(T, signedroot(r))
end

"""
    clebschgordan(T::Type{<:Real} = RationalRoot{BigInt}, j₁, m₁, j₂, m₂, j₃, m₃ = m₁+m₂) -> ::T

Compute the value of the Clebsch-Gordan coefficient <j₁, m₁; j₂, m₂ | j₃, m₃ >
as a type `T` real number. By default, `T = RationalRoot{BigInt}` and `m₃ = m₁+m₂`.

Returns `zero(T)` if the triangle condition `δ(j₁, j₂, j₃)` is not satisfied, but
throws a `DomainError` if the `jᵢ`s and `mᵢ`s are not (half)integer or `abs(mᵢ) > jᵢ`.
"""
clebschgordan(j₁, m₁, j₂, m₂, j₃, m₃ = m₁+m₂) =
    clebschgordan(RRBig, j₁, m₁, j₂, m₂, j₃, m₃)
function clebschgordan(T::Type{<:Real}, j₁, m₁, j₂, m₂, j₃, m₃ = m₁+m₂)
    s = wigner3j(T, j₁, j₂, j₃, m₁, m₂, -m₃)
    iszero(s) && return s
    s *= convert(T, signedroot(RRBig, j₃+j₃+one(j₃)))
    return isodd(convert(Int,j₁ - j₂ + m₃)) ? -s : s
end

"""
    racahV(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₁-m₂) -> ::T

Compute the value of Racah's V-symbol
V(j₁, j₂, j₃; m₁, m₂, m₃) = (-1)^(-j₁+j₂+j₃) * ⎛ j₁  j₂  j₃ ⎞
                                               ⎝ m₁  m₂  m₃ ⎠
as a type `T` real number. By default, `T = RationalRoot{BigInt}` and `m₃ = -m₁-m₂`.

Returns `zero(T)` if the triangle condition `δ(j₁, j₂, j₃)` is not satisfied, but
throws a `DomainError` if the `jᵢ`s and `mᵢ`s are not (half)integer or `abs(mᵢ) > jᵢ`.
"""
racahV(j₁, j₂, j₃, m₁, m₂, m₃ = -m₁-m₂) = racahV(RRBig, j₁, j₂, j₃, m₁, m₂, m₃)
function racahV(T::Type{<:Real}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₁-m₂)
    s = wigner3j(T, j₁, j₂, j₃, m₁, m₂, m₃)
    return isodd(convert(Int, -j₁ + j₂ + j₃)) ? -s : s
end

"""
    wigner6j(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃, j₄, j₅, j₆) -> ::T

Compute the value of the Wigner-6j symbol
    _⎧ j₁  j₂  j₃ ⎫_
     ⎩ j₄  j₅  j₆ ⎭
as a type `T` real number. By default, `T = RationalRoot{BigInt}`.

Returns `zero(T)` if any of triangle conditions `δ(j₁, j₂, j₃)`, `δ(j₁, j₆, j₅)`,
`δ(j₂, j₄, j₆)`, `δ(j₃, j₄, j₅)` are not satisfied, but throws a `DomainError` if
the `jᵢ`s are not integer or halfinteger.
"""
wigner6j(j₁, j₂, j₃, j₄, j₅, j₆) = wigner6j(RRBig, j₁, j₂, j₃, j₄, j₅, j₆)
function wigner6j(T::Type{<:Real}, j₁, j₂, j₃, j₄, j₅, j₆)
    # check validity of `jᵢ`s
    for jᵢ in (j₁, j₂, j₃, j₄, j₅, j₆)
        (ishalfinteger(jᵢ) && jᵢ >= zero(jᵢ)) || throw(DomainError(jᵢ, "invalid jᵢ"))
    end
    return _wigner6j(T, HalfInteger.((j₁, j₂, j₃, j₄, j₅, j₆))...)
end

function _wigner6j(T::Type{<:Real}, j₁::HalfInteger, j₂::HalfInteger, j₃::HalfInteger,
        j₄::HalfInteger, j₅::HalfInteger, j₆::HalfInteger
    )
    # check triangle conditions
    if !(δ(j₁, j₂, j₃) && δ(j₁, j₆, j₅) && δ(j₂, j₄, j₆) && δ(j₃, j₄, j₅))
        return zero(T)
    end
    # canonical form under all 144 symmetries
    return _value(T, Key6j(j₁, j₂, j₃, j₄, j₅, j₆))
end

"""
    racahW(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, J, j₃, J₁₂, J₂₃) -> ::T

Compute the value of Racah's W coefficient
`W(j₁, j₂, J, j₃; J₁₂, J₂₃) = <(j₁,(j₂j₃)J₂₃)J | ((j₁j₂)J₁₂,j₃)J> / sqrt((2J₁₂+1)*(2J₁₃+1))`
as a type `T` real number. By default, `T = RationalRoot{BigInt}`.

Returns `zero(T)` if any of triangle conditions `δ(j₁, j₂, J₁₂)`, `δ(j₂, j₃, J₂₃)`,
`δ(j₁, J₂₃, J)`, `δ(J₁₂, j₃, J)` are not satisfied, but throws a `DomainError` if
the `jᵢ`s and `J`s are not integer or halfinteger.
"""
racahW(j₁, j₂, J, j₃, J₁₂, J₂₃) = racahW(RRBig, j₁, j₂, J, j₃, J₁₂, J₂₃)
function racahW(T::Type{<:Real}, j₁, j₂, J, j₃, J₁₂, J₂₃)
    s = wigner6j(T, j₁, j₂, J₁₂, j₃, J, J₂₃)
    if !iszero(s) && isodd(convert(Int, j₁ + j₂ + j₃ + J))
        return -s
    else
        return s
    end
end

# COMPUTATIONAL ROUTINES
#------------------------
# squared triangle coefficient
function Δ²(j₁, j₂, j₃)
    # also checks the triangle conditions by converting to unsigned integer:
    n1 = copy(primefactorial( convert(UInt, + j₁ + j₂ - j₃) ))
    n2 = primefactorial( convert(UInt, + j₁ - j₂ + j₃) )
    n3 = primefactorial( convert(UInt, - j₁ + j₂ + j₃) )
    num = mul!(mul!(n1, n2), n3)
    den = copy(primefactorial( convert(UInt, j₁ + j₂ + j₃ + 1) ))
    # result
    return divgcd!(num, den)
end

# exact value `(r, s)` of the canonical 3j symbol, such that the 3j symbol equals `s * √r`
function _exact(key::Key3j)
    (; L, X, T, B, S) = key
    J = S + X + L + B - T
    # Regge square of the canonical 3j symbol, with columns 1 and 3 interchanged
    # (yielding a sign (-1)^J), such that the series has S+1 terms:
    #     ⎡ J-S-L  L      S     ⎤
    # R = ⎢ J-X-B  B      X     ⎥
    #     ⎣ T      J-L-B  J-S-X ⎦
    R = (J - S - L, J - X - B, T, L, B, J - L - B, S, X, J - S - X)
    # Racah formula in terms of the Regge square
    β₁, β₂, β₃ = R[7], R[2], R[6]
    α₁, α₂ = R[7] - R[3], R[7] - R[5]

    num = copy(primefactorial(R[1]))
    for k in 2:9
        num = mul!(num, primefactorial(R[k]))
    end
    den = copy(primefactorial(J + 1))
    num, den = divgcd!(num, den)
    snum, rnum = splitsquare(num)
    sden, rden = splitsquare(den)
    snum, sden = divgcd!(snum, sden)
    rnum, rden = divgcd!(rnum, rden)
    s = Base.unsafe_rational(convert(BigInt, snum), convert(BigInt, sden))
    r = Base.unsafe_rational(convert(BigInt, rnum), convert(BigInt, rden))
    s *= compute3jseries(β₁, β₂, β₃, α₁, α₂)
    # sign (-1)^(α₁-α₂) from Racah formula, and (-1)^J from interchanging columns
    if isodd(α₁ - α₂ + J)
        s = -s
    end
    return r, s
end

# exact value `(r, s)` of the canonical 6j symbol, such that the 6j symbol equals `s * √r`
function _exact(key::Key6j)
    (; β₁, β₂, β₃, α₁, α₂, α₃, α₄) = key
    # product of the four squared triangle coefficients
    num = one(PrimeFactorization{UInt32})
    for β in (β₁, β₂, β₃), α in (α₁, α₂, α₃, α₄)
        num = mul!(num, primefactorial(β - α))
    end
    den = copy(primefactorial(α₁ + 1))
    for α in (α₂, α₃, α₄)
        den = mul!(den, primefactorial(α + 1))
    end
    num, den = divgcd!(num, den)
    snum, rnum = splitsquare(num)
    sden, rden = splitsquare(den)
    snum, sden = divgcd!(snum, sden)
    rnum, rden = divgcd!(rnum, rden)
    s = Base.unsafe_rational(convert(BigInt, snum), convert(BigInt, sden))
    r = Base.unsafe_rational(convert(BigInt, rnum), convert(BigInt, rden))
    s *= compute6jseries(β₁, β₂, β₃, α₁, α₂, α₃, α₄)
    return r, s
end

# compute the sum appearing in the 3j symbol
function compute3jseries(β₁, β₂, β₃, α₁, α₂)
    krange = max(α₁, α₂, zero(α₁)):min(β₁, β₂, β₃)
    T = PrimeFactorization{eltype(eltype(factorialtable))}

    nums = Vector{T}(undef, length(krange))
    dens = Vector{T}(undef, length(krange))
    for (i, k) in enumerate(krange)
        num = iseven(k) ? one(T) : neg!(one(T))
        den = copy(primefactorial(k))
        den = mul!(mul!(den, primefactorial(k-α₁)), primefactorial(k-α₂))
        den = mul!(mul!(mul!(den, primefactorial(β₁-k)),
                                    primefactorial(β₂-k)),
                                        primefactorial(β₃-k))
        nums[i], dens[i] = num, den
    end
    den = commondenominator!(nums, dens)
    totalnum = sumlist!(nums)
    for n = 1:length(den.powers)
        p = bigprime(n)
        while den.powers[n] > 0
            q, r = divrem(totalnum, p)
            if iszero(r)
                totalnum = q
                den.powers[n] -= 1
            else
                break
            end
        end
    end
    totalden = convert(BigInt, den)
    return Base.unsafe_rational(totalnum, totalden)
end

# compute the sum appearing in the 6j symbol
function compute6jseries(β₁, β₂, β₃, α₁, α₂, α₃, α₄)
    krange = max(α₁, α₂, α₃, α₄):min(β₁, β₂, β₃)
    T = PrimeFactorization{eltype(eltype(factorialtable))}

    nums = Vector{T}(undef, length(krange))
    dens = Vector{T}(undef, length(krange))
    for (i, k) in enumerate(krange)
        num = iseven(k) ? copy(primefactorial(k+1)) : neg!(copy(primefactorial(k+1)))
        den = copy(primefactorial(k-α₁))
        den = mul!(mul!(mul!(den, primefactorial(k-α₂)),
                                    primefactorial(k-α₃)),
                                        primefactorial(k-α₄))
        den = mul!(mul!(mul!(den, primefactorial(β₁-k)),
                                    primefactorial(β₂-k)),
                                        primefactorial(β₃-k))
        nums[i], dens[i] = divgcd!(num, den)
    end
    den = commondenominator!(nums, dens)
    totalnum = sumlist!(nums)
    for n = 1:length(den.powers)
        p = bigprime(n)
        while den.powers[n] > 0
            q, r = divrem(totalnum, p)
            if iszero(r)
                totalnum = q
                den.powers[n] -= 1
            else
                break
            end
        end
    end
    totalden = convert(BigInt, den)
    return Base.unsafe_rational(totalnum, totalden)
end

function _precompile_()
    @assert precompile(wigner3j, (Type{Float64}, Int, Int, Int, Int, Int, Int))
    @assert precompile(wigner6j, (Type{Float64}, Int, Int, Int, Int, Int, Int))
    @assert precompile(wigner3j, (Type{BigFloat}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}))
    @assert precompile(wigner6j, (Type{BigFloat}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}))
    @assert precompile(wigner3j, (Type{RRBig}, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt))
    @assert precompile(wigner6j, (Type{RRBig}, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt))
end
_precompile_()

end # module
