module WignerSymbols
export δ, Δ, clebschgordan, wigner3j, wigner6j, wigner9j, racahV, racahW, HalfInteger

using HalfIntegers
using RationalRoots
using LRUCache
const RRBig = RationalRoot{BigInt}
import RationalRoots: _convert

include("growinglist.jl")
include("bigint.jl") # additional GMP BigInt functionality not wrapped in Base.GMP.MPZ
include("primefactorization.jl")
convert(BigInt, primefactorial(401)) # trigger compilation and generate some fixed data

const Key3j = Tuple{UInt,UInt,UInt,Int,Int}
const Key6j = NTuple{6,UInt}
const Key9j = NTuple{9,HalfInteger}

const Wigner3j = LRU{Key3j,Tuple{Rational{BigInt},Rational{BigInt}}}(; maxsize = 10^6)
const Wigner6j = LRU{Key6j,Tuple{Rational{BigInt},Rational{BigInt}}}(; maxsize = 10^6)
const Wigner9j = LRU{Key9j,Tuple{Rational{BigInt},Rational{BigInt}}}(; maxsize = 10^6)

function set_buffer3j_size!(; maxsize)
    resize!(Wigner3j; maxsize = maxsize)
end
function set_buffer6j_size!(; maxsize)
    resize!(Wigner6j; maxsize = maxsize)
end
function set_buffer9j_size!(; maxsize)
    resize!(Wigner9j; maxsize = maxsize)
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
        (ishalfinteger(jᵢ) && jᵢ >= zero(jᵢ)) || throw(DomainError("invalid jᵢ", jᵢ))
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

    # we reorder such that j₁ >= j₂ >= j₃ and m₁ >= 0 or m₁ == 0 && m₂ >= 0
    j₁, j₂, j₃, m₁, m₂, m₃, sgn = reorder3j(j₁, j₂, j₃, m₁, m₂, m₃)
    # TODO: do we also want to use Regge symmetries?
    α₁ = convert(Int, j₂ - m₁ - j₃ ) # can be negative
    α₂ = convert(Int, j₁ + m₂ - j₃ ) # can be negative
    β₁ = convert(UInt, j₁ + j₂ - j₃ )
    β₂ = convert(UInt, j₁ - m₁ )
    β₃ = convert(UInt, j₂ + m₂ )

    # extra sign in definition: α₁ - α₂ = j₁ + m₂ - j₂ + m₁ = j₁ - j₂ + m₃
    sgn = isodd(α₁ - α₂) ? -sgn : sgn

    # dictionary lookup or compute
    if haskey(Wigner3j, (β₁, β₂, β₃, α₁, α₂))
        r, s = Wigner3j[(β₁, β₂, β₃, α₁, α₂)]
    else
        s1n, s1d = Δ²(j₁, j₂, j₃)
        s2n = prod(map(primefactorial, (β₂, β₁ - α₁, β₁ - α₂, β₃, β₃ - α₁, β₂ - α₂)))

        snum, rnum = splitsquare(s1n*s2n)
        sden, rden = splitsquare(s1d)
        snum, sden = divgcd!(snum, sden)
        rnum, rden = divgcd!(rnum, rden)
        s = Base.unsafe_rational(convert(BigInt, snum), convert(BigInt, sden))
        r = Base.unsafe_rational(convert(BigInt, rnum), convert(BigInt, rden))
        s *= compute3jseries(β₁, β₂, β₃, α₁, α₂)
        Wigner3j[(β₁, β₂, β₃, α₁, α₂)] = (r,s)
    end
    return _convert(T, sgn*s)*convert(T, signedroot(r))
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
        (ishalfinteger(jᵢ) && jᵢ >= zero(jᵢ)) || throw(DomainError("invalid jᵢ", jᵢ))
    end
    return _wigner6j(T, HalfInteger.((j₁, j₂, j₃, j₄, j₅, j₆))...)
end

function _wigner6j(T::Type{<:Real}, j₁::HalfInteger, j₂::HalfInteger, j₃::HalfInteger,
                                    j₄::HalfInteger, j₅::HalfInteger, j₆::HalfInteger)
    α̂₁ = (j₁, j₂, j₃)
    α̂₂ = (j₁, j₆, j₅)
    α̂₃ = (j₂, j₄, j₆)
    α̂₄ = (j₃, j₄, j₅)

    # check triangle conditions
    if !(δ(α̂₁...) && δ(α̂₂...) && δ(α̂₃...) && δ(α̂₄...))
        return zero(T)
    end
    # reduce
    α₁ = convert(UInt, +(α̂₁...))
    α₂ = convert(UInt, +(α̂₂...))
    α₃ = convert(UInt, +(α̂₃...))
    α₄ = convert(UInt, +(α̂₄...))
    β₁ = convert(UInt, j₁+j₂+j₄+j₅)
    β₂ = convert(UInt, j₁+j₃+j₄+j₆)
    β₃ = convert(UInt, j₂+j₃+j₅+j₆)

    # we should have αᵢ < βⱼ, ∀ i, j and ∑ᵢ αᵢ = ∑ⱼ βⱼ
    # now order them as β₁ >= β₂ >= β₃ >= α₁ >= α₂ >= α₃ >= α₄
    (β₁, β₂, β₃, α₁, α₂, α₃, α₄) = reorder6j(β₁, β₂, β₃, α₁, α₂, α₃, α₄)

    # dictionary lookup or compute
    if haskey(Wigner6j, (β₁, β₂, β₃, α₁, α₂, α₃))
        r, s = Wigner6j[(β₁, β₂, β₃, α₁, α₂, α₃)]
    else
        # order irrelevant: product remains the same under action of reorder6j
        n₁, d₁ = Δ²(α̂₁...)
        n₂, d₂ = Δ²(α̂₂...)
        n₃, d₃ = Δ²(α̂₃...)
        n₄, d₄ = Δ²(α̂₄...)

        snum, rnum = splitsquare(n₁ * n₂ * n₃ * n₄)
        sden, rden = splitsquare(d₁ * d₂ * d₃ * d₄)
        snum, sden = divgcd!(snum, sden)
        rnum, rden = divgcd!(rnum, rden)
        s = Base.unsafe_rational(convert(BigInt, snum), convert(BigInt, sden))
        r = Base.unsafe_rational(convert(BigInt, rnum), convert(BigInt, rden))
        s *= compute6jseries(β₁, β₂, β₃, α₁, α₂, α₃, α₄)

        Wigner6j[(β₁, β₂, β₃, α₁, α₂, α₃)] = (r, s)
    end
    return _convert(T, s) * convert(T, signedroot(r))
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

"""
    wigner9j(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉) -> ::T

Compute the value of the Wigner-9j symbol
⎧ j₁ j₂ j₃ ⎫
⎨	j₄ j₅ j₆ ⎬
⎩	j₇ j₈ j₉ ⎭
as a type `T` real number. By default `T = RationalRoot{BigInt}`.

Returns `zero(T)` if any of the triangle conditions TODO 
are not satisfied, but throws a `DomainError` if
the `jᵢ`s are not integer or halfinteger.
"""
wigner9j(j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉) = wigner9j(RRBig, j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉)
function wigner9j(T::Type{<:Real}, j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉) 
    # check validity of `jᵢ`s
    for jᵢ in (j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉)
        (ishalfinteger(jᵢ) && jᵢ >= zero(jᵢ)) || throw(DomainError("invalid jᵢ", jᵢ))
    end
    return _wigner9j(T, HalfInteger.((j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉))...)
end


function _wigner9j(T::Type{<:Real}, j₁::HalfInteger, j₂::HalfInteger, j₃::HalfInteger, 
                                    j₄::HalfInteger, j₅::HalfInteger, j₆::HalfInteger,
                                    j₇::HalfInteger, j₈::HalfInteger, j₉::HalfInteger)
  
    # check triangle conditions 
    if !(δ(j₁, j₂, j₃) && δ(j₄, j₅, j₆) && δ(j₇, j₈, j₉) && δ(j₁, j₄, j₇) && δ(j₂, j₅, j₈) && δ(j₃, j₆, j₉))
        return zero(T)
    end
   
    # dictionary lookup by regge symmetries
    k, m = reorder9j(j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉)
    if haskey(Wigner9j, k)
        r, s = Wigner9j[k]
        return m * _convert(T, s) * convert(T, signedroot(r))
    end

    # canonicalized entries (m is a restoring sign factor)
    j₁c, j₂c, j₃c, j₄c, j₅c, j₆c, j₇c, j₈c, j₉c = k

    # order irrelevant: product remains the same
    n₁, d₁ = Δ²(j₁c, j₂c, j₃c)
    n₂, d₂ = Δ²(j₄c, j₅c, j₆c)
    n₃, d₃ = Δ²(j₇c, j₈c, j₉c)
    n₄, d₄ = Δ²(j₁c, j₄c, j₇c)
    n₅, d₅ = Δ²(j₂c, j₅c, j₈c)
    n₆, d₆ = Δ²(j₃c, j₆c, j₉c) 

    snum, rnum = splitsquare(n₁ * n₂ * n₃ * n₄ * n₅ * n₆)
    sden, rden = splitsquare(d₁ * d₂ * d₃ * d₄ * d₅ * d₆)
    
    rnum, rden = divgcd!(rnum, rden) 
    rr = Base.unsafe_rational(convert(BigInt, rnum), convert(BigInt, rden)) 

    snum, sden = divgcd!(snum, sden)
    snum2 = compute9jseries(j₁c, j₂c, j₃c, j₄c, j₅c, j₆c, j₇c, j₈c, j₉c)
    for n = 1:length(sden.powers)
        p = bigprime(n)
        while sden.powers[n] > 0
            q, r = divrem(snum2, p)
            if iszero(r)
                snum2 = q
                sden.powers[n] -= 1
            else
                break
            end
        end
    end
    s = Base.unsafe_rational(convert(BigInt, snum)*snum2, convert(BigInt, sden))

    Wigner9j[(j₁c, j₂c, j₃c, j₄c, j₅c, j₆c, j₇c, j₈c, j₉c)] = (rr, s)
    return m * _convert(T, s) * convert(T, signedroot(rr))
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

# reorder parameters determining the 3j symbol to canonical order:
# j₁ >= j₂ >= j₃ and m₁ >= 0 or m₁ == 0 && m₂ >= 0
function reorder3j(j₁, j₂, j₃, m₁, m₂, m₃, sign = one(Int8))
    if j₁ < j₂
        return reorder3j(j₂, j₁, j₃, m₂, m₁, m₃, -sign)
    elseif j₂ < j₃
        return reorder3j(j₁, j₃, j₂, m₁, m₃, m₂, -sign)
    elseif m₁ < zero(m₁)
        return reorder3j(j₁, j₂, j₃, -m₁, -m₂, -m₃, -sign)
    elseif iszero(m₁) && m₂ < zero(m₂)
        return reorder3j(j₁, j₂, j₃, -m₁, -m₂, -m₃, -sign)
    else
        # sign doesn't matter if total J=j₁ + j₂ + j₃ is even
        if iseven(convert(UInt,j₁ + j₂ + j₃))
            sign = one(sign)
        end
        return (j₁, j₂, j₃, m₁, m₂, m₃, sign)
    end
end

# reorder parameters determining the 6j symbol to canonical order:
# β₁ >= β₂ >= β₃ >= α₁ >= α₂ >= α₃ >= α₄
function reorder6j(β₁, β₂, β₃, α₁, α₂, α₃, α₄)
    if β₁ < β₂
        return reorder6j(β₂, β₁, β₃, α₁, α₂, α₃, α₄)
    elseif β₂ < β₃
        return reorder6j(β₁, β₃, β₂, α₁, α₂, α₃, α₄)
    elseif α₁ < α₂
        return reorder6j(β₁, β₂, β₃, α₂, α₁, α₃, α₄)
    elseif α₂ < α₃
        return reorder6j(β₁, β₂, β₃, α₁, α₃, α₂, α₄)
    elseif α₃ < α₄
        return reorder6j(β₁, β₂, β₃, α₁, α₂, α₄, α₃)
    else
        return (β₁, β₂, β₃, α₁, α₂, α₃, α₄)
    end
end

const _perms = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
const _perms9j = [(r, c) for r in _perms, c in _perms]
_permparity(p) = count(i < j && p[i] > p[j] for i in 1:length(p), j in 1:length(p)) % 2

#const _perms9j = [([1, 2, 3], [1, 2, 3]) ([1, 2, 3], [1, 3, 2]) ([1, 2, 3], [2, 1, 3]) ([1, 2, 3], [2, 3, 1]) ([1, 2, 3], [3, 1, 2]) ([1, 2, 3], [3, 2, 1]);
#                  ([1, 3, 2], [1, 2, 3]) ([1, 3, 2], [1, 3, 2]) ([1, 3, 2], [2, 1, 3]) ([1, 3, 2], [2, 3, 1]) ([1, 3, 2], [3, 1, 2]) ([1, 3, 2], [3, 2, 1]);
#                  ([2, 1, 3], [1, 2, 3]) ([2, 1, 3], [1, 3, 2]) ([2, 1, 3], [2, 1, 3]) ([2, 1, 3], [2, 3, 1]) ([2, 1, 3], [3, 1, 2]) ([2, 1, 3], [3, 2, 1]);
#                  ([2, 3, 1], [1, 2, 3]) ([2, 3, 1], [1, 3, 2]) ([2, 3, 1], [2, 1, 3]) ([2, 3, 1], [2, 3, 1]) ([2, 3, 1], [3, 1, 2]) ([2, 3, 1], [3, 2, 1]);
#                  ([3, 1, 2], [1, 2, 3]) ([3, 1, 2], [1, 3, 2]) ([3, 1, 2], [2, 1, 3]) ([3, 1, 2], [2, 3, 1]) ([3, 1, 2], [3, 1, 2]) ([3, 1, 2], [3, 2, 1]);
#                  ([3, 2, 1], [1, 2, 3]) ([3, 2, 1], [1, 3, 2]) ([3, 2, 1], [2, 1, 3]) ([3, 2, 1], [2, 3, 1]) ([3, 2, 1], [3, 1, 2]) ([3, 2, 1], [3, 2, 1])]

# reorder parameters determining the 9j symbol to canonical (lexicographic) order with regge factor
function reorder9j(j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉)

    J = [j₁ j₂ j₃; j₄ j₅ j₆; j₇ j₈ j₉]

    k = (j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉)
    m = 1

    rg = (-1)^sum(k)

    for p in _perms9j
        JJ = J[p...]
        kk = Tuple(reshape(JJ, 9))
        kkT = Tuple(reshape(JJ', 9))

        s = iseven(_permparity(p[1]) + _permparity(p[2])) ? 1 : rg

        if kk < kkT
            if kk < k
                k = kk
                m = s
            end
        else # kk > kkT
            if kkT < k                
                k = kkT
                m = s
            end
        end        
    end
    return k, m
end

# compute the sum appearing in the 3j symbol
function compute3jseries(β₁, β₂, β₃, α₁, α₂)
    krange = max(α₁, α₂, zero(α₁)):min(β₁, β₂, β₃)
    T = PrimeFactorization{eltype(eltype(factorialtable))}

    nums = Vector{T}(undef, length(krange))
    dens = Vector{T}(undef, length(krange))
    for (i, k) in enumerate(krange)
        num = iseven(k) ? one(T) : -one(T)
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

# compute the sum appearing in the 9j symbol
function compute9jseries(a, b, c, d, e, f, g, h, j)
    I1 = max(abs(h - d), abs(b - f), abs(a - j))
    I2 = min(h + d, b + f, a + j)
    krange = I1:I2
 
    s = sum(krange) do k
        p = iseven(BigInt(2k)) ? big(2k + 1) : -big(2k + 1)

        b₁ = let (m₁, m₂, m₃, m₄, m₅, m₆) = (a, b, c, f, j, k)     
            α₁ = convert(BigInt, m₁ + m₅ - m₆)
            α₂ = convert(BigInt, m₁ - m₅ + m₆)
            α₃ = convert(BigInt, -m₁ + m₅ + m₆)
            β₁ = convert(BigInt, m₁ + m₅ + m₆)
            β₂ = convert(BigInt, m₂ + m₄ + m₆)
            β₃ = convert(BigInt, m₃ + m₄ + m₅)
            β₀ = convert(BigInt, m₁ + m₂ + m₃)
            wei9jbracket(α₁, α₂, α₃, β₁, β₂, β₃, β₀)
        end
        
        b₂ = let (m₁, m₂, m₃, m₄, m₅, m₆) = (f, d, e, h, b, k)
            α₁ = convert(BigInt, m₁ + m₅ - m₆)
            α₂ = convert(BigInt, m₁ - m₅ + m₆)
            α₃ = convert(BigInt, -m₁ + m₅ + m₆)
            β₁ = convert(BigInt, m₁ + m₅ + m₆)
            β₂ = convert(BigInt, m₂ + m₄ + m₆)
            β₃ = convert(BigInt, m₃ + m₄ + m₅)
            β₀ = convert(BigInt, m₁ + m₂ + m₃)
            wei9jbracket(α₁, α₂, α₃, β₁, β₂, β₃, β₀)
        end

        b₃ = let (m₁, m₂, m₃, m₄, m₅, m₆) = (h, j, g, a, d, k) 
            α₁ = convert(BigInt, m₁ + m₅ - m₆)
            α₂ = convert(BigInt, m₁ - m₅ + m₆)
            α₃ = convert(BigInt, -m₁ + m₅ + m₆)
            β₁ = convert(BigInt, m₁ + m₅ + m₆)
            β₂ = convert(BigInt, m₂ + m₄ + m₆)
            β₃ = convert(BigInt, m₃ + m₄ + m₅)
            β₀ = convert(BigInt, m₁ + m₂ + m₃)
            wei9jbracket(α₁, α₂, α₃, β₁, β₂, β₃, β₀)
        end
        
        p * b₁ * b₂ * b₃
    end

    return convert(BigInt, s)
end

# Wei square bracket terms appearing the 9j series
function wei9jbracket(α₁::BigInt, α₂::BigInt, α₃::BigInt, β₁::BigInt, β₂::BigInt, β₃::BigInt, β₀::BigInt) 
    p = max(β₁, β₂, β₃, β₀)
    q = min(α₁ + β₂, α₂ + β₃, α₃ + β₀)
    lrange = p:q
    T = PrimeFactorization{eltype(eltype(factorialtable))}

    terms = Vector{T}(undef, length(lrange))
    for (j, l) in enumerate(lrange)
        b₁ = iseven(l) ? copy(primebinomial(big(l + 1), l - β₁)) : neg!(copy(primebinomial(big(l + 1), l - β₁)))
        b₂ = primebinomial(α₁, l - β₂)
        b₃ = primebinomial(α₂, l - β₃)
        b₄ = primebinomial(α₃, l - β₀)

        terms[j] = mul!(mul!(mul!(b₁, b₂), b₃), b₄)
    end
    total = sumlist!(terms)
    return total
end

function _precompile_()
    @assert precompile(wigner3j, (Type{Float64}, Int, Int, Int, Int, Int, Int))
    @assert precompile(wigner6j, (Type{Float64}, Int, Int, Int, Int, Int, Int))
    @assert precompile(wigner9j, (Type{Float64}, Int, Int, Int, Int, Int, Int, Int, Int, Int))
    @assert precompile(wigner3j, (Type{BigFloat}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}))
    @assert precompile(wigner6j, (Type{BigFloat}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}))
    @assert precompile(wigner9j, (Type{BigFloat}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}, Rational{Int}))
    @assert precompile(wigner3j, (Type{RRBig}, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt))
    @assert precompile(wigner6j, (Type{RRBig}, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt))
    @assert precompile(wigner9j, (Type{RRBig}, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt, HalfInt))
end
_precompile_()

end # module
