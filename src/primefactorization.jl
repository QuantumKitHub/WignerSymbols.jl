using Primes: isprime
import Base.divgcd

const primetable = GrowingList([2, 3]; sizehint = 1024)
const factortable = GrowingList([UInt8[], UInt8[1], UInt8[0,1]]; sizehint = 4096)
const factorialtable = GrowingList([UInt32[], UInt32[1], UInt32[1,1]]; sizehint = 4096)
const bigprimetable = GrowingList([GrowingList([big(2)]; sizehint = 2048),
                                    GrowingList([big(3)]; sizehint = 1024)];
                                    sizehint = 1024)
const bigone = big(1)

# Make a prime iterator
struct PrimeIterator
end
primes() = PrimeIterator()

Base.IteratorSize(::Type{PrimeIterator}) = Base.IsInfinite()
Base.IteratorEltype(::Type{PrimeIterator}) = Base.HasEltype()
Base.eltype(::PrimeIterator) = Int

# Get the `n`th prime; store all primes up to the `n`th if not yet available
function prime(n::Int)
    k = min(length(primetable), length(bigprimetable))
    while k < n
        @inbounds p = primetable[k]
        p = p + 2
        while !isprime(p)
            p += 2
        end
        k += 1
        # these lines do not get but set new elements; provided no other task did so earlier
        get!(primetable, k, p)
        bp = big(p)
        bpf = GrowingList{BigInt}((big(p),); sizehint = 4)
        get!(bigprimetable, k, bpf)
        k = min(length(primetable), length(bigprimetable))
         # other threads might have inserted additional entries,
         # make sure they are finished with both primetable and bigprimetable
    end
    @inbounds return primetable[n]
end

Base.iterate(::PrimeIterator, n = 1) = prime(n), n+1

# get primes and their powers as `BigInt`, also cache all results
function bigprime(n::Integer, e::Integer=1)
    e == 0 && return bigone
    p = prime(n) # triggers computation of prime(n) if necessary
    @inbounds powerlist = bigprimetable[n]
    l = length(powerlist)
    @inbounds while l < e
        # compute next prime power as approximate square of existing results
        l += 1
        k = l>>1
        newpower = powerlist[k]*powerlist[l-k]
        get!(powerlist, l, newpower)
        l = length(powerlist) # other threads might have inserted more powers
    end
    @inbounds return powerlist[e]
end

# A custom `Integer` subtype to store an integer as its prime factorization
# mutable to allow in place update of sign
mutable struct PrimeFactorization{U<:Unsigned} <: Integer
    powers::Vector{U}
    sign::Int8
    PrimeFactorization{U}(powers::Vector, sign = one(Int8)) where {U<:Unsigned} =
        new{U}(convert(Vector{U}, powers), sign)
end
# convenience constructor: normalizes powers to have last entry nonzero
PrimeFactorization(powers::Vector{U}, sign = one(Int8)) where {U<:Unsigned} =
    PrimeFactorization{U}(_normalize_powers!(powers), sign)

function _normalize_powers!(v::Vector{<:Integer})
    i = findlast(!iszero, v)
    l = ifelse(i === nothing, 0, i)
    l < length(v) && resize!(v, l)
    return v
end

# define our own factor function, returning an instance of PrimeFactorization
function primefactor(n::Integer)
    iszero(n) && return PrimeFactorization{UInt8}(UInt8[], zero(Int8))
    sn = n < 0 ? -one(Int8) : one(Int8)
    n = abs(n)
    m = length(factortable)
    while m < abs(n)
        m += 1
        powers = UInt8[] # should be sufficient for all integers up to 2^255
        a = m
        for p in primes()
            f = 0
            anext, r = divrem(a, p)
            while r == 0
                f += 1
                a = anext
                anext, r = divrem(a, p)
            end
            push!(powers, f)
            a == 1 && break
        end
        get!(factortable, m, powers)
        m = length(factortable) # other threads may have inserted other entries
    end
    @inbounds return PrimeFactorization{UInt8}(factortable[n], sn)
end

function primefactorial(n::Integer)
    n < 0 && throw(DomainError(n,"primefactorial only works for non-negative numbers"))
    n <= 1 && return PrimeFactorization{UInt32}(UInt32[], one(Int8))
    m = length(factorialtable)
    @inbounds while m < n
        prevfactorial = factorialtable[m]
        m += 1
        f = primefactor(m).powers
        if length(f) > length(prevfactorial) # can at most be 1 larger
            powers = similar(prevfactorial, length(f))
            powers[1:end-1] = prevfactorial
            powers[end] = 0
        else
            powers = copy(prevfactorial)
        end
        for k = 1:length(f)
            powers[k] += f[k]
        end
        get!(factorialtable, m, powers)
        m = length(factorialtable)  # other threads may have inserted other entries
    end
    @inbounds return PrimeFactorization{UInt32}(factorialtable[n])
end

# Methods for PrimeFactorization:
Base.copy(a::PrimeFactorization) = PrimeFactorization(copy(a.powers), a.sign)
function Base.copy!(c::PrimeFactorization, a::PrimeFactorization)
    c.sign = a.sign
    copy!(c.powers, a.powers)
    return c
end

Base.one(::Type{PrimeFactorization{U}}) where {U<:Unsigned} =
    PrimeFactorization{U}(Vector{U}(), one(Int8))
Base.zero(::Type{PrimeFactorization{U}}) where {U<:Unsigned} =
    PrimeFactorization{U}(Vector{U}(), zero(Int8))

one!(c::PrimeFactorization) = (c.sign = one(Int8); empty!(c.powers); return c)
zero!(c::PrimeFactorization) = (c.sign = zero(Int8); empty!(c.powers); return c)

Base.promote_rule(P::Type{<:PrimeFactorization},::Type{<:Integer}) = P
Base.promote_rule(::Type{<:PrimeFactorization},::Type{BigInt}) = BigInt
Base.promote_rule(::Type{PrimeFactorization{U1}},
                    ::Type{PrimeFactorization{U2}}) where {U1<:Unsigned, U2<:Unsigned} = PrimeFactorization{promote_type(U1, U2)}

Base.convert(P::Type{<:PrimeFactorization}, n::Integer) = convert(P, primefactor(n))
function _convert!(x::BigInt, a::PrimeFactorization)
    MPZ.set!(x, bigone)
    for (n, e) in enumerate(a.powers)
        if !iszero(e)
            MPZ.mul!(x, bigprime(n, e))
        end
    end
    return a.sign < 0 ? MPZ.neg!(x) : x
end
Base.convert(::Type{BigInt}, a::PrimeFactorization) = _convert!(one(BigInt), a)
Base.convert(::Type{PrimeFactorization{U}}, a::PrimeFactorization{U}) where {U<:Unsigned} =
    a
Base.convert(::Type{PrimeFactorization{U}}, a::PrimeFactorization) where {U<:Unsigned} =
    PrimeFactorization{U}(convert(Vector{U}, a.powers), a.sign)

Base.:(==)(a::PrimeFactorization, b::PrimeFactorization) =
    a.powers == b.powers && a.sign == b.sign
function Base.:<(a::PrimeFactorization, b::PrimeFactorization)
    if a.sign != b.sign
        return a.sign < b.sign
    elseif a.sign < 0
        return <(-b, -a)
    else
        ag, bg = divgcd(a, b)
        ag == bg && return false
        if length(ag.powers) <= length(bg.powers)
                all(k->ag.powers[k]<bg.powers[k], 1:length(ag.powers))
            return true
        else
            return convert(BigInt, ag) < convert(BigInt, bg)
        end
    end
end

# Methods for PrimeFactorization: only fast multiplication, and lcm and gcd.
# Addition and subtraction will require conversion to BigInt
Base.sign(a::PrimeFactorization) = a.sign

neg!(a::PrimeFactorization) = (a.sign = -a.sign; return a)

function mul!(c::PrimeFactorization, a::PrimeFactorization, b::PrimeFactorization)
    if a.sign == 0 || b.sign == 0
        zero!(c)
    else
        c.sign = a.sign * b.sign
        la = length(a.powers)
        lb = length(b.powers)
        lc = max(la, lb)
        lc === length(c.powers) || resize!(c.powers, lc)
        @inbounds for k = 1:min(la,lb)
            c.powers[k] = +(a.powers[k], b.powers[k])
        end
        if c !== a
            @inbounds for k = lb+1:la
                c.powers[k] = a.powers[k]
            end
        end
        @inbounds for k = la+1:lb
            c.powers[k] = b.powers[k]
        end
    end
    return c
end
# unlike div, this one errors if the a is not divisible by b
function divexact!(c::PrimeFactorization, a::PrimeFactorization, b::PrimeFactorization)
    if iszero(a.sign)
        zero!(c)
    elseif iszero(b.sign)
        throw(DivideError())
    else
        c.sign = a.sign * b.sign
        la = length(a.powers)
        lb = length(b.powers)
        if lb > la
            throw(DivideError())
        end
        lc = la
        if lb == lc
            while lc > 0 && a.powers[lc] == b.powers[lc]
                lc -= 1
            end
        end
        lc == length(c.powers) || resize!(c.powers, lc)
        @inbounds for k = 1:min(lb, lc)
            if b.powers[k] > a.powers[k]
                throw(DivideError())
            end
            c.powers[k] = a.powers[k] - b.powers[k]
        end
        if c !== a
            @inbounds for k = lb+1:lc
                c.powers[k] = a.powers[k]
            end
        end
    end
    return c
end
function gcd!(c::PrimeFactorization, a::PrimeFactorization, b::PrimeFactorization)
    if a.sign == 0
        copy!(c.powers, b.powers)
    elseif b.sign ==0
        copy!(c.powers, a.powers)
    else
        c.sign = one(Int8)
        la = length(a.powers)
        lb = length(b.powers)
        lc = min(la, lb)
        lc === length(c.powers) || resize!(c.powers, lc)
        @inbounds for k = 1:lc
            c.powers[k] = min(a.powers[k], b.powers[k])
        end
    end
    c.sign = one(Int8)
    return c
end
function lcm!(c::PrimeFactorization, a::PrimeFactorization, b::PrimeFactorization)
    if a.sign == 0 || b.sign == 0
        return zero!(c)
    else
        c.sign = one(Int8)
        la = length(a.powers)
        lb = length(b.powers)
        lc = max(la, lb)
        lc === length(c.powers) || resize!(c.powers, lc)
        @inbounds for k = 1:min(la,lb)
            c.powers[k] = max(a.powers[k], b.powers[k])
        end
        if c !== a
            @inbounds for k = lb+1:la
                c.powers[k] = a.powers[k]
            end
        end
        @inbounds for k = la+1:lb
            c.powers[k] = b.powers[k]
        end
    end
    c.sign = one(Int8)
    return c
end
function divgcd!(a::PrimeFactorization, b::PrimeFactorization)
    af, bf = a.powers, b.powers
    for k = 1:min(length(af), length(bf))
        gk = min(af[k], bf[k])
        af[k] -= gk
        bf[k] -= gk
    end
    _normalize_powers!(a.powers)
    _normalize_powers!(b.powers)
    return a, b
end

mul!(a::PrimeFactorization, b::PrimeFactorization) = mul!(a, a, b)
divexact!(a::PrimeFactorization, b::PrimeFactorization) = divexact!(a, a, b)
gcd!(a::PrimeFactorization, b::PrimeFactorization) = gcd!(a, a, b)
lcm!(a::PrimeFactorization, b::PrimeFactorization) = lcm!(a, a, b)

Base.:-(a::PrimeFactorization) = neg!(copy(a))
function Base.:*(a::PrimeFactorization, b::PrimeFactorization)
    P = promote_type(typeof(a), typeof(b))
    if length(a.powers) >= length(b.powers)
        return typeof(a) == P ? mul!(copy(a), b) : mul!(convert(P, a), b)
    else
        return typeof(b) == P ? mul!(copy(b), a) : mul!(convert(P, b), a)
    end
end
function Base.lcm(a::PrimeFactorization, b::PrimeFactorization)
    P = promote_type(typeof(a), typeof(b))
    if length(a.powers) >= length(b.powers)
        return typeof(a) == P ? lcm!(copy(a), b) : lcm!(convert(P, a), b)
    else
        return typeof(b) == P ? lcm!(copy(b), a) : lcm!(convert(P, b), a)
    end
end
function Base.gcd(a::PrimeFactorization, b::PrimeFactorization)
    P = promote_type(typeof(a), typeof(b))
    if length(a.powers) <= length(b.powers)
        return typeof(a) == P ? lcm!(copy(a), b) : lcm!(convert(P, a), b)
    else
        return typeof(b) == P ? lcm!(copy(b), a) : lcm!(convert(P, b), a)
    end
end
Base.divgcd(a::PrimeFactorization, b::PrimeFactorization) = divgcd!(copy(a), copy(b))
# no promotion necessary, should be smaller than a
divexact(a::PrimeFactorization, b::PrimeFactorization) = divexact!(copy(a), b)

# split `a::PrimeFactorization` into a square `s` and a remainder `r`, such that
# `a = s^2 * r` and the powers in the prime factorization of `r` are zero or one
function splitsquare(a::PrimeFactorization)
    r = PrimeFactorization(map(p->convert(UInt8, isodd(p)), a.powers), a.sign)
    s = PrimeFactorization(map(p->(p>>1), a.powers))
    return s, r
end

# given a list of numerators and denominators, compute the common denominator and
# the rescaled numerator after putting all fractions at the same common denominator
function commondenominator!(nums::Vector{P}, dens::Vector{P}) where {P<:PrimeFactorization}
    isempty(nums) && return one(P)
    # accumulate lcm of denominator
    den = copy(dens[1])
    for i = 2:length(dens)
        lcm!(den, dens[i])
    end
    # rescale numerators
    for i = 1:length(nums)
        divexact!(mul!(nums[i], den), dens[i])
    end
    return den
end

# auxiliary function to compute sums of a list of PrimeFactorizations as quickly as possible
function sumlist!(list::Vector{<:PrimeFactorization}, ind = 1:length(list))
    # first compute gcd to take out common factors
    g = copy(list[ind[1]])
    for k in ind
        gcd!(g, list[k])
    end
    for k in ind
        divexact!(list[k], g)
    end
    L = length(ind)
    i = big(1)
    if L > 32
        l = L >> 1
        s = sumlist!(list, first(ind).+(0:l-1))
        s = MPZ.add!(s, sumlist!(list, first(ind).+(l:L-1)))
    else # do sum, add to s
        s = big(0)
        for k in ind
            MPZ.add!(s, _convert!(i, list[k]))
        end
    end
    return MPZ.mul!(s, _convert!(i, g))
end

function primebinomial(n::BigInt, k::BigInt)
    num = copy(primefactorial(n))
    divexact!(num, primefactorial(k))
    return divexact!(num, primefactorial(n-k))
end

