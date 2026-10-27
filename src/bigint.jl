# additional bigint functionality
using Base.GMP.MPZ
using Base.GMP.MPZ: gmpz, mpz_t

# `Base.GMP.libgmp` was added in Julia 1.9
const libgmp = isdefined(Base.GMP, :libgmp) ? Base.GMP.libgmp : :libgmp

divexact!(x::BigInt, a::BigInt, b::BigInt) =
    (ccall((:__gmpz_divexact, libgmp), Cvoid, (mpz_t, mpz_t, mpz_t), x, a, b); x)
divexact(a::BigInt, b::BigInt) = divexact!(BigInt(), a, b)
divexact!(a::BigInt, b::BigInt) = divexact!(a, a, b)

const TMP_BIG = BigInt(0)

function mul!(x::Rational{BigInt}, a::Rational{BigInt}, b::Rational{BigInt})
    MPZ.mul!(x.num, a.num, b.num)
    MPZ.mul!(x.den, a.den, b.den)
    g = MPZ.gcd!(TMP_BIG, x.num, x.den)
    divexact!(x.num, g)
    divexact!(x.den, g)
    return x
end
