# Canonical forms of Wigner symbols under their full symmetry groups, following
# J. Rasch and A. C. H. Yu, SIAM J. Sci. Comput. 25, 1416 (2003).
#
# Both canonical forms can be represented as non-increasing tuples of non-negative
# integers, which are mapped onto a dense linear index using the combinatorial number
# system, see `_tableindex`.

# 3j symbols
#------------
"""
    ReggeSquare(data::NTuple{9,Int})

Regge square of a 3j symbol with arguments `(j₁, j₂, j₃, m₁, m₂, m₃)`, i.e. the 3×3 matrix
```
⎡ -j₁+j₂+j₃  j₁-j₂+j₃  j₁+j₂-j₃ ⎤
⎢  j₁-m₁      j₂-m₂     j₃-m₃   ⎥
⎣  j₁+m₁      j₂+m₂     j₃+m₃   ⎦
```
All entries are non-negative integers, and all row and column sums equal
`J = j₁ + j₂ + j₃`. The 72 symmetries of the 3j symbol correspond to row and column
permutations and transposition of the Regge square. Under an odd permutation of rows or
columns, the 3j symbol picks up a sign `(-1)^J`; transposition leaves it invariant.
"""
struct ReggeSquare
    data::NTuple{9, Int} # column major
end
# entries of the Regge square in column major order
@inline function _reggeentries(j₁, j₂, j₃, m₁, m₂, m₃)
    return (
        -j₁ + j₂ + j₃, j₁ - m₁, j₁ + m₁,
        j₁ - j₂ + j₃, j₂ - m₂, j₂ + m₂,
        j₁ + j₂ - j₃, j₃ - m₃, j₃ + m₃,
    )
end
Base.@propagate_inbounds Base.getindex(R::ReggeSquare, i::Int, j::Int) = R.data[i + 3 * (j - 1)]
sumJ(R::ReggeSquare) = R[1, 1] + R[1, 2] + R[1, 3]

"""
    Key3j(L, X, T, B, S)

Canonical form of a 3j symbol under its 72 symmetries, following Rasch and Yu. It
corresponds to the Regge square
```
⎡ S      L      J-S-L ⎤
⎢ X      B      J-X-B ⎥
⎣ J-S-X  J-L-B  T     ⎦
```
with `L` the largest and `S` the smallest entry, `L ≥ X ≥ T ≥ B ≥ S`, and
`J = S + X + L + B - T`.
"""
struct Key3j
    L::Int
    X::Int
    T::Int
    B::Int
    S::Int
end
_combinatorialkey(k::Key3j) = (k.L, k.X, k.T, k.B, k.S)

"""
    canonicalize(R::ReggeSquare) -> (key::Key3j, odd::Bool)

Bring the Regge square to the canonical form, and return whether this required an odd
permutation of rows and columns.
"""
@inline function canonicalize(R::ReggeSquare)
    # locate largest entry
    imax, jmax, L = 1, 1, R[1, 1]
    for j in 1:3, i in 1:3
        if R[i, j] > L
            imax, jmax, L = i, j, R[i, j]
        end
    end
    # the smallest entry is in the row or in the column of L; if not in the row: transpose
    rowmin = min(R[imax, 1], R[imax, 2], R[imax, 3])
    colmin = min(R[1, jmax], R[2, jmax], R[3, jmax])
    transposed = colmin < rowmin
    @inline e(i, j) = transposed ? R[j, i] : R[i, j]
    i₀, j₀ = transposed ? (jmax, imax) : (imax, jmax)

    # row permutation (p₁, p₂, p₃) with p₁ = i₀: odd iff i₀ == 2
    p₁ = i₀
    p₂, p₃ = i₀ == 1 ? (2, 3) : i₀ == 2 ? (1, 3) : (1, 2)
    odd = i₀ == 2
    # column permutation (q₁, q₂, q₃) with q₂ = j₀: odd iff j₀ != 2
    q₂ = j₀
    q₁, q₃ = j₀ == 1 ? (2, 3) : j₀ == 2 ? (1, 3) : (1, 2)
    odd ⊻= j₀ != 2
    # S at (1,1)
    if e(p₁, q₁) > e(p₁, q₃)
        q₁, q₃ = q₃, q₁
        odd = !odd
    end
    # B = R₂₂ ≤ R₃₂, ties broken by T = R₃₃ ≥ R₂₃
    if e(p₂, q₂) > e(p₃, q₂) || (e(p₂, q₂) == e(p₃, q₂) && e(p₂, q₃) > e(p₃, q₃))
        p₂, p₃ = p₃, p₂
        odd = !odd
    end
    return Key3j(L, e(p₂, q₁), e(p₃, q₃), e(p₂, q₂), e(p₁, q₁)), odd
end

# 6j symbols
#------------
"""
    Key6j(j₁, j₂, j₃, j₄, j₅, j₆)

Canonical form of a 6j symbol under its 144 symmetries, which correspond to arbitrary
permutations of the sums `αᵢ` of the four triads and of the sums `βⱼ` of the three
quadrangles. It stores these sums sorted as `β₁ ≥ β₂ ≥ β₃ ≥ α₁ ≥ α₂ ≥ α₃ ≥ α₄`.

Since `∑ᵢ αᵢ = ∑ⱼ βⱼ`, the 6j symbol is fully determined by the non-increasing tuple
`(β₁-α₄, β₂-α₄, β₃-α₄, β₃-α₃, β₃-α₂, β₃-α₁)`, and every non-increasing tuple of
non-negative integers corresponds to exactly one canonical 6j symbol.
"""
struct Key6j
    β₁::Int
    β₂::Int
    β₃::Int
    α₁::Int
    α₂::Int
    α₃::Int
    α₄::Int
end
@inline function Key6j(j₁, j₂, j₃, j₄, j₅, j₆)
    α₁, α₂, α₃, α₄ = _sort4(
        convert(Int, j₁ + j₂ + j₃), convert(Int, j₁ + j₆ + j₅),
        convert(Int, j₂ + j₄ + j₆), convert(Int, j₃ + j₄ + j₅)
    )
    β₁, β₂, β₃ = _sort3(
        convert(Int, j₁ + j₂ + j₄ + j₅), convert(Int, j₁ + j₃ + j₄ + j₆),
        convert(Int, j₂ + j₃ + j₅ + j₆)
    )
    return Key6j(β₁, β₂, β₃, α₁, α₂, α₃, α₄)
end
function _combinatorialkey(k::Key6j)
    return (k.β₁ - k.α₄, k.β₂ - k.α₄, k.β₃ - k.α₄, k.β₃ - k.α₃, k.β₃ - k.α₂, k.β₃ - k.α₁)
end

# sorting networks, in descending order
@inline _minmax(a, b) = a < b ? (b, a) : (a, b)
@inline function _sort3(a, b, c)
    a, b = _minmax(a, b)
    b, c = _minmax(b, c)
    a, b = _minmax(a, b)
    return a, b, c
end
@inline function _sort4(a, b, c, d)
    a, b = _minmax(a, b)
    c, d = _minmax(c, d)
    a, c = _minmax(a, c)
    b, d = _minmax(b, d)
    b, c = _minmax(b, c)
    return a, b, c, d
end

# Dense index
#-------------
# binomial coefficient for small k and n ≥ 0, without overflow checks
@inline function _binomial(n::Int, k::Int)
    r = 1
    for i in 1:k
        r = (r * (n - k + i)) ÷ i
    end
    return r
end

# The combinatorial number system maps non-increasing tuples k₁ ≥ k₂ ≥ … ≥ k_N ≥ 0 one to
# one onto the positive integers, via 1 + ∑ᵢ binomial(kᵢ + N - i, N - i + 1). Keys are
# thus ordered by their leading entry k₁, and the `binomial(K + N, N)` keys with k₁ ≤ K
# occupy the first positions.
@inline function _tableindex(key::NTuple{N, Int}) where {N}
    i = 1
    for n in 1:N
        @inbounds i += _binomial(key[n] + N - n, N - n + 1)
    end
    return i
end
_tablelength(N::Int, K::Int) = _binomial(K + N, N)
