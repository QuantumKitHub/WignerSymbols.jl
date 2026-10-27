# WignerSymbols

[![CI](https://github.com/Jutho/WignerSymbols.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Jutho/WignerSymbols.jl/actions/workflows/ci.yml)
[![CI (Julia nightly)](https://github.com/Jutho/WignerSymbols.jl/actions/workflows/ci-julia-nightly.yml/badge.svg)](https://github.com/Jutho/WignerSymbols.jl/actions/workflows/ci-julia-nightly.yml)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![codecov.io](http://codecov.io/github/Jutho/WignerSymbols.jl/coverage.svg?branch=master)](http://codecov.io/github/Jutho/WignerSymbols.jl?branch=master)

Compute Wigner's 3j and 6j symbols, and related quantities such as Clebsch-Gordan
coefficients and Racah's symbols.

## What's new in v2

WignerSymbols.jl was updated to version 2.0 on June 16th, 2021. This is the first major
update in several years. The most important change is that WignerSymbols.jl is now
completely thread safe, i.e. you can request Wigner symbols from different threads
simultaneously. The computation of the Wigner symbols is not in itself multithreaded (this
may be added in the future).

WignerSymbols.jl does no longer store the Wigner 3j and 6j symbols in a `Dict` cache, but
rather in an `LRU` cache from [LRUCache.jl](https://github.com/JuliaCollections/LRUCache.jl). 
Hence, it no longer stores all Wigner symbols ever computed, but only the
most recent ones, and it that sense this is a (softly) breaking release. By default, it
stores the $10^6$ most recent ones, which is probably equivalent to storing all of them in
most use cases. This number can be changed via the interface
```julia
WignerSymbols.set_buffer3j_size(; maxsize = ...)
WignerSymbols.set_buffer6j_size(; maxsize = ...)
```
Thus note that there are separate cache buffers for 3j symbols (or Clebsch-Gordan
coefficients, or Racah V coefficients) and 6j symbols (or Racah W coefficients).

For the underlying prime factorizations on which WignerSymbols.jl is based (which are also
cached), a custom type `GrowingList` was implemented that can be expanded indefinitely in a
thread-safe way. While there is some overhead in making the caches thread safe, these
should mostly be compensated (except for maybe in compilation time) by overall improvements
throughout the library, being more careful about unnecessary computations and about memory
consumption for temporary variables. These changes also rely on `Base.unsafe_rational`
which is only available since Julia 1.5, which is now required and thus provides another
good reason for increasing the major version of WignerSymbols.jl. In tests for generating
all Wigner symbols up to a maximal angular momentum value, WignerSymbols version 2
outperforms version 1.x with about ten to tweny percent.

## Installation
Install with the new package manager via `]add WignerSymbols` or
```julia
using Pkg
Pkg.add("WignerSymbols")
```

## Available functions
While the following function signatures are probably self-explanatory, you can query help
for them in the Julia REPL to get further details.
*   `wigner3j(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₂-m₁) -> ::T`
*   `wigner6j(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃, j₄, j₅, j₆) -> ::T`
*   `wigner9j(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃, j₄, j₅, j₆, j₇, j₈, j₉) -> ::T`
*   `clebschgordan(T::Type{<:Real} = RationalRoot{BigInt}, j₁, m₁, j₂, m₂, j₃, m₃ = m₁+m₂) -> ::T`
*   `racahV(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃, m₁, m₂, m₃ = -m₁-m₂) -> ::T`
*   `racahW(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, J, j₃, J₁₂, J₂₃) -> ::T`
*   `δ(j₁, j₂, j₃) -> ::Bool`
*   `Δ(T::Type{<:Real} = RationalRoot{BigInt}, j₁, j₂, j₃) -> ::T`


The package relies on [HalfIntegers.jl](https://github.com/sostock/HalfIntegers.jl) to
support and use arithmetic with half integer numbers, and, since v1.1, on
[RationalRoots.jl](https://github.com/Jutho/RationalRoots.jl) to return the result exactly
as the square root of a `Rational{BigInt}`, which will then be automatically converted to a
suitable floating point value upon further arithmetic, using the `AbstractIrrational`
interface from Julia Base.

## Implementation
Largely based on reading the paper (but not the code):

[1] [H. T. Johansson and C. Forssén, SIAM Journal on Scientific Compututing 38 (2016) 376-384](https://doi.org/10.1137/15M1021908) ([arXiv:1504.08329](https://arxiv.org/abs/1504.08329))

with some additional modifications to further improve efficiency for large `j` (angular
momenta quantum numbers).

In particular, 3j, 6j, and 9j symbols are computed exactly, in the format `√(r) * s` where `r`
and `s` are exactly computed as `Rational{BigInt}`, using an intermediate representation
based on prime number factorization. This exact representation is captured by the
`RationalRoot` type. For further calculations, these values probably need to be converted
to a floating point type. Because of this exact representation, all of the above functions
can be called requesting `BigFloat` precision for the result.

Most intermediate calculations (prime factorizations of numbers and their factorials,
conversion between prime powers and `BigInt`s) are cached to improve the efficiency, but
this can result in large use of memory when querying Wigner symbols for large values of `j`.

Also uses ideas from

[2] [J. Rasch and A. C. H. Yu, SIAM Journal on Scientific Compututing 25 (2003), 1416–1428](https://doi.org/10.1137/S1064827503422932)

for caching the computed 3j and 6j symbols. Wigner 9-j symbols implemented based on 

[3] [L. Wei, New formula for 9-j symbols and their direct calculation, Computers in Physics, 12 (1998), 632–634.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.481.5946&rep=rep1&type=pdf)

with binomials additionally optimized using the methods described in [1].
