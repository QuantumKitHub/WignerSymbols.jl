using Test
using WignerSymbols
using LinearAlgebra
using Random
using Base.Threads
if Base.Sys.iswindows() # not sure why the multithreaded version fails on Windows
    N = 1
else
    N = Base.Threads.nthreads()
end
Random.seed!(1234)

smalljlist = 0:1//2:10
largejlist = 0:1//2:1000

@threads for i = 1:N
    @testset "triangle coefficient, thread $i" begin
        for k = i:N:length(smalljlist)
            j1 = smalljlist[k]
            for j2 in smalljlist
                for j3 = abs(j1-j2):(j1+j2)
                    @test Δ(j1,j2,j3) ≈ sqrt(factorial(big(Int(j1+j2-j3)))*
                                                factorial(big(Int(j1-j2+j3)))*
                                                factorial(big(Int(j2+j3-j1)))/
                                                factorial(big(Int(j1+j2+j3+1))))
                end
            end
        end
    end
end

# test 3j:
#--------
@threads for i = 1:N
    @testset "clebschgordan: orthogonality relations, thread $i" begin
        for k = i:N:length(smalljlist)
            j1 = smalljlist[k]
            for j2 in smalljlist
                d1::Int = 2*j1+1
                d2::Int = 2*j2+1
                M = zeros(Float64, (d1*d2, d1*d2))
                ind1 = 1
                for m1 in -j1:j1, m2 in -j2:j2
                    ind2 = 1
                    @inbounds for j3 in abs(j1-j2):(j1+j2), m3 in -j3:j3
                        M[ind1,ind2] = clebschgordan(j1,m1,j2,m2,j3,m3)
                        ind2 += 1
                    end
                    ind1 += 1
                end
                @test M'*M ≈ one(M)
            end
        end
    end
end

# test recurrence relations: Phys Rev E 57, 7274 (1998)
@threads for i = 1:N
    @testset "wigner3j: recurrence relations, thread $i" begin
        for k = 1:div(8,N)
            j2 = convert(BigFloat, rand(largejlist))
            j3 = convert(BigFloat, rand(largejlist))
            m2 = -convert(BigFloat, rand(-j2:0))
            m3 = convert(BigFloat, rand(-j3:0))

            jmin = max(abs(j2 - j3), abs(m2 + m3)) + 1
            jmax = (j2 + j3) - 1
            # `BigFloat` ranges with `jmax < jmin - 1` throw instead of being empty
            jmin <= jmax || continue
            for j in jmin:jmax
                X = j*sqrt(((j+1)^2-(j2-j3)^2)*((j2+j3+1)^2-(j+1)^2)*((j+1)^2-(m2+m3)^2))
                Y = (2*j+1)*((m2+m3)*(j2*(j2+1)-j3*(j3+1)) - (m2-m3)*j*(j+1))
                Z = (j+1)*sqrt((j^2-(j2-j3)^2)*((j2+j3+1)^2-j^2)*(j^2-(m2+m3)^2))
                tol = 10*max(abs(X),abs(Y),abs(Z))*eps(BigFloat)
                @test (X*wigner3j(BigFloat,j+1,j2,j3,-m2-m3,m2,m3) +
                            Z*wigner3j(BigFloat,j-1,j2,j3,-m2-m3,m2,m3)) ≈
                                (-Y*wigner3j(BigFloat,j,j2,j3,-m2-m3,m2,m3)) atol=tol
            end
        end
    end
end

@threads for i = 1:N
    @testset "wigner3j: orthogonality relations, thread $i" begin
        # equivalent to Clebsch-Gordan orthogonality, now test using Float32
        for k = i:N:length(smalljlist)
            j1 = smalljlist[k]
            for j2 in smalljlist
                d1::Int = 2*j1+1
                d2::Int = 2*j2+1
                M = zeros(Float32, (d1*d2, d1*d2))
                ind2 = 1
                for m1 in -j1:j1, m2 in -j2:j2
                    ind1 = 1
                    @inbounds for j3 in abs(j1-j2):(j1+j2), m3 in -j3:j3
                        d3::Int = 2*j3+1
                        M[ind1,ind2] += sqrt(d3) * wigner3j(Float32, j1, j2, j3, m1, m2, m3)
                        ind1 += 1
                    end
                    ind2 += 1
                end
                @test M'*M ≈ one(M) # orthogonality relation type 1
                @test M*M' ≈ one(M) # orthogonality relation type 2
            end
        end
    end
end

# test 6j
#----------
@threads for i = 1:N
    @testset "wigner6j: orthogonality relations, thread $i" begin
        for k = i:N:length(smalljlist)
            j1 = smalljlist[k]
            for j2 in smalljlist, j4 in smalljlist
                for j5 in max(abs(j1-j2-j4),abs(j1-j2+j4),abs(j1+j2-j4)):(j1+j2+j4)
                    j6range = max(abs(j2-j4),abs(j1-j5)):min((j2+j4),(j1+j5))
                    j3range = max(abs(j1-j2),abs(j4-j5)):min((j1+j2),(j4+j5))
                    @test length(j6range) == length(j3range)
                    M = zeros(Float64, (length(j3range), length(j6range)))
                    for (k2,j6) in enumerate(j6range)
                        for (k1,j3) in enumerate(j3range)
                            M[k1,k2] = sqrt(2*j3+1)*sqrt(2*j6+1)*wigner6j(j1,j2,j3,j4,j5,j6)
                        end
                    end
                    @test M'*M ≈ one(M)
                end
            end
        end
    end
end

@threads for i = 1:N
    @testset "wigner6j: special cases, thread $i" begin
        for k = i:N:length(smalljlist)
            j1 = smalljlist[k]
            for j2 in smalljlist
                j6 = 0
                j4 = j2
                j5 = j1
                for j3 in abs(j1-j2):(j1+j2)
                    @test wigner6j(j1,j2,j3,j4,j5,j6) ≈
                            (-1)^(j1+j2+j3)/sqrt((2*j1+1)*(2*j2+1))
                end
            end
        end
    end
end

@threads for i = 1:N
    @testset "wigner6j: recurrence relation, thread $i" begin
        for k = 1:div(8,N)
            j2 = convert(BigFloat,rand(largejlist))
            j3 = convert(BigFloat,rand(largejlist))
            l1 = convert(BigFloat,rand(largejlist))
            l2 = convert(BigFloat,rand(abs(l1-j3):(l1+j3)))
            l3 = convert(BigFloat,rand(abs(l1-j2):min(l1+j2)))

            for j in intersect(abs(j2-j3):(j2+j3), abs(l2-l3):(l2+l3))
                X = j * sqrt( ((j+1)^2-(j2-j3)^2) * ((j2+j3+1)^2-(j+1)^2) *
                                ((j+1)^2-(l2-l3)^2) * ((l2+l3+1)^2 - (j+1)^2) )
                Y = (2*j+1) * ( j*(j+1)*( -j*(j+1) + j2*(j2+1) + j3*(j3+1) - 2*l1*(l1+1)) +
                                l2*(l2+1)*( j*(j+1) + j2*(j2+1) - j3*(j3+1) ) +
                                l3*(l3+1)*( j*(j+1) - j2*(j2+1) + j3*(j3+1) ) )
                Z = (j+1) * sqrt( (j^2-(j2-j3)^2) * ((j2+j3+1)^2-j^2) *
                                    (j^2-(l2-l3)^2) * ((l2+l3+1)^2-j^2) )
                tol = 10 * max(abs(X), abs(Y), abs(Z)) * eps(BigFloat)
                @test (X*wigner6j(BigFloat, j+1, j2, j3, l1, l2, l3) +
                        Z*wigner6j(BigFloat, j-1, j2, j3, l1, l2, l3)) ≈
                            (-Y*wigner6j(BigFloat, j, j2, j3, l1, l2, l3)) atol=tol
            end
        end
    end
end

@threads for i = 1:N
    @testset "recoupling relation between 3j/CG and 6j/racahW symbols, thread $i" begin
        smallerjlist = 0:1//2:5
        for k = i:N:length(smallerjlist)
            j1 = smallerjlist[k]
            for j2 in smallerjlist, j3 in smallerjlist
                m1range = -j1:j1
                m2range = -j2:j2
                m3range = -j3:j3
                V1 = Array{Float64}(undef, length(m1range),length(m2range),length(m3range))
                V2 = Array{Float64}(undef, length(m1range),length(m2range),length(m3range))
                for J in max(abs(j1-j2-j3),abs(j1-j2+j3),abs(j1+j2-j3)):(j1+j2+j3)
                    J12range = max(abs(j1-j2),abs(J-j3)):min((j1+j2),(J+j3))
                    J23range = max(abs(j2-j3),abs(j1-J)):min((j2+j3),(j1+J))
                    for J12 in J12range, J23 in J23range
                        M = rand(-J:J) # only test for one instance of M in -J:J
                        # should be independent of M anyway
                        fill!(V1,0)
                        fill!(V2,0)
                        for (k1,m1) in enumerate(m1range), (k2,m2) in enumerate(m2range)
                            abs(m1+m2)<=J12 || continue
                            for (k3,m3) in enumerate(m3range)
                                abs(m2+m3)<=J23 || continue
                                m1+m2+m3==M || continue
                                V1[k1,k2,k3] = clebschgordan(j1,m1,j2,m2,J12) *
                                                clebschgordan(J12,m1+m2,j3,m3,J)
                                V2[k1,k2,k3] = clebschgordan(j2,m2,j3,m3,J23) *
                                                clebschgordan(j1,m1,J23,m2+m3,J)
                            end
                        end
                        @test racahW(j1,j2,J,j3,J12,J23) ≈
                                dot(V2,V1)/sqrt((2*J12+1)*(2*J23+1)) atol=10*eps(Float64)
                    end
                end
            end
        end
    end
end

@testset "invalid arguments" begin
    @test_throws DomainError Δ(1 // 3, 1, 1)
    @test_throws DomainError Δ(-1, 1, 1)
    @test_throws DomainError wigner3j(1, 1, 1, 2, -2)
    @test_throws DomainError wigner3j(1 // 3, 1, 1, 0, 0)
    @test_throws DomainError wigner6j(-1, 1, 1, 1, 1, 1)
    @test_throws DomainError wigner6j(0.3, 1, 1, 1, 1, 1)
end

# test symmetries and caching
#-----------------------------
# exact value as a correctly rounded Float64
exactfloat(x) = setprecision(() -> Float64(convert(BigFloat, x)), BigFloat, 512)

@testset "wigner3j: 72 symmetries of the Regge square" begin
    perms = ((1, 2, 3), (2, 3, 1), (3, 1, 2), (2, 1, 3), (1, 3, 2), (3, 2, 1)) # even, then odd
    for j1 in 0:(1 // 2):3, j2 in 0:(1 // 2):3, j3 in abs(j1 - j2):min(j1 + j2, 3), m1 in -j1:j1, m2 in -j2:j2
        m3 = -m1 - m2
        abs(m3) <= j3 || continue
        J = Int(j1 + j2 + j3)
        R = [-j1 + j2 + j3 j1 - j2 + j3 j1 + j2 - j3; j1 - m1 j2 - m2 j3 - m3; j1 + m1 j2 + m2 j3 + m3]
        x = wigner3j(j1, j2, j3, m1, m2, m3)
        x64 = wigner3j(Float64, j1, j2, j3, m1, m2, m3)
        @test x64 === exactfloat(x) + 0.0
        for (kp, p) in enumerate(perms), (kq, q) in enumerate(perms), t in (false, true)
            R′ = R[collect(p), collect(q)]
            t && (R′ = permutedims(R′))
            j′ = ntuple(k -> (R′[2, k] + R′[3, k]) // 2, 3)
            m′ = ntuple(k -> (R′[3, k] - R′[2, k]) // 2, 3)
            sgn = isodd(J) && ((kp > 3) ⊻ (kq > 3)) ? -1 : 1
            @test wigner3j(j′..., m′...) == sgn * x
            @test wigner3j(Float64, j′..., m′...) === sgn * x64 + 0.0
        end
    end
end

@testset "wigner6j: 144 symmetries" begin
    jlist = 0:(1 // 2):2
    for j1 in jlist, j2 in jlist, j3 in jlist, j4 in jlist, j5 in jlist, j6 in jlist
        (δ(j1, j2, j3) && δ(j1, j6, j5) && δ(j2, j4, j6) && δ(j3, j4, j5)) || continue
        x = wigner6j(j1, j2, j3, j4, j5, j6)
        x64 = wigner6j(Float64, j1, j2, j3, j4, j5, j6)
        @test x64 === exactfloat(x) + 0.0
        # Regge symmetry
        s = (j2 + j3 + j5 + j6) // 2
        @test wigner6j(j1, s - j2, s - j3, j4, s - j5, s - j6) == x
        # column permutations and interchange of upper and lower arguments in two columns
        @test wigner6j(j2, j1, j3, j5, j4, j6) == x
        @test wigner6j(j1, j3, j2, j4, j6, j5) == x
        @test wigner6j(j4, j5, j3, j1, j2, j6) == x
        @test wigner6j(Float64, j4, j2, j6, j1, j5, j3) === x64
    end
end

@testset "caches" begin
    W = WignerSymbols
    jmax3 = (@atomic W.Wigner3jFloat.table.maxleading) // 2
    jmax6 = (@atomic W.Wigner6jFloat.table.maxleading) // 2
    # Float64 requests do not fill the exact caches
    empty!(W.Wigner3j)
    empty!(W.Wigner6j)
    x = wigner3j(Float64, 7, 15 // 2, 19 // 2, -3, 1 // 2)
    y = wigner6j(Float64, 7, 15 // 2, 19 // 2, 4, 11 // 2, 13 // 2)
    @test isempty(W.Wigner3j) && isempty(W.Wigner6j)
    @test !iszero(x) && !iszero(y)
    @test x === exactfloat(wigner3j(7, 15 // 2, 19 // 2, -3, 1 // 2))
    @test y === exactfloat(wigner6j(7, 15 // 2, 19 // 2, 4, 11 // 2, 13 // 2))
    @test length(W.Wigner3j) == 1 && length(W.Wigner6j) == 1
    # symbols outside the dense table are stored in the LRU cache, also after resizing
    for jmax in (1, 5, jmax3)
        W.set_table3j_size!(; jmax = jmax)
        W.set_table6j_size!(; jmax = min(jmax, jmax6))
        @test wigner3j(Float64, 7, 15 // 2, 19 // 2, -3, 1 // 2) === x
        @test wigner6j(Float64, 7, 15 // 2, 19 // 2, 4, 11 // 2, 13 // 2) === y
        @test wigner3j(Float64, 201, 300, 150, 5, -7) ===
            exactfloat(wigner3j(201, 300, 150, 5, -7)) + 0.0
        @test wigner6j(Float64, 201, 300, 150, 100, 250, 220) ===
            exactfloat(wigner6j(201, 300, 150, 100, 250, 220)) + 0.0
    end
    @test length(W.Wigner3jFloat.lru) > 0 && length(W.Wigner6jFloat.lru) > 0
    @test_throws DomainError wigner6j(1, 2, 3, 4, 5, 1 // 3)
end

@testset "argument handling" begin
    # different argument types give identical results
    for (j1, j2, j3, m1, m2) in ((1 // 2, 3 // 2, 1, -1 // 2, 3 // 2), (2, 3, 4, -1, 2), (5 // 2, 5 // 2, 2, 1 // 2, -3 // 2))
        x = wigner3j(Float64, j1, j2, j3, m1, m2)
        @test wigner3j(Float64, WignerSymbols.HalfInt.((j1, j2, j3, m1, m2))...) === x
        @test wigner3j(Float64, Float64.((j1, j2, j3, m1, m2))...) === x
        @test wigner3j(Float64, j1, j2, j3, m1, m2, -m1 - m2) === x
        @test wigner3j(j1, j2, j3, m1, m2) == wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
        @test clebschgordan(Float64, j1, m1, j2, m2, j3) === clebschgordan(Float64, j1, m1, j2, m2, j3, m1 + m2)
        @test racahV(Float64, j1, j2, j3, m1, m2) === racahV(Float64, j1, j2, j3, m1, m2, -m1 - m2)
    end
    # vanishing symbols
    @test wigner3j(Float64, 1, 1, 3, 0, 0) === 0.0                 # triangle condition
    @test wigner3j(Float64, 1, 1, 1, 1, 0, 0) === 0.0              # m₁+m₂+m₃ ≠ 0
    @test racahV(1, 1, 3, 0, 0) == 0
    @test wigner6j(Float64, 1, 1, 3, 1, 1, 1) === 0.0
    @test wigner6j(Float64, 1 // 2, 1 // 2, 1 // 2, 1, 1, 1) === 0.0
    @test racahW(1 // 2, 1 // 2, 1, 1 // 2, 1, 1) == 0
    # invalid arguments
    @test_throws DomainError wigner3j(1, 1, 1, 2, -2)       # |m| > j
    @test_throws DomainError wigner3j(1, 1, 1, 1 // 2, -1 // 2) # j - m not integer
    @test_throws DomainError wigner3j(1 // 3, 1, 1, 0, 0)
    @test_throws DomainError wigner3j(-1, 1, 1, 0, 0)
    @test_throws DomainError clebschgordan(1, 0, 1, 1 // 3, 1)
    @test_throws DomainError wigner6j(-1, 1, 1, 1, 1, 1)
    @test_throws DomainError wigner6j(0.3, 1, 1, 1, 1, 1)
end

@testset "triangle coefficient: rounding and arguments" begin
    for (j1, j2, j3) in ((1, 1, 1), (5 // 2, 3, 7 // 2), (40, 31, 20), (150, 201, 300))
        x = Δ(j1, j2, j3)
        @test Δ(Float64, j1, j2, j3) === exactfloat(x)
        @test Δ(Float32, j1, j2, j3) === Float32(exactfloat(x))
        @test Δ(Float64, float.((j1, j2, j3))...) === Δ(Float64, j1, j2, j3)
    end
    @test Δ(Float64, 1, 1, 3) === 0.0
    @test Δ(1, 1 // 2, 1) == 0
    @test_throws DomainError Δ(1 // 3, 1, 1)
    @test_throws DomainError Δ(-1, 1, 1)
end

@testset "GrowingList" begin
    l = WignerSymbols.GrowingList(collect(1:10); sizehint = 4)
    @test length(l) == 10
    @test collect(l) == 1:10
    @test get!(l, 11, 11) == 11
    @test get!(() -> 12, l, 12) == 12
    @test l[12] == 12
    @test_throws ArgumentError get!(l, 14, 14)
end
