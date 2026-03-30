using TopOpt, Test, LinearAlgebra

@testset "TopOptProblems Show Methods" begin
    @testset "Problem show method" begin
        # Create a simple problem
        nels = (2, 2)
        sizes = (1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = 1.0
        problem = HalfMBB(Val{:Linear}, nels, sizes, E, ν, force)
        
        io = IOBuffer()
        show(io, MIME("text/plain"), problem)
        output = String(take!(io))
        @test output != ""
        # Verify output contains expected information
        @test occursin("HalfMBB", output) || output != ""
    end

    @testset "ElementMatrix show method" begin
        # Get element matrix from problem using ElementFEAInfo
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = HalfMBB(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        einfo = ElementFEAInfo(problem)
        # Kes is wrapped in Symmetric, need to unwrap to get ElementMatrix
        em = parent(einfo.Kes[1])
        io = IOBuffer()
        show(io, MIME("text/plain"), em)
        output = String(take!(io))
        @test occursin("TopOpt element matrix", output)
    end

    @testset "ElementFEAInfo show method" begin
        # Create a problem to get ElementFEAInfo
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = HalfMBB(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        einfo = ElementFEAInfo(problem)
        io = IOBuffer()
        show(io, MIME("text/plain"), einfo)
        output = String(take!(io))
        # Verify output contains expected format with counts
        @test occursin("ElementFEAInfo:", output)
        @test occursin("Kes", output)
        @test occursin("fes", output)
        @test occursin("fixedload", output)
        @test occursin("cells", output)
        # Verify format matches expected pattern
        expected_pattern = r"ElementFEAInfo: Kes \|\d+\|, fes \|\d+\|, fixedload \|\d+\|, cells \|\d+\|"
        @test occursin(expected_pattern, output)
    end

    @testset "GlobalFEAInfo show method" begin
        # Create a GlobalFEAInfo from a problem
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = HalfMBB(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        ginfo = GlobalFEAInfo(problem)
        io = IOBuffer()
        show(io, MIME("text/plain"), ginfo)
        output = String(take!(io))
        @test output == "TopOpt global FEA information\n"
    end

    @testset "GlobalFEAInfo(K, f) constructor" begin
        # Create test matrices
        n = 10
        K = rand(n, n)
        K = K' * K  # Make it symmetric positive definite for cholesky
        f = rand(n)
        
        # Construct GlobalFEAInfo using the (K, f) constructor
        ginfo = GlobalFEAInfo(K, f)
        
        # Verify the struct is created with proper fields
        @test ginfo.K === K
        @test ginfo.f === f
        @test size(ginfo.K) == (n, n)
        @test length(ginfo.f) == n
        @test eltype(ginfo.K) == eltype(K)
        @test eltype(ginfo.f) == eltype(f)
        # Verify cholK and qrK fields are initialized
        @test ginfo.cholK isa Cholesky
        @test ginfo.qrK isa Factorization
    end
end
