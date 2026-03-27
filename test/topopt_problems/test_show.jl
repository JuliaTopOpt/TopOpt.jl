using TopOpt, Test

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
end