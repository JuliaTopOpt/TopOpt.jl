using TopOpt, Test

@testset "Functions Show Methods" begin
    # Create a simple problem for testing
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))

    @testset "TrussStress show method" begin
        # Create a truss problem for this test
        nels_truss = (6, 4)
        sizes = (1.0, 1.0)
        truss_problem = PointLoadCantileverTruss(nels_truss, sizes; k_connect=1)
        truss_solver = FEASolver(DirectSolver, truss_problem; xmin=0.001, penalty=PowerPenalty(1.0))
        
        stress_fn = TrussStress(truss_solver)
        io = IOBuffer()
        show(io, MIME("text/plain"), stress_fn)
        output = String(take!(io))
        @test output == "TopOpt truss stress function\n"
    end

    @testset "ElementK show method" begin
        element_k = ElementK(solver)
        io = IOBuffer()
        show(io, MIME("text/plain"), element_k)
        output = String(take!(io))
        @test output == "TopOpt element stiffness matrix construction function\n"
    end

    @testset "Volume show method" begin
        vol = TopOpt.Functions.Volume(solver)
        io = IOBuffer()
        show(io, MIME("text/plain"), vol)
        output = String(take!(io))
        @test output == "TopOpt volume (fraction) function\n"
    end

    @testset "Displacement show method" begin
        disp = Displacement(solver)  # displacement function
        io = IOBuffer()
        show(io, MIME("text/plain"), disp)
        output = String(take!(io))
        @test output == "TopOpt displacement function\n"
    end

    @testset "TrussElementKσ show method" begin
        # Create a truss problem for this test
        nels_truss = (6, 4)
        sizes = (1.0, 1.0)
        truss_problem = PointLoadCantileverTruss(nels_truss, sizes; k_connect=1)
        truss_solver = FEASolver(DirectSolver, truss_problem; xmin=0.001, penalty=PowerPenalty(1.0))
        
        ksigma_fn = TrussElementKσ(truss_problem, truss_solver)
        io = IOBuffer()
        show(io, MIME("text/plain"), ksigma_fn)
        output = String(take!(io))
        @test output == "TopOpt element stress stiffness matrix (Kσ_e) construction function\n"
    end

    @testset "AssembleK show method" begin
        assemble_k = AssembleK(problem)
        io = IOBuffer()
        show(io, MIME("text/plain"), assemble_k)
        output = String(take!(io))
        @test output == "TopOpt global linear stiffness matrix assembly function\n"
    end
end