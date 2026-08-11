using TopOpt, Test
using TopOpt: TopOptProblems
using NonconvexCore: NonconvexCore
using TopOpt.TopOptProblems:
    getmetadata, getpressuredict, getheatfluxdict, getcloaddict, nnodespercell, getfacesets

# Minimal test for Nonconvex.NonconvexCore.getdim(::ComplianceFun) = 1
@testset "getdim ComplianceFun test" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))

    comp = ComplianceFun(solver)

    # Test NonconvexCore.getdim(::ComplianceFun) = 1
    dim = NonconvexCore.getdim(comp)
    @test dim == 1
    println("✓ NonconvexCore.getdim(::ComplianceFun) = $dim")

    # Also test that getdim returns 1 for VolumeFun
    vol = VolumeFun(solver)
    vol_dim = NonconvexCore.getdim(vol)
    @test vol_dim == 1
    println("✓ NonconvexCore.getdim(::VolumeFun) = $vol_dim")
end

# Test for Nonconvex.NonconvexCore.getdim(::MeanComplianceFun) = 1
@testset "getdim MeanComplianceFun test" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    multiload_problem = MultiLoad(problem, F -> begin
        return [F[:, 1] F[:, 1]]
    end)
    solver = FEASolver(
        DirectSolver, multiload_problem; xmin=0.01, penalty=PowerPenaltyFun(3.0)
    )

    mean_comp = MeanComplianceFun(multiload_problem, solver; method=:exact)

    # Test NonconvexCore.getdim(::MeanComplianceFun) = 1
    dim = NonconvexCore.getdim(mean_comp)
    @test dim == 1
    println("✓ NonconvexCore.getdim(::MeanComplianceFun) = $dim")
end

# Test for TopOpt.TopOptProblems.getdim(::LBeam) = 2 (Linear)
@testset "TopOpt.TopOptProblems.getdim LBeam test" begin
    problem = LBeam(Val{:Linear}, Float64; force=1.0)

    # Test TopOpt.TopOptProblems.getdim(::LBeam) = 2
    dim = TopOpt.TopOptProblems.getdim(problem)
    @test dim == 2
    println("✓ TopOpt.TopOptProblems.getdim(::LBeam) = $dim")
end

# Test for TopOpt.TopOptProblems.getdim(::TieBeam) = 2 (Linear)
@testset "TopOpt.TopOptProblems.getdim TieBeam test" begin
    problem = TieBeam(Val{:Linear}, Float64; refine=1, force=1.0)

    # Test TopOpt.TopOptProblems.getdim(::TieBeam) = 2
    dim = TopOpt.TopOptProblems.getdim(problem)
    @test dim == 2
    println("✓ TopOpt.TopOptProblems.getdim(::TieBeam) = $dim")
end

# Test for TieBeam accessor functions
@testset "TieBeam accessor functions" begin
    using TopOpt.TopOptProblems: getpressuredict, getfacesets, nnodespercell, getdh

    # Create a TieBeam problem
    problem = TieBeam(Val{:Linear}, Float64; refine=1, force=1.0)

    # Test nnodespercell(::TieBeam{T,N}) = N
    nnodes = nnodespercell(problem)
    @test nnodes == 4  # Linear quadrilateral elements have 4 nodes
    println("✓ nnodespercell(::TieBeam) = $nnodes")

    # Test getpressuredict(::TieBeam)
    pressure = getpressuredict(problem)
    @test pressure["rightload"] == 2 * 1.0  # 2 * force
    @test pressure["bottomload"] == -1.0    # -force
    println("✓ getpressuredict(::TieBeam) = $pressure")

    # Test getfacesets(::TieBeam)
    facesets = getfacesets(problem)
    @test facesets isa Dict
    println("✓ getfacesets(::TieBeam) returns Dict with $(length(facesets)) entries")
end

# Test for HeatTransferTopOptProblem accessor functions
@testset "HeatTransferTopOptProblem accessor functions" begin
    # Create a HeatConductionProblem using the correct constructor
    # HeatConductionProblem(::Type{Val{CellType}}, nels, sizes, k=1.0; Tleft=0.0, Tright=0.0, heatflux=Dict())
    nels = (10, 10)
    sizes = (1.0, 1.0)
    k = 1.0

    # Quadratic elements (9 nodes per cell in 2D)
    problem = HeatConductionProblem(
        Val{:Quadratic}, nels, sizes, k; Tleft=0.0, Tright=100.0
    )

    # Test getdim
    @test TopOpt.TopOptProblems.getdim(problem) == 2
    println("✓ TopOpt.TopOptProblems.getdim(::HeatConductionProblem) = 2")

    # Test getmetadata
    metadata = getmetadata(problem)
    @test metadata isa TopOpt.TopOptProblems.Metadata
    println("✓ getmetadata(::HeatConductionProblem) returns Metadata")

    # Test getpressuredict - should return empty dict for heat transfer
    pd = getpressuredict(problem)
    @test pd isa Dict{String,Float64}
    @test isempty(pd)
    println("✓ getpressuredict(::HeatConductionProblem) returns empty Dict")

    # Test getheatfluxdict - should return empty dict by default
    hfd = getheatfluxdict(problem)
    @test hfd isa Dict{String,Float64}
    @test isempty(hfd)
    println("✓ getheatfluxdict(::HeatConductionProblem) returns empty Dict")

    # Test getcloaddict - should return empty dict for heat transfer
    cld = getcloaddict(problem)
    @test cld isa Dict{Int,Vector{Float64}}
    @test isempty(cld)
    println("✓ getcloaddict(::HeatConductionProblem) returns empty Dict")

    # Test nnodespercell for quadratic elements (9 nodes in 2D)
    nnodes = nnodespercell(problem)
    @test nnodes == 9
    println("✓ nnodespercell(::HeatConductionProblem with quadratic elements) = $nnodes")

    # Test getfacesets
    fs = getfacesets(problem)
    @test fs isa Dict
    println("✓ getfacesets(::HeatConductionProblem) returns Dict")
end
