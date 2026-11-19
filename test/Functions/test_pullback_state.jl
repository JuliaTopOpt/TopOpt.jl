using TopOpt, Test, LinearAlgebra, SparseArrays, Zygote, Random, ChainRulesCore

# A pullback must describe its own forward evaluation, not the last use of a
# shared workspace. Two distinct designs exercise this through ordinary AD.
function check_composed_pullback(f, n)
    x = collect(range(0.45, 0.75; length=n))
    objective = z -> sum(f(PseudoDensities(z))) + 0.7 * sum(f(PseudoDensities(0.8 .* z)))
    h = 1e-5
    fd = map(eachindex(x)) do i
        step = zeros(n)
        step[i] = h
        (objective(x + step) - objective(x - step)) / (2h)
    end
    ad = Zygote.gradient(objective, x)[1]
    @test all(isfinite, ad)
    @test ad ≈ fd rtol = 1e-4 atol = 1e-6
end

@testset "Pullbacks retain their forward state" begin
    Random.seed!(123)
    p = PointLoadCantilever((4, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    s = FEASolver(DirectSolver, p; xmin=0.001, penalty=PowerPenaltyFun(3.0))
    check_composed_pullback(DisplacementFun(s), length(s.vars))

    tp = HeatConductionProblem(
        (4, 2), (1.0, 1.0), 1.0; Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 1.0)
    )
    ts = FEASolver(DirectSolver, tp; xmin=0.001, penalty=PowerPenaltyFun(3.0))
    check_composed_pullback(TemperatureFun(ts), length(ts.vars))

    F = sparse(hcat(s.elementinfo.fixedload, 2 .* s.elementinfo.fixedload))
    ml = MultiLoad(p, F)
    for method in (:exact, :exact_svd, :trace, :svd_trace)
        @testset "mean $method" begin
            mc = MeanComplianceFun(ml, s; method, nv=4)
            check_composed_pullback(mc, length(s.vars))
        end
    end
    for method in (:exact, :exact_svd, :approx)
        @testset "block $method" begin
            bc = BlockComplianceFun(ml, s; method, nv=4)
            check_composed_pullback(bc, length(s.vars))
        end
    end

    # Resampling must not change a saved pullback's probes or primal states.
    bc = BlockComplianceFun(ml, s; method=:approx, nv=4, sample_once=false)
    x = PseudoDensities(fill(0.6, length(s.vars)))
    value, pb = rrule(bc, x)
    weights = [0.4, 1.2]
    expected = copy(pb(weights)[2].x)
    bc(PseudoDensities(fill(0.4, length(s.vars))))
    @test pb(weights)[2].x ≈ expected
    @test pb(weights)[2].x ≈ expected

    nodes, elements, mats, areas, fixities, cases = load_truss_json(
        joinpath(@__DIR__, "testfile2_compact.json")
    )
    truss = TrussProblem(nodes, elements, cases["0"], fixities, mats, areas)
    truss_solver = FEASolver(DirectSolver, truss; xmin=0.001, penalty=PowerPenaltyFun(3.0))
    check_composed_pullback(TrussStressFun(truss_solver), length(elements))
end
