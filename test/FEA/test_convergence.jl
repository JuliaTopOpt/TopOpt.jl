using TopOpt, Test, LinearAlgebra, Ferrite
using TopOpt: ConvergenceHistory, has_converged, default_convergence

@testset "ConvergenceHistory Construction" begin
    # Test basic construction
    hist = ConvergenceHistory()
    @test length(hist.obj_vals) == 0
    @test length(hist.changes) == 0
    @test hist.convstate === nothing
    
    # Test construction with parameters
    hist2 = ConvergenceHistory(Float64[], Float64[], nothing)
    @test hist2 isa ConvergenceHistory
end

@testset "ConvergenceHistory State Tracking" begin
    hist = ConvergenceHistory()
    
    # Simulate optimization iterations
    for i in 1:10
        push!(hist.obj_vals, 100.0 / i)  # Decreasing objective
        push!(hist.changes, 0.1 / i)    # Decreasing change
    end
    
    @test length(hist.obj_vals) == 10
    @test length(hist.changes) == 10
    @test hist.obj_vals[end] < hist.obj_vals[1]
    @test hist.changes[end] < hist.changes[1]
end

@testset "Default Convergence Check" begin
    # Test with converged history
    hist_conv = ConvergenceHistory()
    for i in 1:5
        push!(hist_conv.obj_vals, 100.0)
        push!(hist_conv.changes, 0.001)
    end
    
    # Should converge if changes are small
    result = default_convergence(hist_conv)
    @test result == true
    
    # Test with non-converged history
    hist_noconv = ConvergenceHistory()
    for i in 1:5
        push!(hist_noconv.obj_vals, 100.0)
        push!(hist_noconv.changes, 0.1)
    end
    
    # Should not converge if changes are large
    result = default_convergence(hist_noconv)
    @test result == false
end

@testset "Has Converged Function" begin
    # Test with empty history
    hist_empty = ConvergenceHistory()
    @test !has_converged(hist_empty)
    
    # Test with converged state
    hist = ConvergenceHistory()
    hist.convstate = true
    @test has_converged(hist)
    
    # Test with non-converged state
    hist2 = ConvergenceHistory()
    hist2.convstate = false
    @test !has_converged(hist2)
    
    # Test with nothing state (should return false)
    hist3 = ConvergenceHistory()
    @test hist3.convstate === nothing
    @test !has_converged(hist3)
end

@testset "Convergence with Different Tolerances" begin
    # Test strict tolerance
    hist_strict = ConvergenceHistory()
    push!(hist_strict.obj_vals, 100.0)
    push!(hist_strict.changes, 0.0001)
    
    # Test lenient tolerance
    hist_lenient = ConvergenceHistory()
    push!(hist_lenient.obj_vals, 100.0)
    push!(hist_lenient.changes, 0.05)
    
    # Default convergence uses internal tolerance
    # Just verify they don't error
    @test default_convergence(hist_strict) isa Bool
    @test default_convergence(hist_lenient) isa Bool
end

@testset "ConvergenceHistory Edge Cases" begin
    # Single iteration
    hist_single = ConvergenceHistory()
    push!(hist_single.obj_vals, 100.0)
    push!(hist_single.changes, 0.001)
    @test length(hist_single.obj_vals) == 1
    
    # Many iterations
    hist_many = ConvergenceHistory()
    for i in 1:1000
        push!(hist_many.obj_vals, 1.0 / i)
        push!(hist_many.changes, 0.01 / i)
    end
    @test length(hist_many.obj_vals) == 1000
    
    # Oscillating objective
    hist_osc = ConvergenceHistory()
    for i in 1:10
        push!(hist_osc.obj_vals, i % 2 == 0 ? 100.0 : 95.0)
        push!(hist_osc.changes, 5.0)
    end
    # Should not converge with oscillating values
    @test !default_convergence(hist_osc)
end

@testset "Custom Convergence Criteria" begin
    # Custom convergence function based on objective value change
    function custom_convergence(hist::ConvergenceHistory; tol=0.01)
        length(hist.obj_vals) < 2 && return false
        
        # Check relative change in objective
        rel_change = abs(hist.obj_vals[end] - hist.obj_vals[end-1]) / abs(hist.obj_vals[end-1])
        return rel_change < tol
    end
    
    hist = ConvergenceHistory()
    push!(hist.obj_vals, 100.0)
    push!(hist.obj_vals, 99.0)  # 1% change
    push!(hist.changes, 1.0)
    
    @test custom_convergence(hist, tol=0.02)
    @test !custom_convergence(hist, tol=0.005)
end