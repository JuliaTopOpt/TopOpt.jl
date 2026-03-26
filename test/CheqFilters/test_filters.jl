using TopOpt, Test, LinearAlgebra, Zygote, Ferrite
using TopOpt: DensityFilter, SensFilter, apply_filter, apply_filter_sens
using Ferrite: Grid, DofHandler

# Helper function to create simple test grid
function create_test_grid(nx=10, ny=10)
    grid = Ferrite.generate_grid(Ferrite.Quadrilateral, (nx, ny), Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0)))
    return grid
end

@testset "DensityFilter Construction" begin
    grid = create_test_grid()
    rmin = 2.0
    
    df = DensityFilter(rmin, grid)
    @test df.rmin == rmin
    @test df.grid === grid
    
    # Test that filter matrix was computed
    @test df.H !== nothing
    @test df.Hs !== nothing
end

@testset "SensFilter Construction" begin
    grid = create_test_grid()
    rmin = 2.0
    
    sf = SensFilter(rmin, grid)
    @test sf.rmin == rmin
    @test sf.grid === grid
end

@testset "DensityFilter Application" begin
    grid = create_test_grid(5, 5)
    rmin = 1.5
    
    df = DensityFilter(rmin, grid)
    
    # Test with uniform density
    x_uniform = ones(Ferrite.getncells(grid))
    x_filtered = apply_filter(df, x_uniform)
    @test x_filtered ≈ x_uniform
    
    # Test with random density
    x_rand = rand(Ferrite.getncells(grid))
    x_filtered = apply_filter(df, x_rand)
    @test length(x_filtered) == length(x_rand)
    @test all(0 .<= x_filtered .<= 1)
    
    # Test with zero density
    x_zero = zeros(Ferrite.getncells(grid))
    x_filtered = apply_filter(df, x_zero)
    @test x_filtered ≈ x_zero
    
    # Test with checkerboard pattern
    x_checker = zeros(Ferrite.getncells(grid))
    for i in 1:length(x_checker)
        x_checker[i] = (i % 2 == 0) ? 1.0 : 0.0
    end
    x_filtered = apply_filter(df, x_checker)
    @test all(0 .<= x_filtered .<= 1)
    @test any(x_filtered .> 0) && any(x_filtered .< 1)
end

@testset "SensFilter Application" begin
    grid = create_test_grid(5, 5)
    rmin = 1.5
    
    sf = SensFilter(rmin, grid)
    
    # Test with uniform sensitivity
    x = ones(Ferrite.getncells(grid))
    sens_uniform = ones(Ferrite.getncells(grid))
    sens_filtered = apply_filter_sens(sf, x, sens_uniform)
    @test sens_filtered ≈ sens_uniform
    
    # Test with random sensitivity
    sens_rand = rand(Ferrite.getncells(grid))
    sens_filtered = apply_filter_sens(sf, x, sens_rand)
    @test length(sens_filtered) == length(sens_rand)
    
    # Test with zero sensitivity
    sens_zero = zeros(Ferrite.getncells(grid))
    sens_filtered = apply_filter_sens(sf, x, sens_zero)
    @test sens_filtered ≈ sens_zero
    
    # Test gradient (chain rule)
    x = fill(0.5, Ferrite.getncells(grid))
    sens = rand(Ferrite.getncells(grid))
    sens_filtered = apply_filter_sens(sf, x, sens)
    @test length(sens_filtered) == length(sens)
end

@testset "Filter with Different Grid Sizes" begin
    for (nx, ny) in [(3, 3), (5, 10), (10, 5)]
        grid = create_test_grid(nx, ny)
        rmin = 1.2
        
        df = DensityFilter(rmin, grid)
        sf = SensFilter(rmin, grid)
        
        x = rand(Ferrite.getncells(grid))
        
        x_f = apply_filter(df, x)
        sens_f = apply_filter_sens(sf, x, ones(length(x)))
        
        @test length(x_f) == Ferrite.getncells(grid)
        @test length(sens_f) == Ferrite.getncells(grid)
    end
end

@testset "Filter Edge Cases" begin
    # Very small grid
    grid_small = create_test_grid(2, 2)
    df_small = DensityFilter(0.5, grid_small)
    x_small = [0.2, 0.5, 0.8, 0.3]
    x_f = apply_filter(df_small, x_small)
    @test length(x_f) == 4
    
    # Large rmin relative to grid
    grid = create_test_grid(5, 5)
    df_large = DensityFilter(5.0, grid)
    x = rand(Ferrite.getncells(grid))
    x_f = apply_filter(df_large, x)
    @test length(x_f) == length(x)
    # With large rmin, filtering should be strong
    @test std(x_f) < std(x)  # Variance should be reduced
end

@testset "Filter Gradients" begin
    grid = create_test_grid(4, 4)
    rmin = 1.5
    
    df = DensityFilter(rmin, grid)
    
    # Test gradient through filter
    x0 = fill(0.5, Ferrite.getncells(grid))
    
    function f_density(x)
        x_f = apply_filter(df, x)
        return sum(x_f.^2)
    end
    
    g = Zygote.gradient(f_density, x0)[1]
    @test length(g) == length(x0)
    @test all(isfinite, g)
end

@testset "Filter Type Stability" begin
    grid = create_test_grid(5, 5)
    rmin = 1.5
    
    df = DensityFilter(rmin, grid)
    x = rand(Ferrite.getncells(grid))
    
    x_f = apply_filter(df, x)
    @test eltype(x_f) == eltype(x)
end

@testset "Filter Independence" begin
    grid1 = create_test_grid(5, 5)
    grid2 = create_test_grid(5, 5)
    
    df1 = DensityFilter(1.5, grid1)
    df2 = DensityFilter(2.0, grid2)
    
    x = rand(Ferrite.getncells(grid1))
    
    x_f1 = apply_filter(df1, x)
    x_f2 = apply_filter(df2, x)
    
    # Different rmin should give different results
    @test !(x_f1 ≈ x_f2)
end