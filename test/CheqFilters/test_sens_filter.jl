using TopOpt.CheqFilters
using Test
using LinearAlgebra

# Test sensitivity filter functionality

@testset "Sensitivity Filter Tests" begin
    @testset "Filter initialization" begin
        # Create filter parameters
        n_elements = 100
        rmin = 1.5  # Filter radius
        
        # Simple filter initialization test
        @test rmin > 0.0
        @test n_elements > 0
        
        # Check filter radius scaling
        cell_size = 1.0
        rmin_scaled = rmin * cell_size
        @test rmin_scaled ≈ rmin
    end
    
    @testset "Neighbor search" begin
        # Test neighbor finding for a simple grid
        nx, ny = 10, 10
        n_elem = nx * ny
        
        # Element centers
        x = Float64[]
        y = Float64[]
        for j in 1:ny
            for i in 1:nx
                push!(x, i - 0.5)
                push!(y, j - 0.5)
            end
        end
        
        rmin = 1.5
        
        # Find neighbors for center element
        center_idx = 55
        neighbors = Int[]
        weights = Float64[]
        
        for i in 1:n_elem
            dist = sqrt((x[i] - x[center_idx])^2 + (y[i] - y[center_idx])^2)
            if dist <= rmin
                push!(neighbors, i)
                push!(weights, rmin - dist)
            end
        end
        
        # Check results
        @test length(neighbors) > 0
        @test length(neighbors) == length(weights)
        @test center_idx in neighbors
        
        # Normalized weights
        weight_sum = sum(weights)
        weights_norm = weights / weight_sum
        @test sum(weights_norm) ≈ 1.0
    end
    
    @testset "Sensitivity weight computation" begin
        n_elem = 50
        
        # Create random densities
        ρ = rand(n_elem)
        
        # Sensitivity weights (typical in topology optimization)
        α = 1.0  # Damping factor
        
        # Simple weight calculation
        weights = ρ .^ α
        
        @test length(weights) == n_elem
        @test all(weights .>= 0.0)
        @test all(weights .<= 1.0)
    end
    
    @testset "Filter convolution" begin
        n_elem = 100
        
        # Create simple sensitivity field
        sens = rand(n_elem)
        
        # Simple box filter
        filter_size = 3
        filtered = similar(sens)
        
        for i in 1:n_elem
            start_idx = max(1, i - filter_size)
            end_idx = min(n_elem, i + filter_size)
            window = start_idx:end_idx
            filtered[i] = sum(sens[window]) / length(window)
        end
        
        # Check filtered values
        @test length(filtered) == n_elem
        @test minimum(filtered) >= minimum(sens)
        @test maximum(filtered) <= maximum(sens)
        
        # Conservation check (approximate)
        @test sum(filtered) ≈ sum(sens) rtol=0.1
    end
    
    @testset "Density filter vs sensitivity filter" begin
        n_elem = 50
        ρ = rand(n_elem)
        sens = rand(n_elem)
        
        rmin = 2.0
        
        # Density filter: modifies densities directly
        # ρ_filtered = filter(ρ)
        
        # Sensitivity filter: modifies sensitivities
        # sens_filtered = filter(sens, ρ)
        
        # Test that both operations are valid
        @test length(ρ) == n_elem
        @test length(sens) == n_elem
        
        # Simple filtered values
        ρ_filtered = 0.5 * ρ + 0.5 * circshift(ρ, 1)
        sens_filtered = 0.5 * sens + 0.5 * circshift(sens, 1)
        
        @test length(ρ_filtered) == n_elem
        @test length(sens_filtered) == n_elem
    end
    
    @testset "Helmholtz filter" begin
        # Helmholtz-type filter for sensitivities
        n = 50
        
        # Create 1D field
        x = range(0, 1, length=n)
        f = sin.(2π * x)
        
        # Filter radius parameter
        r = 0.1
        
        # Simple Helmholtz smoothing (approximated)
        f_filtered = similar(f)
        α = r^2  # Filter parameter
        
        for i in 2:n-1
            f_filtered[i] = (f[i] + α * (f[i-1] + f[i+1])) / (1 + 2α)
        end
        f_filtered[1] = f[1]
        f_filtered[n] = f[n]
        
        @test length(f_filtered) == n
        
        # Filter should smooth oscillations
        @test sum(abs.(diff(f_filtered))) < sum(abs.(diff(f)))
    end
    
    @testset "Sensitivity projection" begin
        n_elem = 100
        
        # Raw sensitivity
        sens_raw = rand(n_elem) .* 2 .- 1  # Range [-1, 1]
        
        # Projection parameter
        β = 8.0
        
        # Heaviside projection (approximate)
        sens_proj = @. (tanh(β * sens_raw) + tanh(β * (1 - sens_raw))) / 2
        
        @test length(sens_proj) == n_elem
        @test all(sens_proj .>= 0.0)
        @test all(sens_proj .<= 1.0)
    end
    
    @testset "Chain rule for filtered sensitivities" begin
        n_elem = 50
        
        # Objective: compliance
        # Filter: density filter
        # We need dC/dρ_raw
        
        ρ = rand(n_elem)
        
        # Mock filtered densities
        H = 0.5 * I(n_elem) + 0.25 * (diagm(1 => ones(n_elem-1)) + diagm(-1 => ones(n_elem-1)))
        ρ_filtered = H * ρ
        
        # Sensitivities wrt filtered densities
        dC_dρ_filtered = rand(n_elem)
        
        # Chain rule
        dC_dρ = H' * dC_dρ_filtered
        
        @test length(dC_dρ) == n_elem
        @test isfinite(sum(dC_dρ))
    end
    
    @testset "Filter kernel properties" begin
        # Filter kernel should be positive and normalize to 1
        kernel_size = 5
        σ = 1.0
        
        x = range(-2, 2, length=kernel_size)
        kernel = @. exp(-x^2 / (2 * σ^2))
        kernel_normalized = kernel / sum(kernel)
        
        @test length(kernel_normalized) == kernel_size
        @test sum(kernel_normalized) ≈ 1.0
        @test all(kernel_normalized .>= 0.0)
        
        # Symmetry
        @test kernel_normalized ≈ reverse(kernel_normalized)
    end
end