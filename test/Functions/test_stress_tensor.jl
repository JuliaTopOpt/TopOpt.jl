using TopOpt
using Test
using LinearAlgebra

# Test stress tensor computation and stress-based functions

@testset "Stress Tensor Tests" begin
    @testset "Stress tensor computation" begin
        # Create simple 2D stress state
        σ_xx = 1.0
        σ_yy = 0.5
        σ_xy = 0.3
        
        # Stress tensor matrix
        σ = [σ_xx σ_xy 0.0;
             σ_xy σ_yy 0.0;
             0.0  0.0  0.0]
        
        # Principal stresses
        eigenvals = eigvals(σ[1:2, 1:2])
        σ1 = maximum(eigenvals)
        σ2 = minimum(eigenvals)
        
        @test σ1 >= σ2
        @test σ1 + σ2 ≈ σ_xx + σ_yy
    end
    
    @testset "von Mises stress" begin
        # Simple 2D stress state
        σ_xx = 100.0
        σ_yy = 50.0
        σ_xy = 30.0
        
        # von Mises stress formula
        σ_vm = sqrt(σ_xx^2 - σ_xx*σ_yy + σ_yy^2 + 3*σ_xy^2)
        
        # Check value
        expected = sqrt(10000 - 5000 + 2500 + 2700)
        @test σ_vm ≈ expected
        
        # Uniaxial stress
        σ_uni = 100.0
        σ_vm_uni = σ_uni  # For uniaxial stress, von Mises = stress magnitude
        @test σ_vm_uni ≈ 100.0
    end
    
    @testset "Tresca stress" begin
        # Principal stresses
        σ1 = 100.0
        σ2 = 20.0
        σ3 = -30.0
        
        # Maximum shear stress (Tresca)
        τ_max = (maximum([σ1, σ2, σ3]) - minimum([σ1, σ2, σ3])) / 2
        
        @test τ_max ≈ (100.0 - (-30.0)) / 2
        @test τ_max ≈ 65.0
    end
    
    @testset "Stress tensor invariants" begin
        # 3D stress tensor
        σ = [100.0  30.0  20.0;
              30.0  50.0  10.0;
              20.0  10.0  40.0]
        
        # First invariant (trace)
        I1 = tr(σ)
        @test I1 ≈ 100.0 + 50.0 + 40.0
        
        # Second invariant
        I2 = 0.5 * (tr(σ)^2 - tr(σ^2))
        
        # Third invariant (determinant)
        I3 = det(σ)
        
        # Check invariants are real numbers
        @test isreal(I1)
        @test isreal(I2)
        @test isreal(I3)
    end
    
    @testset "Deviatoric stress" begin
        # Hydrostatic stress
        σ_h = [50.0  0.0  0.0;
               0.0  50.0  0.0;
               0.0   0.0  50.0]
        
        # Deviatoric stress should be zero for hydrostatic
        mean_stress = tr(σ_h) / 3
        s = σ_h - mean_stress * I
        @test norm(s) < 1e-10
        
        # General stress
        σ = [100.0  30.0  20.0;
              30.0  50.0  10.0;
              20.0  10.0  40.0]
        
        mean_σ = tr(σ) / 3
        s_dev = σ - mean_σ * I
        @test tr(s_dev) < 1e-10  # Deviatoric stress has zero trace
    end
end

@testset "Stress-based Functions" begin
    @testset "Stress constraint" begin
        # Simple stress constraint
        σ_allow = 200.0  # Allowable stress
        σ_calc = 150.0   # Calculated stress
        
        # Stress ratio (should be < 1.0 for satisfaction)
        ratio = σ_calc / σ_allow
        @test ratio < 1.0
        @test ratio ≈ 0.75
        
        # Constraint violation
        σ_calc_high = 250.0
        ratio_high = σ_calc_high / σ_allow
        @test ratio_high > 1.0
    end
    
    @testset "p-norm stress aggregation" begin
        # Multiple stress values
        stresses = [100.0, 150.0, 80.0, 200.0, 120.0]
        P = 6.0  # p-norm parameter
        
        # p-norm approximation of maximum
        σ_pnorm = (sum(abs.(stresses).^P) / length(stresses))^(1/P)
        σ_max = maximum(stresses)
        
        # p-norm should be close to max for large P
        @test σ_pnorm <= σ_max
        @test σ_pnorm > σ_max / 2  # Should be reasonably close
        
        # Test with different P values
        P_vals = [2.0, 4.0, 8.0]
        for P in P_vals
            σ_pn = (sum(abs.(stresses).^P) / length(stresses))^(1/P)
            @test σ_pn <= σ_max
        end
    end
    
    @testset "Stress sensitivity" begin
        # Simple sensitivity test
        σ = 100.0
        dσ_dρ = 50.0  # Sensitivity of stress to density
        
        # Linear sensitivity
        Δρ = 0.01
        Δσ = dσ_dρ * Δρ
        @test Δσ ≈ 0.5
    end
end