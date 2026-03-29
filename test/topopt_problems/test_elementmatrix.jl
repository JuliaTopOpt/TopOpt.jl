using Test
using TopOpt.TopOptProblems: ElementMatrix, rawmatrix, bcmatrix
using StaticArrays
using LinearAlgebra

@testset "ElementMatrix Base.convert" begin
    @testset "StaticMatrix convert - no boundary conditions" begin
        # Test 1: StaticMatrix 2x2 without bc_dofs
        T = Float64
        N = 2
        
        # Create sample element stiffness matrices
        K1 = @SMatrix [4.0 -1.0; -1.0 4.0]
        K2 = @SMatrix [5.0 -2.0; -2.0 5.0]
        Kes = [K1, K2]
        
        # No boundary conditions
        bc_dofs = Int[]
        dof_cells = Dict{Int, Vector{Tuple{Int, Int}}}()
        
        element_Kes = Base.convert(Vector{<:ElementMatrix}, Kes; bc_dofs=bc_dofs, dof_cells=dof_cells)
        
        @test length(element_Kes) == 2
        @test element_Kes[1] isa ElementMatrix{T, <:SMatrix{N,N,T}}
        
        # All masks should be true
        @test all(element_Kes[1].mask .== true)
        @test all(element_Kes[2].mask .== true)
        
        # Verify matrix values are set
        @test element_Kes[1].matrix[1, 1] ≈ 4.0
        @test element_Kes[1].matrix[2, 2] ≈ 4.0
        @test element_Kes[2].matrix[1, 1] ≈ 5.0
        @test element_Kes[2].matrix[2, 2] ≈ 5.0
        
        # meandiag is computed as sum of diagonal (4+4=8, 5+5=10)
        @test element_Kes[1].meandiag ≈ 8.0
        @test element_Kes[2].meandiag ≈ 10.0
    end

    @testset "StaticMatrix 3x3 - no boundary conditions" begin
        T = Float64
        N = 3
        
        # Create 3x3 StaticMatrices
        K1 = @SMatrix [4.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 4.0]
        K2 = @SMatrix [5.0 -2.0 0.0; -2.0 5.0 -2.0; 0.0 -2.0 5.0]
        Kes = [K1, K2]
        
        # No boundary conditions
        bc_dofs = Int[]
        dof_cells = Dict{Int, Vector{Tuple{Int, Int}}}()
        
        element_Kes = Base.convert(Vector{<:ElementMatrix}, Kes; bc_dofs=bc_dofs, dof_cells=dof_cells)
        
        @test length(element_Kes) == 2
        
        # All masks should be true
        @test all(element_Kes[1].mask .== true)
        @test all(element_Kes[2].mask .== true)
        
        # Verify matrices are copied
        @test element_Kes[1].matrix[1, 1] ≈ 4.0
        @test element_Kes[2].matrix[2, 2] ≈ 5.0
    end

    @testset "StaticMatrix Symmetric wrapper - no boundary conditions" begin
        T = Float64
        N = 2
        
        # Create Symmetric wrapped StaticMatrices
        K1 = Symmetric(@SMatrix [4.0 -1.0; -1.0 4.0])
        K2 = Symmetric(@SMatrix [5.0 -2.0; -2.0 5.0])
        Kes = [K1, K2]
        
        bc_dofs = Int[]
        dof_cells = Dict{Int, Vector{Tuple{Int, Int}}}()
        
        element_Kes = Base.convert(Vector{<:ElementMatrix}, Kes; bc_dofs=bc_dofs, dof_cells=dof_cells)
        
        @test length(element_Kes) == 2
        @test element_Kes[1] isa Symmetric{T, <:ElementMatrix{T}}
        
        # All masks should be true
        @test all(element_Kes[1].data.mask .== true)
        @test all(element_Kes[2].data.mask .== true)
        
        # Verify matrix values
        @test element_Kes[1].data.matrix[1, 1] ≈ 4.0
        @test element_Kes[2].data.matrix[2, 2] ≈ 5.0
        
        # meandiag is sum of diagonal
        @test element_Kes[1].data.meandiag ≈ 8.0
        @test element_Kes[2].data.meandiag ≈ 10.0
    end

    @testset "Empty bc_dofs (no boundary conditions)" begin
        N = 2
        K1 = @SMatrix [4.0 -1.0; -1.0 4.0]
        K2 = @SMatrix [5.0 -2.0; -2.0 5.0]
        Kes = [K1, K2]
        
        bc_dofs = Int[]
        dof_cells = Dict{Int, Vector{Tuple{Int, Int}}}()
        
        element_Kes = Base.convert(Vector{<:ElementMatrix}, Kes; bc_dofs=bc_dofs, dof_cells=dof_cells)
        
        @test length(element_Kes) == 2
        
        # All masks should be true
        @test all(element_Kes[1].mask .== true)
        @test all(element_Kes[2].mask .== true)
        
        # Verify matrices are set correctly
        @test element_Kes[1].matrix[1, 1] ≈ 4.0
        @test element_Kes[2].matrix[2, 2] ≈ 5.0
    end

    @testset "Single element matrix - no BCs" begin
        N = 2
        K1 = @SMatrix [4.0 -1.0; -1.0 4.0]
        Kes = [K1]
        
        bc_dofs = Int[]
        dof_cells = Dict{Int, Vector{Tuple{Int, Int}}}()
        
        element_Kes = Base.convert(Vector{<:ElementMatrix}, Kes; bc_dofs=bc_dofs, dof_cells=dof_cells)
        
        @test length(element_Kes) == 1
        @test all(element_Kes[1].mask .== true)
    end

    @testset "Different numeric types" begin
        # Test with Float32
        T = Float32
        N = 2
        
        K1 = @SMatrix T[4.0f0 -1.0f0; -1.0f0 4.0f0]
        K2 = @SMatrix T[5.0f0 -2.0f0; -2.0f0 5.0f0]
        Kes = [K1, K2]
        
        bc_dofs = Int[]
        dof_cells = Dict{Int, Vector{Tuple{Int, Int}}}()
        
        element_Kes = Base.convert(Vector{<:ElementMatrix}, Kes; bc_dofs=bc_dofs, dof_cells=dof_cells)
        
        @test eltype(element_Kes[1].matrix) == Float32
        @test element_Kes[1].meandiag isa Float32
    end

    @testset "RawMatrix and BCMatrix functions" begin
        N = 2
        K1 = @SMatrix [4.0 -1.0; -1.0 4.0]
        Kes = [K1]
        
        bc_dofs = Int[]
        dof_cells = Dict{Int, Vector{Tuple{Int, Int}}}()
        
        element_Kes = Base.convert(Vector{<:ElementMatrix}, Kes; bc_dofs=bc_dofs, dof_cells=dof_cells)
        em = element_Kes[1]
        
        # Test rawmatrix
        @test rawmatrix(em) ≈ K1
        
        # Test bcmatrix (no BCs, should be same as original)
        bcm = bcmatrix(em)
        @test bcm ≈ K1
    end

    @testset "Symmetric ElementMatrix rawmatrix and bcmatrix" begin
        N = 2
        K1 = Symmetric(@SMatrix [4.0 -1.0; -1.0 4.0])
        Kes = [K1]
        
        bc_dofs = Int[]
        dof_cells = Dict{Int, Vector{Tuple{Int, Int}}}()
        
        element_Kes = Base.convert(Vector{<:ElementMatrix}, Kes; bc_dofs=bc_dofs, dof_cells=dof_cells)
        em = element_Kes[1]
        
        # Test rawmatrix on Symmetric wrapper
        rm = rawmatrix(em)
        @test rm isa Symmetric
        @test rm.data ≈ K1.data
    end

    @testset "ElementMatrix struct basic functionality" begin
        # Test creating ElementMatrix directly
        mat = @SMatrix [4.0 -1.0; -1.0 4.0]
        mask = SVector{2, Bool}(true, false)
        meandiag_val = 8.0
        
        em = ElementMatrix(mat, mask, meandiag_val)
        
        @test em.matrix ≈ mat
        @test em.mask == mask
        @test em.meandiag ≈ meandiag_val
        
        # Test rawmatrix
        @test rawmatrix(em) ≈ mat
        
        # Test bcmatrix - should zero out row/col 2 (where mask is false)
        bcm = bcmatrix(em)
        @test bcm[1, 1] ≈ 4.0
        @test bcm[1, 2] ≈ 0.0
        @test bcm[2, 1] ≈ 0.0
        @test bcm[2, 2] ≈ 0.0
    end

    @testset "Symmetric ElementMatrix wrapper basic functionality" begin
        mat = @SMatrix [4.0 -1.0; -1.0 4.0]
        mask = SVector{2, Bool}(true, false)
        meandiag_val = 8.0
        
        em = Symmetric(ElementMatrix(mat, mask, meandiag_val))
        
        # Test rawmatrix - returns Symmetric{SMatrix}, not Symmetric{ElementMatrix}
        rm = rawmatrix(em)
        @test rm isa Symmetric
        @test rm.data ≈ mat
        
        # Test bcmatrix
        bcm = bcmatrix(em)
        @test bcm[1, 1] ≈ 4.0
        @test bcm[1, 2] ≈ 0.0
        @test bcm[2, 1] ≈ 0.0
        @test bcm[2, 2] ≈ 0.0
    end
end