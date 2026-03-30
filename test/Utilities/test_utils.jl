using TopOpt, Test, LinearAlgebra, StaticArrays, Ferrite
using TopOpt: RaggedArray, compliance, meandiag, density, sumdiag
using TopOpt: @params, @forward_property
using Ferrite: DofHandler, Grid, getncells

@testset "RaggedArray" begin
    # Test construction from Vector{Vector}
    vv = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]]
    ra = RaggedArray(vv)
    @test ra.offsets == [1, 3, 6, 7]
    @test ra.values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Test getindex for single index
    @test ra[1] == [1.0, 2.0]
    @test ra[2] == [3.0, 4.0, 5.0]
    @test ra[3] == [6.0]

    # Test getindex for double indices
    @test ra[1, 1] == 1.0
    @test ra[2, 1] == 2.0
    @test ra[1, 2] == 3.0
    @test ra[2, 2] == 4.0
    @test ra[3, 2] == 5.0
    @test ra[1, 3] == 6.0

    # Test setindex!
    ra[2, 2] = 10.0
    @test ra[2, 2] == 10.0

    # Test edge case: single element vectors
    vv_single = [[1.0], [2.0], [3.0]]
    ra_single = RaggedArray(vv_single)
    @test length(ra_single.values) == 3
    @test ra_single[1, 2] == 2.0

    # Test edge case: mixed sizes
    vv_mixed = [[1.0, 2.0, 3.0], Float64[], [4.0]]
    ra_mixed = RaggedArray(vv_mixed)
    @test ra_mixed[1] == [1.0, 2.0, 3.0]
    @test ra_mixed[2] == Float64[]
    @test ra_mixed[3] == [4.0]
end

@testset "@params macro" begin
    # Test basic struct generation
    # The @params macro creates type parameters for each field
    @params struct TestStruct1{T}
        f1::T
        f2::AbstractVector{T}
    end
    
    # Create instance using default constructor
    # The macro generates: struct TestStruct1{T, T1<:T, T2<:AbstractVector{T}}
    ts = TestStruct1(1.0, [2.0, 3.0])
    @test ts.f1 == 1.0
    @test ts.f2 == [2.0, 3.0]

    # Test with type bounds - note: need to respect the type hierarchy
    @params struct TestStruct2{T}
        f1::T
        f2::AbstractVector{<:Real}
        f3
    end
    
    ts2 = TestStruct2(1, [2.0, 3.0], "test")
    @test ts2.f1 == 1
    @test ts2.f2 == [2.0, 3.0]
    @test ts2.f3 == "test"

    # Test with inheritance
    abstract type AbstractTest end
    @params struct TestStruct3{T} <: AbstractTest
        f1::T
    end
    
    ts3 = TestStruct3(1.0)
    @test ts3 isa AbstractTest
end

@testset "compliance" begin
    # Simple test case
    Ke = [2.0 1.0; 1.0 2.0]
    u = [1.0, 2.0]
    dofs = [1, 2]
    comp = compliance(Ke, u, dofs)
    expected = u[1]*Ke[1,1]*u[1] + u[1]*Ke[1,2]*u[2] + u[2]*Ke[2,1]*u[1] + u[2]*Ke[2,2]*u[2]
    @test comp ≈ expected

    # Test with different dof configuration
    u_big = [0.0, 1.0, 2.0, 0.0]
    dofs_offset = [2, 3]
    comp2 = compliance(Ke, u_big, dofs_offset)
    @test comp2 ≈ expected

    # Test type stability
    Ke_f32 = Float32[2.0 1.0; 1.0 2.0]
    u_f32 = Float32[1.0, 2.0]
    comp_f32 = compliance(Ke_f32, u_f32, dofs)
    @test typeof(comp_f32) == Float32
end

@testset "meandiag" begin
    # Test with simple matrix
    K = [1.0 2.0; 3.0 4.0]
    @test meandiag(K) ≈ (abs(1.0) + abs(4.0)) / 2

    # Test with zero diagonal
    K_zero = [0.0 1.0; 1.0 0.0]
    @test meandiag(K_zero) ≈ 0.0

    # Test with larger matrix
    K_large = Matrix{Float64}(I, 5, 5) * 3.0
    @test meandiag(K_large) ≈ 3.0

    # Test with negative diagonal
    K_neg = [-1.0 0.0; 0.0 -2.0]
    @test meandiag(K_neg) ≈ (1.0 + 2.0) / 2
end

@testset "density" begin
    # Test basic calculation
    @test density(0.5, 0.1) ≈ 0.5 * (1 - 0.1) + 0.1
    @test density(0.5, 0.1) ≈ 0.55

    # Test boundary conditions
    @test density(0.0, 0.1) ≈ 0.1  # Minimum density
    @test density(1.0, 0.1) ≈ 1.0   # Maximum density

    # Test with different xmin
    @test density(0.5, 0.01) ≈ 0.5 * 0.99 + 0.01
    @test density(0.5, 0.001) ≈ 0.5 * 0.999 + 0.001
end

@testset "@forward_property macro" begin
    # Create a nested structure to test property forwarding
    using TopOpt: @forward_property
    
    # Use mutable struct for Inner to allow setproperty!
    mutable struct InnerFP
        x::Float64
        y::Float64
    end
    
    struct OuterFP
        inner::InnerFP
    end
    @forward_property OuterFP inner
    
    inner_obj = InnerFP(1.0, 2.0)
    outer_obj = OuterFP(inner_obj)
    
    # Test getproperty forwarding
    @test outer_obj.x == 1.0
    @test outer_obj.y == 2.0
    
    # Test setproperty! forwarding
    outer_obj.x = 3.0
    @test outer_obj.x == 3.0
    @test inner_obj.x == 3.0
end

@testset "sumdiag" begin
    # Test with StaticMatrix
    K_static = @SMatrix [1.0 2.0; 3.0 4.0]
    @test sumdiag(K_static) ≈ 1.0 + 4.0
    
    # Test with Symmetric StaticMatrix
    K_sym = Symmetric(@SMatrix [1.0 2.0; 2.0 4.0])
    @test sumdiag(K_sym) ≈ 1.0 + 4.0
    
    # Test with 3x3 matrix
    K_3x3 = @SMatrix [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0]
    @test sumdiag(K_3x3) ≈ 6.0
    
    # Compare with manual sum
    K_test = @SMatrix [5.0 1.0; 2.0 3.0]
    @test sumdiag(K_test) == sum(diag(K_test))
end

@testset "visualize fallback function" begin
    # Test that visualize throws informative error when Makie is not loaded
    err = try
        visualize(42)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("visualize", err.msg)
    @test occursin("Int64", err.msg)
    @test occursin("Makie", err.msg)

    # Test with different types
    err_str = try
        visualize("hello")
        nothing
    catch e
        e
    end
    @test err_str isa ErrorException
    @test occursin("String", err_str.msg)

    # Test with keyword arguments
    err_kw = try
        visualize(42; color=:red)
        nothing
    catch e
        e
    end
    @test err_kw isa ErrorException
    @test occursin("Int64", err_kw.msg)

    # Test that visualize is exported
    @test isdefined(@__MODULE__, :visualize)
end
