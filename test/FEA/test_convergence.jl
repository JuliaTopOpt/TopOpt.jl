using TopOpt, Test, LinearAlgebra, Ferrite
using TopOpt: DefaultCriteria, EnergyCriteria
using IterativeSolvers: ConvergenceHistory, isconverged

@testset "DefaultCriteria Construction" begin
    # Test basic construction
    criteria = DefaultCriteria()
    @test criteria isa DefaultCriteria
end

@testset "EnergyCriteria Construction" begin
    # Test basic construction
    criteria = EnergyCriteria()
    @test criteria isa EnergyCriteria{Float64}
    @test criteria.energy == 0.0
    
    # Test construction with custom type
    criteria_f32 = EnergyCriteria{Float32}(0.0f0)
    @test criteria_f32 isa EnergyCriteria{Float32}
    
    # Test construction with custom value
    criteria_val = EnergyCriteria(100.0)
    @test criteria_val.energy == 100.0
end

@testset "EnergyCriteria Mutation" begin
    criteria = EnergyCriteria()
    
    # Test that energy field can be mutated
    criteria.energy = 50.0
    @test criteria.energy == 50.0
    
    criteria.energy = -10.0
    @test criteria.energy == -10.0
end
