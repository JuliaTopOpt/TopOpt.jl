using Test, TopOpt, LinearAlgebra
using TopOpt.TrussTopOptProblems
using TopOpt.TrussTopOptProblems: load_truss_json
using TopOpt.FEA: simulate

@testset "simulate function - truss problems" begin
    # Use lowercase directory path since that's how the file system has it
    ins_dir = joinpath(@__DIR__, "..", "truss_topopt_problems", "instances", "fea_examples")
    
    # Test 1: Basic truss simulation
    @testset "Basic truss simulation" begin
        # Load a simple truss problem from JSON
        file_path = joinpath(ins_dir, "mgz_truss1.json")
        node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(file_path)
        
        # Create a truss problem (note: loads come before fixities in constructor)
        loads = load_cases["0"]
        problem = TrussTopOptProblems.TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs)
        
        # Full material topology
        topology = ones(TopOpt.TrussTopOptProblems.getncells(problem))
        
        result = simulate(problem, topology)
        
        @test hasfield(typeof(result), :u)
        @test hasfield(typeof(result), :comp)
        @test result.comp > 0
        @test length(result.u) > 0
    end

    # Test 2: Different truss topologies
    @testset "Different truss topologies" begin
        file_path = joinpath(ins_dir, "mgz_truss2.json")
        node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(file_path)
        loads = load_cases["0"]
        problem = TrussTopOptProblems.TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs)
        
        full_topology = ones(TopOpt.TrussTopOptProblems.getncells(problem))
        partial_topology = fill(0.5, TopOpt.TrussTopOptProblems.getncells(problem))
        
        result_full = simulate(problem, full_topology)
        result_partial = simulate(problem, partial_topology)
        
        @test result_full.comp > 0
        @test result_partial.comp > 0
    end

    # Test 3: Truss with void elements
    @testset "Truss with void elements" begin
        file_path = joinpath(ins_dir, "mgz_truss3.json")
        node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(file_path)
        loads = load_cases["0"]
        problem = TrussTopOptProblems.TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs)
        
        topology = ones(TopOpt.TrussTopOptProblems.getncells(problem))
        # Remove some elements
        topology[1:2] .= 0.0
        
        result = simulate(problem, topology)
        
        @test result.comp > 0
    end
end
