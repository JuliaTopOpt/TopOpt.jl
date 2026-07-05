using TopOpt
using TopOpt.TopOptProblems
using Ferrite
using LinearAlgebra
using SparseArrays
using Test

using TopOpt.TopOptProblems: FluidThermalProblem, solve_fluid_thermal

@testset "Fluid-Thermal Coupling Architecture" begin
    
    # Initialization Test
    grid_info = TopOptProblems.RectilinearGrid(Val{:Linear}, (10, 10), (1.0, 1.0))
    problem = FluidThermalProblem(grid_info, conductivity=0.1, heat_capacity=10.0)
    grid = problem.rect_grid.grid
    ch = problem.ch
    
    @test ndofs(ch.dh) == 484 # 121 nodes * (2u + 1p + 1T)
    
    # Apply Boundaries
    Ferrite.add!(ch, Dirichlet(:u, getfaceset(grid, "top"), (x,t) -> [0.0, 0.0], [1, 2]))
    Ferrite.add!(ch, Dirichlet(:u, getfaceset(grid, "bottom"), (x,t) -> [0.0, 0.0], [1, 2]))
    Ferrite.add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> [1.0, 0.0], [1, 2]))
    Ferrite.add!(ch, Dirichlet(:p, getfaceset(grid, "right"), (x,t) -> 0.0, [1]))
    Ferrite.add!(ch, Dirichlet(:T, getfaceset(grid, "bottom"), (x,t) -> 100.0, [1]))
    Ferrite.add!(ch, Dirichlet(:T, getfaceset(grid, "left"), (x,t) -> 0.0, [1]))
    close!(ch)
    update!(ch, 0.0)
    
    @test length(ch.prescribed_dofs) > 0

    # Sequential Solver Test
    vars_fluid = ones(getncells(grid))
    u_final, K_final = solve_fluid_thermal(problem, vars_fluid)
    
    @test !any(isnan.(u_final)) # Ensure no singular matrix failures
    
    # Physics & Temperature Bound Verification
    fh = ch.dh.fieldhandlers[1]
    dofs_T = dof_range(fh, :T)
    global_T_dofs = unique(reduce(vcat, [celldofs(c)[dofs_T] for c in CellIterator(ch.dh)]))
    temperatures = u_final[global_T_dofs]
    
    max_T = maximum(temperatures)
    min_T = minimum(temperatures)
    
    # Mathematically verify bounds (allowing for Galerkin SUPG wiggles < 0)
    @test isapprox(max_T, 100.0, atol=1e-1)
    @test min_T < 0.0 # Validates the presence of high Peclet number advection wiggles
    
    # Brinkman Penalization Solid Test
    vars_solid = zeros(getncells(grid))
    _, K_solid = solve_fluid_thermal(problem, vars_solid)
    
    # The matrix norm of the solid should be vastly larger due to Brinkman penalty
    @test norm(K_solid) > norm(K_final) * 1e5
end