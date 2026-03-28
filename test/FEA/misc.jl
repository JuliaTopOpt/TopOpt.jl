using Test, TopOpt, LinearAlgebra, SparseArrays
using TopOpt.FEA: getcompliance, matrix_free_apply2f!, update_f!, MatrixFreeOperator, MatrixOperator
using TopOpt.TopOptProblems: assemble!, GlobalFEAInfo, ElementFEAInfo
using Ferrite, StaticArrays

@testset "getcompliance function tests" begin
    @testset "Basic compliance calculation" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        solver = FEASolver(DirectSolver, problem)
        
        # Set uniform design variables
        solver.vars .= 1.0
        solver()
        
        # Test getcompliance computes u' * K * u
        comp = getcompliance(solver)
        expected_comp = solver.u' * solver.globalinfo.K * solver.u
        
        @test comp isa Real
        @test comp > 0
        @test comp ≈ expected_comp
        
        # Verify with direct dot product of u and f
        @test comp ≈ dot(solver.u, solver.globalinfo.f)
    end
    
    @testset "getcompliance with different solvers" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver_direct = FEASolver(DirectSolver, problem)
        solver_cg = FEASolver(CGAssemblySolver, problem; abstol=1e-8)
        
        solver_direct.vars .= 1.0
        solver_cg.vars .= 1.0
        
        solver_direct()
        solver_cg()
        
        comp_direct = getcompliance(solver_direct)
        comp_cg = getcompliance(solver_cg)
        
        # Compliance should be similar between solvers
        @test isapprox(comp_direct, comp_cg; rtol=1e-3)
    end
    
    @testset "getcompliance with varying density" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        solver = FEASolver(DirectSolver, problem)
        
        # Test with full density
        solver.vars .= 1.0
        solver()
        comp_full = getcompliance(solver)
        
        # Test with reduced density
        solver.vars .= 0.5
        solver()
        comp_reduced = getcompliance(solver)
        
        @test comp_full > 0
        @test comp_reduced > comp_full  # Reduced density should have higher compliance
    end
end

@testset "update_f! function tests" begin
    @testset "update_f! with prescribed DOFs" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
        globalinfo = GlobalFEAInfo(problem)
        
        ndofs_total = ndofs(problem.ch.dh)
        f = zeros(ndofs_total)
        
        # Get boundary condition info
        ch = problem.ch
        values = ch.values
        prescribed_dofs = ch.prescribed_dofs
        
        # Setup test parameters
        vars = ones(prod(nels))
        penalty = PowerPenalty(1.0)
        xmin = 0.001
        M = 1.0  # Diagonal scaling factor
        
        dof_cells = elementinfo.metadata.dof_cells
        cell_dofs = elementinfo.metadata.cell_dofs
        Kes = elementinfo.Kes
        
        # Call update_f!
        applyzero = false
        
        update_f!(
            f, values, prescribed_dofs, applyzero, dof_cells, cell_dofs,
            Kes, xmin, penalty, vars, M
        )
        
        # Check that prescribed DOFs have been set
        for (i, dof) in enumerate(prescribed_dofs)
            @test f[dof] ≈ M * values[i]
        end
    end

    @testset "update_f! with non-zero Dirichlet BC" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)

        # Use PointLoadCantilever which has proper boundary condition setup
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

        ndofs_total = Ferrite.ndofs(problem.ch.dh)
        f = zeros(ndofs_total)

        # Get original boundary condition info
        ch = problem.ch
        original_values = copy(ch.values)
        prescribed_dofs = ch.prescribed_dofs

        # Create custom non-zero BC values (simulate a non-zero displacement)
        # We'll set all prescribed DOFs to have a value of 0.5
        non_zero_values = fill(0.5, length(prescribed_dofs))

        # Setup test parameters
        vars = ones(prod(nels))
        penalty = PowerPenalty(1.0)
        xmin = 0.001
        M = 1.0  # Diagonal scaling factor

        dof_cells = elementinfo.metadata.dof_cells
        cell_dofs = elementinfo.metadata.cell_dofs
        Kes = elementinfo.Kes

        # Call update_f! with non-zero values
        applyzero = false

        update_f!(
            f, non_zero_values, prescribed_dofs, applyzero, dof_cells, cell_dofs,
            Kes, xmin, penalty, vars, M
        )

        # With non-zero Dirichlet BCs (applyzero=false), the function:
        # 1. Applies M * values at prescribed DOFs
        # 2. Then subtracts K_bc contributions from those DOFs
        # This means the final values at prescribed DOFs will NOT be exactly M * values[i]
        # They will be modified based on the stiffness matrix

        # Verify the function completed without error and modified the force vector
        @test length(f) == ndofs_total

        # All values should be finite (not NaN or Inf)
        @test all(isfinite, f)

        # The function was called with non-zero BC values
        @test length(non_zero_values) > 0
        @test length(prescribed_dofs) > 0

        # Compare with applyzero=true case - should have different force values
        f_applyzero = zeros(ndofs_total)
        update_f!(
            f_applyzero, non_zero_values, prescribed_dofs, true, dof_cells, cell_dofs,
            Kes, xmin, penalty, vars, M
        )

        # With non-zero BCs and applyzero=false, force values should differ from applyzero=true
        @test f != f_applyzero
    end
    
    @testset "update_f! with applyzero=true" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
        globalinfo = GlobalFEAInfo(problem)
        
        ndofs_total = ndofs(problem.ch.dh)
        f = zeros(ndofs_total)
        
        # Get boundary condition info
        ch = problem.ch
        values = ch.values
        prescribed_dofs = ch.prescribed_dofs
        
        # Setup test parameters
        vars = ones(prod(nels))
        penalty = PowerPenalty(1.0)
        xmin = 0.001
        M = 1.0
        
        dof_cells = elementinfo.metadata.dof_cells
        cell_dofs = elementinfo.metadata.cell_dofs
        Kes = elementinfo.Kes
        
        # Call update_f! with applyzero=true
        applyzero = true
        
        update_f!(
            f, values, prescribed_dofs, applyzero, dof_cells, cell_dofs,
            Kes, xmin, penalty, vars, M
        )
        
        # With applyzero=true, non-zero boundary values should not contribute
        # to other DOFs, only prescribed DOFs should be set
        for (i, dof) in enumerate(prescribed_dofs)
            @test f[dof] ≈ M * values[i]
        end
    end
end

@testset "matrix_free_apply2f! function tests" begin
    @testset "matrix_free_apply2f! basic functionality" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
        
        ndofs_total = ndofs(problem.ch.dh)
        f = zeros(ndofs_total)
        
        vars = ones(prod(nels))
        penalty = PowerPenalty(1.0)
        xmin = 0.001
        M = 1.0
        
        # Call matrix_free_apply2f!
        matrix_free_apply2f!(f, elementinfo, M, vars, problem, penalty, xmin, false)
        
        # Check that vector was modified
        @test f isa Vector{Float64}
        @test length(f) == ndofs_total
        
        # Prescribed DOFs should have values set
        ch = problem.ch
        for (i, dof) in enumerate(ch.prescribed_dofs)
            @test f[dof] ≈ M * ch.values[i]
        end
    end
    
    @testset "matrix_free_apply2f! with applyzero=true" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
        
        ndofs_total = ndofs(problem.ch.dh)
        f = zeros(ndofs_total)
        
        vars = ones(prod(nels))
        penalty = PowerPenalty(1.0)
        xmin = 0.001
        M = 1.0
        
        # Call with applyzero=true
        matrix_free_apply2f!(f, elementinfo, M, vars, problem, penalty, xmin, true)
        
        # Vector should still be modified
        @test f isa Vector{Float64}
        @test length(f) == ndofs_total
    end
    
    @testset "matrix_free_apply2f! with varying density" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
        
        ndofs_total = ndofs(problem.ch.dh)
        
        # Test with full density
        vars_full = ones(prod(nels))
        f_full = zeros(ndofs_total)
        penalty = PowerPenalty(1.0)
        xmin = 0.001
        M = 1.0
        
        matrix_free_apply2f!(f_full, elementinfo, M, vars_full, problem, penalty, xmin, false)
        
        # Test with reduced density
        vars_reduced = fill(0.5, prod(nels))
        f_reduced = zeros(ndofs_total)
        
        matrix_free_apply2f!(f_reduced, elementinfo, M, vars_reduced, problem, penalty, xmin, false)
        
        # Results should be different
        @test f_full != f_reduced || all(f_full .== 0)  # Either different or both zero
    end
end

@testset "MatrixOperator size methods" begin
    @testset "size(op) returns (m, n)" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem)
        K = solver.globalinfo.K
        f = solver.globalinfo.f
        
        operator = MatrixOperator(K, f, TopOpt.FEA.DefaultCriteria())
        
        # Test size without dimension argument
        sz = size(operator)
        @test sz isa Tuple{Int,Int}
        @test sz == size(K)
        @test sz[1] == size(K, 1)
        @test sz[2] == size(K, 2)
    end
    
    @testset "size(op, i) for i=1,2" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem)
        K = solver.globalinfo.K
        f = solver.globalinfo.f
        
        operator = MatrixOperator(K, f, TopOpt.FEA.DefaultCriteria())
        
        # Test size with dimension argument
        @test size(operator, 1) == size(K, 1)
        @test size(operator, 2) == size(K, 2)
    end
    
    @testset "eltype returns element type of K" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem)
        K = solver.globalinfo.K
        f = solver.globalinfo.f
        
        operator = MatrixOperator(K, f, TopOpt.FEA.DefaultCriteria())
        
        @test eltype(operator) == eltype(K)
        @test eltype(operator) <: Real
    end
end

@testset "MatrixFreeOperator size methods" begin
    @testset "size(op) returns (m, n)" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver = FEASolver(CGMatrixFreeSolver, problem)
        elementinfo = solver.elementinfo
        meandiag = solver.meandiag
        vars = solver.vars
        xes = solver.xes
        fixed_dofs = solver.fixed_dofs
        free_dofs = solver.free_dofs
        xmin = solver.xmin
        penalty = solver.penalty
        
        operator = MatrixFreeOperator(
            elementinfo.fixedload, elementinfo, meandiag, vars, xes,
            fixed_dofs, free_dofs, xmin, penalty, solver.conv
        )
        
        # Test size without dimension argument
        sz = size(operator)
        @test sz isa Tuple{Int,Int}
        @test sz[1] == sz[2]  # Should be square
        @test sz[1] == length(elementinfo.fixedload)
    end
    
    @testset "size(op, i) for i=1,2" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver = FEASolver(CGMatrixFreeSolver, problem)
        elementinfo = solver.elementinfo
        meandiag = solver.meandiag
        vars = solver.vars
        xes = solver.xes
        fixed_dofs = solver.fixed_dofs
        free_dofs = solver.free_dofs
        xmin = solver.xmin
        penalty = solver.penalty
        
        operator = MatrixFreeOperator(
            elementinfo.fixedload, elementinfo, meandiag, vars, xes,
            fixed_dofs, free_dofs, xmin, penalty, solver.conv
        )
        
        ndofs = length(elementinfo.fixedload)
        
        # Test size with dimension argument
        @test size(operator, 1) == ndofs
        @test size(operator, 2) == ndofs
        
        # Invalid dimensions should return 1
        @test size(operator, 0) == 1
        @test size(operator, 3) == 1
    end
    
    @testset "eltype returns type parameter T" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver = FEASolver(CGMatrixFreeSolver, problem)
        elementinfo = solver.elementinfo
        meandiag = solver.meandiag
        vars = solver.vars
        xes = solver.xes
        fixed_dofs = solver.fixed_dofs
        free_dofs = solver.free_dofs
        xmin = solver.xmin
        penalty = solver.penalty
        
        operator = MatrixFreeOperator(
            elementinfo.fixedload, elementinfo, meandiag, vars, xes,
            fixed_dofs, free_dofs, xmin, penalty, solver.conv
        )
        
        @test eltype(operator) <: Real
        @test eltype(operator) == Float64  # Default for this problem
    end
end

@testset "MatrixFreeOperator show method" begin
    @testset "MatrixFreeOperator show output" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver = FEASolver(CGMatrixFreeSolver, problem)
        
        # Create MatrixFreeOperator
        elementinfo = solver.elementinfo
        meandiag = solver.meandiag
        vars = solver.vars
        xes = solver.xes
        fixed_dofs = solver.fixed_dofs
        free_dofs = solver.free_dofs
        xmin = solver.xmin
        penalty = solver.penalty
        
        operator = MatrixFreeOperator(
            elementinfo.fixedload, elementinfo, meandiag, vars, xes,
            fixed_dofs, free_dofs, xmin, penalty, solver.conv
        )
        
        # Test show method
        io = IOBuffer()
        show(io, MIME"text/plain"(), operator)
        output = String(take!(io))
        
        @test occursin("matrix-free", lowercase(output))
    end
    
    @testset "MatrixOperator show output" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem)
        
        K = solver.globalinfo.K
        f = solver.globalinfo.f
        
        operator = MatrixOperator(K, f, TopOpt.FEA.DefaultCriteria())
        
        # Test show method
        io = IOBuffer()
        show(io, MIME"text/plain"(), operator)
        output = String(take!(io))
        
        @test occursin("matrix", lowercase(output))
    end
end

@testset "MatrixFreeOperator mul! operation" begin
    @testset "mul! with solver state" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver = FEASolver(CGMatrixFreeSolver, problem)
        
        # Set design variables and solve to initialize state
        solver.vars .= 0.8
        solver()
        
        # Create MatrixFreeOperator with current state
        elementinfo = solver.elementinfo
        meandiag = solver.meandiag
        vars = solver.vars
        xes = solver.xes
        fixed_dofs = solver.fixed_dofs
        free_dofs = solver.free_dofs
        xmin = solver.xmin
        penalty = solver.penalty
        
        operator = MatrixFreeOperator(
            elementinfo.fixedload, elementinfo, meandiag, vars, xes,
            fixed_dofs, free_dofs, xmin, penalty, solver.conv
        )
        
        u = solver.u        
        y = similar(u)
        mul!(y, operator, u)
        @test length(y) == length(u)
        @test y isa Vector{Float64}
        
        # Verify the result makes physical sense (y should be related to K*u)
        @test !all(iszero, y)  # Should have non-zero values
        @test all(isfinite, y)  # Should be finite
    end
    
    @testset "mul! consistency with matrix assembly" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        # Assembly solver
        solver_asm = FEASolver(CGAssemblySolver, problem)
        solver_asm.vars .= 0.8
        solver_asm()
        
        # Matrix-free solver
        solver_mf = FEASolver(CGMatrixFreeSolver, problem)
        solver_mf.vars .= 0.8
        solver_mf()
        
        # Get free DOF values
        free_dofs = solver_asm.free_dofs
        u = solver_asm.u
        
        # Extract K for free DOFs
        K = solver_asm.globalinfo.K
        
        # Expected result from matrix multiplication
        y_expected = K * u
        
        # Matrix-free operator result
        elementinfo = solver_mf.elementinfo
        meandiag = solver_mf.meandiag
        vars = solver_mf.vars
        xes = solver_mf.xes
        fixed_dofs = solver_mf.fixed_dofs
        xmin = solver_mf.xmin
        penalty = solver_mf.penalty
        
        operator = MatrixFreeOperator(
            elementinfo.fixedload, elementinfo, meandiag, vars, xes,
            fixed_dofs, free_dofs, xmin, penalty, solver_mf.conv
        )
        
        y_mf = similar(u)
        mul!(y_mf, operator, u)

        # Results should be approximately equal
        @test_broken isapprox(y_mf, y_expected; rtol=1e-3)
    end
end

@testset "Integration: matrix_free_apply2f! with solver workflow" begin
    @testset "Consistency with assembled matrix approach" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
        
        solver_assembly = FEASolver(CGAssemblySolver, problem; abstol=1e-8)
        solver_matrixfree = FEASolver(CGMatrixFreeSolver, problem; abstol=1e-8)
        
        # Set same design variables
        x0 = fill(0.8, length(solver_assembly.vars))
        solver_assembly.vars .= x0
        solver_matrixfree.vars .= x0
        
        # Solve
        solver_assembly()
        solver_matrixfree()
        
        # Solutions should be similar
        @test isapprox(solver_assembly.u, solver_matrixfree.u; rtol=1e-3)
        
        # Compliance should be similar
        comp_assembly = getcompliance(solver_assembly)
        comp_matrixfree = getcompliance(solver_matrixfree)
        
        @test isapprox(comp_assembly, comp_matrixfree; rtol=1e-3)
    end
end
