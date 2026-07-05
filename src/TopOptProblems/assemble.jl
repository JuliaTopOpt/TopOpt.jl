function assemble(
    problem::AbstractTopOptProblem,
    elementinfo::ElementFEAInfo,
    vars=ones(floattype(problem), getncells(getdh(problem).grid)),
    penalty=PowerPenalty(floattype(problem)(3.0)),
    xmin=floattype(problem)(0.001),
)
    T = floattype(problem)
    dim = getdim(problem)
    globalinfo = GlobalFEAInfo(problem)
    assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)
    return globalinfo
end

# Assembly for all problem types
# For structural: fes contains body forces (penalized), fixedload contains concentrated/distributed loads (not penalized)
# For heat transfer: fes is zeros (no body forces), fixedload contains heat source (not penalized)
# 
# Note: ρ should be a full density vector (length = nel) with values already accounting for 
# black (density=1) and white (density=xmin) elements. Use FixedElementProjector to map 
# free variables to full densities.
function assemble!(
    globalinfo::GlobalFEAInfo,
    problem::AbstractTopOptProblem,
    elementinfo::ElementFEAInfo,
    ρ=ones(floattype(problem), getncells(getdh(problem).grid)),
    penalty=PowerPenalty(floattype(problem)(3.0)),
    xmin=floattype(problem)(0.001);
    assemble_f=true,
)
    T = floattype(problem)
    dim = getdim(problem)
    ch = problem.ch
    dh = ch.dh
    K, f = globalinfo.K, globalinfo.f
    if assemble_f
        f .= elementinfo.fixedload
    end
    Kes, fes = elementinfo.Kes, elementinfo.fes

    _K = K isa Symmetric ? K.data : K
    _K.nzval .= 0
    assembler = Ferrite.AssemblerSparsityPattern(_K, f, Int[], Int[])

    global_dofs = zeros(Int, ndofs_per_cell(dh))
    fe = zeros(typeof(fes[1]))
    Ke = zeros(T, size(rawmatrix(Kes[1])))

    celliterator = CellIterator(dh)
    for (i, cell) in enumerate(celliterator)
        # get global_dofs for cell#i
        celldofs!(global_dofs, dh, i)
        fe = fes[i]
        _Ke = rawmatrix(Kes[i])
        Ke = _Ke isa Symmetric ? _Ke.data : _Ke

        # Apply density interpolation
        if PENALTY_BEFORE_INTERPOLATION
            px = density(penalty(ρ[i]), xmin)
        else
            px = penalty(density(ρ[i], xmin))
        end
        Ke = px * Ke
        if assemble_f
            fe = px * fe
            Ferrite.assemble!(assembler, global_dofs, Ke, fe)
        else
            Ferrite.assemble!(assembler, global_dofs, Ke)
        end
    end

    #* apply boundary condition
    TK = eltype(K)
    _K = TK <: Symmetric ? K.data : K
    apply!(_K, f, ch)

    return nothing
end

function assemble_f(
    problem::StiffnessTopOptProblem{dim,T},
    elementinfo::ElementFEAInfo{dim,T},
    vars::AbstractVector{T},
    penalty,
    xmin=T(1) / 1000,
) where {dim,T}
    f = get_f(problem, vars)
    assemble_f!(f, problem, elementinfo, vars, penalty, xmin)
    return f
end
get_f(problem, vars::Array) = zeros(floattype(problem), ndofs(problem.ch.dh))

function assemble_f!(
    f::AbstractVector,
    problem::StiffnessTopOptProblem,
    elementinfo::ElementFEAInfo,
    ρ::AbstractVector,
    penalty,
    xmin,
)
    fes = elementinfo.fes

    dof_cells = elementinfo.metadata.dof_cells

    update_f!(f, fes, elementinfo.fixedload, dof_cells, ρ, penalty, xmin)
    return f
end

function update_f!(f::Vector, fes, fixedload, dof_cells, ρ, penalty, xmin)
    @inbounds for dofidx in 1:length(f)
        f[dofidx] = fixedload[dofidx]
        r = dof_cells.offsets[dofidx]:(dof_cells.offsets[dofidx + 1] - 1)
        for i in r
            cellidx, localidx = dof_cells.values[i]
            if PENALTY_BEFORE_INTERPOLATION
                px = density(penalty(ρ[cellidx]), xmin)
            else
                px = penalty(density(ρ[cellidx], xmin))
            end
            f[dofidx] += px * fes[cellidx][localidx]
        end
    end

    return nothing
end

function assemble_f!(f::AbstractVector, problem, dloads)
    metadata = problem.metadata
    dof_cells = metadata.dof_cells
    update_f!(f, dof_cells, dloads)
    return f
end

function update_f!(f::Vector, dof_cells, dloads)
    for dofidx in 1:length(f)
        r = dof_cells.offsets[dofidx]:(dof_cells.offsets[dofidx + 1] - 1)
        for i in r
            cellidx, localidx = dof_cells.values[i]
            f[dofidx] += dloads[cellidx][localidx]
        end
    end
    return nothing
end

function assemble_f(
    problem::StokesFlowProblem{dim, T},
    vars::AbstractVector{T},
    penalty,
    xmin::T
) where {dim, T}
    
    dh = problem.ch.dh
    μ = problem.viscosity
    α_max = problem.alpha_max
    
    # P1-P1 Interpolations! Both are linear now.
    ip_geo = Lagrange{dim, RefCube, 1}()
    ip_u = Lagrange{dim, RefCube, 1}()
    ip_p = Lagrange{dim, RefCube, 1}()
    qr = QuadratureRule{dim, RefCube}(3)

    cv_u = CellVectorValues(qr, ip_u, ip_geo)
    cv_p = CellScalarValues(qr, ip_p, ip_geo)
    
    fh = dh.fieldhandlers[1] 
    ndof_per_cell = ndofs_per_cell(dh)
    Ke = zeros(T, ndof_per_cell, ndof_per_cell)
    
    dofs_u = dof_range(fh, :u)
    dofs_p = dof_range(fh, :p)
    
    K = create_sparsity_pattern(dh)
    assembler = start_assemble(K)
    
    for (e, cell) in enumerate(CellIterator(dh))
        reinit!(cv_u, cell)
        reinit!(cv_p, cell)
        fill!(Ke, 0.0)
        
        density = vars[e]
        α_e = α_max * (1.0 - density^3) 
        
        for q_point in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, q_point)
            
            # --- VELOCITY BLOCK ---
            for i in 1:length(dofs_u)
                ∇N_u_i = shape_gradient(cv_u, q_point, i)
                N_u_i = shape_value(cv_u, q_point, i)
                for j in 1:length(dofs_u)
                    ∇N_u_j = shape_gradient(cv_u, q_point, j)
                    N_u_j = shape_value(cv_u, q_point, j)
                    
                    viscous = μ * sum(∇N_u_i .* ∇N_u_j)
                    brinkman = α_e * sum(N_u_i .* N_u_j)
                    Ke[dofs_u[i], dofs_u[j]] += (viscous + brinkman) * dΩ 
                end
            end
            
            # --- PRESSURE-VELOCITY BLOCK & STABILIZATION ---
            for i in 1:length(dofs_u)
                div_N_u_i = shape_divergence(cv_u, q_point, i)
                for j in 1:length(dofs_p)
                    N_p_j = shape_value(cv_p, q_point, j)
                    val = -N_p_j * div_N_u_i * dΩ
                    Ke[dofs_u[i], dofs_p[j]] += val
                    Ke[dofs_p[j], dofs_u[i]] += val 
                end
            end

            # --- THE STABILIZATION TRICK ---
            # Prevents the matrix from becoming singular with P1-P1 elements
            for j in 1:length(dofs_p)
                N_p_j = shape_value(cv_p, q_point, j)
                Ke[dofs_p[j], dofs_p[j]] -= 1e-6 * N_p_j * N_p_j * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    
    f = zeros(T, ndofs(dh))
    return K, f
end

function assemble_coupled(
    problem::FluidThermalProblem{dim, T},
    vars::AbstractVector{T},
    u_global::AbstractVector{T} 
) where {dim, T}
    
    dh = problem.ch.dh
    μ = problem.viscosity
    α_max = problem.alpha_max
    k_cond = problem.conductivity
    ρ_cp = problem.heat_capacity
    
    ip_geo = Lagrange{dim, RefCube, 1}()
    ip_u = Lagrange{dim, RefCube, 1}()
    ip_p = Lagrange{dim, RefCube, 1}()
    ip_T = Lagrange{dim, RefCube, 1}()
    qr = QuadratureRule{dim, RefCube}(3)

    cv_u = CellVectorValues(qr, ip_u, ip_geo)
    cv_p = CellScalarValues(qr, ip_p, ip_geo)
    cv_T = CellScalarValues(qr, ip_T, ip_geo)
    
    fh = dh.fieldhandlers[1] 
    ndof_per_cell = ndofs_per_cell(dh)
    Ke = zeros(T, ndof_per_cell, ndof_per_cell)
    
    dofs_u = dof_range(fh, :u)
    dofs_p = dof_range(fh, :p)
    dofs_T = dof_range(fh, :T)
    
    K = create_sparsity_pattern(dh)
    assembler = start_assemble(K)
    
    for (e, cell) in enumerate(CellIterator(dh))
        reinit!(cv_u, cell)
        reinit!(cv_p, cell)
        reinit!(cv_T, cell)
        fill!(Ke, 0.0)
        
        density = vars[e]
        α_e = α_max * (1.0 - density^3) 
        u_e = u_global[celldofs(cell)[dofs_u]]
        
        for q_point in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, q_point)
            vel_vec = function_value(cv_u, q_point, u_e)
            
            # Stokes Flow
            for i in 1:length(dofs_u)
                ∇N_u_i = shape_gradient(cv_u, q_point, i)
                N_u_i = shape_value(cv_u, q_point, i)
                for j in 1:length(dofs_u)
                    ∇N_u_j = shape_gradient(cv_u, q_point, j)
                    N_u_j = shape_value(cv_u, q_point, j)
                    Ke[dofs_u[i], dofs_u[j]] += (μ * sum(∇N_u_i .* ∇N_u_j) + α_e * sum(N_u_i .* N_u_j)) * dΩ 
                end
            end
            
            # Pressure-Velocity & P1-P1 Penalty Stabilization
            for i in 1:length(dofs_u)
                div_N_u_i = shape_divergence(cv_u, q_point, i)
                for j in 1:length(dofs_p)
                    val = -shape_value(cv_p, q_point, j) * div_N_u_i * dΩ
                    Ke[dofs_u[i], dofs_p[j]] += val
                    Ke[dofs_p[j], dofs_u[i]] += val 
                end
            end
            for j in 1:length(dofs_p)
                N_p_j = shape_value(cv_p, q_point, j)
                Ke[dofs_p[j], dofs_p[j]] -= 1e-6 * N_p_j * N_p_j * dΩ
            end
            
            # Thermal Advection-Diffusion
            for i in 1:length(dofs_T)
                N_T_i = shape_value(cv_T, q_point, i)
                ∇N_T_i = shape_gradient(cv_T, q_point, i)
                for j in 1:length(dofs_T)
                    N_T_j = shape_value(cv_T, q_point, j)
                    ∇N_T_j = shape_gradient(cv_T, q_point, j)
                    
                    conduction = k_cond * dot(∇N_T_i, ∇N_T_j)
                    advection = ρ_cp * N_T_i * dot(vel_vec, ∇N_T_j)
                    
                    Ke[dofs_T[i], dofs_T[j]] += (conduction + advection) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    
    f = zeros(T, ndofs(dh))
    return K, f
end

# Wrapper to execute the Sequential Picard Iteration solver
function solve_fluid_thermal(problem::FluidThermalProblem, vars::AbstractVector)
    ch = problem.ch
    total_dofs = ndofs(ch.dh)
    
    # Step A: Pure Fluid (Zero Advection)
    K_fluid, f_fluid = assemble_coupled(problem, vars, zeros(total_dofs))
    apply!(K_fluid, f_fluid, ch)
    u_fluid = K_fluid \ f_fluid
    
    # Step B: Advection-Diffusion 
    K_coupled, f_coupled = assemble_coupled(problem, vars, u_fluid)
    apply!(K_coupled, f_coupled, ch)
    u_final = K_coupled \ f_coupled
    
    return u_final, K_coupled
end

#=
function update_f!(f::CuVector, dof_cells, dloads)
    args = (f, dof_cells.offsets, dof_cells.values, dloads)
    callkernel(dev, assemble_kernel2, args)
    CUDAdrv.synchronize(ctx)

    return
end

function assemble_kernel2(f, dof_cells_offsets, dof_cells_values, dloads)
    i = @thread_global_index()
    offset = @total_threads()
    @inbounds while i <= length(f)
        r = dof_cells_offsets[i] : dof_cells_offsets[i+1]-1
        for i in r
            cellidx, localidx = dof_cells_values[i]
            f[i] += dloads[cellidx][localidx]
        end
        i += offset
    end
    return
end
=#
