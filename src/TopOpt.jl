module TopOpt

using JuAFEM

export make_rect_grid, StiffnessProblem, get_fe, assemble!, compliance, simulate, History

abstract type AbstractGrid{dim, T} end
struct StiffnessProblem{_dim, T, G <: AbstractGrid{_dim, T}}
    dim::Int
    E::T
    ν::T
    grid::G
    problem::Int
end

function StiffnessProblem(grid::G, dim=2, problem=1, E=1., ν=0.3) where {G <: AbstractGrid}
    return StiffnessProblem{dim, typeof(ν), G}(dim, E, ν, grid, problem)
end

struct RectGrid{dim, T, G <: JuAFEM.Grid} <: AbstractGrid{dim, T}
    grid::G
    corners::NTuple{2, Vec{dim,T}}
end

struct History{T<:AbstractFloat, I<:Integer}
    c_hist::Vector{T}
    v_hist::Vector{T}
    x_hist::Vector{Vector{T}}
    add_hist::Vector{I}
    rem_hist::Vector{I}
end

const sp_fields = [:E, :ν]
macro unpack(sp)
    esc(Expr(:block, [:($f = sp.$f) for f in sp_fields]...))
end

left(rectgrid::RectGrid, x) = x[1] ≈ rectgrid.corners[1][1]
right(rectgrid::RectGrid, x) = x[1] ≈ rectgrid.corners[2][1]
bottom(rectgrid::RectGrid, x) = x[2] ≈ rectgrid.corners[1][2]
top(rectgrid::RectGrid, x) = x[2] ≈ rectgrid.corners[2][2]
back(rectgrid::RectGrid, x) = x[3] ≈ rectgrid.corners[1][3]
front(rectgrid::RectGrid, x) = x[3] ≈ rectgrid.corners[2][3]
middlex(rectgrid::RectGrid, x) = x[1] ≈ (rectgrid.corners[1][1] + rectgrid.corners[2][1]) / 2
middley(rectgrid::RectGrid, x) = x[2] ≈ (rectgrid.corners[1][2] + rectgrid.corners[2][2]) / 2
middlez(rectgrid::RectGrid, x) = x[3] ≈ (rectgrid.corners[1][3] + rectgrid.corners[2][3]) / 2

@generated function make_rect_grid(nels::NTuple{dim,Int}, sizes::NTuple{dim,T}) where {dim,T}
    # Draw rectangular grid
    geoshape = dim == 2 ? Quadrilateral : Hexahedron
    return quote
        corner1 = Vec{$dim}(fill(0., $dim))
        corner2 = Vec{$dim}((nels .* sizes))
        grid = generate_grid($geoshape, nels, corner1, corner2);
        rectgrid = RectGrid(grid, (corner1, corner2))    
        return rectgrid
    end
end

function get_fe(sp::StiffnessProblem{dim,T}, force::Float64) where {dim, T}
    E = sp.E
    ν = sp.ν

    # Create stiffness tensor
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    C = SymmetricTensor{4, dim}(g)

    # Shape functions and quadrature rule
    order = 2
    interpolation = Lagrange{dim, RefCube, 1}()
    quad_rule = QuadratureRule{dim, RefCube}(order)
    fe_values = CellScalarValues(quad_rule, interpolation)
    face_quad_rule = QuadratureRule{dim-1, RefCube}(order)
    facevalues = FaceScalarValues(face_quad_rule, interpolation)
    
    # Create displacement field u
    dh = DofHandler(sp.grid.grid)
    push!(dh, :u, dim) # Add a displacement field
    close!(dh)

    # Initialize K with its sparsity pattern
    K = create_symmetric_sparsity_pattern(dh)

    # Boundary conditions
    f = zeros(T, JuAFEM.ndofs(dh))
    dbc = DirichletBoundaryConditions(dh)
    local fnode::Int
    if sp.problem == 1
        addnodeset!(dh.grid, "fixed_all", x -> left(sp.grid, x));
        addnodeset!(dh.grid, "down_force", x -> right(sp.grid, x) && middley(sp.grid, x));
        addcellset!(dh.grid, "black", x -> right(sp.grid, x) && middley(sp.grid, x); all = false)
        #addcellset!(dh.grid, "black", x -> right(sp.grid, x) && left(sp.grid, x))
        addcellset!(dh.grid, "white", x -> right(sp.grid, x) && left(sp.grid, x))
        
        add!(dbc, :u, getnodeset(dh.grid, "fixed_all"), (x,t) -> [0.0 for i in 1:dim], collect(1:dim))

        fnode = Tuple(getnodeset(sp.grid.grid, "down_force"))[1]
        f_dof = JuAFEM.dofs_node(dh, fnode)[2]
        f[f_dof] = force
    elseif sp.problem == 2
        addnodeset!(dh.grid, "fixed_u1", x -> left(sp.grid, x));
        addnodeset!(dh.grid, "fixed_u2", x -> bottom(sp.grid, x) && right(sp.grid, x));
        addnodeset!(dh.grid, "down_force", x -> top(sp.grid, x) && left(sp.grid, x));
        #addnodeset!(dh.grid, "down_force", x -> x[2] ≈ 0.019 && left(sp.grid, x));
        addcellset!(dh.grid, "black", x -> right(sp.grid, x) && left(sp.grid, x))
        addcellset!(dh.grid, "white", x -> right(sp.grid, x) && left(sp.grid, x))
        #addcellset!(dh.grid, "white", x -> top(sp.grid, x) && left(sp.grid, x))
        
        add!(dbc, :u, getnodeset(dh.grid, "fixed_u1"), (x,t) -> [0.0], [1])
        add!(dbc, :u, getnodeset(dh.grid, "fixed_u2"), (x,t) -> [0.0], [2])

        fnode = Tuple(getnodeset(sp.grid.grid, "down_force"))[1]
        f_dof = JuAFEM.dofs_node(dh, fnode)[2]
        f[f_dof] = force
    elseif sp.problem == 3
        addnodeset!(dh.grid, "fixed_u1", x -> left(sp.grid, x));
        addnodeset!(dh.grid, "fixed_u2", x -> top(sp.grid, x) && right(sp.grid, x));
        addnodeset!(dh.grid, "down_force", x -> top(sp.grid, x) && left(sp.grid, x));
        #addnodeset!(dh.grid, "down_force", x -> x[2] ≈ 0.019 && left(sp.grid, x));
        addcellset!(dh.grid, "black", x -> right(sp.grid, x) && left(sp.grid, x))
        addcellset!(dh.grid, "white", x -> right(sp.grid, x) && left(sp.grid, x))
        #addcellset!(dh.grid, "white", x -> top(sp.grid, x) && left(sp.grid, x))
        
        add!(dbc, :u, getnodeset(dh.grid, "fixed_u1"), (x,t) -> [0.0], [1])
        add!(dbc, :u, getnodeset(dh.grid, "fixed_u2"), (x,t) -> [0.0], [2])

        fnode = Tuple(getnodeset(sp.grid.grid, "down_force"))[1]
        f_dof = JuAFEM.dofs_node(dh, fnode)[2]
        f[f_dof] = force
    end
    close!(dbc)
    t = 0.0
    update!(dbc, t)

    # Calculate element stiffness matrices
    Kesize::Int = dim*getnbasefunctions(fe_values)
    nel = getncells(dh.grid)
    Kes = [Symmetric(zeros(T, Kesize, Kesize),:U) for i = 1:nel]
    n_basefuncs = getnbasefunctions(fe_values)
    Ke_e = zeros(T, dim, dim)
    for (k, cell) in enumerate(CellIterator(dh))
        reinit!(fe_values, cell)
        for q_point in 1:getnquadpoints(fe_values)
            for a in 1:n_basefuncs
                for b in 1:n_basefuncs
                    ∇ϕa = shape_gradient(fe_values, q_point, a)
                    ∇ϕb = shape_gradient(fe_values, q_point, b)
                    Ke_e .= dotdot(∇ϕa, C, ∇ϕb) * getdetJdV(fe_values, q_point)
                    for d1 in 1:dim, d2 in 1:dim
                        if dim*(b-1) + d2 >= dim*(a-1) + d1
                            Kes[k].data[dim*(a-1) + d1, dim*(b-1) + d2] += Ke_e[d1,d2]
                        end
                    end
                end
            end
        end
    end

    return dh, dbc, K, Kes, f
end

function assemble!(K, f, densities, xmin, Kes, dh, dbc, sp)
    K.data.nzval .= 0.
    assembler = start_assemble(K)
    k = 1
    for (i,cell) in enumerate(CellIterator(dh))
        if i ∈ getcellset(sp.grid.grid, "black")
            JuAFEM.assemble!(assembler, celldofs(cell), Kes[i])
        elseif i ∈ getcellset(sp.grid.grid, "white")
            JuAFEM.assemble!(assembler, celldofs(cell), xmin .* Kes[i])  
        else
            JuAFEM.assemble!(assembler, celldofs(cell), densities[k] .* Kes[i])
            k += 1
        end
    end
    apply!(K, f, dbc)
    return 
end

compliance(ue, Ke) = ue' * Ke * ue 

function meandiag(K::AbstractMatrix)
    z = zero(eltype(K))
    for i in 1:size(K, 1)
        z += abs(K[i, i])
    end
    return z / size(K, 1)
end

function simulate(sp, force, topology=ones(getncells(sp.grid.grid)))
    dh, dbc, K, Kes, f = get_fe(sp, force)
    for (i,cell) in enumerate(CellIterator(dh))
        if i ∈ getcellset(sp.grid.grid, "white")
            topology[i] = 0.
        elseif i ∈ getcellset(sp.grid.grid, "black")
            topology[i] = 1.
        end
    end
    K.data.nzval .= 0.
    assembler = start_assemble(K)
    for (i,cell) in enumerate(CellIterator(dh))
        if topology[i] ≉  0.
            JuAFEM.assemble!(assembler, celldofs(cell), topology[i] .* Kes[i])
        end
    end
    apply!(K, f, dbc)
    m = meandiag(K)
    for i in 1:size(K,1)
        if K[i,i] ≈ 0.
            K[i,i] = m
        end
    end
    u = Symmetric(K) \ f
    comp = dot(u, f)
    return u, comp
end

function simulate(dh, dbc, K, Kes, f, densities=ones(getncells(dh.grid)), hard=false)
    black_cells = getcellset(dh.grid, "black")
    white_cells = getcellset(dh.grid, "white")
    fixedcells = black_cells ∪ white_cells
    nfc = length(fixedcells)
    nel = getncells(dh.grid)
    topology = zeros(nel)

    if length(densities) == nel
        for (i,cell) in enumerate(CellIterator(dh))
            if i ∈ white_cells
                topology[i] = 0.
            elseif i ∈ black_cells
                topology[i] = 1.
            elseif hard
                topology[i] = round(densities[i])
            else
                topology[i] = densities[i]
            end
        end
    elseif length(densities) == nel - nfc
        k = 1
        for (i,cell) in enumerate(CellIterator(dh))
            if i ∈ white_cells
                topology[i] = 0.
            elseif i ∈ black_cells
                topology[i] = 1.
            elseif hard
                topology[i] = round(densities[k])
                k += 1
            else
                topology[i] = densities[k]
                k += 1
            end
        end
    else
        throw("Topology is not of an appropriate size.")
    end

    K.data.nzval .= 0.
    assembler = start_assemble(K)
    for (i,cell) in enumerate(CellIterator(dh))
        if topology[i] ≉  0.
            JuAFEM.assemble!(assembler, celldofs(cell), topology[i] .* Kes[i])
        end
    end
    apply!(K, f, dbc)
    m = meandiag(K)
    for i in 1:size(K,1)
        if K[i,i] ≈ 0.
            K[i,i] = m
        end
    end
    zero_diags = 0
    for i in 1:size(K,1)
        if K[i,i] ≈ 0.    
            zero_diags += 1
        end
    end
    if size(K,1) > rank(Array(K)) # Disconnected
        return zeros(f), Inf
    end

    u = Symmetric(K) \ f
    comp = dot(u, f)
    return u, comp
end

end