function gettypes(
    ::Type{T}, # number type
    ::Type{Val{:Static}}, # matrix type
    ::Type{Val{Kesize}}, # matrix size
) where {T, Kesize}
    return SMatrix{Kesize,  Kesize, T, Kesize^2}, SVector{Kesize, T}
end
function gettypes(
    ::Type{T}, # number type
    ::Type{Val{:SMatrix}}, # matrix type
    ::Type{Val{Kesize}}, # matrix size
) where {T, Kesize}
    return SMatrix{Kesize,  Kesize, T, Kesize^2}, SVector{Kesize, T}
end
function gettypes(
    ::Type{T}, # number type
    ::Type{Val{:MMatrix}}, # matrix type
    ::Type{Val{Kesize}}, # matrix size
) where {T, Kesize}
    return MMatrix{Kesize,  Kesize, T, Kesize^2}, MVector{Kesize, T}
end
function gettypes(
    ::Type{BigFloat}, # number type
    ::Type{Val{:Static}}, # matrix type
    ::Type{Val{Kesize}}, # matrix size
) where {Kesize}
    return SizedMatrix{Kesize, Kesize, BigFloat, Kesize^2}, SizedVector{Kesize, BigFloat}
end
function gettypes(
    ::Type{BigFloat}, # number type
    ::Type{Val{:SMatrix}}, # matrix type
    ::Type{Val{Kesize}}, # matrix size
) where {Kesize}
    return SizedMatrix{Kesize, Kesize, BigFloat, Kesize^2}, SizedVector{Kesize, BigFloat}
end
function gettypes(
    ::Type{BigFloat}, # number type
    ::Type{Val{:MMatrix}}, # matrix type
    ::Type{Val{Kesize}}, # matrix size
) where {Kesize}
    return SizedMatrix{Kesize, Kesize, BigFloat, Kesize^2}, SizedVector{Kesize, BigFloat}
end
function gettypes(
    ::Type{T}, # number type
    ::Any, # matrix type
    ::Any, # matrix size
) where {T}
    return Matrix{T}, Vector{T}
end

initialize_K(sp::StiffnessTopOptProblem) = Symmetric(create_sparsity_pattern(sp.ch.dh))

initialize_f(sp::StiffnessTopOptProblem{dim, T}) where {dim, T} = zeros(T, ndofs(sp.ch.dh))

function make_Kes_and_fes(problem, quad_order=2)
    make_Kes_and_fes(problem, quad_order, Val{:Static})
end
function make_Kes_and_fes(problem, ::Type{Val{mat_type}}) where mat_type
    make_Kes_and_fes(problem, 2, Val{mat_type})
end
function make_Kes_and_fes(problem, quad_order, ::Type{Val{mat_type}}) where {mat_type}
    T = floattype(problem)
    dim = getdim(problem)
    geom_order = getgeomorder(problem)
    dh = getdh(problem)
    E = getE(problem)
    ν = getν(problem)
    ρ = getdensity(problem)

    refshape = Ferrite.getrefshape(dh.field_interpolations[1])

    λ = E*ν / ((1 + ν) * (1 - 2*ν))
    μ = E / (2*(1 + ν))
    δ(i,j) = i == j ? T(1) : T(0)
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    C = SymmetricTensor{4, dim}(g)

    # Shape functions and quadrature rule
    interpolation_space = Lagrange{dim, refshape, geom_order}()
    quadrature_rule = QuadratureRule{dim, refshape}(quad_order)
    cellvalues = CellScalarValues(quadrature_rule, interpolation_space)
    facevalues = FaceScalarValues(QuadratureRule{dim-1, refshape}(quad_order), interpolation_space)

    # Calculate element stiffness matrices
    n_basefuncs = getnbasefunctions(cellvalues)
    
    Kesize = dim*n_basefuncs
    MatrixType, VectorType = gettypes(T, Val{mat_type}, Val{Kesize})
    Kes, weights = _make_Kes_and_weights(dh, Tuple{MatrixType, VectorType}, Val{n_basefuncs}, Val{dim*n_basefuncs}, C, ρ, quadrature_rule, cellvalues)
    dloads = _make_dloads(weights, problem, facevalues)

    return Kes, weights, dloads, cellvalues, facevalues
end

const g = [0., 9.81, 0.] # N/kg or m/s^2

# Element stiffness matrices are StaticArrays
# `weights` : a vector of `xdim` vectors, element_id => self-weight load vector
function _make_Kes_and_weights(dh::DofHandler{dim, N, T}, ::Type{Tuple{MatrixType, VectorType}},
    ::Type{Val{n_basefuncs}}, ::Type{Val{Kesize}}, C, ρ, quadrature_rule, cellvalues) where {
    dim, N, T, MatrixType <: StaticArray, VectorType, n_basefuncs, Kesize}
    # Calculate element stiffness matrices
    nel = getncells(dh.grid)
    body_force = ρ .* g # Force per unit volume
    Kes = Symmetric{T, MatrixType}[]
    sizehint!(Kes, nel)
    weights = [zeros(VectorType) for i in 1:nel]
    Ke_e = zeros(T, dim, dim)
    fe = zeros(T, Kesize)
    Ke_0 = Matrix{T}(undef, Kesize, Kesize)
    celliterator = CellIterator(dh)
    for (k, cell) in enumerate(celliterator)
        Ke_0 .= 0
        reinit!(cellvalues, cell)
        fe = weights[k]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for b in 1:n_basefuncs
                ∇ϕb = shape_gradient(cellvalues, q_point, b)
                ϕb = shape_value(cellvalues, q_point, b)
                for d2 in 1:dim
                    fe = @set fe[(b-1)*dim + d2] += ϕb * body_force[d2] * dΩ
                    for a in 1:n_basefuncs
                        ∇ϕa = shape_gradient(cellvalues, q_point, a)
                        Ke_e .= dotdot(∇ϕa, C, ∇ϕb) * dΩ
                        for d1 in 1:dim
                            #if dim*(b-1) + d2 >= dim*(a-1) + d1
                            Ke_0[dim*(a-1) + d1, dim*(b-1) + d2] += Ke_e[d1,d2]
                            #end
                        end
                    end
                end
            end
        end
        weights[k] = fe
        if MatrixType <: SizedMatrix # Work around because full constructor errors
            push!(Kes, Symmetric(SizedMatrix{Kesize,Kesize,T}(Ke_0)))
        else
            push!(Kes, Symmetric(MatrixType(Ke_0)))
        end
    end
    return Kes, weights
end
# Fallback
function _make_Kes_and_weights(
    dh::DofHandler{dim, N, T},
    ::Type{Tuple{MatrixType, VectorType}},
    ::Type{Val{n_basefuncs}},
    ::Type{Val{Kesize}},
    C,
    ρ,
    quadrature_rule,
    cellvalues,
) where {
    dim, N, T, MatrixType, VectorType, n_basefuncs, Kesize,
}
    # Calculate element stiffness matrices
    nel = getncells(dh.grid)
    body_force = ρ .* g # Force per unit volume
    Kes = let Kesize=Kesize, nel=nel
        [Symmetric(zeros(T, Kesize, Kesize), :U) for i = 1:nel]
    end
    weights = let Kesize=Kesize, nel=nel
        [zeros(T, Kesize) for i = 1:nel]
    end
    Ke_e = zeros(T, dim, dim)    
    celliterator = CellIterator(dh)
    for (k, cell) in enumerate(celliterator)
        reinit!(cellvalues, cell)
        fe = weights[k]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for b in 1:n_basefuncs
                ∇ϕb = shape_gradient(cellvalues, q_point, b)
                ϕb = shape_value(cellvalues, q_point, b)
                for d2 in 1:dim
                    fe[(b-1)*dim + d2] += ϕb * body_force[d2] * dΩ
                    for a in 1:n_basefuncs
                        ∇ϕa = shape_gradient(cellvalues, q_point, a)
                        Ke_e .= dotdot(∇ϕa, C, ∇ϕb) * dΩ
                        for d1 in 1:dim
                            #if dim*(b-1) + d2 >= dim*(a-1) + d1
                            Kes[k].data[dim*(a-1) + d1, dim*(b-1) + d2] += Ke_e[d1,d2]
                            #end
                        end
                    end
                end
            end
        end
    end
    return Kes, weights
end

"""
    _make_dload(problem)

Assemble a sparse vector for boundary (face) distributed loads
"""
function _make_dloads(fes, problem, facevalues)
    dim = getdim(problem)
    N = nnodespercell(problem)
    T = floattype(problem)
    dloads = deepcopy(fes)
    for i in 1:length(dloads)
        if eltype(dloads) <: SArray
            dloads[i] = zero(eltype(dloads))
        else
            dloads[i] .= 0
        end
    end
    pressuredict = getpressuredict(problem)
    dh = getdh(problem)
    grid = dh.grid
    boundary_matrix = grid.boundary_matrix
    cell_coords = zeros(Ferrite.Vec{dim, T}, N)
    n_basefuncs = getnbasefunctions(facevalues)
    for k in keys(pressuredict)
        t = -pressuredict[k] # traction = negative the pressure
        faceset = getfacesets(problem)[k]
        for (cellid, faceid) in faceset
            boundary_matrix[faceid, cellid] || throw("Face $((cellid, faceid)) not on boundary.")
            fe = dloads[cellid]
            getcoordinates!(cell_coords, grid, cellid)
            reinit!(facevalues, cell_coords, faceid)
            for q_point in 1:getnquadpoints(facevalues)
                dΓ = getdetJdV(facevalues, q_point) # Face area
                normal = getnormal(facevalues, q_point) # Nomral vector at quad point
                for i in 1:n_basefuncs
                    ϕ = shape_value(facevalues, q_point, i) # Shape function value
                    for d = 1:dim
                        if fe isa SArray
                            fe = @set fe[(i-1)*dim + d] += ϕ * t * normal[d] * dΓ
                        else
                            fe[(i-1)*dim + d] += ϕ * t * normal[d] * dΓ
                        end
                    end
                end
            end
            dloads[cellid] = fe
        end
    end
    
    return dloads
end

"""
    make_cload(problem)

Assemble a sparse vector for concentrated loads
"""
function make_cload(problem)
    T = floattype(problem)
    dim = getdim(problem)
    cloads = getcloaddict(problem)
    dh = getdh(problem)
    metadata = getmetadata(problem)
    node_dofs = metadata.node_dofs
    inds = Int[]
    vals = T[]
    for nodeidx in keys(cloads)
        for (dofidx, force) in enumerate(cloads[nodeidx])
            if force != 0
                dof = node_dofs[(nodeidx-1)*dim+dofidx]
                push!(inds, dof)
                push!(vals, force)
            end
        end
    end
    return sparsevec(inds, vals, ndofs(dh))
end
