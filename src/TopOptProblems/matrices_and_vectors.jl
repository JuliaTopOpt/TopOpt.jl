function gettypes(
    ::Type{T}, # number type
    ::Type{Val{:Static}}, # matrix type
    ::Type{Val{Kesize}}; # matrix size
    hyperelastic = false,
) where {T,Kesize}
    return hyperelastic ? 
    (SMatrix{Kesize,Kesize,T,Kesize^2}, SVector{Kesize,T}, SMatrix{3,3,T,3^2}) : 
    (SMatrix{Kesize,Kesize,T,Kesize^2}, SVector{Kesize,T})
end
function gettypes(
    ::Type{T}, # number type
    ::Type{Val{:SMatrix}}, # matrix type
    ::Type{Val{Kesize}}; # matrix size
    hyperelastic = false,
) where {T,Kesize}
    if hyperelastic 
        return SMatrix{Kesize,Kesize,T,Kesize^2}, SVector{Kesize,T}, SMatrix{3,3,T,3^2}
    else
        return SMatrix{Kesize,Kesize,T,Kesize^2}, SVector{Kesize,T}
    end
end
function gettypes(
    ::Type{T}, # number type
    ::Type{Val{:MMatrix}}, # matrix type
    ::Type{Val{Kesize}}; # matrix size
    hyperelastic = false,
) where {T,Kesize}
    if hyperelastic
        return MMatrix{Kesize,Kesize,T,Kesize^2}, MVector{Kesize,T}, MMatrix{3,3,T,3^2}
    else
        return MMatrix{Kesize,Kesize,T,Kesize^2}, MVector{Kesize,T}
    end
end
function gettypes(
    ::Type{BigFloat}, # number type
    ::Type{Val{:Static}}, # matrix type
    ::Type{Val{Kesize}}; # matrix size
    hyperelastic = false,
) where {Kesize}
    if hyperelastic
        return SizedMatrix{Kesize,Kesize,BigFloat,Kesize^2}, SizedVector{Kesize,BigFloat}, SizedMatrix{3,3,BigFloat,3^2}
    else 
        return SizedMatrix{Kesize,Kesize,BigFloat,Kesize^2}, SizedVector{Kesize,BigFloat}
    end
end
function gettypes(
    ::Type{BigFloat}, # number type
    ::Type{Val{:SMatrix}}, # matrix type
    ::Type{Val{Kesize}}; # matrix size
    hyperelastic = false,
) where {Kesize}
    if hyperelastic
        return SizedMatrix{Kesize,Kesize,BigFloat,Kesize^2}, SizedVector{Kesize,BigFloat}, SizedMatrix{3,3,BigFloat,3^2}
    else 
        return SizedMatrix{Kesize,Kesize,BigFloat,Kesize^2}, SizedVector{Kesize,BigFloat}
    end
end
function gettypes(
    ::Type{BigFloat}, # number type
    ::Type{Val{:MMatrix}}, # matrix type
    ::Type{Val{Kesize}}; # matrix size
    hyperelastic = false,
) where {Kesize}
    if hyperelastic
        return SizedMatrix{Kesize,Kesize,BigFloat,Kesize^2}, SizedVector{Kesize,BigFloat}, SizedMatrix{3,3,BigFloat,3^2}
    else
        return SizedMatrix{Kesize,Kesize,BigFloat,Kesize^2}, SizedVector{Kesize,BigFloat}
    end
end
function gettypes(
    ::Type{T}, # number type
    ::Any, # matrix type
    ::Any; # matrix size
    hyperelastic = false,
) where {T}
    if hyperelastic
        return Matrix{T}, Vector{T}, Matrix{T}
    else
        return Matrix{T}, Vector{T}
    end
end

initialize_K(sp::StiffnessTopOptProblem;symmetric::Bool=true) = symmetric ? Symmetric(create_sparsity_pattern(sp.ch.dh)) : create_sparsity_pattern(sp.ch.dh)

initialize_f(sp::StiffnessTopOptProblem{dim,T}) where {dim,T} = zeros(T, ndofs(sp.ch.dh))

function make_Kes_and_fes(problem, quad_order=2)
    return make_Kes_and_fes(problem, quad_order, Val{:Static})
end
function make_Kes_and_fes(problem, ::Type{Val{mat_type}}) where {mat_type}
    return make_Kes_and_fes(problem, 2, Val{mat_type})
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

    λ = E * ν / ((1 + ν) * (1 - 2 * ν))
    μ = E / (2 * (1 + ν))
    δ(i, j) = i == j ? T(1) : T(0)
    g(i, j, k, l) = λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
    C = SymmetricTensor{4,dim}(g)

    # Shape functions and quadrature rule
    interpolation_space = Lagrange{dim,refshape,geom_order}()
    quadrature_rule = QuadratureRule{dim,refshape}(quad_order)
    cellvalues = CellScalarValues(quadrature_rule, interpolation_space)
    facevalues = FaceScalarValues(
        QuadratureRule{dim - 1,refshape}(quad_order), interpolation_space
    )

    # Calculate element stiffness matrices
    n_basefuncs = getnbasefunctions(cellvalues)

    Kesize = dim * n_basefuncs
    MatrixType, VectorType = gettypes(T, Val{mat_type}, Val{Kesize})
    Kes, weights = _make_Kes_and_weights(
        dh,
        Tuple{MatrixType,VectorType},
        Val{n_basefuncs},
        Val{dim * n_basefuncs},
        C,
        ρ,
        quadrature_rule,
        cellvalues,
    )
    dloads = _make_dloads(weights, problem, facevalues)

    return Kes, weights, dloads, cellvalues, facevalues
end

function make_Kes_and_fes_hyperelastic(mp, problem, u, quad_order, ::Type{Val{mat_type}}, ts) where {mat_type}
    T = floattype(problem)
    dim = getdim(problem)
    geom_order = getgeomorder(problem)
    dh = getdh(problem)
    E = getE(problem)
    ν = getν(problem)
    ρ = getdensity(problem)

    refshape = Ferrite.getrefshape(dh.field_interpolations[1])

    #λ = E * ν / ((1 + ν) * (1 - 2 * ν))
    #μ = E / (2 * (1 + ν))
    #δ(i, j) = i == j ? T(1) : T(0)
    #g(i, j, k, l) = λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
    #C = SymmetricTensor{4,dim}(g)

    # Shape functions and quadrature rule
    interpolation_space = Lagrange{dim,refshape,geom_order}()
    quadrature_rule = QuadratureRule{dim,refshape}(quad_order)
    cellvalues = CellScalarValues(quadrature_rule, interpolation_space) # JGB: change from CellScalarValues 
    cellvaluesV = CellVectorValues(quadrature_rule, interpolation_space) # JGB: change from CellScalarValues 
    facevalues = FaceScalarValues(
        QuadratureRule{dim - 1,refshape}(quad_order), interpolation_space
    ) # JGB: change from CellScalarValues 
    facevaluesV = FaceVectorValues(
        QuadratureRule{dim - 1,refshape}(quad_order), interpolation_space
    )

    # Calculate element stiffness matrices
    n_basefuncs = getnbasefunctions(cellvalues)

    Kesize = dim * n_basefuncs
    MatrixType, VectorType, MatrixTypeF = gettypes(T, Val{mat_type}, Val{Kesize}; hyperelastic = true)
    Kes, weights, ges, Fes = _make_Kes_and_weights_hyperelastic(
        dh,
        Tuple{MatrixType,VectorType,MatrixTypeF},
        Val{n_basefuncs},
        Val{dim * n_basefuncs},
        #C,
        mp,
        u,
        ρ,
        quadrature_rule,
        cellvalues,
        cellvaluesV,
    )
    dloads, ges2 = _make_dloads_hyperelastic(weights, problem, facevalues, facevaluesV, ges, ts)
    ges += ges2

    return Kes, weights, dloads, ges, Fes, cellvalues, facevalues # switched to ges2 to solve error where 2x ges2 with no contribution from dload
end

const g = [0.0, 9.81, 0.0] # N/kg or m/s^2

# Element stiffness matrices are StaticArrays
# `weights` : a vector of `xdim` vectors, element_id => self-weight load vector
function _make_Kes_and_weights(
    dh::DofHandler{dim,N,T},
    ::Type{Tuple{MatrixType,VectorType}},
    ::Type{Val{n_basefuncs}},
    ::Type{Val{Kesize}},
    C,
    ρ,
    quadrature_rule,
    cellvalues,
) where {dim,N,T,MatrixType<:StaticArray,VectorType,n_basefuncs,Kesize}
    # Calculate element stiffness matrices
    nel = getncells(dh.grid)
    body_force = ρ .* g # Force per unit volume
    Kes = Symmetric{T,MatrixType}[]
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
                    fe = @set fe[(b - 1) * dim + d2] += ϕb * body_force[d2] * dΩ
                    for a in 1:n_basefuncs
                        ∇ϕa = shape_gradient(cellvalues, q_point, a)
                        Ke_e .= dotdot(∇ϕa, C, ∇ϕb) * dΩ
                        for d1 in 1:dim
                            #if dim*(b-1) + d2 >= dim*(a-1) + d1
                            Ke_0[dim * (a - 1) + d1, dim * (b - 1) + d2] += Ke_e[d1, d2]
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
    dh::DofHandler{dim,N,T},
    ::Type{Tuple{MatrixType,VectorType}},
    ::Type{Val{n_basefuncs}},
    ::Type{Val{Kesize}},
    C,
    ρ,
    quadrature_rule,
    cellvalues,
) where {dim,N,T,MatrixType,VectorType,n_basefuncs,Kesize}
    # Calculate element stiffness matrices
    nel = getncells(dh.grid)
    body_force = ρ .* g # Force per unit volume
    Kes = let Kesize = Kesize, nel = nel
        [Symmetric(zeros(T, Kesize, Kesize), :U) for i in 1:nel]
    end
    weights = let Kesize = Kesize, nel = nel
        [zeros(T, Kesize) for i in 1:nel]
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
                    fe[(b - 1) * dim + d2] += ϕb * body_force[d2] * dΩ
                    for a in 1:n_basefuncs
                        ∇ϕa = shape_gradient(cellvalues, q_point, a)
                        Ke_e .= dotdot(∇ϕa, C, ∇ϕb) * dΩ
                        for d1 in 1:dim
                            #if dim*(b-1) + d2 >= dim*(a-1) + d1
                            Kes[k].data[dim * (a - 1) + d1, dim * (b - 1) + d2] += Ke_e[
                                d1, d2
                            ]
                            #end
                        end
                    end
                end
            end
        end
    end
    return Kes, weights
end

function Ψ(C, mp) # JGB: add to .ipynb
    μ = mp.μ
    λ = mp.λ
    I1 = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (I1 - 3) - μ * log(J) + λ / 2 * (J - 1)^2 # Ferrite.jl version
    #return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J-1)^2 # Bower version
    #Cnew = @MArray C
    #I1bar = Ic*J^-2/3
    #I1bar = Ic*det(C)^-1/3
    #return μ / 2 * (I1bar - 3) + 0.5*(λ + 2μ/3)*(J-1)^2 # ABAQUS/Bower version
end

function constitutive_driver(C, mp) # JGB removed type ::NeoHook from mp
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end

# Element stiffness matrices are StaticArrays
# `weights` : a vector of `xdim` vectors, element_id => self-weight load vector
function _make_Kes_and_weights_hyperelastic(
    dh::DofHandler{dim,N,T},
    ::Type{Tuple{MatrixType,VectorType,MatrixTypeF}},
    ::Type{Val{n_basefuncs}},
    ::Type{Val{Kesize}},
    mp,
    u,
    ρ,
    quadrature_rule,
    cellvalues,
    cellvaluesV,
) where {dim,N,T,MatrixType<:StaticArray,VectorType,MatrixTypeF<:StaticArray,n_basefuncs,Kesize}
    # Calculate element stiffness matrices
    nel = getncells(dh.grid)
    body_force = ρ .* g # Force per unit volume
    #Kes = Symmetric{T,MatrixType}[] # JGB: scary (is this sucker symmetric?)
    Kes = Vector{MatrixType}()
    sizehint!(Kes, nel)
    weights = [zeros(VectorType) for i in 1:nel]
    ges = [zeros(VectorType) for i in 1:nel]
    Fes = [zeros(MatrixTypeF) for _ in 1:nel]
    Ke_0 = Matrix{T}(undef, Kesize, Kesize)
    celliterator = CellIterator(dh)
    for (k, cell) in enumerate(celliterator)
        Ke_0 .= 0
        #Ke_02 .= 0
        global_dofs = celldofs(cell) # new
        ue = u[global_dofs] # new
        reinit!(cellvalues, cell)
        reinit!(cellvaluesV, cell)
        fe = weights[k]
        ge = ges[k]
        Fe = Fes[k]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            ∇u = function_gradient(cellvaluesV, q_point, ue) # JGB add (NEEDS TO BE CHECKED!!)
            F = one(∇u) + ∇u # JGB add 
            Fe += F*dΩ
            C = tdot(F) # JGB add 
            S, ∂S∂C = constitutive_driver(C, mp) # JGB add 
            P = F ⋅ S # JGB add 
            I = one(S) # JGB add 
            ∂P∂F =  otimesu(I, S) + 2 * otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I) # JGB add (neither P or F is symmetric so ∂P∂F will not either)
            for b in 1:Kesize
                ∇ϕb = shape_gradient(cellvaluesV, q_point, b) # JGB: like ∇δui
                ϕb = shape_value(cellvaluesV, q_point, b) # JGB: like δui
                ∇ϕb∂P∂F = ∇ϕb ⊡ ∂P∂F # Hoisted computation # JGB add
                #fe = @set fe[(b - 1) * dim + d2] += ϕb * body_force[d2] * dΩ # weird but probably fine... just not in ferrite.jl example code (leaving in case of zygote issues)
                fe = @set fe[b] += ϕb ⋅ body_force * dΩ
                ge = @set ge[b] += ( ∇ϕb ⊡ P - ϕb ⋅ body_force ) * dΩ  # Add contribution to the residual from this test function
                for a in 1:Kesize
                    ∇ϕa = shape_gradient(cellvaluesV, q_point, a) # JGB: like ∇δuj
                    Ke_0[a,b] += (∇ϕb∂P∂F ⊡ ∇ϕa) * dΩ
                end
            end
        end
        weights[k] = fe
        ges[k] = ge
        Fes[k] = Fe
        if MatrixType <: SizedMatrix # Work around because full constructor errors
            #push!(Kes, Symmetric(SizedMatrix{Kesize,Kesize,T}(Ke_0)))
            push!(Kes, SizedMatrix{Kesize,Kesize,T}(Ke_0))
        else
            #push!(Kes, Symmetric(MatrixType(Ke_0)))
            push!(Kes, MatrixType(Ke_0))
        end
    end
    return Kes, weights, ges, Fes # weights is fes
end
# Fallback
#= bring this up to speed at a later time to match with
function _make_Kes_and_weights(
    dh::DofHandler{dim,N,T},
    ::Type{Tuple{MatrixType,VectorType}},
    ::Type{Val{n_basefuncs}},
    ::Type{Val{Kesize}},
    C,
    ρ,
    quadrature_rule,
    cellvalues,
) where {dim,N,T,MatrixType,VectorType,n_basefuncs,Kesize}
    # Calculate element stiffness matrices
    nel = getncells(dh.grid)
    body_force = ρ .* g # Force per unit volume
    Kes = let Kesize = Kesize, nel = nel
        [Symmetric(zeros(T, Kesize, Kesize), :U) for i in 1:nel]
    end
    weights = let Kesize = Kesize, nel = nel
        [zeros(T, Kesize) for i in 1:nel]
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
                    fe[(b - 1) * dim + d2] += ϕb * body_force[d2] * dΩ
                    for a in 1:n_basefuncs
                        ∇ϕa = shape_gradient(cellvalues, q_point, a)
                        Ke_e .= dotdot(∇ϕa, C, ∇ϕb) * dΩ
                        for d1 in 1:dim
                            #if dim*(b-1) + d2 >= dim*(a-1) + d1
                            Kes[k].data[dim * (a - 1) + d1, dim * (b - 1) + d2] += Ke_e[
                                d1, d2
                            ]
                            #end
                        end
                    end
                end
            end
        end
    end
    return Kes, weights
end=#

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
    cell_coords = zeros(Ferrite.Vec{dim,T}, N)
    n_basefuncs = getnbasefunctions(facevalues)
    for k in keys(pressuredict)
        t = -pressuredict[k] # traction = negative the pressure
        faceset = getfacesets(problem)[k]
        for (cellid, faceid) in faceset
            boundary_matrix[faceid, cellid] ||
                throw("Face $((cellid, faceid)) not on boundary.")
            fe = dloads[cellid]
            getcoordinates!(cell_coords, grid, cellid)
            reinit!(facevalues, cell_coords, faceid)
            for q_point in 1:getnquadpoints(facevalues)
                dΓ = getdetJdV(facevalues, q_point) # Face area
                normal = getnormal(facevalues, q_point) # Nomral vector at quad point
                for i in 1:n_basefuncs
                    ϕ = shape_value(facevalues, q_point, i) # Shape function value
                    for d in 1:dim
                        if fe isa SArray
                            fe = @set fe[(i - 1) * dim + d] += ϕ * t * normal[d] * dΓ
                        else
                            fe[(i - 1) * dim + d] += ϕ * t * normal[d] * dΓ
                        end
                    end
                end
            end
            dloads[cellid] = fe
        end
    end

    return dloads
end

function _make_dloads_hyperelastic(fes, problem, facevalues, facevaluesV, ges, ts)
    dim = getdim(problem)
    N = nnodespercell(problem)
    T = floattype(problem)
    dloads = deepcopy(fes)
    ges2 = deepcopy(ges)
    for i in 1:length(dloads)
        if eltype(dloads) <: SArray
            dloads[i] = zero(eltype(dloads))
        else
            dloads[i] .= 0
        end
    end
    for i in 1:length(ges2)
        if eltype(ges2) <: SArray
            ges2[i] = zero(eltype(ges2))
        else
            ges2[i] .= 0
        end
    end
    pressuredict = getpressuredict(problem; ts=ts)
    dh = getdh(problem)
    grid = dh.grid
    boundary_matrix = grid.boundary_matrix
    cell_coords = zeros(Ferrite.Vec{dim,T}, N)
    n_basefuncs = getnbasefunctions(facevalues)
    for k in keys(pressuredict)
        t = -pressuredict[k] # traction = negative the pressure
        faceset = getfacesets(problem)[k]
        for (cellid, faceid) in faceset
            boundary_matrix[faceid, cellid] ||
                throw("Face $((cellid, faceid)) not on boundary.")
            fe = dloads[cellid]
            ge = ges2[cellid]
            getcoordinates!(cell_coords, grid, cellid)
            reinit!(facevalues, cell_coords, faceid)
            for q_point in 1:getnquadpoints(facevalues)
                dΓ = getdetJdV(facevalues, q_point) # Face area
                normal = getnormal(facevalues, q_point) # Nomral vector at quad point
                for i in 1:n_basefuncs
                    ϕ = shape_value(facevalues, q_point, i) # Shape function value
                    for d in 1:dim
                        if fe isa SArray
                            fe = @set fe[(i - 1) * dim + d] += ϕ * t * normal[d] * dΓ
                        else
                            fe[(i - 1) * dim + d] += ϕ * t * normal[d] * dΓ
                        end
                        if ge isa SArray
                            ge = @set ge[(i - 1) * dim + d] -= ϕ * t * normal[d] * dΓ
                        else
                            ge[(i - 1) * dim + d] -= ϕ * t * normal[d] * dΓ
                        end
                    end
                end
            end
            dloads[cellid] = fe
            ges2[cellid] = ge
        end
    end

    return dloads, ges2
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
                dof = node_dofs[(nodeidx - 1) * dim + dofidx]
                push!(inds, dof)
                push!(vals, force)
            end
        end
    end
    return sparsevec(inds, vals, ndofs(dh))
end

function make_cload_hyperelastic(problem,ts)
    T = floattype(problem)
    dim = getdim(problem)
    cloads = getcloaddict(problem; ts=ts)
    dh = getdh(problem)
    metadata = getmetadata(problem)
    node_dofs = metadata.node_dofs
    inds = Int[]
    vals = T[]
    for nodeidx in keys(cloads)
        for (dofidx, force) in enumerate(cloads[nodeidx])
            if force != 0
                dof = node_dofs[(nodeidx - 1) * dim + dofidx]
                push!(inds, dof)
                push!(vals, force)
            end
        end
    end
    return sparsevec(inds, vals, ndofs(dh))
end
