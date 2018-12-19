const RectilinearPointLoad{dim, T, N, M} = Union{PointLoadCantilever{dim, T, N, M}, HalfMBB{dim, T, N, M}, LBeam{T, N, M}}

struct ElementMatrix{T, TM <: AbstractMatrix{T}, TMask} <: AbstractMatrix{T}
    matrix::TM
    mask::TMask
    meandiag::T
end
ElementMatrix(matrix, mask) = ElementMatrix(matrix, mask, sumdiag(matrix)/size(matrix, 1))

const StaticMatrices{m,T} = Union{StaticMatrix{m,m,T}, Symmetric{T, <:StaticMatrix{m,m,T}}}
@generated function sumdiag(K::StaticMatrices{m,T}) where {m,T}
    return reduce((ex1,ex2) -> :($ex1 + $ex2), [:(K[$j,$j]) for j in 1:m])
end

Base.size(m::ElementMatrix) = size(m.matrix)
Base.getindex(m::ElementMatrix, i...) = m.matrix[i...]

rawmatrix(m::ElementMatrix) = m.matrix
rawmatrix(m::Symmetric{T, <:ElementMatrix{T}}) where {T} = Symmetric(m.data.matrix)
@generated function bcmatrix(m::ElementMatrix{T, TM}) where {dim, T, TM <: StaticMatrix{dim, dim, T}}
    expr = Expr(:tuple)
    for j in 1:dim, i in 1:dim
        push!(expr.args, :(ifelse(m.mask[$i] && m.mask[$j], m.matrix[$i,$j], zero(T))))
    end
    return :($(Expr(:meta, :inline)); $TM($expr))
end
@generated function bcmatrix(m::Symmetric{T, <:ElementMatrix{T, TM}}) where {dim, T, TM <: StaticMatrix{dim, dim, T}}
    expr = Expr(:tuple)
    for j in 1:dim, i in 1:dim
        push!(expr.args, :(ifelse(m.data.mask[$i] && m.data.mask[$j], m.data.matrix[$i,$j], zero(T))))
    end
    return :($(Expr(:meta, :inline)); Symmetric($TM($expr)))
end

struct ElementFEAInfo{dim, T, TKe<:AbstractMatrix{T}, Tfe<:AbstractVector{T}, TKes<:AbstractVector{TKe}, Tfes<:AbstractVector{Tfe}, Tcload<:AbstractVector{T}, refshape, TCV<:CellValues{dim, T, refshape}, dimless1, TFV<:FaceValues{dimless1, T, refshape}, TMeta<:Metadata, TBoolVec<:AbstractVector, TIndVec<:AbstractVector{Int}, TCells}
    Kes::TKes
    fes::Tfes
    fixedload::Tcload
    cellvolumes::Tcload
    cellvalues::TCV
    facevalues::TFV
    metadata::TMeta
    black::TBoolVec
    white::TBoolVec
    varind::TIndVec
    cells::TCells
end
function ElementFEAInfo(sp, quad_order=2, ::Type{Val{mat_type}}=Val{:Static}) where {mat_type} 
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(sp, quad_order, Val{mat_type})
    element_Kes = make_element_Kes(Kes, sp.ch.prescribed_dofs, sp.metadata.dof_cells)
    fixedload = Vector(make_cload(sp))
    assemble_f!(fixedload, sp, dloads)
    cellvolumes = get_cell_volumes(sp, cellvalues)
    cells = sp.ch.dh.grid.cells
    ElementFEAInfo(element_Kes, weights, fixedload, cellvolumes, cellvalues, facevalues, sp.metadata, sp.black, sp.white, sp.varind, cells)
end
function make_element_Kes(Kes::AbstractVector{TMorSymm}, bc_dofs, dof_cells) where {N, T, TM <: StaticMatrix{N, N, T}, TMorSymm <: Union{TM, Symmetric{T, TM}}}
    fill_matrix = zero(TM)
    fill_mask = ones(SVector{N, Bool})
    if TMorSymm <: Symmetric
        element_Kes = fill(Symmetric(ElementMatrix(fill_matrix, fill_mask)), length(Kes))
    else
        element_Kes = fill(ElementMatrix(fill_matrix, fill_mask), length(Kes))
    end
    for i in bc_dofs
        d_cells = dof_cells[i]
        for c in d_cells
            (cellid, localdof) = c
            if TMorSymm <: Symmetric
                Ke = element_Kes[cellid].data
            else
                Ke = element_Kes[cellid]
            end
            new_Ke = @set Ke.mask[localdof] = false
            element_Kes[cellid] = Symmetric(new_Ke)
        end
    end
    for e in 1:length(element_Kes)
        if eltype(element_Kes) <: Symmetric
            Ke = element_Kes[e].data
            matrix = Kes[e].data
            Ke = @set Ke.matrix = matrix
            element_Kes[e] = Symmetric(@set Ke.meandiag = sumdiag(Ke.matrix))
        else
            Ke = element_Kes[e]
            matrix = Kes[e]
            Ke = @set Ke.matrix = matrix
            element_Kes[e] = @set Ke.meandiag = sumdiag(Ke.matrix)
        end
    end
    element_Kes
end

function make_element_Kes(Kes::AbstractVector{TM}, bc_dofs, dof_cells) where {T, TM <: AbstractMatrix{T}, TMorSymm <: Union{TM, Symmetric{T, TM}}}
    N = size(Kes[1], 1)
    fill_matrix = zero(TM)
    fill_mask = ones(Bool, N)
    if TM <: Symmetric
        element_Kes = [Symmetric(deepcopy(ElementMatrix(fill_matrix, fill_mask))) for i in 1:length(Kes)]
    else
        element_Kes = [deepcopy(ElementMatrix(fill_matrix, fill_mask)) for i in 1:length(Kes)]
    end
    for i in bc_dofs
        d_cells = dof_cells[i]
        for c in d_cells
            (cellid, localdof) = c
            if TM <: Symmetric
                Ke = element_Kes[cellid].data
            else
                Ke = element_Kes[cellid]
            end
            Ke.mask[localdof] = false
        end
    end
    element_Kes
end

function get_cell_volumes(sp::StiffnessTopOptProblem{dim, T}, cellvalues) where {dim, T}
    dh = sp.ch.dh
    cellvolumes = zeros(T, getncells(dh.grid))
    for (i, cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues, cell)
        cellvolumes[i] = sum(JuAFEM.getdetJdV(cellvalues, q_point) for q_point in 1:JuAFEM.getnquadpoints(cellvalues))
    end
    return cellvolumes
end

mutable struct GlobalFEAInfo{T, TK<:AbstractMatrix{T}, Tf<:AbstractVector{T}}
    K::TK
    f::Tf
end

GlobalFEAInfo(::Type{T}) where T = GlobalFEAInfo{T}()

GlobalFEAInfo() = GlobalFEAInfo{Float64}()

GlobalFEAInfo{T}() where T = GlobalFEAInfo{T, SparseMatrixCSC{T, Int}, Vector{T}}(sparse(zeros(T, 0, 0)), zeros(T, 0))

GlobalFEAInfo(sp::StiffnessTopOptProblem) = GlobalFEAInfo(make_empty_K(sp), make_empty_f(sp))

make_empty_K(sp::StiffnessTopOptProblem) = Symmetric(create_sparsity_pattern(sp.ch.dh))

make_empty_f(sp::StiffnessTopOptProblem{dim, T}) where {dim, T} = zeros(T, ndofs(sp.ch.dh))

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

    refshape = JuAFEM.getrefshape(dh.field_interpolations[1])

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
    
    Kes, weights = _make_Kes_and_weights(dh, Val{mat_type}, Val{n_basefuncs}, Val{dim*n_basefuncs}, C, ρ, quadrature_rule, cellvalues)
    dloads = _make_dloads(weights, problem, facevalues)

    return Kes, weights, dloads, cellvalues, facevalues
end

const g = [0., 9.81, 0.] # N/kg or m/s^2

function _make_Kes_and_weights(dh::DofHandler{dim, N, T}, ::Type{Val{mat_type}}, ::Type{Val{n_basefuncs}}, ::Type{Val{ndofs_per_cell}}, C, ρ, quadrature_rule, cellvalues) where {dim, N, T, mat_type, n_basefuncs, ndofs_per_cell}
    # Calculate element stiffness matrices
    Kesize = ndofs_per_cell
    nel = getncells(dh.grid)
    body_force = ρ .* g # Force per unit volume
    
    if !(T === BigFloat)
        if mat_type === :Static || mat_type === :SMatrix
            MatrixType = SMatrix{Kesize, Kesize, T, Kesize^2}
            VectorType = SVector{Kesize, T}
        elseif mat_type === :MMatrix
            MatrixType = MMatrix{Kesize, Kesize, T, Kesize^2}
            VectorType = MVector{Kesize, T}
        else
            MatrixType = Matrix{T}
            VectorType = Vector{T}
        end
    else
        if mat_type === :Static || mat_type === :SMatrix  || mat_type === :MMatrix
            MatrixType = SizedMatrix{Kesize, Kesize, T, Kesize^2}
            VectorType = SizedVector{Kesize, T}
        else
            MatrixType = Matrix{T}
            VectorType = Vector{T}
        end
    end

    if MatrixType <: StaticArray
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
    else
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
    end
    return Kes, weights
end

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
    cell_coords = zeros(JuAFEM.Vec{dim, T}, N)
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
