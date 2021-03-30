"""
    struct ElementFEAInfo{dim, T}
        Kes::AbstractVector{<:AbstractMatrix{T}}
        fes::AbstractVector{<:AbstractVector{T}}
        fixedload::AbstractVector{T}
        cellvolumes::AbstractVector{T}
        cellvalues::CellValues{dim, T}
        facevalues::FaceValues{<:Any, T}
        metadata::Metadata
        black::AbstractVector
        white::AbstractVector
        varind::AbstractVector{Int}
        cells
    end

An instance of the `ElementFEAInfo` type stores element information such as:
- `Kes`: the element stiffness matrices,
- `fes`: the element load vectors,
- `cellvolumes`: the element volumes,
- `cellvalues` and `facevalues`: two `Ferrite` types that facilitate cell and face iteration and queries.
- `metadata`: that stores degree of freedom (dof) to node mapping, dof to cell mapping, etc.
- `black`: a `BitVector` such that `black[i]` is 1 iff element `i` must be part of any feasible design.
- `white`: a `BitVector` such that `white[i]` is 1 iff element `i` must never be part of any feasible design.
- `varind`: a vector such that `varind[i]` gives the decision variable index of element `i`.
- `cells`: the cell connectivities.
"""
@params struct ElementFEAInfo{dim, T}
    Kes::AbstractVector{<:AbstractMatrix{T}}
    fes::AbstractVector{<:AbstractVector{T}}
    fixedload::AbstractVector{T}
    cellvolumes::AbstractVector{T}
    cellvalues::CellValues{dim, T, <:Any}
    facevalues::FaceValues{<:Any, T, <:Any}
    metadata::Metadata
    black::AbstractVector
    white::AbstractVector
    varind::AbstractVector{Int}
    cells
end

function Base.show(io::Base.IO, ::MIME"text/plain", efeainfo::ElementFEAInfo)
    print(io, "ElementFEAInfo: Kes |$(length(efeainfo.Kes))|, fes |$(length(efeainfo.fes))|, fixedload |$(length(efeainfo.fixedload))|, cells |$(length(efeainfo.cells))|")
end

"""
    ElementFEAInfo(sp, quad_order=2, ::Type{Val{mat_type}}=Val{:Static}) where {mat_type}

Constructs an instance of `ElementFEAInfo` from a stiffness problem `sp` using a Gaussian quadrature order of `quad_order`. The element matrix and vector types will be:
1. `SMatrix` and `SVector` if `mat_type` is `:SMatrix` or `:Static`, the default,
2. `MMatrix` and `MVector` if `mat_type` is `:MMatrix`, or
3. `Matrix` and `Vector` otherwise.

The static matrices and vectors are more performant and GPU-compatible therefore they are used by default.
"""
function ElementFEAInfo(
    sp,
    quad_order = 2,
    ::Type{Val{mat_type}} = Val{:Static},
) where {mat_type} 
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(
        sp,
        quad_order,
        Val{mat_type},
    )
    element_Kes = convert(
        Vector{<:ElementMatrix},
        Kes;
        bc_dofs = sp.ch.prescribed_dofs,
        dof_cells = sp.metadata.dof_cells,
    )
    fixedload = Vector(make_cload(sp))
    assemble_f!(fixedload, sp, dloads)
    cellvolumes = get_cell_volumes(sp, cellvalues)
    cells = sp.ch.dh.grid.cells
    ElementFEAInfo(
        element_Kes,
        weights,
        fixedload,
        cellvolumes,
        cellvalues,
        facevalues,
        sp.metadata,
        sp.black,
        sp.white,
        sp.varind,
        cells,
    )
end

"""
    struct GlobalFEAInfo{T, TK<:AbstractMatrix{T}, Tf<:AbstractVector{T}, Tchol}
        K::TK
        f::Tf
        cholK::Tchol
    end

An instance of `GlobalFEAInfo` hosts the global stiffness matrix `K`, the load vector `f` and the cholesky decomposition of the `K`, `cholK`.
"""
@params mutable struct GlobalFEAInfo{T}
    K::AbstractMatrix{T}
    f::AbstractVector{T}
    cholK
    qrK
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::GlobalFEAInfo) = println("TopOpt global FEA information")

"""
    GlobalFEAInfo(::Type{T}=Float64) where {T}

Constructs an empty instance of `GlobalFEAInfo` where the field `K` is an empty sparse matrix of element type `T` and the field `f` is an empty dense vector of element type `T`.
"""
GlobalFEAInfo(::Type{T}=Float64) where {T} = GlobalFEAInfo{T}()
function GlobalFEAInfo{T}() where {T}
    return GlobalFEAInfo(sparse(zeros(T, 0, 0)), zeros(T, 0), cholesky(one(T)), qr(one(T)))
end

"""
    GlobalFEAInfo(sp::StiffnessTopOptProblem)

Constructs an instance of `GlobalFEAInfo` where the field `K` is a sparse matrix with the correct size and sparsity pattern for the problem instance `sp`. The field `f` is a dense vector of the appropriate size. The values in `K` and `f` are meaningless though and require calling the function `assemble!` to update.
"""
function GlobalFEAInfo(sp::StiffnessTopOptProblem)
    K = initialize_K(sp)
    f = initialize_f(sp)
    return GlobalFEAInfo(K, f)
end
function GlobalFEAInfo(
    K::Union{AbstractSparseMatrix, Symmetric{<:Any, <:AbstractSparseMatrix}},
    f,
)
    chol = cholesky(spdiagm(0=>ones(size(K, 1))))
    qrfact = qr(spdiagm(0=>ones(size(K, 1))))
    return GlobalFEAInfo{eltype(K), typeof(K), typeof(f), typeof(chol), typeof(qrfact)}(K, f, chol, qrfact)
end

"""
    GlobalFEAInfo(K, f)

Constructs an instance of `GlobalFEAInfo` with global stiffness matrix `K` and load vector `f`.
"""
function GlobalFEAInfo(K, f)
    chol = cholesky(Matrix{eltype(K)}(I, size(K)...))
    qrfact = qr(Matrix{eltype(K)}(I, size(K)...))
    return GlobalFEAInfo(K, f, chol, qrfact)
end

"""
    get_cell_volumes(sp::StiffnessTopOptProblem{dim, T}, cellvalues)

Calculates an approximation of the element volumes by approximating the volume integral of 1 over each element using Gaussian quadrature. `cellvalues` is a `Ferrite` struct that facilitates the computation of the integral. To initialize `cellvalues` for an element with index `cell`, `Ferrite.reinit!(cellvalues, cell)` can be called. Calling `Ferrite.getdetJdV(cellvalues, q_point)` then computes the value of the determinant of the Jacobian of the geometric basis functions at the point `q_point` in the reference element. The sum of such values for all integration points is the volume approximation.
"""
function get_cell_volumes(sp::StiffnessTopOptProblem{dim, T}, cellvalues) where {dim, T}
    dh = sp.ch.dh
    cellvolumes = zeros(T, getncells(dh.grid))
    for (i, cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues, cell)
        cellvolumes[i] = sum(Ferrite.getdetJdV(cellvalues, q_point) for q_point in 1:Ferrite.getnquadpoints(cellvalues))
    end
    return cellvolumes
end
