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
- `cellvalues` and `facevalues`: two `JuAFEM` types that facilitate cell and face iteration and queries.
- `metadata`: that stores degree of freedom (dof) to node mapping, dof to cell mapping, etc.
- `black`: a `BitVector` such that `black[i]` is 1 iff element `i` must be part of any feasible design.
- `white`: a `BitVector` such that `white[i]` is 1 iff element `i` must never be part of any feasible design.
- `varind`: a vector such that `varind[i]` gives the decision variable index of element `i`.
- `cells`: the cell connectivities,
"""
@params struct ElementFEAInfo{dim, T}
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

mutable struct GlobalFEAInfo{T, TK<:AbstractMatrix{T}, Tf<:AbstractVector{T}, Tchol}
    K::TK
    f::Tf
    cholK::Tchol
end
GlobalFEAInfo(::Type{T}) where {T} = GlobalFEAInfo{T}()
GlobalFEAInfo() = GlobalFEAInfo{Float64}()
function GlobalFEAInfo{T}() where {T}
    return GlobalFEAInfo(sparse(zeros(T, 0, 0)), zeros(T, 0), cholesky(one(T)))
end
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
    return GlobalFEAInfo{eltype(K), typeof(K), typeof(f), typeof(chol)}(K, f, chol)
end
function GlobalFEAInfo(K, f)
    chol = cholesky(Matrix{eltype(K)}(I, size(K)...))
    return GlobalFEAInfo{eltype(K), typeof(K), typeof(f), typeof(chol)}(K, f, chol)
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
