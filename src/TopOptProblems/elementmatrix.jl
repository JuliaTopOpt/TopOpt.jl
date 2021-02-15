"""
    struct ElementMatrix{T, TM <: AbstractMatrix{T}} <: AbstractMatrix{T}
        matrix::TM
        mask
        meandiag::T
    end

An element stiffness matrix. `matrix` is the unconstrained element stiffness matrix. `mask` is a `BitVector` where `mask[i]` is 1 iff the local degree of freedom `i` is not constrained by a Dirichlet boundary condition. `meandiag` is the mean of the diagonal of the unconstrained element stiffness matrix.
"""
@params struct ElementMatrix{T, TM <: AbstractMatrix{T}} <: AbstractMatrix{T}
    matrix::TM
    mask
    meandiag::T
end
ElementMatrix(matrix, mask) = ElementMatrix(matrix, mask, sumdiag(matrix)/size(matrix, 1))
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::ElementMatrix) = println("TopOpt element matrix")

"""
    rawmatrix(m::ElementMatrix)

Returns the unconstrained element stiffness matrix `m.matrix`.
"""
rawmatrix(m::ElementMatrix) = m.matrix
rawmatrix(m::Symmetric{T, <:ElementMatrix{T}}) where {T} = Symmetric(m.data.matrix)

"""
    bcmatrix(m::ElementMatrix{T, TM}) where {dim, T, TM <: StaticMatrix{dim, dim, T}}

Returns the constrained element stiffness matrix where the elements in the rows and columns corresponding to any local degree of freedom with a Dirichlet boundary condition are replaced by 0.
"""
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

Base.size(m::ElementMatrix) = size(m.matrix)
Base.getindex(m::ElementMatrix, i...) = m.matrix[i...]

"""
    convert(::Type{Vector{<:ElementMatrix}}, Kes::Vector{<:AbstractMatrix})

Converts the element stiffness matrices `Kes` from an abstract vector of matrices to a vector of instances of the type `ElementMatrix`.
"""
function Base.convert(
    ::Type{Vector{<:ElementMatrix}},
    Kes::Vector{TM};
    bc_dofs,
    dof_cells,
) where {
    N, T, TM <: StaticMatrix{N, N, T},
}
    fill_matrix = zero(TM)
    fill_mask = ones(SVector{N, Bool})
    element_Kes = fill(ElementMatrix(fill_matrix, fill_mask), length(Kes))
    for i in bc_dofs
        d_cells = dof_cells[i]
        for c in d_cells
            (cellid, localdof) = c
            Ke = element_Kes[cellid]
            new_Ke = @set Ke.mask[localdof] = false
            element_Kes[cellid] = Symmetric(new_Ke)
        end
    end
    for e in 1:length(element_Kes)
        Ke = element_Kes[e]
        matrix = Kes[e]
        Ke = @set Ke.matrix = matrix
        element_Kes[e] = @set Ke.meandiag = sumdiag(Ke.matrix)
    end
    return element_Kes
end
function Base.convert(
    ::Type{Vector{<:ElementMatrix}},
    Kes::Vector{Symmetric{T, TM}};
    bc_dofs,
    dof_cells,
) where {
    N, T, TM <: StaticMatrix{N, N, T},
}
    fill_matrix = zero(TM)
    fill_mask = ones(SVector{N, Bool})
    element_Kes = fill(Symmetric(ElementMatrix(fill_matrix, fill_mask)), length(Kes))
    for i in bc_dofs
        d_cells = dof_cells[i]
        for c in d_cells
            (cellid, localdof) = c
            Ke = element_Kes[cellid].data
            new_Ke = @set Ke.mask[localdof] = false
            element_Kes[cellid] = Symmetric(new_Ke)
        end
    end
    for e in 1:length(element_Kes)
        Ke = element_Kes[e].data
        matrix = Kes[e].data
        Ke = @set Ke.matrix = matrix
        element_Kes[e] = Symmetric(@set Ke.meandiag = sumdiag(Ke.matrix))
    end
    return element_Kes
end
function Base.convert(
    ::Type{Vector{<:ElementMatrix}},
    Kes::Vector{TM};
    bc_dofs,
    dof_cells,
) where {
    T, TM <: AbstractMatrix{T},
}
    N = size(Kes[1], 1)
    fill_matrix = zero(TM)
    fill_mask = ones(Bool, N)
    element_Kes = [deepcopy(ElementMatrix(fill_matrix, fill_mask)) for i in 1:length(Kes)]
    for i in bc_dofs
        d_cells = dof_cells[i]
        for c in d_cells
            (cellid, localdof) = c
            Ke = element_Kes[cellid]
            Ke.mask[localdof] = false
        end
    end
    return element_Kes
end
function Base.convert(
    ::Type{Vector{<:ElementMatrix}},
    Kes::Vector{Symmetric{T, TM}};
    bc_dofs,
    dof_cells,
) where {
    T, TM <: AbstractMatrix{T},
}
    N = size(Kes[1], 1)
    fill_matrix = zero(TM)
    fill_mask = ones(Bool, N)
    element_Kes = [
        Symmetric(deepcopy(ElementMatrix(fill_matrix, fill_mask))) for i in 1:length(Kes)
    ]
    for i in bc_dofs
        d_cells = dof_cells[i]
        for c in d_cells
            (cellid, localdof) = c
            Ke = element_Kes[cellid].data
            Ke.mask[localdof] = false
        end
    end
    return element_Kes
end

for TM in (:(StaticMatrix{m, m, T}), :(Symmetric{T, <:StaticMatrix{m, m, T}}))
    @eval begin
        @generated function sumdiag(K::$TM) where {m,T}
            return reduce((ex1,ex2) -> :($ex1 + $ex2), [:(K[$j,$j]) for j in 1:m])
        end
    end
end
@doc """
sumdiag(K::Union{StaticMatrix, Symmetric{<:Any, <:StaticMatrix}})

Computes the sum of the diagonal of the static matrix `K`.
""" sumdiag
