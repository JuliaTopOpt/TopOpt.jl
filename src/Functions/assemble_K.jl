mutable struct AssembleK{
    T,Tp<:StiffnessTopOptProblem,TK<:AbstractMatrix{T},Tg<:AbstractVector{<:Integer}
} <: AbstractFunction{T}
    problem::Tp
    K::TK
    global_dofs::Tg # preallocated dof vector for a cell
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::AssembleK)
    return println("TopOpt global linear stiffness matrix assembly function")
end

function AssembleK(problem::StiffnessTopOptProblem)
    dh = problem.ch.dh
    k = ndofs_per_cell(dh)
    global_dofs = zeros(Int, k)
    return AssembleK(problem, initialize_K(problem), global_dofs)
end

"""
    assembleK(Kes) = (Σ_ei x_e * K_ei)

Forward-pass function call.
"""
function (ak::AssembleK{T})(Kes::AbstractVector{<:AbstractMatrix{T}}) where {T}
    @unpack problem, K, global_dofs = ak
    dh = problem.ch.dh
    if K isa Symmetric
        K.data.nzval .= 0
        assembler = Ferrite.AssemblerSparsityPattern(K.data, T[], Int[], Int[])
    else
        K.nzval .= 0
        assembler = Ferrite.AssemblerSparsityPattern(K, T[], Int[], Int[])
    end
    Ke = zeros(T, size(Kes[1]))
    TK = eltype(Kes)
    for (i, _) in enumerate(CellIterator(dh))
        celldofs!(global_dofs, dh, i)
        Ke = TK isa Symmetric ? Kes[i].data : Kes[i]
        Ferrite.assemble!(assembler, global_dofs, Ke)
    end
    return copy(K)
end

"""
    ChainRulesCore.rrule(ak::AssembleK{T}, Kes)

`rrule` for autodiff. 
    
Let's consider composite function `g(F(...))`, where
`F` can be a struct-valued, vector-valued, or matrix-valued function.
In the case here, `F = AssembleK`. Then `rrule` wants us to find `g`'s derivative
w.r.t each *output* of `F`, given `g`'s derivative w.r.t. each *input* of `F`.
Here, `F: K_e -> K_ij = sum_e K_e_ij`. Then `dK_ij/dK_e_ij = 1`.
And we know `Delta_ij = dg/dK_ij`.

And our goal for `rrule` is to go from `dg/dKij` to `dg/dK_e_ij`, 
which has the same structure as the input `K_e`.

    dg/dK_e_ij = sum_i'j' dg/dK_i'j' dK_i'j'/dK_e_ij
    		   = dg/dK_ij dK_ij/dK_e_ij
    		   # i, j above are global indices
    		   # i, j below are local indices
    		   = Delta[global_dofs[i], global_dofs[j]]

    (dK_i'j'/dK_e_ij = 0 unless i' == i, j' == j, (i, j) in e)

which can be shortened as:

    dg/dK_e = Delta[global_dofs, global_dofs]
"""
function ChainRulesCore.rrule(
    ak::AssembleK{T}, Kes::AbstractVector{<:AbstractMatrix{T}}
) where {T}
    @unpack problem, K, global_dofs = ak
    dh = problem.ch.dh
    # * forward-pass
    K = ak(Kes)
    n_dofs = length(global_dofs)
    function assembleK_pullback(Δ)
        ΔKes = [zeros(T, n_dofs, n_dofs) for _ in 1:getncells(dh.grid)]
        for (ci, _) in enumerate(CellIterator(dh))
            celldofs!(global_dofs, dh, ci)
            ΔKes[ci] = Δ[global_dofs, global_dofs]
        end
        return Tangent{typeof(ak)}(; problem=NoTangent(), K=Δ, global_dofs=NoTangent()),
        ΔKes
    end
    return K, assembleK_pullback
end
