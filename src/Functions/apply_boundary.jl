"""
    apply_boundary_with_zerodiag!(Kσ, ch)

Apply boundary condition to a matrix. Zero-out the corresponding [i,:] and [:,j] with
i, j ∈ ch.prescribed_dofs.

This function is typically used with the stress stiffness matrix `Kσ`. More info about this can be found at: 
https://github.com/JuliaTopOpt/TopOpt.jl/wiki/Applying-boundary-conditions-to-the-stress-stiffness-matrix
"""
function apply_boundary_with_zerodiag!(Kσ, ch)
    T = eltype(Kσ)
    Ferrite.apply!(Kσ, T[], ch, true)
    for i = 1:length(ch.values)
        d = ch.prescribed_dofs[i]
        Kσ[d, d] = zero(T)
    end
    return Kσ
end

function ChainRulesCore.rrule(::typeof(apply_boundary_with_zerodiag!), Kσ, ch)
    project_to = ChainRulesCore.ProjectTo(Kσ)
    function pullback_fn(Δ)
        return NoTangent(), apply_boundary_with_zerodiag!(project_to(Δ), ch), NoTangent()
    end
    return apply_boundary_with_zerodiag!(Kσ, ch), pullback_fn
end
"""
Derivations for `rrule` of `apply_boundary_with_zerodiag!`

g(F(K)), F: K1 -> K2

dg/dK1_ij = dg/dK2_i'j' * dK2_i'j'/dK1_ij
          = Delta[i',j'] * dK2_i'j'/dK1_ij

dK2_i'j'/dK1_ij = 0, if i' or j' in ch.prescribed_dofs
                = 1, otherwise

dg/dK1_ij = 0, if i or j in ch.prescribed_dofs
          = Delta[i,j], otherwise
"""

########################################

"""
    apply_boundary_with_meandiag!(K, ch)

Apply boundary condition to a matrix. Zero-out the corresponding [i,:] and [:,j] with
i, j ∈ ch.prescribed_dofs, then fill in K[i,i] for i ∈ ch.prescribed_dofs with the
mean diagonal of the original matrix.
"""
function apply_boundary_with_meandiag!(
    K::Union{SparseMatrixCSC,Symmetric},
    ch::ConstraintHandler,
)
    Ferrite.apply!(K, eltype(K)[], ch, false)
    return K
end

function ChainRulesCore.rrule(::typeof(apply_boundary_with_meandiag!), K, ch)
    project_to = ChainRulesCore.ProjectTo(K)
    diagK = diag(K)
    jac_meandiag = sign.(diagK) / length(diagK)
    function pullback_fn(Δ)
        Δ_ch_diagsum = zero(eltype(K))
        for i = 1:length(ch.values)
            d = ch.prescribed_dofs[i]
            Δ_ch_diagsum += Δ[d, d]
        end
        ΔK = project_to(Δ)
        apply_boundary_with_zerodiag!(ΔK, ch)
        for i = 1:size(K, 1)
            ΔK[i, i] += Δ_ch_diagsum * jac_meandiag[i]
        end
        return NoTangent(), ΔK, NoTangent()
    end
    return apply_boundary_with_meandiag!(K, ch), pullback_fn
end
"""
Derivations for `rrule` of `apply_boundary_with_meandiag!`
g(F(K)), F: K1 -> K2

dg/dK1_ij = sum_i'j' dg/dK2_i'j' * dK2_i'j'/dK1_ij
          = sum_i'j' Delta[i',j'] * dK2_i'j'/dK1_ij

If i' != j' and (i' or j' in ch.prescribed_dofs)
    dK2_i'j'/dK1_ij = 0
If i' != j' and !(i' or j' in ch.prescribed_dofs)
    If i' == i and j' == j
        dK2_i'j'/dK1_ij = 1
    If i' != i or j' != j
        dK2_i'j'/dK1_ij = 0
If i' == j' and !(i' or j' in ch.prescribed_dofs)
    If i' == i and j' == j (# and i == j)
        # includes the case i == j and !(i in ch.prescribed_dofs)
        dK2_i'j'/dK1_ij = 1
    If i' != i or j' != j
        dK2_i'j'/dK1_ij = 0
If i' == j' and (i' or j' in ch.prescribed_dofs)
    If i == j
        # includes the case i == j and !(i in ch.prescribed_dofs)
        dK2_i'j'/dK1_ij = d(meandiag(K1))/dK1_ii
    If i != j
        dK2_i'j'/dK1_ij = 0

We have to compute d(meandiag(K1))/dK1_ii, i in ch.prescribed_dofs.
    d(meandiag(K1))/dK1_ii = sign(K1_ii) / size(K1, 1)

If i != j and i or j in ch.prescribed_dofs
    dg/dK1_ij = 0
If i != j and !(i or j in ch.prescribed_dofs)
    dg/dK1_ij = Delta[i, j]
If i == j and i in prescribed_dofs
    dg/dK1_ii = 
        sum_{i' in prescribed_dofs} Delta[i',i'] * d(meandiag(K1))/dK1_ii
If i == j and !(i in prescribed_dofs)
    dg/dK1_ii = Delta[i, i] + 
        sum_{i' in prescribed_dofs} Delta[i',i'] * d(meandiag(K1))/dK1_ii
"""

########################################

# TODO apply!(K, f, ch) if needed
