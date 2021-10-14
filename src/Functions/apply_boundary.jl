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
    for i in 1:length(ch.values)
        d = ch.prescribed_dofs[i]
        Kσ[d, d] = zero(T)
    end
    return Kσ
end

"""
g(F(K)), F: K1 -> K2

dg/dK1_ij = dg/dK2_i'j' * dK2_i'j'/dK1_ij
          = Delta[i',j'] * dK2_i'j'/dK1_ij

dK2_i'j'/dK1_ij = 0, if i' or j' in ch.prescribed_dofs
                = 1, otherwise

dg/dK1_ij = 0, if i or j in ch.prescribed_dofs
          = Delta[i,j], otherwise
"""
function ChainRulesCore.rrule(::typeof(apply_boundary_with_zerodiag!), Kσ, ch)
    project_to = ChainRulesCore.ProjectTo(Kσ)
    function pullback_fn(Δ)
        return NoTangent(), apply_boundary_with_zerodiag!(project_to(Δ), ch) , NoTangent()
    end
    return apply_boundary_with_zerodiag!(Kσ, ch), pullback_fn
end

########################################

"""
    apply_boundary_with_meandiag!(K, ch)

Apply boundary condition to a matrix. Zero-out the corresponding [i,:] and [:,j] with
i, j ∈ ch.prescribed_dofs, then fill in K[i,i] for i ∈ ch.prescribed_dofs with the
mean diagonal of the original matrix.
"""
function apply_boundary_with_meandiag!(K::Union{SparseMatrixCSC,Symmetric}, ch::ConstraintHandler)
    Ferrite.apply!(K, eltype(K)[], ch, false)
    return K
end

"""
g(F(K)), F: K1 -> K2

dg/dK1_ij = dg/dK2_i'j' * dK2_i'j'/dK1_ij
          = Delta[i',j'] * dK2_i'j'/dK1_ij

dK2_i'j'/dK1_ij = 0, if i' or j' in ch.prescribed_dofs and i' != j'
                = d(meandiag(K1))/dK1_ij, if i' or j' in ch.prescribed_dofs and i' == j'
                = 1, otherwise

We have to compute d(meandiag(K1))/dK1_ii, i in ch.prescribed_dofs.

dg/dK1_ij = 0, if i or j in ch.prescribed_dofs and i' != j'
          = Delta[i,j] * d(meandiag(K1))/dK1_ij, if i or j in ch.prescribed_dofs and i' == j'
          = Delta[i,j], otherwise
"""
function ChainRulesCore.rrule(::typeof(apply_boundary_with_meandiag!), K, ch)
    project_to = ChainRulesCore.ProjectTo(K)
    function pullback_fn(Δ)
        ΔK = project_to(Δ)
        apply_boundary_with_meandiag!(ΔK, ch)
        # d(meandiag(K1))/dK1_ii
        meandiag_vec_fn = x -> sum(abs.(x))/length(x)
        jac_meandiag = ForwardDiff.derivative(meandiag_vec_fn, diag(K))
        for i in 1:length(ch.values)
            d = ch.prescribed_dofs[i]
            ΔK[d, d] = Δ[d,d] * jac_meandiag[d]
        end
        return NoTangent(), ΔK, NoTangent()
    end
    return apply_boundary_with_meandiag!(K, ch), pullback_fn
end

########################################

# TODO apply!(K, f, ch) if needed