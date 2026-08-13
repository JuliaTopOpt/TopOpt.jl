# Relaxation helpers for stress-constrained topology optimization.
# Differentiable (AD-safe via Zygote) and intended to wrap the element-wise
# stress vector returned by `von_mises_stress_function`. The qp-approach
# relaxation is available through the `stress_exponent` keyword of
# `von_mises_stress_function`.

"""
    epsilon_relaxed(σv, ρ, σlim, ε)

Element-wise ε-relaxed stress constraint values (each value must be `≤ 0`):

    g_e = ρ_e (σ_e / σ_lim - 1) - ε

with element von Mises stresses `σv`, densities `ρ` (pass the raw filtered
design variables, *without* the `xmin` floor: `g` is linear in `ρ`, so at
`ρ = 0` the constraint equals `-ε` exactly and the relaxation is independent
of the void floor; the floor still enters through the computed stress field),
stress limit `σlim`, and relaxation parameter `ε > 0`. Multiplying the
constraint residual by `ρ_e`
widens the degenerate subspaces of the feasible domain: for sufficiently small
`ρ_e` the constraint is satisfied regardless of the stress, which makes the
singular optima of stress-constrained topology optimization [SvedGinos1968](@cite)
reachable by gradient-based optimizers [ChengGuo1997](@cite) and
[DuysinxBendsøe1998](@cite). As `ε → 0` the original constraints
`σ_e ≤ σ_lim` on the material domain are recovered; a continuation strategy
decreasing `ε` between optimization runs is common, though the global optimum
of the relaxed problem can jump discontinuously with `ε`
[StolpeSvanberg2001b](@cite). The qp-approach [Bruggi2008](@cite) is the
alternative relaxation, available through the `stress_exponent` keyword of
[`von_mises_stress_function`](@ref). The returned values are signed, so
aggregate them with a signed-safe smooth-max such as the Kreisselmeier–
Steinhauser function (`logsumexp(γ .* g) / γ`), not a p-norm.
"""
epsilon_relaxed(σv, ρ, σlim, ε) = @. ρ * (σv / σlim - 1) - ε
