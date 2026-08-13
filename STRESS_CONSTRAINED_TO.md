# Stress-constrained topology optimization: literature notes and TopOpt.jl audit

This document collects (a) the standard theory of stress-constrained topology
optimization (TO) needed to work on TopOpt.jl's stress features, (b) an audit
of the `global_stress` tutorial whose original output design was wrong, and
(c) the changes made to fix the library and the tutorials.

References are listed at the end; citations use bracketed keys matching
`docs/biblio/ref.bib` where available.

## 1. Problem statement

The classical formulation (Duysinx & Bendsøe 1998):

    min_ρ  V(ρ) = Σ_e ρ_e v_e
    s.t.   K(ρ) u = f
           σ_vm,e(u) ≤ σ_lim     for every element e
           0 ≤ ρ_e ≤ 1

Two independent difficulties make this much harder than compliance
minimization:

1. **Singular optima** (the *stress singularity problem*). First observed by
   Sved & Ginos (1968) on a three-bar truss: the true optimum requires
   removing a member, but the member's stress constraint is violated
   throughout the removal path, so gradient-based optimizers cannot reach it.
   Kirsch (1989/1990) characterized such optima as lying in degenerate
   subspaces of the feasible domain; see Rozvany & Birker (1994) and Rozvany
   (2001) for continuum discussions.

2. **Locality of stress.** There is one constraint per element (or quadrature
   point) — as many constraints as design variables. The adjoint trick that
   makes compliance cheap (one adjoint solve per *global* response) gives no
   savings, so local stress constraints are expensive; aggregation is the
   standard remedy (§4).

## 2. Microscopic vs. macroscopic stress

With SIMP stiffness interpolation `E(ρ) = ρ^p E_0` (stiffness exponent `p`,
typically 3), two stress notions must be distinguished:

- **Macroscopic (homogenized) stress**: `σ_macro = ρ^p C_0 : ε` — the average
  stress of the effective material. Goes to zero with density; not meaningful
  for failure of the solid phase.
- **Microscopic stress** (Duysinx & Bendsøe 1998): derived from the behavior
  of stress in a porous layered composite. The microscopic stress should be
  (i) inversely related to density and (ii) finite as ρ → 0. Writing
  `σ_mic = ρ^{p-q} C_0 : ε`, consistency (finite limit at ρ = 0) requires
  `q = p`, i.e.

      σ_mic = C_0 : ε        (stress computed with the BASE stiffness)

  This is exactly what `StressTensorFun` / `von_mises_stress_function(solver)`
  compute in TopOpt.jl. It is the physically consistent choice — and it is
  precisely the choice that produces singular optima: at zero density the
  microscopic stress is finite (generically nonzero), so the constraint
  `σ_mic ≤ σ_lim` forbids a continuous path that removes a stressed member.

In practice, under a fixed load and SIMP stiffness, the equilibrium strain of
a thinning member grows as `ε ∝ ρ^{-p}`, so its microscopic stress grows as
`ρ^{-p}`: the constraint bites *harder* the more the member is thinned, and
the optimum of the unrelaxed problem generically has large gray regions the
optimizer cannot remove (Duysinx & Bendsøe 1998).

## 3. Relaxation: making singular optima reachable

Both classical relaxations perturb the feasible domain so that low-density
elements automatically satisfy the constraint.

### 3.1 ε-relaxation (Cheng & Guo 1997; Duysinx & Bendsøe 1998)

Replace the local constraint by

    g_e = ρ_e (σ_e / σ_lim − 1) − ε ≤ 0,     ε > 0.

For sufficiently small ρ_e the constraint is satisfied regardless of the
stress, which opens the degenerate subspaces. As ε → 0 the original problem
is recovered (Cheng & Guo proved convergence of the relaxed optimum to the
true optimum); a continuation strategy decreases ε between runs. Caveat:
Stolpe & Svanberg (2001) showed the global optimum of the relaxed problem can
jump discontinuously as ε → 0, so trajectory-following does not guarantee the
global optimum. Duysinx & Sigmund (1998) use the variant
`ρ(σ/(ρ^n σ_lim) − 1) ≤ ε(1 − ρ)`.

TopOpt.jl: `epsilon_relaxed(σv, ρ, σlim, ε)` evaluates the `g_e` vector
(`src/Functions/stress_relaxation.jl`). Pass the raw (filtered) design
densities, not floored physical densities: `g` is linear in `ρ`, so at
`ρ = 0` the constraint is exactly `−ε` and AD gradients are trivially finite.

### 3.2 The qp-approach / relaxed stress (Bruggi 2008; Le et al. 2010)

Penalize the stress measure with an exponent `q` lower than the stiffness
exponent `p`:

    σ̃_e = ρ_e^{p−q} …  (Verbart's constraint form)   ⟺   relaxed stress
    σ̃_e = ρ_e^{q} σ_mic,e   with q > 0, typically q = 1/2 at p = 3
        (Le et al. 2010's "SIMP-motivated stress definition")

Because `σ̃_e → 0` as `ρ → 0`, void elements always satisfy
`σ̃_e ≤ σ_lim`. The relaxation parameter is `ε_qp = p − q`; Bruggi (2008)
shows ε-relaxation and the qp-approach perturb the feasible domain similarly,
with qp perturbing more smoothly over the whole density range (ε-relaxation
perturbs mainly near ρ = 0). The qp form does not converge to the original
problem as ε_qp → 0 (the true optimum is not in the limit feasible set), but
this is irrelevant in practice: it is used with a fixed, finite ε_qp.

TopOpt.jl: `von_mises_stress_function(solver; stress_exponent=q)` returns
`σ̃_e = ρ_e^q σ_e` with `ρ_e = xmin + (1 − xmin) x_e` (the physical density —
deliberately *not* the penalized stiffness density, so the relaxation is
independent of the SIMP exponent `p` and of the `PENALTY_BEFORE_INTERPOLATION`
preference). The `xmin` floor is required for AD safety: `d/dx[x^q]` diverges
at `x = 0` for fractional `q`, and MMA iterates routinely sit on the lower
bound.

### 3.3 Interaction with the void floor (xmin)

With a density floor `xmin`, "vanishing" elements keep stiffness
`xmin` (penalty-before-interpolation) or `xmin^p` (after), so their strain —
hence microscopic stress — can be large. The relaxation must overpower the
void stress:

- ε-relaxation (floored ρ): void constraint inactive iff
  `σ_void ≤ σ_lim (1 + ε/xmin)` — ε must scale *linearly* with xmin.
  With raw (unfloored) densities there is no such dependence at all.
- qp relaxed stress: void stress suppressed by `xmin^q`, i.e. inactive iff
  `σ_void ≤ σ_lim xmin^{−q}` — q depends on xmin only *logarithmically*
  (xmin: 1e-3 → 1e-6 maps q: 0.5 → 0.25 for the same suppression).

Neither relaxation parameter needs to track the `PENALTY_BEFORE_INTERPOLATION`
choice definitionally; the only coupling is numerical, through the void stress
magnitude.

### 3.4 Unified view (Verbart et al. 2017)

Aggregating a reformulated, design-independent constraint set with a
*lower-bound* aggregation function relaxes and aggregates in one step,
removing the separate relaxation parameter. Not (yet) implemented in
TopOpt.jl.

## 4. Aggregation: making thousands of constraints cheap

Approximate `max_e σ_e` by a smooth function:

- **P-norm** (Duysinx & Sigmund 1998): `σ_PN = (Σ_e σ_e^P)^{1/P} ≥ max σ`,
  approaching the max as P grows (typical P = 6–16). **Always use the
  normalized form** `σ_PN = ( (1/N) Σ σ_e^P )^{1/P}`: the raw norm scales
  with `N^{1/P}` and mixes bulk and peak stress. The normalized form
  *under*estimates the max (`σ_PN ≤ max σ`, equality only for uniform
  stress), so Le et al. (2010) rescale it by an adaptive factor
  `c ← max(σ)/σ_PN` updated every iteration, giving `c·σ_PN ≈ max σ`.
  P-norm gradients are spread over all moderately stressed elements —
  numerically the most robust choice.
- **KS function** (Kreisselmeier & Steinhauser 1979; Yang & Chen 1996):
  `σ_KS = max σ + log( Σ_e exp(γ(σ_e − max σ)) )/γ`, an *upper* bound with
  slack `≤ log(N)/γ` — `σ_KS ≤ σ_lim` certifies `max σ ≤ σ_lim`. No
  normalization factor needed, but the softmax gradient concentrates on the
  few peak elements as γ grows, which destabilizes MMA (bang-bang
  oscillation); moderate γ (10–30) is a practical compromise. Unlike the
  p-norm, KS handles signed values and can therefore aggregate ε-relaxed
  constraint values `g_e` directly.
- **Regional / block aggregation** (París et al. 2009; Le et al. 2010):
  several interlaced regional p-norms instead of one global measure — better
  local control at a few× the adjoint cost. Not implemented here.

## 5. Benchmarks and expected designs

The canonical benchmark is the **L-bracket** (fixed top edge, downward load at
the end of the horizontal arm): the re-entrant corner is a stress
concentrator, and stress-constrained designs characteristically **round the
corner** while compliance-driven designs keep it sharp. MBB and T-junction
are also common. Bruggi & Duysinx (2013+) advocate reference test cases with
fixed geometry/materials for fair comparison.

Reference parameter set (Le et al. 2010): SIMP `p = 3` fixed (no penalty
continuation), relaxed stress with `q = 1/2`, density filter for length
scale, p-norm `P ≈ 8` with adaptive normalization, MMA with a few hundred
iterations.

Two physical notes that matter for reproducibility:

- **Point loads and support edges are singular stress sources.** A point
  force on a continuum gives a stress field that the mesh regularizes to
  `~ f/element_size` — mesh-dependent and topology-independent. Standard
  handling: distribute the load over a few nodes/elements (TopOpt.jl:
  `LBeam(...; load_width=w)`), or exclude the load vicinity from the
  aggregation, or use a passive solid load pad.
- **The gray boundary layer.** With a density filter, boundary elements have
  intermediate ρ; their relaxed stress underreports the true stress by
  `ρ^{-q}` and their microscopic stress is inflated accordingly. Report
  stresses on solid elements only (e.g. ρ > 0.5), as in da Silva et al.
  (2019).

## 6. Audit: why the old `global_stress` tutorial produced a non-standard design

The old tutorial (pre-fix) optimized a 60×20 point-load cantilever:

```julia
threshold = 2 * maximum(stress(filter(PseudoDensities(ones(N)))))
obj    = x -> volfrac(...) + 1e-4 * comp(...)
constr = x -> norm(stress(filter(PseudoDensities(x))), 5) - threshold
```

with a SIMP penalty continuation 1 → 3, 50 MMA iterations per stage. The
deployed CI log showed the mechanism of failure:

- The final design had **volume fraction 1.0 — the full solid block** —
  and the constraint ended **violated** (`constr = +0.041`), while the
  tutorial prose claimed the opposite.
- Root cause: the raw 5-norm over N = 1200 elements is a *bulk-dominated*
  measure (`N^{1/5} ≈ 4.1`); numerically, `‖σ_solid‖₅ ≈ 2.05·max σ_solid`, so
  the threshold `2·max σ_solid` made the constraint **infeasible even for the
  full-material design**. MMA parked all variables at the upper bound for 150
  iterations (KKT residual frozen at ~5e31).
- Secondary issues: no relaxation (unrelaxed microscopic stress, so removing
  material only raises the measure); `p = 5` far below the useful range;
  threshold anchored to the point-load singularity of the solid design;
  compliance regularizer biasing the design; 50 iterations per stage;
  no verification output.

Every one of these is a standard pitfall with a standard fix in the
literature; the fixed tutorial applies them.

## 7. Changes made (this branch)

Library:

- `von_mises_stress_function(solver; stress_exponent=q)` — qp relaxed stress
  (default `q = 0` reproduces the old behavior).
- `epsilon_relaxed(σv, ρ, σlim, ε)` — ε-relaxed constraint values.
- `LBeam(...; load_width=w)` — distribute the load over `w` right-edge nodes
  (also works for `:Quadratic` grids); `getcloaddict` splits the force
  equally (resultant and torque conserved).
- `ext/TopOptMakieExt`: `visualize` now works for problems with facet-based
  Dirichlet BCs (`LBeam`, `TieBeam`) by expanding `FacetIndex` BC sets to
  their nodes when drawing supports.

Tutorials:

- `global_stress.qmd` rewritten around the L-bracket with all three
  approaches: (1) relaxed stress + normalized p-norm + adaptive c, (2)
  relaxed stress + KS, (3) ε-relaxation + KS. All three converge to the
  rounded-corner design; verification numbers are printed.
- `local_stress.qmd`: uses the relaxed stress and a calibrated threshold.

## 8. Remaining gaps / future work

- No regional/block aggregation (Le et al. 2010; París et al. 2009).
- No unified aggregation-relaxation (Verbart et al. 2017).
- ε-relaxation with aggregated constraints converges slowly under MMA;
  augmented-Lagrangian with the summed positive-part measure (Fancello &
  Pereira 2004/2006) is the literature's remedy.
- No stress recovery / nodal averaging option for stress evaluation at jagged
  boundaries (da Silva et al. 2019 discuss the accuracy issue and the
  projection-sharpness remedy).
- A distributed-load option for the other built-in problems
  (`PointLoadCantilever`, `HalfMBB`) would help stress work beyond the
  L-bracket.

## 9. References

- Sved & Ginos (1968). Structural optimization under multiple loading. Int.
  J. Mech. Sci. 10:803–805.
- Kreisselmeier & Steinhauser (1979). Systematic control design by optimizing
  a vector performance index. IFAC Proc. 12(7):113–117.
- Kirsch (1989/1990). On singular topologies in optimum structural design.
- Yang & Chen (1996). Stress-based topology optimization. Struct. Optim.
  12:98–105.
- Cheng & Guo (1997). ε-relaxed approach in structural topology optimization.
  Struct. Optim. 13:258–266.
- Duysinx & Bendsøe (1998). Topology optimization of continuum structures
  with local stress constraints. Int. J. Numer. Meth. Eng. 43:1453–1478.
- Duysinx & Sigmund (1998). New developments in handling stress constraints
  in optimal material distribution. 7th AIAA/USAF/NASA/ISSMo Symp.
- Rozvany & Birker (1994); Rozvany (2001). On singular optima and the
  topology of the feasible domain.
- Stolpe & Svanberg (2001). On the trajectories of the epsilon-relaxation
  approach for stress-constrained truss topology optimization. Struct.
  Multidiscip. Optim. 21:140–151.
- Pereira, Fancello & Barcellos (2004); Fancello & Pereira (2006).
  ε-relaxation with global measures and augmented Lagrangian.
- Bruggi (2008). On an alternative approach to stress constraints relaxation
  in topology optimization. Struct. Multidiscip. Optim. 36:125–141.
- París, Navarrina, Colominas & Casteleiro (2009). Topology optimization of
  continuum structures with local and global stress constraints. Struct.
  Multidiscip. Optim. 39:419–437.
- Le, Norato, Bruns, Ha & Tortorelli (2010). Stress-based topology
  optimization for continua. Struct. Multidiscip. Optim. 41:605–620.
- Holmberg, Torstenfelt & Klarbring (2013). Stress constrained topology
  optimization. Struct. Multidiscip. Optim. 48:33–47.
- Verbart, Langelaar & van Keulen (2017). A unified aggregation and
  relaxation approach for stress-constrained topology optimization. Struct.
  Multidiscip. Optim. 55:663–679.
- da Silva, Beck & Sigmund (2019). Stress-constrained topology optimization
  considering uniform manufacturing uncertainties. Comput. Methods Appl.
  Mech. Eng. 344:512–537.
