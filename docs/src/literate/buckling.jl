# # Buckling-constrained topology optimization
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`buckling.ipynb`](@__NBVIEWER_ROOT_URL__/examples/buckling.ipynb)
#-
# ## Commented Program
#
# This example demonstrates buckling-constrained topology optimization on a
# truss structure. The goal is to minimize compliance while ensuring that the
# combined stiffness matrix `K + c·Kσ` stays positive semidefinite — i.e. the
# structure does not buckle under the applied load multiplied by the buckling
# load factor `c`.
#
# The building blocks used here are the differentiable functions `ElementK`,
# `AssembleK`, `TrussElementKσ`, `apply_boundary_with_zerodiag!`, and
# `apply_boundary_with_meandiag!`, which are documented in the
# [Functions](@ref) page.
#md # The full program, without comments, can be found in the next [section](@id buckling-plain-program).

using TopOpt, LinearAlgebra, Zygote
using Nonconvex, NonconvexSemidefinite

# ### Load the truss geometry from JSON
# We use the same 2-D truss geometry as the mixed-integer example. The JSON file
# defines node coordinates, element connectivity, supports, and load cases.
ndim = 2
node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
    joinpath(@__DIR__, "..", "data", "tim_$(ndim)d.json")
)
ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
loads = load_cases["0"]
problem = TrussProblem(
    Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
)

# ### Parameter settings
xmin = 0.0001   # minimum density
p = 1.0         # SIMP penalty exponent
V = 0.5         # maximum volume fraction
x0 = fill(1.0, ncells)  # initial design (all solid)

# ### FEA solver and buckling helpers
# We need five differentiable building blocks:
#
# 1. `Compliance` — the objective to minimize.
# 2. `Displacement` — solves the FEA system to get the nodal displacements `u`.
# 3. `ElementK` — computes per-element stiffness matrices from the design.
# 4. `AssembleK` — assembles element matrices into a global sparse matrix.
# 5. `TrussElementKσ` — computes per-element geometric (stress) stiffness
#    matrices from the displacements and the design.
#
# Together, `K + c·Kσ` must remain PSD, which is enforced as a
# semidefinite constraint.
solver = FEASolver(DirectSolver, problem; xmin=xmin)
ch = problem.ch

comp = TopOpt.Compliance(solver)
dp = TopOpt.Displacement(solver)
assemble_k = TopOpt.AssembleK(problem)
element_k = ElementK(solver)
truss_element_kσ = TrussElementKσ(problem, solver)

# ### Buckling constraint
# The buckling constraint constructs the combined matrix `K + c·Kσ` and
# returns it as a dense array for the SDP solver. `c` is the buckling load
# multiplier — the factor by which the applied load is scaled before checking
# stability.
c = 1.0

function buckling_matrix_constr(x)
    xd = PseudoDensities(x)
    # Solve for displacements under the current design
    u = dp(xd)
    # Element stiffness matrices -> global stiffness matrix
    Kes = element_k(xd)
    K = assemble_k(Kes)
    K = apply_boundary_with_meandiag!(K, ch)
    # Element geometric stiffness matrices -> global Kσ
    Kσs = truss_element_kσ(u, xd)
    Kσ = assemble_k(Kσs)
    Kσ = apply_boundary_with_zerodiag!(Kσ, ch)
    return Array(K + c * Kσ)
end

# Sanity-check: the initial (all-solid) design should be stable.
S0 = buckling_matrix_constr(x0)
println("Initial design minimum eigenvalue: $(minimum(eigen(S0).values))")

# ### Volume constraint
vol_constr(x) = sum(x) / length(x) - V

# ### Optimization
# First, minimize compliance without the buckling constraint to get a
# baseline design.
obj = x -> comp(PseudoDensities(x))

model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, vol_constr)

alg = MMA87()
options = MMAOptions(; maxiter=100, tol=Tolerance(; kkt=1e-4))
r1 = optimize(model, alg, x0; options)
println("Compliance-only result: obj=$(obj(r1.minimizer)), vol=$(vol_constr(r1.minimizer))")

# Now add the semidefinite buckling constraint and re-optimize. The
# `SDPBarrierAlg` uses an interior-point sub-solver (here IPOPT) to handle the
# PSD constraint via a log-barrier.
Nonconvex.@load Ipopt
add_sd_constraint!(model, buckling_matrix_constr)
alg2 = SDPBarrierAlg(; sub_alg=IpoptAlg())
options2 = SDPBarrierOptions(; sub_options=IpoptOptions(; max_iter=200), keep_all=true)
r2 = optimize(model, alg2, x0; options=options2)
println("Buckling-constrained result: obj=$(obj(r2.minimizer)), vol=$(vol_constr(r2.minimizer))")

# ### Check stability
# Compare the minimum eigenvalue of the combined stiffness matrix before and
# after the buckling-constrained optimization.
S1 = buckling_matrix_constr(r1.minimizer)
S2 = buckling_matrix_constr(r2.minimizer)
ev1 = eigen(S1).values
ev2 = eigen(S2).values
println("Compliance-only min eigenvalue: $(minimum(ev1))")
println("Buckling-constrained min eigenvalue: $(minimum(ev2))")

#md # ## [Plain Program](@id buckling-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [buckling.jl](buckling.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```