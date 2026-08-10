# # Mixed-integer truss topology optimization example
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`mixed_integer_truss.ipynb`](@__NBVIEWER_ROOT_URL__/examples/mixed_integer_truss.ipynb)
#-
# ## Commented Program
#
# This example solves a truss compliance-minimization problem in which
# every cell's density is restricted to be exactly `0` or `1` (a binary
# design). The mixed-integer nonlinear program is handled by
# [Juniper.jl](https://github.com/lanl-ANS/Juniper.jl) with IPOPT as the
# continuous relaxation solver, accessed through `Nonconvex.jl`.
#md # The full program, without comments, can be found in the next [section].

using TopOpt, LinearAlgebra, StatsFuns
using Makie
using CairoMakie
# alternatively, `using GLMakie`

Nonconvex.@load Juniper

# ### Load the truss geometry from JSON
# The same `tim_2d.json` file used by the problem types page
# defines the node coordinates, element connectivity, supports, and load
# cases for a 2-D truss.
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
xmin = 0.0001 # minimum density
x0 = fill(1.0, ncells) # initial design
p = 1.0 # penalty
V = 0.5 # maximum volume fraction

# ### FEA solver and compliance objective
solver = FEASolver(DirectSolver, problem; xmin=xmin)
comp = TopOpt.Compliance(solver)

function obj(x)
    # minimize compliance
    return comp(PseudoDensities(x))
end
function constr(x)
    # volume fraction constraint
    return sum(x) / length(x) - V
end

# ### Optimization with integer variables
# `integer=trues(length(x0))` forces every density to be binary.
m = Model(obj)
addvar!(m, zeros(length(x0)), ones(length(x0)); integer=trues(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = JuniperIpoptOptions()
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(m, JuniperIpoptAlg(), x0; options=options)

# ### Results
@show obj(r.minimizer)
@show constr(r.minimizer)

# ### (Optional) Visualize the result using Makie.jl
fig = visualize(problem; solver.u, topology=r.minimizer, default_exagg_scale=0.0)
Makie.display(fig)

#md # ## Plain Program
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [mixed_integer_truss.jl](mixed_integer_truss.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```