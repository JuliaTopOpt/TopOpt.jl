# # Heat conduction (conductivity tree) example
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`heat_tree.ipynb`](@__NBVIEWER_ROOT_URL__/examples/heat_tree.ipynb)
#-
# ## Commented Program
#
# This example minimizes the thermal compliance of a heat-conduction problem:
# distributed heat flux enters the top edge, the bottom edge is held at
# `T = 0`, and the sides are insulated. The optimal layout is the classic
# branching "conductivity tree" (Bendsøe & Sigmund, *Topology Optimization*,
# §1.3).
#md # The full program, without comments, can be found in the next [section](@ref heat_tree-plain-program).

using TopOpt

# ### Define the problem
# `HeatTree` sets up the standard benchmark: flux `q` on the top boundary,
# `T = 0` on the bottom, free left/right.
nels = (80, 40)
k = 1.0
q = 1.0
problem = HeatTree(Val{:Linear}, nels, (1.0, 1.0), k; q=q)

# ### Parameter settings
V = 0.3      # volume fraction
xmin = 1e-3  # minimum density
rmin = 2.0

x0 = fill(V, TopOpt.getncells(problem))
solver = FEASolver(DirectSolver, problem; xmin=xmin, penalty=PowerPenalty(3.0))
comp = ThermalCompliance(solver)
filter = DensityFilter(solver; rmin=rmin)
obj = x -> comp(filter(PseudoDensities(x)))
# Define volume constraint
volfrac = Volume(solver)
constr = x -> volfrac(filter(PseudoDensities(x))) - V
model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr)
alg = MMA87()

# ### Optimize
options = MMAOptions(; tol=Tolerance(; kkt=1e-4), maxiter=200)
res = optimize(model, alg, x0; options)

@show obj(res.minimizer)
@show constr(res.minimizer)

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add("Makie")` first and either `Pkg.add("CairoMakie")` or `Pkg.add("GLMakie")`
using Makie
using CairoMakie
# alternatively, `using GLMakie`
fig = visualize(problem; topology=res.minimizer)
Makie.display(fig)

#md # ## [Plain Program](@id heat_tree-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [heat_tree.jl](heat_tree.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```