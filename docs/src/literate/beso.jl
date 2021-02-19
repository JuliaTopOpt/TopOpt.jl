# # BESO optimization example: HalfMBB Beam 
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`point_load_cantilever.ipynb`](@__NBVIEWER_ROOT_URL__/examples/point_load_cantilever.ipynb)
#-
# ## Commented Program
#
# Now we solve the problem in JuAFEM. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref beso-plain-program).

using TopOpt

# ### Define the problem
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0; # downward force

nels = (10, 10)
problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), E, v, f)

# ### Define the FEA Solver and penalty functions
solver = FEASolver(Displacement, Direct, problem, xmin = 0.01, penalty = TopOpt.PowerPenalty(3.0))

# ### Define the compliance objective function and volume fraction constraint
comp = Compliance(problem, solver)
volfrac = Volume(problem, solver)
sensfilter = SensFilter(solver, rmin = 4.0)
beso = BESO(comp, IneqConstraint(volfrac, 0.5), sensfilter)
x0 = ones(length(solver.vars))

# ### Run optimization
resul = beso(x0)

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add(Makie)` first
# using TopOpt.TopOptProblems.Visualization: visualize
# fig = visualize(
#     problem; topology = result.topology, default_exagg_scale = 0.07,
#     scale_range = 10.0, vector_linewidth = 3, vector_arrowsize = 0.5,
# )
# Makie.display(fig)

#md # ## [Plain Program](@id beso-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [point_load_cantilever.jl](point_load_cantilever.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```