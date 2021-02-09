# # PointLoadCantilever
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`point_load_cantilever.ipynb`](@__NBVIEWER_ROOT_URL__/examples/point_load_cantilever.ipynb)
#-
# ## Commented Program
#
# Now we solve the problem in JuAFEM. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref point-load-cantilever-plain-program).

using TopOpt, Makie
import GeometryBasics

# ### Define the problem
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0; # downward force

nels = (40, 20, 20) 
problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0, 1.0), E, v, f);

# ### Parameter settings
V = 0.3 # volume fraction
xmin = 0.001 # minimum density
rmin = 4.0; # density filter radius

# ### Define a finite element solver
penalty = TopOpt.PowerPenalty(3.0)
solver = FEASolver(Displacement, Direct, problem, xmin = xmin,
    penalty = penalty);

# ### Define compliance objective
obj = Objective(TopOpt.Compliance(problem, solver, filterT = DensityFilter,
    rmin = rmin, tracing = true, logarithm = false));

# ### Define volume constraint
constr = Constraint(TopOpt.Volume(problem, solver, filterT = DensityFilter, rmin = rmin), V);

# ### Define subproblem optimizer
mma_options = options = MMA.Options(maxiter = 3000, 
    tol = MMA.Tolerances(kkttol = 0.001))
convcriteria = MMA.KKTCriteria()
optimizer = MMAOptimizer(obj, constr, MMA.MMA87(),
    ConjugateGradient(), options = mma_options,
    convcriteria = convcriteria);

# ### Define SIMP optimizer
simp = SIMP(optimizer, penalty.p);

# ### Solve
x0 = fill(1.0, length(solver.vars))
result = simp(x0);

# ### Visualize the result using Makie.jl
using TopOpt.TopOptProblems.Visualization: visualize
fig = visualize(problem; topology=result.topology, 
    default_exagg_scale=0.07, scale_range=10.0, vector_linewidth=3, vector_arrowsize=0.5)
Makie.display(fig)
# or convert it to a Mesh
# result_mesh = GeometryBasics.Mesh(problem, result.topology);
# Makie.mesh(result_mesh)

#md # ## [Plain Program](@id point-load-cantilever-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [point_load_cantilever.jl](point_load_cantilever.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```