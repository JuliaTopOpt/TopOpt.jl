# # SIMP example: Point Load Cantilever 
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`simp.ipynb`](@__NBVIEWER_ROOT_URL__/examples/simp.ipynb)
#-
# ## Commented Program
#
# What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref simp-plain-program).

using TopOpt

# ### Define the problem
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0; # downward force

nels = (30, 10, 10)
problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0, 1.0), E, v, f);

# See also the detailed API of `PointLoadCantilever`:
#md # ```@docs
#md # TopOpt.TopOptProblems.PointLoadCantilever
#md # ```

# ### Parameter settings
V = 0.3 # volume fraction
xmin = 1e-6 # minimum density
rmin = 2.0; # density filter radius

# ### Define a finite element solver
penalty = TopOpt.PowerPenalty(3.0)
solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)

# ### Define compliance objective
comp = TopOpt.Compliance(solver)
filter = DensityFilter(solver; rmin=rmin)
obj = x -> comp(filter(PseudoDensities(x)))

# ### Define volume constraint
volfrac = TopOpt.Volume(solver)
constr = x -> volfrac(filter(PseudoDensities(x))) - V

# ### Define subproblem optimizer
x0 = fill(V, length(solver.vars))
model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr)
alg = MMA87()
convcriteria = Nonconvex.KKTCriteria()
options = MMAOptions(;
    maxiter=3000, tol=Nonconvex.Tolerance(; x=1e-3, f=1e-3, kkt=0.001), convcriteria
)
r = optimize(model, alg, x0; options)

@show obj(r.minimizer)

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add("Makie")` first and either `Pkg.add("CairoMakie")` or `Pkg.add("GLMakie")`
using Makie
using CairoMakie
# alternatively, `using GLMakie`
fig = visualize(
    problem;
    topology=r.minimizer,
    default_exagg_scale=0.07,
    scale_range=10.0,
    vector_linewidth=3,
    vector_arrowsize=0.5,
)
Makie.display(fig)

# or convert it to a Mesh
# Need to run `using Pkg; Pkg.add(GeometryBasics)` first
using Makie, GeometryBasics
result_mesh = GeometryBasics.Mesh(problem, r.minimizer);
Makie.mesh(result_mesh)

#md # ## [Plain Program](@id simp-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [simp.jl](simp.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
