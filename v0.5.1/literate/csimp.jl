# # Continuous SIMP example
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`csimp.ipynb`](@__NBVIEWER_ROOT_URL__/examples/csimp.ipynb)
#-
# ## Commented Program
#
# What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref csimp-plain-program).

using TopOpt

# ### Define the problem
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problems = Any[
    PointLoadCantilever(Val{:Linear}, (60, 20, 20), (1.0, 1.0, 1.0), E, v, f),
    PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f),
    HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
    LBeam(Val{:Linear}, Float64, force = f),
    TieBeam(Val{:Quadratic}, Float64),
]
problem_names =
    ["3d cantilever beam", "cantilever beam", "half MBB beam", "L-beam", "tie-beam"]

i = 2
println(problem_names[i])
problem = problems[i]

# ### Parameter settings
V = 0.5 # volume fraction
xmin = 0.001 # minimum density
rmin = 3.0
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
reuse = true # adaptive penalty flag

convcriteria = Nonconvex.GenericCriteria()
penalty = TopOpt.PowerPenalty(1.0)
pcont = Continuation(penalty, steps = steps, xmin = xmin, pmax = 5.0)

# NOTE: non-convexity + computational error lead to different solutions that satisfy the KKT tolerance
mma_options = options = MMAOptions(maxiter = 1000)
maxtol = 0.01 # maximum tolerance
mintol = 0.0001 # minimum tolerance
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
mma_options_gen = TopOpt.MMAOptionsGen(
    steps = steps,
    initial_options = mma_options,
    ftol_gen = ExponentialContinuation(a, b, 0.0, steps + 1, mintol),
)
csimp_options = TopOpt.CSIMPOptions(
    steps = steps,
    options_gen = mma_options_gen,
    p_gen = pcont,
    reuse = reuse,
)

# ### Define a finite element solver
solver = FEASolver(Direct, problem, xmin = xmin, penalty = penalty)

# ### Define compliance objective
comp = Compliance(problem, solver)
filter = if problem isa TopOptProblems.TieBeam
    identity
else
    DensityFilter(solver, rmin = rmin)
end
obj = x -> comp(filter(x))

# ### Define volume constraint
volfrac = TopOpt.Volume(problem, solver)
constr = x -> volfrac(filter(x)) - V

# ### Define subproblem optimizer
x0 = fill(V, length(solver.vars))
optimizer =
    Optimizer(obj, constr, x0, MMA87(), options = mma_options, convcriteria = convcriteria)

# ### Define continuation SIMP optimizer
simp = SIMP(optimizer, solver, penalty.p)
cont_simp = ContinuationSIMP(simp, steps, csimp_options)

# ### Solve
result = cont_simp(x0)

@show result.convstate
@show optimizer.workspace.iter
@show result.objval

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add(Makie)` first
# ```julia
# using TopOpt.TopOptProblems.Visualization: visualize
# fig = visualize(problem; topology = result.topology,
#     problem; topology = result.topology, default_exagg_scale = 0.07,
#     scale_range = 10.0, vector_linewidth = 3, vector_arrowsize = 0.5,
# )
# Makie.display(fig)
# ```

#md # ## [Plain Program](@id csimp-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [csimp.jl](csimp.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
