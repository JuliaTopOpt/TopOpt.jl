# # Global stress objective example
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`global_stress.ipynb`](@__NBVIEWER_ROOT_URL__/examples/global_stress.ipynb)
#-
# ## Commented Program
#
# What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref global-stress-plain-program).

using TopOpt, LinearAlgebra

# ### Define the problem
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 3.0 # filter radius

problems = Any[
    PointLoadCantilever(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
    HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
]
problem_names = ["Cantilever beam", "Half MBB beam", "L-beam", "Tie-beam"]

i = 1
println(problem_names[i])
problem = problems[i]

# ### Parameter settings
V = 0.5 # volume fraction
xmin = 0.001 # minimum density
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
convcriteria = Nonconvex.KKTCriteria()
penalty = TopOpt.PowerPenalty(1.0)

# ### Define a finite element solver
solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)

# ### Define **stress** objective
# Notice that gradient is derived automatically by automatic differentiation (Zygote.jl)!
stress = TopOpt.von_mises_stress_function(solver)
filter = if problem isa TopOptProblems.TieBeam
    identity
else
    DensityFilter(solver; rmin=rmin)
end
volfrac = TopOpt.Volume(problem, solver)

x0 = ones(length(solver.vars))
threshold = 2 * maximum(stress(filter(x0)))

obj = x -> volfrac(filter(x))
constr = x -> norm(stress(filter(x)), 5) - threshold
options = MMAOptions(; maxiter=2000, tol=Nonconvex.Tolerance(; kkt=1e-4))

# ### Define subproblem optimizer
x0 = fill(0.5, length(solver.vars))
optimizer = Optimizer(obj, constr, x0, MMA87(); options=options, convcriteria=convcriteria)

# ### Define continuation SIMP optimizer
simp = SIMP(optimizer, solver, 3.0)

# ### Solve
result = simp(x0)

@show result.convstate
@show optimizer.workspace.iter
@show result.objval

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add(Makie)` first
# ```julia
# using TopOpt.TopOptProblems.Visualization: visualize
# fig = visualize(problem; topology = result.topology)
# Makie.display(fig)
# ```

#md # ## [Plain Program](@id global-stress-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [global-stress.jl](global_stress.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
