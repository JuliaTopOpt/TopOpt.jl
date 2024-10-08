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
    LBeam(Val{:Linear}, Float64; force=f),
    TieBeam(Val{:Quadratic}, Float64),
]
problem_names = [
    "3d cantilever beam", "cantilever beam", "half MBB beam", "L-beam", "tie-beam"
]

i = 2
println(problem_names[i])
problem = problems[i]

# ### Parameter settings
V = 0.5 # volume fraction
xmin = 0.001 # minimum density
rmin = 3.0

convcriteria = Nonconvex.KKTCriteria()
x0 = fill(V, TopOpt.getncells(problem))
penalty = TopOpt.PowerPenalty(1.0)
solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
comp = Compliance(solver)
filter = if problem isa TopOptProblems.TieBeam
    identity
else
    DensityFilter(solver; rmin=rmin)
end
obj = x -> comp(filter(PseudoDensities(x)))
# Define volume constraint
volfrac = Volume(solver)
constr = x -> volfrac(filter(PseudoDensities(x))) - V
model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr)
alg = MMA87()

nsteps = 4
ps = range(1.0, 5.0; length=nsteps + 1)
# exponentially decaying tolerance from 10^-2 to 10^-4
tols = exp10.(range(-2, -4; length=nsteps + 1))
x = x0
for j in 1:(nsteps + 1)
    global convcriteria
    p = ps[j]
    tol = tols[j]
    TopOpt.setpenalty!(solver, p)
    options = MMAOptions(; tol=Tolerance(; kkt=tol), maxiter=1000, convcriteria)
    res = optimize(model, alg, x; options)
    global x = res.minimizer
end

@show obj(x)
@show constr(x)

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add("Makie")` first and either `Pkg.add("CairoMakie")` or `Pkg.add("GLMakie")`
using Makie
using CairoMakie
# alternatively, `using GLMakie`
fig = visualize(
    problem;
    topology=x,
    default_exagg_scale=0.07,
    scale_range=10.0,
    vector_linewidth=3,
    vector_arrowsize=0.5,
)
Makie.display(fig)

#md # ## [Plain Program](@id csimp-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [csimp.jl](csimp.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
