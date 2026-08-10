# # Neural-network parametrized topology optimization example
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`neural.ipynb`](@__NBVIEWER_ROOT_URL__/examples/neural.ipynb)
#-
# ## Commented Program
#
# Instead of optimizing the density of each cell directly, this example
# parametrizes the design with a small feed-forward neural network
# (`Flux.jl`). The network maps cell centroids to a density, so the design
# field is implicit in the network weights. We optimize the weights under a
# volume constraint using a sequence of IPOPT subproblems: first a
# feasibility restoration step, then an augmented-Lagrangian-style loop that
# increases the penalty on constraint violation.
#md # The full program, without comments, can be found in the next [section].

using TopOpt, Zygote, ChainRulesCore
using Flux
Nonconvex.@load Ipopt

# ### Define the problem
E = 1.0 # Young's modulus
v = 0.3 # Poisson's ratio
f = 1.0 # downward force

problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f)

V = 0.5
xmin = 0.0001
rmin = 4.0
p = 3.0

solver = FEASolver(DirectSolver, problem; xmin)
nvars = length(solver.vars)
x0 = fill(V, nvars)

penalty = TopOpt.PowerPenalty(p)
solver = FEASolver(DirectSolver, problem; xmin, penalty)
filter = DensityFilter(solver; rmin)
comp = Compliance(solver)
volfrac = Volume(solver)

# ### Neural-network parametrization
# A small MLP maps the 2-D cell centroid to a scalar density (sigmoid output).
m = 20
act = leakyrelu
nn = Chain(Dense(2, m, act), Dense(m, m, act), Dense(m, m, act), Dense(m, 1, sigmoid))
nn_model = NeuralNetwork(nn, problem)
tf = TrainFunction(nn_model)
p0 = nn_model.init_params
tf(p0)

obj = p -> comp(filter(tf(p)))
constr = p -> volfrac(filter(tf(p))) - V

# ### Feasibility restoration
# Start with a feasibility problem: minimize `(constr + 0.1)²` to drive the
# design toward the volume constraint before optimizing compliance.
alg = IpoptAlg()
options = IpoptOptions(; max_iter=20)
model1 = Model()
nparams = length(p0)
addvar!(model1, fill(-100.0, nparams), fill(100.0, nparams))
set_objective!(model1, p -> (constr(p) + 0.1)^2)
res1 = optimize(model1, alg, p0; options)

# ### Augmented-Lagrangian-style refinement
# Each iteration raises `μ` and adds a log-barrier on the constraint, then
# re-optimizes from the previous solution.
μ = 1.0
res2 = res1
for _ in 1:3
    global μ *= 2
    global res2
    model2 = Model()
    addvar!(model2, fill(-100.0, nparams), fill(100.0, nparams))
    set_objective!(model2, p -> μ * obj(p) - log(max(0, -constr(p))))
    options = IpoptOptions(; max_iter=20)
    res2 = optimize(model2, alg, res2.minimizer; options)
    @show extrema(tf(res2.minimizer))
    @show obj(res2.minimizer)
    @show constr(res2.minimizer)
end

# ### Sanity-check the gradients
Zygote.gradient(obj, p0)
Zygote.gradient(constr, p0)

tf(res2.minimizer)

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add("Makie")` first and either `Pkg.add("CairoMakie")` or `Pkg.add("GLMakie")`
using Makie
using CairoMakie
# alternatively, `using GLMakie`
topology = filter(tf(res2.minimizer))
visualize(problem; topology)

#md # ## Plain Program
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [neural.jl](neural.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```